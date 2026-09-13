PUCK PEAK - TECHNICAL HANDOVER DOCUMENT

For the next AI or human who has to touch this thing at 2am.
Entry point: `app.py`. Most logic lives in `nhl/`.
This is the active long-form handover doc. Exact cache migration details live in
`foundation_phase.md`; the pre-foundation writeup in
`docs/archive/pre_foundation_architecture_overview.md` is historical only.

SECTION 1 - ARCHITECTURE OVERVIEW
---------------------------------
This is a modular Streamlit app.
There is no backend, no database, and no auth.
Streamlit reruns `app.py` top-to-bottom on every interaction, so persistent state lives in
`st.session_state`.

Expensive local artifacts still live behind `@st.cache_data`, but NHL HTTP is no longer a
simple "`@st.cache_data` only" story. The normal API-backed read path is:

`public loader/helper -> @st.cache_data -> NHLClient.get() -> shared NHLCache -> HTTP`

`NHLCache` lives in `nhl/cache.py` and uses on-disk `diskcache` at `.cache/nhl_api` when
available, with an in-process dict fallback if `diskcache` is not installed.

The app has four data sources/artifacts:
- LIVE: NHL public APIs
- LOCAL: `nhl_historical_seasons.parquet`
- LOCAL: `win_prob_weights.json`
- LOCAL RUNTIME CACHE: `.cache/nhl_api` shared disk cache when `diskcache` is available

The parquet file powers KNN projection, historical baselines, and Season Snapshot age-rarity
comparisons. Without it, the app still renders real data, but projection, baseline, and rarity
features degrade.

The rarity layer now depends on additive parquet columns `Shots` and `TotalTOIMins` so clicked
season snapshots can support `SH%`, `TOI`, percentile ranking, and top-age leaderboards without
changing the old projection/baseline columns or row semantics.

The JSON weight artifact powers two things: pregame win probability in the right-rail
predictions panel, and the simulated Stanley Cup odds on the Current Standings board. The
Streamlit app never trains that model at runtime. It loads frozen weights, rebuilds team
ratings from league-wide game data, scores upcoming matchups and simulates the rest of the
season. See SECTION 10A.

Those prediction cards are now clickable matchup-context surfaces. A normal click opens a
`Matchup History` modal with the last 10 head-to-head meetings, a plain-text win summary,
and stacked matchup cards. The primary trigger is a small JS bridge mounted through
`st.components.v2.component()`; the old `mh=AWY,HOME` query-param path stays as fallback.

`app.py` is the session-state coordinator and render pass. It:
- attempts to start the optional process-local background cache warmer early through `start_background_warmer()`; this is a no-op unless the env flag enables it
- loads URL params once
- seeds session state
- auto-loads a live, recent, or next-scheduled game once per session when appropriate
- runs the older session-local `async_preloader.py` warm-up once per session for non-active categories
- renders sidebar and controls
- dispatches to `process_players()` or `process_teams()`
- renders the chart-column `Chart season` picker, the right-rail predictions area, and the comparison panel

`app.py` runs a three-phase render pass: a slot phase that reserves three `st.container()` slots before any pipeline call (fixing the page layout order without painting any placeholder content), a fetch phase that runs `process_players()` / `process_teams()` synchronously, and a mount phase that renders into the same slot through the `@st.fragment`-wrapped helpers in `nhl/fragments.py`. No shimmer skeletons are pre-painted, because those flashed on every full rerun. The slots MUST stay `st.container()` and must not go back to `st.empty()`: an Empty delta is not inert, the frontend renders it as a bare `<div data-testid="stEmpty">`, so re-emitting one each run took the node at that path from Block back to Empty, React tore the subtree down, and the chart / tabs / right rail sat blank for the whole pipeline before popping back at mount. (An older version of this file claimed an empty slot produces no delta until it is filled. It does, and that blanking was the "the page redraws itself on every click" bug fixed in v1.01.6 - measured at 3 Empty deltas per run before, 0 after.) A Block delta re-sent at the same path reconciles instead, so the previous content stays on screen until the new content replaces it. The fragments exist so post-load widget interactions only rerun the chart, detail tabs, or predictions panel they sit in.

SECTION 2 - FILE STRUCTURE
--------------------------
Top level:
- `app.py` - session-state orchestrator and render pass
- `foundation_phase.md` - authoritative cache/foundation notes and migration details
- `cache_strategy.md` - design rationale for the `NHLClient` / shared-cache direction
- `scraper.py` - manual historical parquet refresh
- `train_win_prob.py` - offline trainer AND backtester for the win-probability model and the season simulator; writes the artifact only if its gates pass. `--phase2-report` re-runs the xG / goalie evaluation (report only, never writes the artifact; see SECTION 10A)
- `.cache/xg_training/` - trainer-only local cache of parsed play-by-play shots (gitignored, ~14 MB for 2017-18..2025-26); not part of the runtime NHLCache
- `nhl_historical_seasons.parquet` - historical seasons used by baselines and KNN
- `win_prob_weights.json` - version-2 model artifact: logistic-regression weights over Elo / shrunk form / back-to-back features, rating hyperparameters, overtime model, simulator noise, the `goal_model` block behind the 60-minute and puck-line markets (present only when its gate passed), and the full backtest report
- `.cache/nhl_api/` - shared runtime disk cache directory when `diskcache` is installed
- `docs/archive/pre_foundation_architecture_overview.md` - archived pre-foundation architecture note; keep it historical
- `requirements.txt` - FULLY PINNED on purpose; see the dependency rule below

DEPENDENCY RULE - do not loosen these pins
------------------------------------------
`docker compose build` resolves fresh wheels on every build, so an unpinned entry means
production runs versions nobody tested. That is exactly how plotly 7.0.0 reached production
while 6.5.2 was tested locally.

The load-bearing one is plotly. Streamlit renders `st.plotly_chart` with its OWN bundled
plotly.js (3.3.1 for streamlit 1.54.0, inside `streamlit/static/static/js/`) and only
serializes the figure on the Python side. plotly 6.5.2 targets plotly.js 3.3.1 - the matched
pair. plotly 7.x targets 4.0.0 and regenerates its validators and colour handling against that
schema. Before bumping plotly, compare `plotly.offline.get_plotlyjs_version()` against the
version string in Streamlit's bundle.

`nhl/` modules:
- `__init__.py` - package index docstring
- `constants.py` - shared constants and metric sets
- `styles.py` - stylesheet delivery and UI asset helpers.  The sheet itself is
  `assets/puckpeak.css`, handed to streamlit's media file manager and pulled into the page
  with a one-line `@import`; only a ~1.4 KB critical block is inlined per run
- `era.py` - era multipliers and historical adjustment helpers
- `cache.py` - shared cache backend, TTL tiers, and `effective_ttl()` helpers
- `api.py` - central `NHLClient` with rate limiting, retry, deduplication, and cache-aware HTTP
- `data_loaders.py` - app-facing parquet loaders and NHL data wrappers built on the cache/client layer
- `rarity.py` - historical age-rarity ranking, role splits, and top-season leaderboard payloads
- `baselines.py` - historical and team baseline builders
- `knn_engine.py` - KNN projection and fallback logic.  `run_knn_projection()` is memoized on
  a fingerprint of its inputs (bounded LRU, 256 entries): it runs once per player on every
  rerun and was the dominant CPU cost of an interaction.  Plain dicts rather than
  `@st.cache_data`, same reason as `_CAREER_YEARS_MEMO` - this module has no Streamlit
  dependency, and its DataFrame args are not cheaply hashable by Streamlit's hasher.  The
  career frame is fingerprinted by hashing its actual CELL VALUES, not a summary: a summary
  would collide across genuinely different careers and the memo would serve a wrong
  projection, which is worse than the recompute it saves.  Results are deep-copied on the
  way out because callers append `proj_rows` into frames and stash `clone_names`.
- `team_ratings.py` - league game table (label from `wins`, never goals), Elo, shrunk team form, back-to-back flags; shared by the trainer, the runtime and the simulator
- `win_prob.py` - model contract: feature names, artifact validation, scalar / vectorized / decomposed scoring, overtime probability
- `season_sim.py` - Monte Carlo season + playoff simulator (division / wild-card seeding, exact best-of-7) and the NHL-payload adapters feeding it
- `goal_model.py` - corrected-Poisson score distribution (tie inflation, empty-net transfers) solved to match the win probability; prices the 60-minute result and puck line; numpy only
- `ledger.py` - prediction ledger (SQLite in `PUCKPEAK_DATA_DIR`): pregame rows frozen at puck drop, grading, free NHL partner-odds parsing and margin removal, live track-record metrics
- `xg.py` - OFFLINE RESEARCH ONLY (the app never imports it): play-by-play shot parsing, logistic expected-goals scoring, per-game xG / goalie summaries, point-in-time goalie ratings and projected starters, used by `train_win_prob.py --phase2-report`
- `player_pipeline.py` - full player processing path
- `team_pipeline.py` - team processing path
- `controls.py` - top controls expander
- `sidebar.py` - sidebar UI and add/remove flows
- `dialog.py` - chart click dialogs and matchup-history modal
- `chart.py` - Plotly render, baseline overlay, share link, native point-click dispatch, and dialog routing
- `comparison.py` - Overview / Current Standings tabs, the chart-season picker renderer, clickable predictions panel, and live standings board markup
- `fragments.py` - `@st.fragment` wrappers around `render_chart`, `render_detail_tabs`, `render_predictions_panel`, and the sidebar FAQ button so widget interactions stay scoped
- `ui_state.py` - session-state helpers plus the one-slot dialog mutex (`begin_script_run()`,
  `begin_dialog_run(scope)`); see the DIALOG SLOT rules in SECTION 4
- `stanley_cup.py` - current-standings board builder: merges standings with the season projection (odds / preseason / champion / standings-only modes)
- `url_params.py` - compact share-link encode/decode with legacy-link sanitization and canonicalization
- `schedule.py` - live defaults (live > finished > soonest upcoming, preseason included), upcoming games, featured players, matchup-history loading, runtime win-prob inference, the shared team-ratings snapshot, and the cached season projection
- `cache_warmer.py` - optional process-local daemon warmer for shared-cache live / seasonal / historical paths
- `async_preloader.py` - older session-local additive preloader for non-active categories inside the current worker

SECTION 3 - EXTERNAL API ENDPOINTS
----------------------------------
Search:
`https://search.d3.nhle.com/api/v1/search/player`

Player landing payload:
`https://api-web.nhle.com/v1/player/{player_id}/landing`

Player game log payload:
`https://api-web.nhle.com/v1/player/{player_id}/game-log/{season_id}/{game_type_id}`

Season skater summary:
`https://api.nhle.com/stats/rest/en/skater/summary`

Season goalie summary:
`https://api.nhle.com/stats/rest/en/goalie/summary`

Roster:
`https://api-web.nhle.com/v1/roster/{team_abbr}/current`

Scoreboard now:
`https://api-web.nhle.com/v1/scoreboard/now`

Scores by date:
`https://api-web.nhle.com/v1/score/{date}`

Club stats:
`https://api-web.nhle.com/v1/club-stats/{team_abbr}/now`

Team summary:
`https://api.nhle.com/stats/rest/en/team/summary`
(with `isGame=true` it returns one row per team per game - league-wide in one request per season/game type)

Team 5v5 shot attempts (per game with `isGame=true`):
`https://api.nhle.com/stats/rest/en/team/summaryshooting`

League schedule (every game of a season, one request; gameType 9 exhibition rows must be filtered out):
`https://api.nhle.com/stats/rest/en/game?cayenneExp=season={season_id} and gameType>=2`

Playoff bracket and live series state (year = calendar year the playoffs end; `series` is empty before the playoffs):
`https://api-web.nhle.com/v1/playoff-bracket/{year}`

Standings metadata and dated standings (trainer only, for historical division membership):
- `https://api-web.nhle.com/v1/standings-season`
- `https://api-web.nhle.com/v1/standings/{date}`

Betting-partner odds (cache warmer only, for the ledger's market benchmark; never displayed):
`https://api-web.nhle.com/v1/partner-game/{country}/now` - games of the next odds date with moneyline,
60-minute 3-way, puck line and totals. CA/US partners quote American odds, SE/FI decimal odds.

Play-by-play (trainer `--phase2-report` only; ~300 KB per game, never fetched by the app):
`https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play`
Unblocked shots carry x/y, shotType, goalieInNetId and situationCode. `situationCode` digits are
[away goalie in net][away skaters][home skaters][home goalie in net] (verified against official goal
strengths; a pulled goalie's sixth skater is an extra attacker, scored as even strength).
`homeTeamDefendingSide` is missing in 2017-18, so the attacking net is inferred from offensive-zone
shots. A blocked-shot event is owned by the BLOCKING team.

Records APIs:
- `https://records.nhl.com/site/api/skater-career-scoring-regular-season`
- `https://records.nhl.com/site/api/skater-career-scoring-playoff`
- `https://records.nhl.com/site/api/goalie-career-stats`
- `https://records.nhl.com/site/api/goalie-career-playoff-stats`

All are undocumented. All are wrapped in try/except. Keep the fallbacks.

SECTION 3A - PUBLIC INTERFACES
------------------------------
Stable external surfaces for this repo:
- Streamlit entrypoint: `streamlit run app.py`
- Compact share-link query params handled by `url_params.py`:
  `cat`, `sk_m`, `go_m`, `tm_m`, `sp`, `cs`, `xm`, `lg`, `sm`, `pr`, `era`,
  `cu`, `bl`, `pf`, `pt_s`, `pt_g`, `pt_t`, `pl`, `tm`
- Legacy shared-link params remain supported for backward compatibility:
  `sk`, `go`, and `mh`

Behavior contract:
- outbound share links stay compact and ID-based (`pl` player IDs, `tm` team abbreviations)
- inbound legacy `id|name` / `abbr|name` values are sanitized at ingest
- known player IDs and team abbreviations are canonicalized to trusted display names before render

Non-public surfaces:
- request / loader helpers inside `data_loaders.py`, `schedule.py`, and related modules are
  internal implementation details, not stable public APIs

SECTION 4 - SESSION STATE
-------------------------
`app.py` seeds these persistent keys up front:

Core state:
- `players`
- `teams`
- `stat_category`
- `season_type`
- `x_axis_mode`
- `chart_season`
- `league_filter`

Toggles:
- `do_smooth`
- `do_predict`
- `do_era`
- `do_cumul_toggle`
- `do_base`
- `do_prime`

Comparison tab memory:
- `panel_tab_skater`
- `panel_tab_goalie`
- `panel_tab_team`

One-shot guards:
- `_url_loaded`
- `_default_loaded`
- `_preloaded`
- `_dialog_opened_this_run`   (see "DIALOG SLOT" below - do not set this by hand)
- `_dialog_run_token`         (bumped once per full script run)
- `_dialog_scope_token_*`     (one per fragment: chart, detail_tabs, predictions)
- `_pending_matchup_history`
- `_last_matchup_history_trigger_nonce`
- `_last_identity_card_trigger_nonce`
- `_last_handled_chart_click_nonce`

DIALOG SLOT - read this before changing anything that opens a dialog
--------------------------------------------------------------------
Only one `st.dialog` may open per rerun, so `nhl/ui_state.py` keeps a one-slot mutex in
`_dialog_opened_this_run`. Six call sites take the slot with `mark_dialog_opened_this_run()`,
and as of v1.01.5 that is genuinely all six - the sidebar FAQ button used to open
`show_app_guide()` with no gate at all, which happened to be safe only because it always fired
on a fresh full run. Releasing the slot is the part that is easy to get wrong.

Rules:
- `app.py` calls `begin_script_run()` once per full script run. That clears the slot and bumps
  `_dialog_run_token`.
- Every `@st.fragment` wrapper in `nhl/fragments.py` calls `begin_dialog_run("<scope>")` at its
  top. That clears the slot ONLY when this scope has already run under the current token - i.e.
  only on a genuine fragment-scoped rerun.
- Never assign `_dialog_opened_this_run` directly, in app code or in tests.

Why it is shaped that way (both failure modes are real, one shipped):
- Resetting only in `app.py` is what caused the outage: a fragment rerun never re-executes
  top-level `app.py`, and the Player Details / Team Details / Matchup History dialogs use the
  default `on_dismiss="ignore"` so closing them reruns nothing either. The flag latched True and
  every later dialog was silently swallowed - chart clicks and player cards alike.
- Resetting unconditionally in every fragment is wrong the other way: on a full run the
  fragments execute in sequence, so a later one would clear a reservation an earlier one had
  already taken and two dialogs would try to open in one run.
The run-token check is what satisfies both. `tests/test_dialog_run_scope.py` pins the behaviour.

Related gate-ordering rule: check the gates BEFORE recording a click nonce as handled, or a
blocked click is destroyed instead of retried. One deliberate exception - in
`_show_chart_dialog_from_trigger()` a click blocked by `suppress_dialogs` DOES burn its nonce,
because that is intentional arbitration (a matchup dialog is taking the run) and the bridge's
trigger value survives reruns; leaving it unburned would pop the chart dialog open on the next
rerun out of nowhere. A merely busy slot is transient and must not consume the click.

Season-mode memory:
- `_pre_season_chart_x_axis_mode`
- `_pre_season_league_filter`
- `_pre_season_do_era`

Notes:
- URL params are applied once, before defaults settle.
- Shared links can carry compact player IDs and team abbreviations.
- Legacy shared-link display names are sanitized at ingest, and known players / teams are
  canonicalized before the sidebar renders them.
- Missing URL params leave defaults alone.
- Extra widget keys like metric selectors are created later by Streamlit widgets.
- `do_cumul` is derived per render, not stored.
- Leaving selected-season mode restores the saved x-axis, league filter, and era toggle through
  `_restore_pre_season_state()`, including the invalid-season fallback that resets
  `chart_season` to `All`.

SECTION 5 - STRICT SKATER / GOALIE SPLIT
----------------------------------------
This rule matters more than your feelings:

`is_goalie = raw_df['Saves'].sum() > 0 or raw_df['Wins'].sum() > 0`

That check runs on the raw player frame before category-specific charting.
If the app is in Skater mode, goalies are skipped.
If the app is in Goalie mode, skaters are skipped.
Never cross-plot them.

SECTION 6 - DATA PIPELINE (PER PLAYER, PER RENDER)
--------------------------------------------------
`process_players()` runs this order:

1. Fetch raw player data.
   - Normal mode uses `get_player_raw_stats()`.
   - Selected-season mode uses `get_player_season_game_log()` and real NHL game logs.
   - Overview cards can also call cached season-summary leaderboards for league rank text.
2. Apply the skater/goalie gatekeeper.
3. Filter to the selected leagues.
4. If `Era` is on for skaters, apply NHLe to non-NHL `Points`, `Goals`, and `Assists`. If `Era` is off, keep league scoring raw.
5. Filter to `Regular`, `Playoffs`, or `Both`.
6. Apply era adjustment.
7. Branch by chart mode.
   - Selected-season mode: keep one row per real game, sort by `GameDate` + `GameId`, build `CumGP`, preserve `Age` plus exact game metadata (`GameId`, `GameDate`, teams, home/road) for clicks and peak copy.
   - Career Games Played mode: group by `SeasonYear`, build `CumGP`, keep `Age` for clicks.
   - Age mode: group by `Age`, preserve latest `SeasonYear`, compute rate stats.
8. Detect peak before projection, cumsum, and smoothing.
9. If allowed, project to age 40.
   - KNN path uses `run_knn_projection()`.
   - Fallback path uses `run_linear_fallback()`.
   - `TOI` is the exception: it is now KNN-only for skaters and never uses the linear fallback.
   - Current in-progress seasons are pace-adjusted inside the KNN step before clone matching.
10. Apply cumulative mode in Age mode only.
11. Apply 3-season rolling smoothing when enabled.
12. Split real vs projected traces.

Projection gate:
- not in Games Played mode
- not in selected-season mode
- `do_predict` is on
- max age is below 40
- metric is not in `NO_PROJECTION_METRICS`
- thin-data guard passes minimum seasons and GP thresholds
- `TOI` also requires modern TOI-bearing history: at least 3 usable `1997+` seasons with nonzero
  `TotalTOIMins`, at least 120 GP across those seasons, and a usable TOI row at the player's latest age

Split behavior:
- real trace uses `Age <= max_age`
- projection trace uses `Age >= max_age`
- the last real point is duplicated into the projected trace for visual continuity

SECTION 7 - HYBRID KNN PROJECTION ENGINE
----------------------------------------
Only runs in Age mode.

KNN rules that matter:
- distance metric is L1
- clone pool is top 10
- clone weights are equal
- clone/prior blend is fixed at 80/20
- percent-change clamp is `[-0.12, +0.25]`
- GP is intentionally excluded from KNN

Matching flow:
1. Filter historical rows by position where possible.
2. Pivot by `PlayerID x Age`.
3. Use mean for rate stats and sum for counting stats.
4. Keep only ages shared by the live player and the historical pivot.
5. Rank players by vectorized L1 distance.
6. Project future ages by mapping clone movement onto the live player.

Projection behavior:
- additive-delta path is used for `+/-`, `GAA`, and `Save %`
- multiplicative path is used for most counting stats
- sparse ages use non-zero fallback targets
- late ages use stabilization instead of trusting tiny clone pools
- stat caps apply every projected year
- `GAA` uses a floor, not a ceiling

GP note:
- the engine still contains a 4-phase durability fallback for GP
- normal app flow suppresses GP and SH% projection via `NO_PROJECTION_METRICS = {'GP', 'SH%'}`
- `TOI` is now handled through a separate KNN-only projection policy instead of the blanket suppression list

TOI note:
- historical TOI projection coverage starts in `1997`
- the historical TOI KNN pool ignores zero-TOI rows and requires at least 40 GP in a season
- goalie TOI is still out of scope because the parquet does not carry usable goalie TOI history

SECTION 8 - BASELINE ENGINE
---------------------------
Source: `nhl_historical_seasons.parquet`

Historical baseline pools:
- skaters require at least 40 GP in a season
- goalies require at least 20 GP in a season

Families:
- `Skater`
- `Goalie`

Construction:
1. Take the 75th percentile by age.
2. Smooth with a centered rolling window.
3. Shape late tails so sparse old-age noise does not create fake rebounds.

Tail rules:
- after age 31, rising skater and counting-stat curves use the old `prev * 0.92` guard
- goalie `Save %` and `GAA` do not use that multiplicative rule
- skater late tails blend against recent trusted decline so they do not look like a fake staircase
- goalie `Save %` late tails stay curved and age-aware

Rendering rules:
- player mode uses historical skater or goalie baselines
- team mode uses team baselines
- Games Played mode disables baselines because the stored baseline index is age-based

SECTION 8B - STYLESHEET DELIVERY AND RERUN FEEDBACK
---------------------------------------------------
Where the CSS lives:
- `assets/puckpeak.css` is the whole stylesheet and the thing you edit.  It used to be a 77 KB
  `_CSS` string inside `nhl/styles.py`.
- `inject_css()` hands the bytes to streamlit's media file manager and emits
  `<style>@import url("/media/<sha224>.css?v=1");</style>` followed by a ~1.4 KB critical block.
  Measured through the real MediaFileHandler: `Content-Type: text/css`,
  `Cache-Control: max-age=315360000` (the `?v=1` is what flips tornado into that mode; safe
  because the path is a content hash), gzipped to 13.7 KB on the wire.
- `inject_header_bb_logo()` does the same for `assets/BB.png`, replacing a 101 KB base64 data URI.

Three rules that are load-bearing here:

1. DO NOT move this to streamlit's static file endpoint.  `AppStaticFileHandler.set_extra_headers`
   forces `Content-Type: text/plain` plus `X-Content-Type-Options: nosniff` for every extension
   outside `SAFE_APP_STATIC_FILE_EXTENSIONS`, and `.css` is not on that list.  Browsers refuse a
   `text/plain` stylesheet in standards mode, for `@import` and `<link>` alike, and no rename
   fixes it.  `MediaFileHandler` has no such override.  `server.enableStaticServing` stays off.

2. DO NOT memoize the media URL.  `script_runner.py` calls `media_file_mgr.clear_session_refs()`
   at the start of every FULL run and `remove_orphaned_files()` at the end, so a file nobody
   re-registered that run is collected.  Fragment reruns deliberately skip the clear, so they do
   not need to re-register.  `_media_url()` is idempotent and content-addressed, so calling it on
   every run is free and always yields the same URL.

3. `app.py` must keep EXACTLY THREE top-level style injections (`inject_css`,
   `inject_mobile_dropdown_fix`, `inject_header_bb_logo`).  `markdown` is not in the frontend's
   `GLOBAL_ELEMENTS` list, so each `<style>` element container occupies a flex gap in the main
   block container, and `.block-container { padding-top: 3.85rem }` is tuned around that count.
   Collapsing them into one call shifts the whole page up.  A test pins the count.
   Related trap: `st.html()` with style-only content routes to the event container, which has NO
   layout footprint at all, and it sanitizes with DOMPurify's html profile - whose allowlist
   contains `style` but not `link`.  Keep these on `st.markdown`.

4. NEGATIVE MARGINS STEAL CLICKS.  The layout pulls sections together with negative margins on
   zero-height anchor divs (`div.element-container:has(#comparison-detail-layout)` at -3.7rem,
   `[data-testid="stHorizontalBlock"]:has(#comparison-season-filter)` at -2.4rem, and the
   `.faq-btn-anchor` rule).  The anchor has no height, so nothing shrinks - the block simply
   OVERFLOWS out of its container and lands on its neighbour.  Overflowing content still
   hit-tests, and a later DOM sibling wins, so the covered element goes dead to the mouse while
   still looking fine.  This shipped: the detail/tabs stack covered the bottom of the Metric
   Selections popover button and Plotly's positioned `.svg-container` covered the top, leaving
   only the upper third clickable.  Fix is `position: relative` + `z-index` on the element that
   must stay clickable - it changes nothing visually.  If you add another negative-margin pull,
   check what it now overlaps.

Why the page used to flash on every click:
- streamlit 1.54 marks elements stale and applies
  `STALE_STYLES = {opacity: .33, transition: "opacity 1s ease-in .5s"}`.  `isElementStale()`
  returns true for EVERY element when the state is `RERUN_REQUESTED`, and on a `RUNNING` full run
  for every element whose `scriptRunId` is older than the current one.  On a fragment rerun it
  only matches elements carrying that fragment's id.
- That is the whole explanation for "card clicks are silent but the category switch flashes": the
  click bridges are fragment-scoped, the sidebar widgets are not.
- The declaration is removed wholesale when the new delta lands - transition included - so the
  page fades out over a second and then snaps back instantly.  That asymmetry is what reads as a
  redraw rather than a load.
- `assets/puckpeak.css` pins `[data-testid="stElementContainer"][data-stale="true"]` back to
  opacity 1.  The baseweb tab list and tab buttons need their own rule: they get `STALE_STYLES`
  inline from component overrides, not through the `data-stale` attribute.
- Replacing it: a 2px bar on `[data-testid="stApp"][data-test-script-state="running"]::after` and
  the same for `"rerunRequested"`.  That attribute is streamlit's own run-state readout, so it
  costs one attribute match instead of a `:has()` scan and it covers fragment reruns too.
  `"initial"` is excluded on purpose - a cold load already has the skeleton and the branded
  `stSpinner` bar.  A 220ms `animation-delay` keeps fast reruns completely silent.
- Worth knowing before optimizing payload again: streamlit's ForwardMsg cache already
  deduplicates any cacheable `new_element` delta at or above `global.minCachedMessageSize`
  (10 KB) from the SECOND rerun of a connection onward - the browser sends
  `cachedMessageHashes` with every rerun request and the server swaps in a ~60-byte reference.
  So big inline blobs are not a per-click cost; they are a cold-load and reconnect cost.

SECTION 9 - PLOTLY RENDERING GUARDRAILS
---------------------------------------
Visual rules:
- real data = solid colored line with filled markers
- projection = dotted player-colored line with open markers
- baseline = dashed white semi-transparent line with tiny markers

CHART IDENTITY - two different things, do not merge them again:
- `CHART_WIDGET_KEY` is a CONSTANT passed as `key=` to `st.plotly_chart`. Streamlit hashes the
  full figure spec into the element id regardless of the key (`plotly_chart.py` passes
  `key_as_main_identity=False`), and the frontend uses that id as the chart's React key. A
  constant key therefore means the chart remounts if and only if the figure actually changed.
- `chart_instance_id` (`_build_chart_instance_id`) identifies the PLOTTED DATA and backs the
  click-bridge staleness guard in `_parse_chart_click_trigger`, the JS rebind guard, and the
  toolbar / share-button DOM ids. It folds in board, metric, category, season type, season,
  x-axis and ALL six view toggles plus `league_filter`.
- Neither may depend on `sidebar_keys`. It used to: `search_term`, `top_selected`, `team_abbr`
  and `roster_player` were in the widget key, so typing in the search box or merely browsing
  another team's roster remounted the chart and reloaded the 22 KB JS iframe while the figure
  JSON was byte-identical. `render_chart` still accepts `sidebar_keys` for signature stability
  and deliberately ignores it.
- The toggles are in `chart_instance_id` for a reason beyond tidiness: the old key omitted
  `do_era`, `do_cumul`, `do_base`, `do_prime`, `league_filter` and (player mode) `season_type`,
  so flipping any of them replaced the plot DOM node while the bridge's `data` prop stayed
  equal - the v2 component's effect never re-ran and the `plotly_click` handler went silently
  missing. Masked because the native `on_select` path is tried first.
- Use a stable digest, not `hash()`: Python salts string hashing per process and this value is
  interpolated into the iframe srcdoc and DOM ids.

JS listener rule - the chart div does NOT always outlive the script that decorates it:
- The `.js-plotly-plot` node survives a rerun only when the figure is unchanged. Any real data
  change produces a new element id and a fresh mount, so the `components.html()` script block
  must handle both cases.
- Any `plot.on(...)` or `parent.addEventListener(...)` therefore MUST be idempotent, or listeners
  stack one layer per rerun until the chart crawls. This shipped once: an unguarded
  `plotly_relayout` bind made the chart progressively unresponsive after a few dialog cycles.
- Guards in place: `plot.__nhlRelayoutClampBound`, `parent.__nhlChartResizeBound`,
  `targetPlot._hoverRectObserver`, and `plot.__nhlChartClickBridgeInstanceId` in the click bridge.
  Follow that pattern for anything new.
- `Plotly.relayout` is ASYNC. A re-entrancy flag cleared on the same synchronous tick is already
  false by the time the resulting `plotly_relayout` event fires and guards nothing - clear it in
  the promise callback instead. `guardedRelayout()` does this correctly; copy it.

Chart duties handled in `chart.py`:
- concatenate processed frames
- add baseline overlays when enabled
- link each player's projected trace to the same legend toggle so one click hides or shows both
- render compact chart header text
- inject JS pan / zoom clamping
- use Streamlit's native `on_select="rerun"` with `selection_mode="points"` to capture point
  clicks; works in both localhost and Streamlit Cloud sandboxed iframes without any JS bridge
- `_handle_native_chart_selection()` consumes the selection event, resolves the trace name
  from `fig.data[curve_number].name`, and calls `_dispatch_chart_click_point()` directly
- dispatch routes into `show_season_details()` or `show_team_game_details()`
- offer a Copy link control using compact URL params
- tune `hovermode='closest'` and a larger `hoverdistance` so taps near the visible line resolve to the nearest point more reliably
- suppress chart clicks when another modal is already reserved for the rerun (gate check runs
  before the dedup key is written, so a suppressed rerun does not poison the click)
- deduplicate by a `curve|point|x|y` selection key in `_last_handled_chart_click_nonce`;
  deselection (empty points payload) clears the key so the same point is re-clickable without
  a chart remount — click background to reset, then click the point again

`comparison.py` defines `render_chart_season_picker()` and keeps it synced with the canonical
`st.session_state["chart_season"]` value, but `app.py` places that picker in the left chart column
immediately above the main chart. The right rail is reserved for predictions and detail panels.

Games Played mode chart specifics:
- x-axis uses `CumGP`
- selected-season mode says `Game`; career GP mode says `Career Game`
- single-season click payload stores `Age`, `GameId`, `GameDate`, and `GameType` so the dialog can resolve the exact game
- selected-season mode keeps peak highlights anchored to the real game number, not age

SECTION 9A - SEASON SNAPSHOT AGE RARITY
---------------------------------------
Scope:
- player age-clicks only
- historical NHL regular-season rows only
- never projection clicks
- never baseline clicks
- never exact one-game snapshot clicks

Flow:
1. `chart.py` passes `do_era` into `dialog.show_season_details()`.
2. `dialog.py` resolves the clicked age snapshot and collapses traded stints with
   `rarity.collapse_player_snapshot_rows()`.
3. The dialog picks the collapsed `NHL + Regular` row for the rarity target.
4. `rarity.get_age_rarity_summary()` loads the historical parquet, rebuilds rate stats, applies
   era logic only when the visible metric is actually era-adjusted, then ranks the clicked row
   against the same-age historical pool.
5. The returned payload drives one callout card under `Career Subtotals`, including:
   - overall percentile / rank / sample size
   - skater role split (`forwards` or `defensemen`) when applicable
   - compact top-5 leaderboard from the same overall comparison pool

Dependencies:
- `rarity.py` depends on `data_loaders.load_historical_data()` for the historical pool
- `rarity.py` depends on `data_loaders.get_player_identity_summary()` for top-5 player names
- `rarity.py` depends on `era.metric_is_era_adjusted()` and `era.apply_era_to_hist()` so the
  rarity card and chart stay aligned about what `Era` changes

Ranking rules:
- higher is better for all supported metrics except `GAA`
- percentile uses a midrank formula, so near-perfect seasons can be `99.96th percentile` without
  being literally `100.0`
- the compact top-5 leaderboard follows the same overall pool as the main `#rank of n` line,
  not the role-split sub-line

SECTION 10 - CACHING STRATEGY
-----------------------------
This section is the current runtime summary. For exact migration history, per-function rollout
notes, and tier tables, use `foundation_phase.md`.

Current mental model:
- permanent local artifacts still use `@st.cache_data` directly: `load_historical_data()`, `load_win_prob_weights()`, baseline builders, and other parquet-derived helpers stay process-local and are read from disk rather than HTTP
- most NHL API-backed public helpers in `data_loaders.py` and `schedule.py` still expose `@st.cache_data` wrappers, but those wrappers now sit in front of `NHLClient`, not in place of it
- `nhl/api.py` is the HTTP chokepoint: `get_client().get()` owns the shared cache lookup, `requests.Session`, per-domain token-bucket rate limiting, retry with exponential backoff + jitter, and in-flight request deduplication by cache key
- `nhl/cache.py` is the shared cache layer: default backend is `diskcache` in `.cache/nhl_api` with LRU eviction and a 200 MB size limit; if `diskcache` is missing, the app falls back to a bounded in-process dict cache
- the practical read path for API-backed data is `@st.cache_data -> NHLClient -> shared diskcache -> HTTP`; local artifact loaders stop at the first layer because they never leave the machine
- `fetch_all_time_records()` is the notable special case: page fetches go through `NHLClient` for rate limiting and retry, then the assembled result is stored directly in the shared cache with `get_cache()`

Shared-cache tier behavior in `nhl/cache.py`:
- `T0_TTL = None` for permanent local artifacts and process-lifetime data derived from local files
- `T1_TTL = 86400` for historical / effectively immutable remote data such as records, top-50 lists, and past-season payloads
- `T2_DEFAULT_TTL = 3600` for semi-static seasonal data such as rosters, standings, player landing payloads, and current-season summaries
- `T3_DEFAULT_TTL = 120` for live or near-real-time score surfaces
- `effective_ttl(season_year)` promotes closed seasons to `T1` and leaves the current season on `T2`, so past season logs and summaries naturally harden into the slower-refresh tier

What matters operationally:
- `st.cache_data` is now the outer per-process short-circuit, not the whole caching architecture
- the shared disk cache is the cross-session / cross-worker layer on the same filesystem and is the main reason repeat HTTP calls can be avoided across reruns and warm workers
- identical in-flight requests with the same cache key are collapsed inside `NHLClient`, so the app does not stampede the same endpoint during concurrent cold paths
- retry only happens for transient failures (`429`, `500`, `502`, `503`, `504`, connection errors, timeouts); `400`, `403`, and `404` are treated as non-retryable
- per-domain rate defaults are conservative and live in `nhl/api.py`; env vars can override them without code changes

Background warming now has two different jobs:
- `cache_warmer.py` is the main optional shared-cache warmer. When `PUCKPEAK_CACHE_WARMER_ENABLED=1`, `app.py` starts live, seasonal, and historical daemon loops that warm the same public functions the UI already uses.
- `async_preloader.py` still exists, but it is the older additive session-local helper. It runs once per session to warm non-active categories in the current worker and should not be described as the primary cache architecture anymore.

SECTION 10A - PREGAME WIN PROBABILITY AND STANLEY CUP ODDS
----------------------------------------------------------
Architecture split:
- offline training + backtesting only: `train_win_prob.py` (the only scikit-learn user)
- shared feature engineering: `nhl/team_ratings.py` (identical code for trainer, runtime and simulator)
- model contract and scoring: `nhl/win_prob.py`
- season simulator: `nhl/season_sim.py`
- runtime wiring: `nhl/schedule.py` (`get_current_team_ratings`, `get_game_win_probabilities`,
  `get_season_projection`), board assembly in `nhl/stanley_cup.py`, markup in `nhl/comparison.py`

LABEL RULE - read before touching the trainer
The outcome label is the home team's `wins` flag. NEVER derive it from goals: the stats API leaves
the shootout goal out of `goalsFor`, so a goal comparison records every shootout as a tie. The v1
model did exactly that - every home shootout win (65-119 shootouts a season) became an away win,
home advantage collapsed to 50.4%, and on 2025-26 (out of sample) it scored no better than always
picking the home team (log loss 0.6922 vs 0.6921). `tests/test_team_ratings.py` pins the rule.

Features (`team_ratings.MODEL_FEATURES`; the artifact scores an ordered subset chosen by backtest):
- `elo_diff` - margin-of-victory Elo, shootout = one-goal win, regressed toward 1505 every season,
  franchise-keyed so ratings survive relocation (ARI -> UTA)
- `goal_diff_shrunk_diff`, `sat_share_shrunk_diff` (5v5 shot attempts), `sog_share_shrunk_diff` -
  season-to-date averages blended with a prior worth `form_prior_weight` games (last season's value
  pulled toward the league mean). There is no minimum-games gate: game 1 of a season is scored from
  the prior, which replaced the old "fall back to last season's features" hack
- `home_back_to_back`, `away_back_to_back` - played the calendar day before (from the schedule)

Offline trainer (`python train_win_prob.py`, ~40 s, ~40 requests):
- fetches 2017-18 through the last completed season (regular season + playoffs + 5v5 shot attempts)
- rolling-origin backtest: each test season (2021-22 onward) is predicted by a model whose Elo grid
  point, prior weight, feature set and C were chosen on earlier seasons only (nested: inner
  validation = last training season)
- GAME GATE: mean log loss must beat the previous design (v1 features, label fixed) by >= 0.005 on
  the same games, and no single season may be worse
- fits the overtime model (P(tied after 60) vs |logit|, plus the shootout share of OT games)
- SIMULATOR GATE: calibrates per-simulation team strength noise on 20 historical checkpoints
  (opening night, Dec 1, Feb 1, Mar 15 of 2021-22..2025-26) and requires P(make playoffs) log loss
  to beat a naive points-pace simulation
- reports playoff-game and best-of-7 series backtests
- writes `win_prob_weights.json` only when both gates pass (exit code 1 otherwise)
- retrain every offseason (after the Final, before opening night) so the newest season is included

Backtest at v1.01.8 (log loss, lower is better; previous design / new model on the same games):
2021-22 0.6624 / 0.6425, 2022-23 0.6698 / 0.6526, 2023-24 0.6734 / 0.6573, 2024-25 0.6756 / 0.6631,
2025-26 0.6908 / 0.6895. Mean 0.6744 -> 0.6610. Playoff games 0.6697 (n=433, calibrated). Series:
66.7% of 75 series called correctly. Simulator P(make playoffs) log loss 0.353 vs points pace 0.397.
The exact numbers live in the artifact's `validation_metrics`.

Runtime rules:
- never retrain inside Streamlit; load frozen weights once through `load_win_prob_weights()`
  (a version-1 artifact is rejected and predictions switch off rather than mis-score)
- `get_current_team_ratings()` replays three completed seasons plus the current one from
  `get_league_game_table()` (3 league-wide requests per season, completed seasons 24h cached) and
  returns one snapshot for all 32 teams, shared by the cards and the simulator
- `get_game_win_probabilities(away, home, game_id, game_type)` scores a matchup from that snapshot
  plus schedule back-to-back flags and returns percentages, raw probabilities, fair decimal odds
  (1/p, no margin), a driver label, games played and an `early_season` flag (< 10 GP)
- preseason / exhibition games (`game_type` not 2 or 3) get NO prediction; the card says so
- there is no goalie adjustment any more: the old save% overlay was an untested hand-tuned constant
  applied to the most-winning goalie, not the starter
- surface the result in clickable predictions cards for up to 8 upcoming games; there is still no
  quick-add action
- clicking a card should open the matchup-history modal, not mutate the player/team board

Season simulator rules (`get_season_projection()`, 10,000 simulations, ~0.5 s warm, warmed hourly
by the cache warmer's seasonal cycle):
- regular season: every unplayed game from `get_league_schedule()` is played with the model;
  losers get an overtime point per the overtime model; each simulation draws a team strength offset
  ~N(0, sd) on the logit scale, sd interpolated from `strength_sd_preseason` to `strength_sd_late` by
  season progress. Without that noise the favourite's odds are badly overconfident
- tiebreaks: points, regulation wins, regulation + OT wins, wins, then random
- playoffs: top 3 per division + 2 wild cards per conference; the better division winner draws the
  second wild card; home ice by record; series resolved with the exact 2-2-1-1-1 best-of-7 DP
- once all eight first-round series are set, the live bracket from `get_playoff_bracket()` replaces
  seeding: decided series stay decided, in-progress series are conditioned on their score
- states: `projection` (phase `preseason` / `regular_season` / `playoffs`), `champion` (Cup
  decided - covers the summer until the September rollover), `unavailable`
- preseason: `/standings/now` still holds last season's table; only division membership is used and
  the board shows projected points instead of the stale record
- the seed is derived from the standings timestamp and remaining-game count, so reruns never jitter
- known weakness: at the mid-March checkpoint the model sim is marginally worse than points pace
  (0.185 vs 0.180); it wins clearly earlier in the season

Matchup-history runtime rules:
- `schedule.get_matchup_history()` walks backward through franchise-aware team seasons, not raw
  single-season opponent strings
- the modal shows the latest 10 meetings across regular season and playoffs, newest first
- `comparison.py` mounts a JS click bridge with `st.components.v2.component()` and intercepts
  prediction-card clicks before navigation so the modal feels in-app instead of like a full refresh
- GOTCHA: that bridge MUST be mounted inside `predictions_fragment`, never at top-level `app.py`
  scope. Streamlit scopes a rerun to a fragment only when the widget that changed belongs to that
  fragment, so a top-level mount made every prediction-card click trigger a FULL script rerun -
  the only click on the page that did, since the chart and identity bridges are fragment-scoped.
  `render_predictions_panel()` mounts it itself; the `matchup_history_bridge_mounted` flag exists
  so a caller that already mounted it can say so and avoid a duplicate key.
- GOTCHA: the prediction-card overlay must NOT carry an `href`. It is absolutely positioned over
  the whole card (card content is `pointer-events: none`), so every click lands on it. With an
  href, any click arriving before the bridge listener attached followed it as a real document
  navigation - tearing down the websocket and starting a fresh session, which looked like "the
  page re-rendered itself". It is now `role="button" tabindex="0"` and the bridge handles
  Enter/Space. `_build_live_game_card_href()` still exists, but only for shareable deep links.
- the old `mh=AWY,HOME` query-param contract remains as a no-JS fallback
- `dialog.show_matchup_history()` adds a plain-text summary of wins by each team above the cards

Guardrails:
- odds are model probabilities, not bookmaker prices: "fair odds" carry no margin and must never be
  presented as a betting line
- the early-season note must stay while either team has fewer than 10 games
PHASE 4 - PREDICTION LEDGER AND MARKET BENCHMARK (v1.02.1, `nhl/ledger.py`) - built with free data only
Why: a paid prediction product needs a record nobody can edit in hindsight, and "close to betting
sites" has to be measured, not claimed. No paid data: the owner declined all paid sources.
- LEDGER: SQLite file `prediction_ledger.sqlite3` in `PUCKPEAK_DATA_DIR` (default `.data/`, gitignored,
  dockerignored). Production MUST keep the `puckpeak_data:/app/.data` volume from
  docker-compose.yml - without it the public track record resets on every deploy
- CAPTURE: `schedule.capture_prediction_ledger()` runs from the cache warmer's live cycle (every
  ~5 min, warmer only; page renders never write). For every regular-season or playoff game starting
  within 36 h it writes the card's numbers (moneyline, 60-minute result, puck line, early-season
  flag, `model_version` = artifact `generated_at_utc`) and keeps refreshing them until puck drop.
  `record_prediction` refuses writes at/after the scheduled start (checked in Python AND in the SQL
  upsert), so the ledger holds the last pregame numbers. `first_captured_utc` / `captured_utc` show
  when. Local runs without the warmer record nothing
- MARKET: the same cycle reads the NHL API's free betting-partner feeds
  `https://api-web.nhle.com/v1/partner-game/{CA,US,SE,FI,CZ}/now` (FanDuel, DraftKings, Unibet,
  Veikkaus, Tipsport; no key, no payment) and stores each partner's latest pregame prices per game
  (frozen at puck drop the same way = near-closing lines). GOTCHA: North American partners quote
  American odds, European partners decimal odds - `ledger.odds_to_probability` tells them apart by
  value (American odds are never between -100 and +100; decimal NHL prices are never >= 100). Margin
  is removed per partner by proportional normalization, then partners are averaged per game
- GRADING: the same cycle grades every logged game found in the completed-game table (moneyline
  result, result type, final and regulation goals)
- TRACK RECORD (`schedule.get_track_record()`, cached 10 min, read-only): the predictions panel
  shows a "Track record" block: this season's live record (games, % winners picked, log loss vs
  coin flip) only once 20 games are graded, the betting market's log loss on exactly the same games
  once 20 of them have market prices, and the backtest line (clearly labelled) from the artifact.
  Live and backtest numbers are never mixed
- NEVER shown: bookmaker names, odds or links (gambling advertising in the EU/CZ). Market prices are
  used only for the aggregate comparison
- LIMITS: the partner feeds only cover the next odds date, so there is no free historical closing-
  line data; the benchmark accumulates from the 2026-27 opener. The only free archive
  (sportsbookreviewsonline NHL archive) is gone; every other historical source is paid
- first read (2026-09-13, openers): 60-minute draw prices within 1.5 points of the market on all
  five games; the moneyline disagreed most where offseason moves matter (CAR 66% vs market 53%)

PHASE 3 - 60-MINUTE RESULT AND PUCK LINE (v1.02.0, `nhl/goal_model.py`)
Why a corrected distribution: independent Poisson badly misprices hockey. On 2017-18..2025-26
regulation scores it predicts ~17% ties (observed 21-25%) and ~30% one-goal games (observed ~18%);
late empty-net goals turn one- and two-goal leads into bigger wins, and totals are under-dispersed.
Model, per game:
- total rate = 2 x league scoring level (`scoring_environment`: regulation goals per team-game,
  season-to-date blended with last season, prior 400 team-games). Per-team scoring rates were
  tested and added nothing over the league level for these markets (differences <= 0.0006)
- grid = Poisson x Poisson, x (1 + tie_inflation) on the diagonal, then a share of two-goal leads
  and then one-goal leads gains a goal (`lead2_transfer`, `lead1_transfer`), with `rate_scale` keeping
  the mean. Shape fitted by maximum likelihood on training regulation scores (two alternating passes
  with the split below)
- the home share of the rate is solved (bisection) so regulation home win + draw x P(home wins
  the tie-break) equals the Phase 1 win probability; the tie-break model is a logistic on the
  win-probability logit. So the card's moneyline, 60-minute result and puck line cannot disagree
- regulation goals from the stats API: an OT goal comes off the winner; shootout goals were never
  in `goalsFor`. Graded totals add one goal for any OT/SO decision (sportsbook rule)
Backtest (each season priced with models fitted on earlier seasons):
  1X2 log loss   model / league rate / moneyline-with-fixed-draw-rate
  2021-22 1.0176 / 1.0653 / 1.0184   2022-23 1.0287 / 1.0734 / 1.0335   2023-24 1.0230 / 1.0583 / 1.0257
  2024-25 1.0319 / 1.0534 / 1.0325   2025-26 1.0723 / 1.0829 / 1.0752
  puck line Brier (home -1.5 and away -1.5) beat their league rates in every season;
  playoff draws 22.3% predicted vs 22.4% observed (433 games)
MARKET GATE: 1X2 and both puck-line sides must beat the league rate in every test season. If it
fails, the trainer omits `goal_model` and the cards show no markets.
TOTALS ARE NOT PUBLISHED: over/under 5.5 and 6.5 did not beat the league's plain over-rate in every
season (Brier 0.2453 vs 0.2456 even after nested selection over 144 variants of rates, priors and
shrinkage). `expected_total` exists internally; do not surface it as a prediction without a new
passing backtest.
Runtime: `get_current_team_ratings()` returns `scoring_environment`; `get_game_win_probabilities()`
adds `markets` (`regulation` home/draw/away, `puck_line` home/away -1.5) when the artifact has a goal
model; the card popover shows both markets with fair decimal odds (no margin), away / draw / home
left to right to match the card, favourite at -1.5.

PHASE 2 RESULT (v1.01.9) - expected goals and goalies did NOT improve the model; not shipped
Tested with `python train_win_prob.py --phase2-report` (reproducible; first run fetches ~12,000
play-by-play payloads into `.cache/xg_training`, ~25 min; later runs take ~30 s):
- own logistic xG model on 1,025,773 unblocked shots (distance, angle, shot type, rebound, rush,
  strength, empty net), trained on 2017-18..2020-21 only: AUC 0.74-0.77 out of sample. Calibration
  drifts in the tracking era (goals per xG 0.99 in 2021-22 falling to 0.87 in 2025-26), which
  largely cancels in shares
- starters: first goalie to face a shot == boxscore starter in 60/60 checked games. Projected
  starter (most starts in the last 10, backup on the second night of a back-to-back) matched the
  actual starter only 66% of the time - modern tandems
- candidate features: shrunk 5v5 xG share, all-situations xG share, projected starter GSAx per 30
  shots (shrunk, season-decayed), and xG replacing the shot-attempt/shot shares
- nested rolling backtest on the same games: mean log-loss gain -0.0001 (need +0.002); per season
  +0.0015, -0.0017, +0.0002, -0.0008, +0.0000. Forcing the best feature set without selection gains
  only 0.0005. Goalie ratings added nothing even when fed the ACTUAL starter (a hindsight upper
  bound), and so did a "backup is starting" flag - so confirmed-starter data would not have
  helped this model either
- conclusion: at game level, team shot shares + goal differential + Elo already carry what team xG
  adds, and goalie form is too noisy to predict single games. The runtime keeps the Phase 1
  features; nothing from `nhl/xg.py` is imported by the app
- GOTCHA for future experiments: `FEATURE_SETS` in the trainer are built from `MODEL_FEATURES` at
  import. Registering an experimental feature in `MODEL_FEATURES` silently leaks it into the
  "baseline" sets. Keep research features outside it (as `xg.RESEARCH_FEATURES` does)

Not separately cached, but intentionally fan out from `get_player_landing()`:
- `get_player_headshot()`
- `get_player_current_team()`
- `get_player_roster_info()`
- `get_player_hero_image()`
- `get_player_awards()`
- `get_player_league_abbrevs()`

SECTION 11 - GAMES PLAYED MODE
------------------------------
Purpose: compare careers by accumulated games instead of age, or one selected NHL season by real game number.

Behavior:
- career mode groups by `SeasonYear`, not `Age`
- selected-season mode keeps one row per game
- x-axis is `CumGP`
- counting stats become cumulative totals by game count when cumulative mode is on
- rate stats become rolling visible averages in games mode
- `Age` is preserved for click dialogs, but single-season clicks resolve by exact game identity instead of only age
- Team selected-season mode is a season-progress branch, not the old franchise games view reused.
- Team selected-season metric values are season-to-date after each game: counting stats are cumulative, rate stats are running rates, and `PP%` falls back to the running mean of game PP% because the public team game feed does not expose PP chances.
- Team selected-season clicks now open a team game snapshot dialog with the matchup card and one-row team snapshot table.

Normal app-flow restrictions:
- no projection
- no baseline
- selected-season mode still allows cumulative display, but comparison cards must use the last visible cumulative value instead of summing cumulative rows again

The pipeline still keeps the age metadata, but the single-season dialog now keys off the exact clicked game and can show matchup, score, venue/time, and the player's one-game stat line.

SECTION 12 - MODULAR PACKAGE STRUCTURE
--------------------------------------
Import shape:
- leaf-ish modules: `constants`, `styles`, `era`, `url_params`
- runtime/cache layer: `cache`, `api`, `data_loaders`, `schedule`, `baselines`, `cache_warmer`
- pure processing: `knn_engine`, `player_pipeline`, `team_pipeline`, `team_ratings`, `goal_model`, `win_prob`, `season_sim` (import order: `constants` <- `team_ratings`, `goal_model` <- `win_prob` <- `season_sim`)
- additive preload helper: `async_preloader`
- UI: `controls`, `sidebar`, `dialog`, `chart`, `comparison`, `fragments`
- `app.py` ties everything together

Module responsibilities:
- `constants.py` - shared URLs, metric lists, caps, floors, league multipliers
- `era.py` - scoring-era multipliers and historical adjustment helpers
- `cache.py` - shared cache backend, TTL constants, `effective_ttl()`, and the diskcache/dict fallback
- `api.py` - `NHLClient` singleton, request session, rate limiting, retry/backoff, in-flight deduplication, and shared-cache integration
- `data_loaders.py` - local artifact loaders plus app-facing NHL data wrappers; most HTTP should route through `api.py`
- `rarity.py` - age-rarity ranking payloads, role splits, and top-season leaderboard assembly
- `baselines.py` - cached historical and team baseline builders
- `knn_engine.py` - clone matching, hybrid-delta projection, stat caps, fallback projection;
  `run_knn_projection()` memoized on a value-hash fingerprint (see SECTION 2)
- `team_ratings.py` - league game table, Elo, shrunk team form, back-to-back flags, runtime team snapshot; pure pandas/numpy
- `win_prob.py` - artifact validation, scalar / vectorized scoring, linear decomposition for the simulator, overtime probability
- `season_sim.py` - vectorized season + playoff Monte Carlo, exact series DP, bracket / standings / schedule adapters
- `goal_model.py` - regulation score grid, moneyline-matching split, 60-minute and puck-line pricing, league scoring level
- `ledger.py` - prediction ledger storage and grading, partner-odds parsing (American and decimal), market consensus, track record
- `xg.py` - offline research only: play-by-play parsing, xG scoring, per-game xG and goalie summaries, `GoalieHistory` point-in-time ratings, `choose_starter`, `research_game_features` (Phase 2 report)
- `player_pipeline.py` - end-to-end player pipeline and peak metadata
- `player_pipeline.py` now owns the extra TOI projection gate and the modern-coverage filtering that
  keeps zero-TOI historical rows out of clone matching
- `team_pipeline.py` - end-to-end team pipeline, including selected-season team season-progress mode
- `controls.py` - top control surface; returns `(metric, do_cumul)`
- `sidebar.py` - player/team add flows plus sidebar status widgets
- `dialog.py` - player clicks, team game snapshot clicks, matchup-history modal, projection, and baseline dialogs
- `dialog.py` now inserts the rarity callout directly under `Career Subtotals` in player age snapshots
- `chart.py` - figure assembly, baseline overlay, share-link button, Plotly click bridge, and player/team click dispatch
- `comparison.py` - season-aware Overview / Current Standings tabs, the chart-season picker renderer, JS click bridges (prediction-card and identity-card), clickable predictions panel, and live standings board wrapper
- `fragments.py` - `@st.fragment`-decorated wrappers around `render_chart`, `render_detail_tabs`, `render_predictions_panel`, and the sidebar FAQ button; called during the mount phase (the FAQ one from `render_sidebar`) so widget interactions only rerun the affected panel
- `stanley_cup.py` - standings-board assembly with simulated playoff / Cup odds, preseason projection and champion modes
- `url_params.py` - compact share-link encoder/decoder with legacy-link sanitization and canonicalization
- `schedule.py` - live/recent matchup detection, upcoming games, featured players, matchup history, runtime pregame win-prob inference, team-ratings snapshot and season projection
- `cache_warmer.py` - optional live / seasonal / historical daemon warmer for shared-cache entry points
- `async_preloader.py` - older per-session non-active-category warm-up inside the current worker

Key integration notes:
- `app.py` calls `start_background_warmer()` during startup, but the warmer is disabled unless `PUCKPEAK_CACHE_WARMER_ENABLED` is truthy. The `Dockerfile` sets it to `1` (mirrored in `docker-compose.yml`) so production always warms; it stays off by default for local and test runs. Do not remove it from the image - without the warmer, the first visitor after every container restart pays the cold `fetch_all_time_records` fetch (~11s) inside their own page load.
- `app.py` still calls `preload_all_categories()` once per session after default seeding; treat that as additive latency smoothing, not the primary cache strategy
- `schedule.py` only auto-seeds the board on first session load and only if a shared URL did not already populate players or teams
- `_find_game_from_data()` prefers a live game, then the most recently finished one, then the soonest upcoming game (preseason included). The upcoming pass is what keeps the landing page populated through the offseason - without it the board seeds empty from the Cup final until opening night
- `get_upcoming_games()` reads `/v1/scoreboard/now` first (~11 days in one request, and it rolls its own window forward to the next games during the offseason) and only walks individual `/v1/score/{date}` days when that came up short. The default window is 60 days, wide enough to span the September gap between the last preseason game and opening night; the old 14-day window returned nothing at all in early September
- `get_game_win_probabilities()` never falls back to another season's features: the shrunk-form prior covers opening night, and the result's `early_season` flag drives the card's "ratings still lean on last season" note. Preseason games return `None` and the card reads "Exhibition — no prediction."
- `cache_warmer._run_seasonal_cycle()` warms `get_season_projection()` so no visitor pays for 10,000 simulated seasons
- `cache_warmer._run_live_cycle()` runs `capture_prediction_ledger()`; the warmer is the ledger's only writer
- the season is resolved through `current_season_year()` in `nhl/constants.py`, called per use rather than read as an import-time constant. A container started before the September rollover would otherwise serve the previous season for its entire lifetime
- `comparison.py` stores tab memory per category via `panel_tab_skater`, `panel_tab_goalie`, and `panel_tab_team`
- `comparison.py` now prefers a JS trigger from `st.components.v2.component()` for prediction-card
  clicks and falls back to the `mh` query param only when the JS bridge does not fire
- `chart.py` uses Streamlit's native `on_select="rerun"` for point clicks; `comparison.py` keeps
  the prediction-card and identity-card bridges on the `st.components.v2.component()` pattern
- `comparison.py` and `chart.py` share a per-rerun dialog guard through `ui_state.py` so chart
  dialogs, player-card dialogs, and matchup-history dialogs do not collide in one rerun. READ THE
  "DIALOG SLOT" RULES in SECTION 4 before touching anything that opens a dialog - the guard is subtle and
  it has already caused one production outage
- `comparison.py` renders the predictions rail, but `app.py` owns the visible placement of the
  chart-season picker above the main chart
- Team all-time cards and team season discovery must use franchise lineage (`TEAM_LINEAGES` /
  `FranchiseAbbrev`), not raw historical `teamAbbrev` fragments.
- `train_win_prob.py` is the only place that should import `scikit-learn` for this feature; runtime scoring must stay numpy/pandas only
- selected-season Overview cards prefer league-wide season rank text from the summary endpoints and fall back to the old game-log scope label if rank data is unavailable
- Team chart-season options now come from `load_all_team_seasons()` history for the selected franchises, not from player landing payloads.
- Team selected-season share links now rely on the same forced-games-mode URL logic as skater and goalie season mode.
- `url_params.py` supports compact ID-only links, legacy `id|name` / `abbr|name` links, sanitizes
  legacy display names at ingest, and handles the chart-season selector without redundantly
  encoding forced games mode
- normal NHL API call sites should go through `get_client().get()`; `discover_all_leagues()` is the intentional audit-helper exception that still uses direct `requests.get()`
- `scraper.py` must keep the historical parquet additive-only; `Shots` and `TotalTOIMins` are now required for full rarity coverage, but old baseline / KNN columns must keep their meaning
- age-rarity top-5 names intentionally reuse cached player landing data through `get_player_identity_summary()` instead of scraping a second historical names artifact

That is the architecture. No magic, just disciplined pandas.

SECTION 13 - DEPLOYMENT (DOCKER + HETZNER)
------------------------------------------
Production lives at https://puckpeak.com on a Hetzner Cloud VPS (Ubuntu 24.04, Docker CE)
accessible over SSH as `ssh hetzner` (IP `91.98.195.150`).

Pipeline order:
`GitHub (main)` -> `git pull on VPS` -> `docker compose up -d --build` -> `puck-peak container (:8501)` -> `Caddy container (:443)` -> user

Repo artifacts (committed):
- `Dockerfile` - single-stage `python:3.11-slim`; installs `requirements.txt`; runs
  `streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true`
- `docker-compose.yml` - single service `puck-peak`, joins external network `web`, no host
  port publish, named volume `nhl_cache` mounted at `/app/.cache/nhl_api`
- `.dockerignore` - excludes `.git`, `.cache`, `tests`, `debug`, `docs`, scraper/trainer
  scripts, and project markdown metadata so those stay out of the image. `assets/` must keep
  shipping: it holds `puckpeak.css` and `BB.png`, which are read at import and served from
  `/media/`. Only `assets/PP.psd` is excluded.
- No Caddy change is needed for the stylesheet. `/media/*` is a core streamlit endpoint on the
  same origin and port as `/_stcore/stream`, so any Caddyfile that reverse-proxies the site at
  all passes it through. The only thing that could break it is a `Content-Security-Policy` whose
  `style-src` omits `'self'`, which would block the `@import`.

Server-side layout:
- `/opt/puck-peak/` - this repo, cloned from GitHub
- `/opt/caddy/` - standalone Caddy stack (`docker-compose.yml` + `Caddyfile`); terminates
  TLS and reverse-proxies `puckpeak.com` -> `puck-peak:8501` over the shared `web` network,
  redirects `www.puckpeak.com` -> apex
- Docker network `web` is external / shared; created once on the host with
  `docker network create web`

Caching across restarts:
- Docker named volume `nhl_cache` persists the `diskcache` directory. `NHLCache` in
  `nhl/cache.py` keeps its shared disk cache there, so live / seasonal / historical
  entries survive container rebuilds. `@st.cache_data` process-local wrappers warm
  themselves from the disk cache after the first rerun.

Deploy workflow (post-initial setup):
```
ssh hetzner
cd /opt/puck-peak
git pull
docker compose up -d --build
docker image prune -f
```

Rollback: `git revert <commit>` + `docker compose up -d --build`, or
`docker compose down` to stop entirely. Dropping the cache volume
(`docker compose down -v`) is optional; the app rewarms on next start.

No secrets, no env-driven config, no CI/CD pipeline. The NHL APIs are public and the
app is read-only, so the image and the host need no credentials. If that ever changes,
add a `.env` (git-ignored) and wire it in through `env_file:` in the compose service.
