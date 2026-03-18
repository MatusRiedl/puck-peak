# Puck Peak - Pre-Foundation Architecture and Scaling Analysis

Archived note: this document captures the pre-`NHLClient`, pre-shared-cache
architecture and the target-state scaling analysis that informed the later
foundation and caching work. It is historical context only.

Current implemented source of truth:
- `foundation_phase.md`

## 1. Current Runtime Architecture

### Entrypoint & Execution Model

Streamlit runs `app.py` top-to-bottom on **every user interaction** (click, toggle, sidebar change). There is no backend server, no database, no auth. All state lives in `st.session_state` (per-browser-tab) and all heavy computation is gated behind `@st.cache_data`.

### How Caching Works Today

All caching uses **`@st.cache_data`**, which is a Streamlit-managed in-process LRU cache keyed by function name + serialized arguments. There are three tiers:

| Tier | TTL | Purpose | Examples |
|------|-----|---------|----------|
| **Permanent** | None (lives until process restarts) | Local artifacts, baselines, rarity | `load_historical_data()`, `load_win_prob_weights()`, `get_historical_baselines()` |
| **Hourly** | `ttl=3600` | Player/team data that changes slowly | `get_player_landing()`, `get_player_raw_stats()`, `load_all_team_seasons()`, `get_id_to_name_map()`, `search_player()`, `get_team_roster()` |
| **5-minute** | `ttl=300` | Live/near-real-time game data | `get_live_or_recent_game()`, `get_upcoming_games()`, `get_game_win_probabilities()` |

**Critical limitation**: `st.cache_data` is **per-process**. On Streamlit Cloud (or any multi-worker deployment), each worker process maintains its own cache. Caches are **not shared across users** unless they happen to land on the same worker. Every cold process start triggers a full set of API calls to warm its cache.

### Local Artifacts (Loaded Once Per Process)

| File | Loader | Cache | Purpose |
|------|--------|-------|---------|
| `nhl_historical_seasons.parquet` | `load_historical_data()` | Permanent | KNN projection, baselines, age-rarity |
| `win_prob_weights.json` | `load_win_prob_weights()` | Permanent | Pregame win-probability inference |

These are read from disk once and stay in memory for the process lifetime. No API calls involved.

### Background Preloading

`async_preloader.py` fires once per session (`_preloaded` guard). It spawns daemon threads to warm caches for the two categories the user is **not** currently viewing:

- If viewing Skaters: preloads Goalie name/clone maps + Team season data
- If viewing Goalies: preloads Skater name/clone maps + Team season data
- If viewing Teams: preloads both Skater and Goalie name/clone maps

This reduces perceived latency on category switches **within a single process**, but does nothing for cross-user cache sharing.

---

## 2. All External NHL API Endpoints

### data_loaders.py

| Function | Endpoint | TTL | Notes |
|----------|----------|-----|-------|
| `get_player_landing()` | `api-web.nhle.com/v1/player/{id}/landing` | 1h | Central player metadata source; fan-out to headshot, team, roster info, awards, league abbrevs |
| `get_player_raw_stats()` | (via `get_player_landing()`) | 1h | Parses `seasonTotals` from landing payload |
| `get_player_season_game_log()` | `api-web.nhle.com/v1/player/{id}/game-log/{season}/{type}` | 1h | 2 calls per player-season (reg + playoffs) |
| `get_player_available_nhl_seasons()` | (via `get_player_landing()`) | 1h | Parses landing payload |
| `search_player()` | `search.d3.nhle.com/api/v1/search/player` | 1h | D3 player search |
| `get_team_roster()` | `api-web.nhle.com/v1/roster/{abbr}/current` | 1h | Current roster for sidebar |
| `load_all_team_seasons()` | `api.nhle.com/stats/rest/en/team/summary` | 1h | 3 calls: team list + reg summary + playoff summary |
| `get_team_season_game_log()` | `api.nhle.com/stats/rest/en/team/summary` (per-game) | 1h | 2 calls per team-season (reg + playoffs) |
| `get_season_leaderboard()` | `api.nhle.com/stats/rest/en/skater/summary` or `goalie/summary` | 1h | Up to 2 calls per leaderboard (Both = reg + playoffs) |
| `get_current_nhl_standings()` | `api-web.nhle.com/v1/standings/now` | 1h | Standings + PP% join |
| `fetch_all_time_records()` | `records.nhl.com/site/api/skater-career-scoring-*` or `goalie-career-*` | 1h | Paginated; up to ~40 pages for full career records |
| `get_top_50()` | `records.nhl.com/site/api/skater-career-scoring-regular-season` | 1h | Top 50 skaters by metric |
| `get_top_50_goalies()` | `records.nhl.com/site/api/goalie-career-stats` | 1h | Top 50 goalies by wins |
| `get_team_trophy_summary()` | `records.nhl.com/site/api/franchise-team-totals` + `franchise-season-results` | 1h | 2 calls |
| `get_team_all_time_stats()` | (via `load_all_team_seasons()`) | 1h | Derived from cached team data |

### schedule.py

| Function | Endpoint | TTL | Notes |
|----------|----------|-----|-------|
| `get_live_or_recent_game()` | `api-web.nhle.com/v1/scoreboard/now` + `score/{date}` | 5m | 1 call (scoreboard), up to 8 fallback date calls |
| `get_upcoming_games()` | `api-web.nhle.com/v1/score/{date}` | 5m | Up to 15 sequential date calls |
| `get_game_details()` | `api-web.nhle.com/v1/score/{date}` | 5m | 1 call per game lookup |
| `get_matchup_history()` | (via `get_team_season_game_log()` + `get_game_details()`) | 1h | Walks backward through seasons; up to ~10 season + 10 game-detail calls |
| `get_game_win_probabilities()` | (via `get_team_season_game_log()` + `_get_cached_club_stats()`) | 5m | 4 calls per matchup (2 game logs + 2 club stats) |
| `_get_cached_club_stats()` / `_fetch_club_stats()` | `api-web.nhle.com/v1/club-stats/{abbr}/now` | 1h | 1 call per team |
| `get_featured_players()` | (via `_get_cached_club_stats()`) | 1h | 2 calls (one per team) |

### Summary: Unique External Domains

1. **`api-web.nhle.com`** -- Player landing, game logs, scoreboard, scores, rosters, club stats, standings
2. **`api.nhle.com`** -- Stats REST API for season summaries (skater, goalie, team)
3. **`search.d3.nhle.com`** -- Player search
4. **`records.nhl.com`** -- All-time career records, franchise history

---

## 3. High-Level Architecture Diagram (Current State)

```
Browser Tab (User A)          Browser Tab (User B)
       |                             |
       v                             v
  +-----------+                +-----------+
  | Streamlit |                | Streamlit |
  | Process 1 |                | Process 2 |   (separate workers on Cloud)
  +-----------+                +-----------+
       |                             |
       |  st.cache_data (in-proc)    |  st.cache_data (in-proc)
       |  [NOT shared]               |  [NOT shared]
       |                             |
       +--------+    +---------------+
                |    |
                v    v
        +------------------+
        |  NHL Public APIs |    (undocumented, no auth, no SLA)
        |                  |
        |  api-web.nhle.com|
        |  api.nhle.com    |
        |  search.d3.nhle  |
        |  records.nhl.com |
        +------------------+

        +------------------+
        |  Local Artifacts |    (read once per process from disk)
        |                  |
        |  .parquet        |
        |  .json           |
        +------------------+
```

---

## 4. Pain Points & Bottlenecks

### API Rate-Limit / IP-Ban Risk

- **No outbound rate limiting.** Every `requests.get()` fires immediately. Under concurrent load, N users x M API calls can spike to hundreds of requests/minute to the same undocumented NHL endpoints.
- **No request deduplication.** If 10 users request the same player within the same second, 10 identical `get_player_landing()` calls go out before any cache entry is written (Streamlit cache is not lock-aware across threads/processes).
- **`get_upcoming_games()` is the worst offender**: up to 15 sequential HTTP calls per invocation, each to a different date URL. Under 5-minute TTL, this fires frequently.
- **`fetch_all_time_records()` paginates** through the full records database (~20k+ skaters) with page_size=500, meaning ~40+ round-trips on first call per process.
- **`get_matchup_history()` walks backward** through multiple seasons, making 1 `get_team_season_game_log()` call per season plus 1 `get_game_details()` call per game found.

### Cache Isolation

- `st.cache_data` is **per-process, in-memory only**. On Streamlit Cloud's multi-worker setup, each process builds its own warm cache independently. There is zero sharing.
- A cold deploy or process recycle means every user triggers a full cache-warming storm simultaneously.
- The `async_preloader` helps within one session but cannot warm caches for other processes.

### Cold-Start Amplification

On a fresh process, the first user triggers:
1. `load_historical_data()` -- parquet read (~instant, local)
2. `load_win_prob_weights()` -- JSON read (~instant, local)
3. `get_live_or_recent_game()` -- 1-9 API calls
4. `get_featured_players()` -- 2 API calls (club stats)
5. `preload_all_categories()` -- background: `get_id_to_name_map()` x2 + `get_clone_details_map()` x2 + `load_all_team_seasons()` -- each of these calls `fetch_all_time_records()` which paginates through ~40 pages
6. Pipeline execution: `get_player_landing()` + `get_player_raw_stats()` per player
7. Predictions panel: `get_upcoming_games()` (up to 15 calls) + `get_game_win_probabilities()` per game (4 calls each)

**Worst-case cold start**: 80-120+ outbound API calls before the first chart renders.

### Sequential HTTP Patterns

- `get_upcoming_games()`: loops day-by-day, each day is a blocking `requests.get()`
- `get_matchup_history()`: loops season-by-season
- `fetch_all_time_records()`: loops page-by-page
- None of these use `asyncio` or concurrent futures for parallelism

### No Retry / Backoff

All API calls use bare `requests.get()` with a 5-15s timeout inside `try/except`. There is no exponential backoff, no retry on transient failures, and no circuit breaker. A brief NHL API hiccup returns empty data that gets cached for the full TTL.

---

## 5. Proposed Production Architecture (Text Diagram)

```
                    Browser Tabs (N concurrent users)
                              |
                              v
                    +-------------------+
                    |   Load Balancer   |
                    +-------------------+
                              |
                    +---------+---------+
                    |                   |
               +----------+       +----------+
               | Streamlit|       | Streamlit|    (N workers)
               | Worker 1 |       | Worker 2 |
               +----------+       +----------+
                    |                   |
                    +------- + ---------+
                             |
                    +--------v---------+
                    |  Shared Cache    |    Redis / Valkey / DiskCache
                    |  (cross-worker)  |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  API Proxy Layer |    rate-limiter + dedup + retry
                    |  (singleton)     |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  NHL Public APIs |
                    +------------------+

                    +------------------+
                    |  Local Artifacts |    .parquet, .json (read-only)
                    +------------------+
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **Load Balancer** | Distribute users across Streamlit workers; sticky sessions optional |
| **Streamlit Workers** | Run `app.py`, render UI, read from shared cache |
| **Shared Cache** | Cross-worker TTL cache (Redis, Valkey, or disk-based). Replaces per-process `st.cache_data` for all API-backed functions. Local-artifact caches stay in-process (permanent, no sharing needed). |
| **API Proxy Layer** | Single chokepoint for all outbound NHL requests. Enforces rate limits (e.g., max 30 req/s to nhle.com), deduplicates in-flight identical requests, retries with exponential backoff, and circuit-breaks on sustained failures. Can be a sidecar process, a simple FastAPI service, or an in-process singleton with `asyncio`. |
| **Local Artifacts** | Parquet + JSON on shared filesystem or baked into container image. Read once per process, cached permanently in-process. No change needed. |

### Key Changes from Current State

1. **Replace `st.cache_data` with shared cache** for all API-fetching functions (~25 functions). Keep `st.cache_data` for in-process permanent caches (parquet, baselines, weights).
2. **Add an API proxy/gateway** that enforces rate limits and deduplicates concurrent identical requests. Even a simple `threading.Lock` per unique request key would prevent the stampede.
3. **Background refresh job**: A single cron/scheduler process that pre-warms the shared cache on a schedule (e.g., refresh standings every 5 min, refresh records every hour) so user requests are always cache hits.
4. **Parallelize sequential HTTP patterns**: Convert `get_upcoming_games()`, `get_matchup_history()`, and `fetch_all_time_records()` to use `concurrent.futures.ThreadPoolExecutor` for parallel fetches within rate limits.
5. **Add retry with backoff**: Wrap all `requests.get()` calls with tenacity or a simple retry decorator (3 attempts, exponential backoff, jitter).

---

## 6. Migration Priority

| Priority | Change | Impact | Effort |
|----------|--------|--------|--------|
| **P0** | Add outbound rate limiting (per-domain token bucket) | Prevents IP ban | Low |
| **P0** | Add retry + backoff to all `requests.get()` calls | Prevents stale empty caches from transient failures | Low |
| **P1** | Shared cache layer (Redis or diskcache) for the ~25 API-backed cached functions | Eliminates redundant API calls across workers | Medium |
| **P1** | Request deduplication (collapse concurrent identical in-flight requests) | Prevents cold-start stampede | Medium |
| **P2** | Background cache warmer (cron job) | Users always hit warm cache | Medium |
| **P2** | Parallelize sequential HTTP loops | Faster cold-start, better UX | Low-Medium |
| **P3** | API proxy as a separate service | Clean separation, centralized monitoring | High |
