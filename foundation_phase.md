# Foundation Phase — Usage Notes

## NHLClient

```python
from nhl.api import get_client

client = get_client()          # singleton, thread-safe lazy init
data = client.get(
    url="https://api-web.nhle.com/v1/player/8471675/landing",
    cache_key="player_landing:8471675",
    ttl=7200,                  # seconds (or use tier constants)
    timeout=10,                # HTTP timeout in seconds
)
# Returns parsed JSON dict, or None on failure.
```

`get()` handles cache lookup, request deduplication, per-domain rate limiting, and retry with exponential backoff internally.

### Invalidation

```python
client.invalidate("player_landing:*")   # fnmatch glob; returns count deleted
```

## Shared Cache (diskcache)

| Setting | Default | Env var |
|---------|---------|---------|
| Directory | `.cache/nhl_api` | `NHL_CACHE_DIR` |
| Size limit | 200 MB | — |
| Eviction | LRU | — |
| Backend | diskcache (SQLite) | Falls back to in-process dict if diskcache is missing |

Direct access (rarely needed):

```python
from nhl.cache import get_cache
cache = get_cache()            # same singleton used by NHLClient
cache.get("some_key")          # returns value or None
cache.set("some_key", data, ttl=3600)
cache.backend_name             # "diskcache" or "dict"
cache.stats()                  # {"hits": N, "misses": M}
```

## TTL Tier Constants

```python
from nhl.cache import T0_TTL, T1_TTL, T2_DEFAULT_TTL, T3_DEFAULT_TTL, effective_ttl

T0_TTL            # None — permanent (static artifacts)
T1_TTL            # 86400 — 24h (historical/immutable)
T2_DEFAULT_TTL    # 3600  — 1h  (semi-static seasonal)
T3_DEFAULT_TTL    # 120   — 2m  (live/near-real-time)

effective_ttl(2020)   # 86400 (past season → T1)
effective_ttl(2025)   # 3600  (current season → T2)
```

## Rate Limiting

Per-domain token buckets with defaults:

| Domain | Rate | Env var override |
|--------|------|-----------------|
| `api-web.nhle.com` | 20 req/s | `NHL_RATE_LIMIT_API_WEB_NHLE_COM` |
| `api.nhle.com` | 15 req/s | `NHL_RATE_LIMIT_API_NHLE_COM` |
| `records.nhl.com` | 10 req/s | `NHL_RATE_LIMIT_RECORDS_NHL_COM` |
| `search.d3.nhle.com` | 5 req/s | `NHL_RATE_LIMIT_SEARCH_D3_NHLE_COM` |

Unknown domains get a conservative 5 req/s default.

## Retry Configuration

| Setting | Default | Env var |
|---------|---------|--------|
| Max attempts | 3 | `NHL_MAX_RETRIES` |
| Base delay | 1.0s | `NHL_RETRY_BASE_DELAY` |

Retries on 429, 500, 502, 503, 504, ConnectionError, Timeout. Respects `Retry-After` header on 429. Does not retry 400, 403, 404.

---

## Phase 2a — First Migration Batch

Three functions now route HTTP through `NHLClient.get()` + diskcache, while keeping their existing `@st.cache_data` decorators as an in-process layer.

### Migrated Functions

| Function | File | Cache Key | diskcache TTL | st.cache_data TTL | Tier |
|----------|------|-----------|---------------|-------------------|------|
| `get_player_landing(pid)` | `nhl/data_loaders.py` | `player_landing:{pid}` | 7200s (2h) | 7200s (2h) | T2 |
| `get_player_raw_stats(pid, name)` | `nhl/data_loaders.py` | N/A — uses `get_player_landing` for HTTP | N/A | 7200s (2h) | T2 |
| `get_live_or_recent_game()` | `nhl/schedule.py` | `scoreboard` / `score:{YYYY-MM-DD}` | 120s (2m) | 120s (2m) | T3 |

### What Changed

- **`get_player_landing`**: `requests.get()` replaced with `get_client().get()`. NHLClient handles retry, rate limiting, dedup, and diskcache. Returns `{}` on failure (contract preserved).
- **`get_player_raw_stats`**: No HTTP change. Only `@st.cache_data` TTL bumped from 1h → 2h to match T2 spec. The function's HTTP dependency flows through `get_player_landing`.
- **`get_live_or_recent_game`**: The old `_find_game_from_url` (fetch + parse) was split into `_find_game_from_data` (pure parser) and NHLClient fetches in the caller. Scoreboard and per-date score endpoints each get their own cache key at T3 TTL.

### Read Path (Two-Layer Cache)

```
Request → st.cache_data (in-process) → NHLClient.get → diskcache (SQLite) → HTTP
```

Both layers use synchronised TTLs. `st.cache_data` avoids re-entering `NHLClient.get()` within the same Streamlit process/TTL window. diskcache shares results across processes and survives restarts.

### How to Migrate Additional Functions

1. Replace `requests.get(url, timeout=N).json()` with `get_client().get(url=url, cache_key="...", ttl=..., timeout=N)`.
2. Choose a cache key from the scheme in `cache_strategy.md` §2.
3. Pick TTL: use `effective_ttl(season_year)` for season-aware functions, `T3_DEFAULT_TTL` for live data, or a fixed value for the rest.
4. Handle `None` return from `NHLClient.get()` — convert to the function's existing failure value (`{}`, empty DataFrame, `None`, etc.).
5. Keep `@st.cache_data` as an extra layer for now. Match its TTL to the diskcache TTL.
6. For pure-compute functions that call an already-migrated HTTP function (like `get_player_raw_stats` calls `get_player_landing`), only adjust the `@st.cache_data` TTL — no NHLClient usage needed.
7. Add tests mocking `get_client` to verify the cache key and TTL are passed correctly.

---

## Phase 2b — Broad Data Loader Migration

All remaining direct `requests.get` call sites in `nhl/data_loaders.py` (12 sites) and `nhl/schedule.py` (3 sites) now route through `NHLClient.get()`. The only exception is `discover_all_leagues` (audit helper, not called in the main app flow).

### Migrated Functions

| Function | File | Cache Key | TTL | Tier |
|----------|------|-----------|-----|------|
| `load_all_team_seasons` | data_loaders.py | `team_list` / `team_summary:{gt_id}` | T1 / T2 | mixed |
| `_paginate_records` | data_loaders.py | (none — rate-limit only) | — | — |
| `fetch_all_time_records` | data_loaders.py | `records:{cat}:{s_type}` | T1 (86400) | T1 |
| `get_top_50` | data_loaders.py | `top50_skater:{metric}` | T1 (86400) | T1 |
| `get_top_50_goalies` | data_loaders.py | `top50_goalie` | T1 (86400) | T1 |
| `search_player` | data_loaders.py | `search:{normalized_query}` | T2 (3600) | T2 |
| `get_team_roster` | data_loaders.py | `roster:{abbr}` | T2 (3600) | T2 |
| `get_current_nhl_standings` | data_loaders.py | `standings` | T2 (3600) | T2 |
| `get_team_trophy_summary` | data_loaders.py | `franchise_team_totals` / `franchise_season_results` | T1 (86400) | T1 |
| `_fetch_team_game_summary_rows` | data_loaders.py | `team_game_summary:{tid}:{sid}:{gt}` | `effective_ttl` | T1/T2 |
| `_fetch_season_summary_rows` | data_loaders.py | `season_summary:{cat}:{sid}:{gt}` | `effective_ttl` | T1/T2 |
| `get_player_season_game_log` | data_loaders.py | `player_gamelog:{pid}:{yr}:{gt}` | `effective_ttl` | T1/T2 |
| `get_upcoming_games` | schedule.py | `score:{date}` per date | T3 (120) | T3 |
| `get_game_details` | schedule.py | `score:{date}` (T3 today, T1 past) | T3/T1 | T3/T1 |
| `_fetch_club_stats` | schedule.py | `club_stats:{abbr}` | T2 (3600) | T2 |

### Two-Layer Cache (Read Path)

```
Request → st.cache_data (in-process) → NHLClient.get → diskcache (SQLite) → HTTP
```

All migrated functions retain `@st.cache_data` as an in-process short-circuit. The disk cache shares results across processes and survives restarts.

### `fetch_all_time_records` — Explicit Disk Cache

This function paginates across many API pages, then assembles the result. Per-page caching is not useful, so individual page fetches use `NHLClient.get()` without a cache key (getting rate limiting and retry only). The assembled result is cached via direct `get_cache().set()` under key `records:{cat}:{s_type}` with T1 TTL.

### Score Date Key Sharing

`get_live_or_recent_game`, `get_game_details`, and `get_upcoming_games` all share the `score:{date}` cache key for the NHL score endpoint. Any of these populating the cache benefits the others.

### Deferred

- `discover_all_leagues(sample_player_ids)` — audit/debug helper with no app callers. Still uses `requests.get` directly. `import requests` is retained in `data_loaders.py` for this single function.

### Test Coverage

- 5 existing tests updated from `@patch("nhl.*.requests.get")` to `@patch("nhl.*.get_client")`.
- 7 new tests in `BroadMigrationTests` (test_data_loaders.py) verifying cache keys, TTLs, and None handling.
- 4 new tests in `ScheduleMigrationTests` (test_schedule.py) verifying cache keys and failure paths.
- Total: 56 tests passing in test_data_loaders.py + test_schedule.py.

---

## Phase 3 - Cache Warmer and Background Refresh

An optional process-local cache warmer now runs in daemon threads and warms the
same public functions the UI already uses. This keeps the shared diskcache warm
through the normal app path while preserving the existing `st.cache_data` layer
inside each Streamlit worker.

### Startup and Enablement

- Disabled by default. The app behaves as before unless `PUCKPEAK_CACHE_WARMER_ENABLED=1`.
- The warmer starts from `app.py` through `start_background_warmer()`.
- A module-level lock/flag ensures only one warmer starts per worker process.
- The existing per-session `async_preloader.py` path remains in place for now.

### Warmed Functions by Tier

| Tier | Thread | Default interval | Env var | Warmed public functions |
|------|--------|------------------|---------|-------------------------|
| Live | `cache-warmer-live` | 300s (5m) | `PUCKPEAK_CACHE_WARMER_LIVE_INTERVAL_SECONDS` | `get_live_or_recent_game()`, `get_featured_players(home, away)` when a matchup is found, `get_upcoming_games(limit=8, days_ahead=14)` |
| Seasonal | `cache-warmer-seasonal` | 3600s (1h) | `PUCKPEAK_CACHE_WARMER_SEASONAL_INTERVAL_SECONDS` | `load_all_team_seasons()`, `get_current_nhl_standings()`, seeded `get_team_roster()` calls for `EDM`, `PIT`, `WSH`, `COL`, `TOR`, `NYR`, seeded `get_player_landing()` calls for McDavid `8478402`, Crosby `8471675`, Ovechkin `8471214`, MacKinnon `8477492`, Matthews `8479318`, Shesterkin `8478048` |
| Historical | `cache-warmer-historical` | 21600s (6h) | `PUCKPEAK_CACHE_WARMER_HISTORICAL_INTERVAL_SECONDS` | `get_id_to_name_map("Skater")`, `get_clone_details_map("Skater")`, `get_id_to_name_map("Goalie")`, `get_clone_details_map("Goalie")`, `get_top_50("Points")`, `get_top_50("Goals")`, `get_top_50("Assists")`, `get_top_50_goalies()` |

### Safety and Load-Shaping

- Each tier runs sequentially on its own daemon thread. The warmer does not fan out parallel API bursts.
- All tasks are wrapped in exception handling and only log failures. A warm-up error never crashes the app process.
- Initial startup is staggered with jitter to reduce thundering-herd deploy behavior:
  - live: 0-15s
  - seasonal: 15-60s
  - historical: 60-180s
- The warmer uses the existing `NHLClient`, so it still respects request deduplication, retry, and per-domain rate limiting.

### Tuning Notes

- The default intervals are conservative and safe for local development and production.
- Most default intervals are at or above the public wrapper TTLs, so the warmer usually refreshes after the in-process cache can expire instead of just re-touching hot `st.cache_data` entries.
- Lower intervals improve hit rate and freshness for active users, but they also increase background API traffic. If production load is light, prefer the defaults before making the live or seasonal tiers more aggressive.
