# Puck Peak — Caching & Data Strategy

## 1. Data Taxonomy

Every piece of data the app touches falls into one of four tiers.

| Tier | Description | TTL / Refresh | Examples |
|------|-------------|---------------|----------|
| **T0 — Static artifacts** | Local files baked into the repo. Never change at runtime. | Permanent (process lifetime) | `nhl_historical_seasons.parquet`, `win_prob_weights.json` |
| **T1 — Historical / immutable** | Past-season stats, career records, franchise history. Data that will never change again (a player's 2018-19 season line is final). | 24 hours, or manual refresh | `fetch_all_time_records()`, `get_id_to_name_map()`, `get_clone_details_map()`, `get_team_trophy_summary()`, completed-season slices of `load_all_team_seasons()`, past-season `get_season_leaderboard()`, past-season `get_player_season_game_log()`, past-season `get_team_season_game_log()` |
| **T2 — Semi-static seasonal** | Current-season aggregates, rosters, standings, player landing pages. Changes after each game night but is stable within a day. | 1–4 hours | `get_player_landing()`, `get_player_raw_stats()`, `get_team_roster()`, current-season `get_season_leaderboard()`, current-season `load_all_team_seasons()`, `get_current_nhl_standings()`, `_get_cached_club_stats()`, `search_player()` |
| **T3 — Live / near-real-time** | Today's scoreboard, upcoming schedule, in-progress win probabilities. | 2–5 minutes | `get_live_or_recent_game()`, `get_upcoming_games()`, `get_game_details()`, `get_game_win_probabilities()`, `get_featured_players()` |

### Key insight: season-awareness splits T1 from T2

Many functions accept a `season_year` argument. If `season_year < current_season`, the result is **T1** (immutable). If `season_year == current_season`, it is **T2** (semi-static). The cache layer should exploit this: same function, different TTL based on whether the season is closed.

```
def effective_ttl(season_year, current_season):
    if season_year < current_season:
        return 86400          # T1: 24 hours (effectively permanent during a session)
    return 3600               # T2: 1 hour
```

---

## 2. Per-Call-Site Cache Design

### 2.1 T0 — Static Artifacts (no changes needed)

| Function | Cache Key | TTL | Backend | Notes |
|----------|-----------|-----|---------|-------|
| `load_historical_data()` | `hist_data` | Permanent | In-process only (`st.cache_data`, no TTL) | ~5-10 MB DataFrame. Read from disk once per process. No sharing needed — every worker reads the same parquet file. |
| `load_win_prob_weights()` | `win_prob_weights` | Permanent | In-process only | ~15 KB JSON. Same rationale. |

**Recommendation**: Leave these exactly as-is. They are already optimal.

---

### 2.2 T1 — Historical / Immutable

These are the biggest wins. Once fetched, they should never be re-fetched until explicitly invalidated.

| Function | Cache Key Format | TTL | Backend | Size | Notes |
|----------|-----------------|-----|---------|------|-------|
| `fetch_all_time_records(cat, s_type)` | `records:{cat}:{s_type}` | 24h | **Disk (SQLite or JSON)** | 500 KB–1 MB each (4 combos) | Heaviest API burden (~40 paginated calls). Disk-cache eliminates cold-start storms. |
| `get_id_to_name_map(cat)` | `id_name_map:{cat}` | 24h | **Disk** | 500 KB–1 MB each | Derived from `fetch_all_time_records`. Cache the derived map directly. |
| `get_clone_details_map(cat)` | `clone_details:{cat}` | 24h | **Disk** | 500 KB–1 MB each | Same derivation. Cache the result, not just the source. |
| `get_top_50(metric)` | `top50_skater:{metric}` | 24h | **Disk** | ~2 KB | Tiny, but still saves an API call. |
| `get_top_50_goalies()` | `top50_goalie` | 24h | **Disk** | ~2 KB | Same. |
| `get_team_trophy_summary()` | `team_trophies` | 24h | **Disk** | ~3 KB | Stanley Cup data changes once a year. |
| `get_season_leaderboard(cat, yr, type)` [past season] | `leaderboard:{cat}:{yr}:{type}` | 24h | **Disk** | 50–100 KB | Immutable once season closes. |
| `get_player_season_game_log(pid, name, yr)` [past season] | `player_gamelog:{pid}:{yr}` | 24h | **Disk** | 30–50 KB | Immutable once season closes. |
| `get_team_season_game_log(abbr, yr)` [past season] | `team_gamelog:{abbr}:{yr}` | 24h | **Disk** | 10–20 KB | Immutable once season closes. |

**Total disk footprint for T1**: ~10–15 MB. Trivial.

---

### 2.3 T2 — Semi-Static Seasonal

| Function | Cache Key Format | TTL | Backend | Size | Notes |
|----------|-----------------|-----|---------|------|-------|
| `get_player_landing(pid)` | `player_landing:{pid}` | 2h | **Disk** | 30–100 KB | Central source for ~8 derived accessors. Cache the landing blob; derive in-process. |
| `get_player_raw_stats(pid, name)` | `player_raw:{pid}` | 2h | **Disk** | 50–80 KB | Derived from landing + NHLe transform. Cache result to avoid recompute. |
| `get_team_roster(abbr)` | `roster:{abbr}` | 4h | **Disk** | ~10 KB | Rosters change at most a few times per season. |
| `load_all_team_seasons()` | `all_team_seasons` | 2h | **Disk** | 3–4 MB | Single large payload. Disk-backed avoids re-fetch on process recycle. |
| `get_current_nhl_standings()` | `standings` | 1h | **Disk** | ~25 KB | Updates after each game night. 1h is fine. |
| `get_season_leaderboard(cat, yr, type)` [current season] | `leaderboard:{cat}:{yr}:{type}` | 1h | **Disk** | 50–100 KB | Same key as T1 but shorter TTL when yr == current. |
| `get_player_season_game_log(pid, name, yr)` [current season] | `player_gamelog:{pid}:{yr}` | 1h | **Disk** | 30–50 KB | Same split. |
| `get_team_season_game_log(abbr, yr)` [current season] | `team_gamelog:{abbr}:{yr}` | 1h | **Disk** | 10–20 KB | Same split. |
| `_get_cached_club_stats(abbr)` | `club_stats:{abbr}` | 2h | **Disk** | ~10 KB | Per-team current-season roster stats. |
| `search_player(query)` | `search:{query_normalized}` | 2h | **Disk** | ~5 KB | Search results shift slowly. Normalize query (lowercase, stripped). |

**Total disk footprint for T2**: ~5–15 MB active, rotating.

---

### 2.4 T3 — Live / Near-Real-Time

| Function | Cache Key Format | TTL | Backend | Size | Notes |
|----------|-----------------|-----|---------|------|-------|
| `get_live_or_recent_game()` | `live_game` | 2 min | **In-memory shared** | ~1 KB | All users see the same featured game. Short TTL for liveness. |
| `get_upcoming_games(limit, days)` | `upcoming:{limit}:{days}` | 5 min | **In-memory shared** | ~18 KB | Schedule doesn't change within minutes. |
| `get_game_details(date, gid)` | `game_detail:{date}:{gid}` | 2 min during game day, 24h for past dates | **Disk** (past) / **In-memory** (today) | ~2 KB | Past game details are immutable. |
| `get_game_win_probabilities(away, home)` | `win_prob:{away}:{home}` | 5 min | **In-memory shared** | ~1 KB | Depends on latest team stats. |
| `get_featured_players(home, away)` | `featured:{home}:{away}` | 5 min | **In-memory shared** | ~2 KB | Derived from club stats. |
| `get_matchup_history(away, home, limit)` | `matchup_hist:{away}:{home}:{limit}` | 1h | **Disk** | ~30 KB | Historical matchups don't change quickly. Bump from 5m to 1h. |

---

## 3. Data Access Layer (DAL) Design

All external API access should be funneled through a single module: **`nhl/api.py`**.

### 3.1 Module Structure

```
nhl/api.py
├── NHLClient                     # Singleton. Owns the requests.Session, rate limiter, and cache.
│   ├── .session: requests.Session
│   ├── .rate_limiters: dict[str, TokenBucket]    # One per NHL domain
│   ├── .cache: DiskCache                         # Shared backend (diskcache or SQLite)
│   ├── .inflight: dict[str, Future]              # Request deduplication
│   ├── .lock: threading.Lock                     # Guards inflight map
│   │
│   ├── get(url, params, cache_key, ttl) -> dict | None
│   │   """Single entry point for all outbound HTTP.
│   │      1. Check cache (disk/memory) by cache_key.
│   │      2. If miss, acquire rate-limit token for url's domain.
│   │      3. Deduplicate: if same cache_key is already in-flight, await that Future.
│   │      4. Execute request with retry + backoff.
│   │      5. Store result in cache.
│   │      6. Return parsed JSON.
│   │   """
│   │
│   ├── get_many(requests: list[RequestSpec]) -> list[dict | None]
│   │   """Parallel fetch with ThreadPoolExecutor, respecting rate limits."""
│   │
│   ├── warm(keys: list[str]) -> None
│   │   """Pre-populate cache for a list of known keys. Used by warmer job."""
│   │
│   └── invalidate(pattern: str) -> int
│       """Remove cache entries matching a glob pattern. Returns count removed."""
│
├── get_client() -> NHLClient     # Module-level accessor. Lazy-init singleton.
│
└── Rate-limit defaults:
    RATE_LIMITS = {
        "api-web.nhle.com": TokenBucket(rate=20, per=1.0),   # 20 req/s
        "api.nhle.com":     TokenBucket(rate=15, per=1.0),   # 15 req/s
        "records.nhl.com":  TokenBucket(rate=10, per=1.0),   # 10 req/s
        "search.d3.nhle.com": TokenBucket(rate=5, per=1.0),  #  5 req/s
    }
```

### 3.2 How Existing Functions Change

Each function in `data_loaders.py` and `schedule.py` that currently does `requests.get(url, timeout=N)` changes to:

```python
# Before
@st.cache_data(ttl=3600)
def get_player_landing(player_id):
    resp = requests.get(f"{STATS_URL}{player_id}/landing", timeout=5)
    return resp.json()

# After
def get_player_landing(player_id):
    client = get_client()
    return client.get(
        url=f"{STATS_URL}{player_id}/landing",
        cache_key=f"player_landing:{player_id}",
        ttl=7200,   # 2 hours for T2
    )
```

The `@st.cache_data` decorator is **removed** from all API-fetching functions. The DAL's `client.get()` handles caching, rate limiting, retry, and deduplication internally. `st.cache_data` is retained **only** for T0 static artifact loaders and for any pure-compute transforms that are expensive but don't touch the network.

### 3.3 Retry & Backoff Policy

```
Retry strategy (applied inside NHLClient.get):
  - Max attempts: 3
  - Backoff: exponential (1s, 2s, 4s) + jitter (±0.5s)
  - Retry on: HTTP 429, 500, 502, 503, 504, ConnectionError, Timeout
  - Do NOT retry: 400, 403, 404 (bad request / not found — cache the miss)
  - On final failure: return None (existing code already handles None/empty gracefully)
  - On 429 specifically: respect Retry-After header if present, else back off 5s
```

### 3.4 Request Deduplication

When multiple threads/users request the same `cache_key` simultaneously:

```
Thread A: client.get("player_landing:8471675") → cache miss → creates Future, starts HTTP
Thread B: client.get("player_landing:8471675") → cache miss → sees in-flight Future → awaits it
Thread C: client.get("player_landing:8471675") → cache miss → sees in-flight Future → awaits it
                                                                  (only 1 HTTP call happens)
```

Implementation: a dict of `{cache_key: concurrent.futures.Future}` guarded by a `threading.Lock`. The lock is held only to check/insert the future — not during the HTTP call itself.

### 3.5 Rate Limiter

Token-bucket algorithm per domain. Each domain gets its own bucket with configurable rate (tokens/second) and burst (max tokens). `client.get()` acquires a token before issuing the request; if the bucket is empty, it blocks (with a timeout) until a token is available. This ensures that even under concurrent load from multiple users, the aggregate outbound rate to any NHL domain stays within bounds.

```python
class TokenBucket:
    def __init__(self, rate: float, per: float = 1.0, burst: int | None = None):
        self.rate = rate            # tokens per `per` seconds
        self.per = per
        self.burst = burst or int(rate * 2)
        self.tokens = self.burst
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 10.0) -> bool:
        """Block until a token is available or timeout."""
        ...
```

---

## 4. Shared Backend Cache

### 4.1 Recommended Store: `diskcache`

For this specific app, **`diskcache`** (Python library backed by SQLite) is the best fit.

**Why diskcache over alternatives:**

| Store | Pros | Cons | Verdict |
|-------|------|------|---------|
| **`diskcache`** | Zero infrastructure. Single pip install. Cross-process via filesystem. Automatic size management. Native TTL support. SQLite WAL mode handles concurrent reads well. | Single-machine only. Not suitable for multi-node. | **Best for Streamlit Cloud (single node, multiple workers)** |
| **Redis / Valkey** | Fast. Multi-node. Mature TTL. Pub/sub for invalidation. | Requires a running server. Not available on Streamlit Cloud free tier. Adds ops burden. | Overkill until you need multi-node or real-time invalidation |
| **Pure in-memory** (`st.cache_data`) | Zero overhead. Already working. | Per-process isolation. Lost on restart. Cold-start storms. | Current state — the problem we're solving |
| **Plain JSON/Parquet on disk** | Simple. Portable. | No built-in TTL. No concurrency handling. Manual cleanup. Key-value semantics are clunky with flat files. | Too primitive for 30+ cache keys with varying TTLs |
| **PostgreSQL** | ACID. Rich queries. | Heavy. Needs a running server. Data is mostly blobs, not relational. | Wrong tool for this job |
| **SQLite (raw)** | Lightweight. Cross-process. | Need to build TTL, serialization, eviction yourself. | `diskcache` already does this — don't reinvent it |

### 4.2 diskcache Configuration

```python
import diskcache

cache = diskcache.Cache(
    directory=".cache/nhl_api",   # Relative to app root. Gitignored.
    size_limit=200 * 1024 * 1024, # 200 MB max (generous; actual usage ~30 MB)
    eviction_policy="least-recently-used",
    disk_min_file_size=0,          # Store everything in SQLite (not loose files)
    statistics=True,               # Track hit/miss rates for monitoring
)
```

### 4.3 Cache Read/Write Flow

```
client.get(url, cache_key, ttl):
    ┌─────────────────────┐
    │ 1. diskcache.get()  │  ← O(1) SQLite lookup
    │    Hit? Return it.  │
    └────────┬────────────┘
             │ Miss
    ┌────────v────────────┐
    │ 2. Check inflight   │  ← threading.Lock (microseconds)
    │    map for cache_key│
    │    In-flight? Await.│
    └────────┬────────────┘
             │ Not in-flight
    ┌────────v────────────┐
    │ 3. Acquire rate-    │  ← TokenBucket.acquire() (may block up to ~1s)
    │    limit token      │
    └────────┬────────────┘
             │ Token acquired
    ┌────────v────────────┐
    │ 4. HTTP GET + retry │  ← requests.Session with retry (up to 3 attempts)
    └────────┬────────────┘
             │ Response
    ┌────────v────────────┐
    │ 5. diskcache.set()  │  ← Store with TTL
    │    Remove inflight  │
    │    Return result    │
    └────────────────────-┘
```

### 4.4 Two-Tier Strategy: Disk + In-Process

For hot-path data that's read many times per rerun (e.g., `load_all_team_seasons()`, `get_id_to_name_map()`), add a thin in-process LRU on top of the disk cache:

```python
@functools.lru_cache(maxsize=64)
def _inmem_get(cache_key, _generation):
    """In-process LRU. _generation is bumped when disk cache is updated."""
    return disk_cache.get(cache_key)
```

This avoids repeated SQLite reads within the same rerun cycle while still sharing across processes via disk. The `_generation` parameter (a monotonic counter stored in diskcache) lets us invalidate the in-process cache when the disk value changes.

For T3 live data, skip the in-process LRU (TTL is too short to benefit).

---

## 5. Evolving `async_preloader.py` into a Backend Warmer

### 5.1 Current State

`async_preloader.py` fires once per session, spawning daemon threads to warm `st.cache_data` for the two categories the user isn't currently viewing. It only helps the single process it runs in.

### 5.2 Target State

Replace it with a **startup warmer** + **periodic refresh** that populates the **shared disk cache**.

#### Startup Warmer (runs once when the app boots)

```python
# nhl/cache_warmer.py

def warm_on_startup():
    """Called once from app.py at import time (or via st.cache_resource).
    Populates disk cache with T1 and T2 data that every user will need.
    Runs in a background thread so it doesn't block the first page render.
    """
    client = get_client()

    # T1: Historical records (heaviest, most important to pre-cache)
    for cat in ("Skater", "Goalie"):
        for s_type in ("Regular", "Playoffs"):
            client.warm(f"records:{cat}:{s_type}")

    # T1: Derived maps
    for cat in ("Skater", "Goalie"):
        client.warm(f"id_name_map:{cat}")
        client.warm(f"clone_details:{cat}")

    # T2: Team seasons (used by many downstream functions)
    client.warm("all_team_seasons")

    # T2: Standings
    client.warm("standings")

    # T2: Top 50 sidebar data
    for metric in ("points", "goals", "assists"):
        client.warm(f"top50_skater:{metric}")
    client.warm("top50_goalie")
```

#### Periodic Refresh (background thread)

```python
def periodic_refresh():
    """Runs in a daemon thread. Refreshes T2 and T3 cache entries
    on a schedule so that user requests are almost always cache hits.
    """
    while True:
        try:
            # T3: Live data (every 2 minutes)
            refresh_live_data()

            # T2: Seasonal data (every 2 hours, but staggered)
            if minutes_since_last_seasonal_refresh() > 120:
                refresh_seasonal_data()

        except Exception:
            log.exception("Periodic refresh failed")

        time.sleep(120)  # Check every 2 minutes
```

#### How This Replaces `async_preloader.py`

| Old (`async_preloader`) | New (`cache_warmer`) |
|------------------------|---------------------|
| Per-session, per-process | Per-app, shared across all processes |
| Only warms category the user isn't viewing | Warms everything proactively |
| Results in `st.cache_data` (lost on restart) | Results in diskcache (survives restarts) |
| No periodic refresh | Background thread refreshes T2/T3 on schedule |
| Cold-start = 80-120 API calls | Cold-start = 0 API calls (disk cache hit) |

#### Triggering the Warmer

```python
# app.py (top-level, runs once per process)
@st.cache_resource
def _init_backend():
    from nhl.cache_warmer import warm_on_startup, periodic_refresh
    import threading
    threading.Thread(target=warm_on_startup, daemon=True).start()
    threading.Thread(target=periodic_refresh, daemon=True).start()

_init_backend()
```

`st.cache_resource` ensures this runs exactly once per process. The threads are daemons so they die when the process exits.

---

## 6. Future: Dedicated API Proxy Service

### 6.1 When to Introduce It

A dedicated proxy becomes worthwhile when:
- The app runs on **multiple nodes** (diskcache can't share across machines).
- You want **centralized observability** (request logs, hit rates, error rates) without instrumenting every Streamlit worker.
- NHL API behavior changes and you need a **single place** to adapt (e.g., new auth requirements, endpoint migrations).

For a single-node Streamlit Cloud deployment, the DAL + diskcache approach in sections 3-4 is sufficient. The proxy is a later evolution.

### 6.2 Architecture

```
Streamlit Workers ──→ API Proxy (FastAPI sidecar) ──→ NHL APIs
                           │
                     Redis / diskcache
                     Rate limiter
                     Request dedup
                     Retry + backoff
                     Metrics endpoint
```

The proxy would expose the same URL structure as the NHL APIs (or a simplified version) so that `NHLClient.get()` only needs to change its base URL:

```python
# Before (direct)
NHL_BASE = "https://api-web.nhle.com"

# After (proxied)
NHL_BASE = os.getenv("NHL_PROXY_URL", "https://api-web.nhle.com")
```

### 6.3 Structuring the DAL for Low-Friction Proxy Adoption

The `NHLClient` in `nhl/api.py` already isolates all HTTP concerns. To make the proxy transition seamless:

1. **All NHL URLs are defined in `constants.py`** (already the case). The proxy just overrides these base URLs via environment variables.
2. **`NHLClient.get()`** is the single outbound choke point. Redirecting it to a proxy is a one-line config change.
3. **Cache keys are URL-independent.** They use semantic keys (`player_landing:8471675`) not URL-based keys. This means switching from direct→proxy doesn't invalidate the cache.
4. **Rate limiting moves to the proxy.** `NHLClient` can detect it's talking to a proxy (via env var) and skip client-side rate limiting, letting the proxy handle it centrally.

---

## 7. Concrete Refactoring Plan

### Phase 1: Foundation (P0)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 1.1 | Create `nhl/api.py` with `NHLClient` class: `requests.Session`, retry/backoff (tenacity or manual), per-domain `TokenBucket` rate limiter, request deduplication via in-flight futures | New: `nhl/api.py` | Medium |
| 1.2 | Add `diskcache` dependency and create cache initialization in `NHLClient.__init__()` | `requirements.txt`, `nhl/api.py` | Low |
| 1.3 | Add `.cache/` to `.gitignore` | `.gitignore` | Trivial |

### Phase 2: Migration (P1)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 2.1 | Migrate `data_loaders.py` API calls: replace `requests.get()` + `@st.cache_data` with `client.get()` for all ~22 API-calling functions. Apply season-aware TTL logic for functions that accept `season_year`. | `nhl/data_loaders.py` | High (largest file, most call sites) |
| 2.2 | Migrate `schedule.py` API calls: same treatment for ~7 functions | `nhl/schedule.py` | Medium |
| 2.3 | Remove `@st.cache_data` from all migrated functions. Keep it only on `load_historical_data()` and `load_win_prob_weights()`. | `nhl/data_loaders.py`, `nhl/schedule.py` | Low |
| 2.4 | Keep thin `@st.cache_data` wrappers (short TTL, no API call) on derived/compute-only functions that are called many times per rerun but just read from the DAL cache (e.g., `get_player_identity_summary`, `get_team_identity_summary`). These prevent redundant deserialization within a single rerun. | `nhl/data_loaders.py` | Low |

### Phase 3: Warmer & Preloader (P2)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 3.1 | Create `nhl/cache_warmer.py` with `warm_on_startup()` and `periodic_refresh()` | New: `nhl/cache_warmer.py` | Medium |
| 3.2 | Replace `async_preloader.py` usage in `app.py` with `cache_warmer` initialization via `st.cache_resource` | `nhl/async_preloader.py`, `app.py` | Low |
| 3.3 | Add a CLI entry point for manual cache warming (`python -m nhl.cache_warmer`) for deployment scripts / cron | `nhl/cache_warmer.py` | Low |

### Phase 4: Parallelization (P2)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 4.1 | Parallelize `get_upcoming_games()`: use `NHLClient.get_many()` to fetch multiple dates concurrently | `nhl/schedule.py` | Low |
| 4.2 | Parallelize `fetch_all_time_records()`: fetch pages concurrently (first page to get total, then remaining in parallel) | `nhl/data_loaders.py` | Low |
| 4.3 | Parallelize `get_matchup_history()`: fetch game details concurrently after identifying games | `nhl/schedule.py` | Low |

### Phase 5: Proxy (P3, future)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 5.1 | Create `proxy/` FastAPI service with its own rate limiter + Redis cache | New directory | High |
| 5.2 | Add `NHL_PROXY_URL` env var support to `NHLClient` | `nhl/api.py` | Trivial |
| 5.3 | Add Docker Compose / deployment config for proxy sidecar | New files | Medium |

---

## 8. Backing Store Trade-Offs (Summary)

| Dimension | `st.cache_data` (current) | `diskcache` (recommended) | Redis | PostgreSQL |
|-----------|--------------------------|--------------------------|-------|------------|
| **Sharing** | Per-process only | Cross-process (same machine) | Cross-machine | Cross-machine |
| **Persistence** | Lost on restart | Survives restarts | Survives restarts (with persistence) | Survives restarts |
| **TTL support** | Built-in | Built-in | Built-in | Manual (needs cleanup job) |
| **Infrastructure** | None | None (pip install) | Requires running server | Requires running server |
| **Streamlit Cloud compatible** | Yes | Yes (filesystem access available) | No (no Redis on free tier) | No (no DB on free tier) |
| **Concurrency** | Thread-safe within process | SQLite WAL handles multi-process reads | Excellent | Excellent |
| **Latency** | Nanoseconds (in-memory) | ~1-5 ms (SQLite read) | ~1-2 ms (network) | ~2-5 ms (network + SQL) |
| **Complexity** | Zero | Low | Medium | High |
| **Multi-node** | No | No | Yes | Yes |
| **Cost** | Free | Free | $15-50/mo (managed) | $15-50/mo (managed) |

**Recommendation**: Start with `diskcache`. It solves the cross-process sharing problem with zero infrastructure, works on Streamlit Cloud, and the API surface is similar enough to Redis that migrating later (if needed for multi-node) is straightforward. The 1-5ms read latency is negligible compared to the 100-500ms NHL API calls it replaces.

If the app moves to a multi-node deployment, swap `diskcache` for Redis by changing only the cache backend in `NHLClient.__init__()`. The cache key format, TTL logic, and DAL interface stay identical.

---

## 9. Migration Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **Stale cache serves wrong data** | Season-aware TTL logic ensures current-season data refreshes hourly. `invalidate()` method available for manual purge. Warmer refreshes proactively. |
| **diskcache corruption** | SQLite WAL mode is crash-safe. Worst case: delete `.cache/nhl_api/` and let it rebuild (cold start, not data loss). |
| **Rate limiter too aggressive** | Start with generous limits (20 req/s to api-web, 15 to api, 10 to records). Tune down if NHL returns 429s. Log rate-limit waits for observability. |
| **Migration breaks existing functions** | Migrate one function at a time. Each function can independently use old (`st.cache_data`) or new (`client.get()`) path. No big-bang cutover needed. |
| **Cache size grows unbounded** | `diskcache` LRU eviction with 200 MB cap. Actual usage ~30 MB, so there's 6x headroom. |
| **diskcache not available on Streamlit Cloud** | Fallback: `NHLClient` detects missing filesystem access and degrades to `st.cache_data` as backend. This preserves all other benefits (rate limiting, retry, dedup). |
