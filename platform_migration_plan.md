# Puck Peak Platform Migration Plan

This document turns the Streamlit app into a phased migration program for:

- a real web product
- an Android app
- login and user accounts
- premium feature monetization
- production-grade caching and deployment on Hetzner

It is designed so each phase can be executed as a separate mini-project in a new AI context window.

## Short Answer

Yes, the current project can be ported with the features you want.

But the port is not "take Streamlit and press export". The reusable part is mostly the Python analytics and NHL data logic. The part that needs to be rebuilt is the app shell: UI, auth, billing, user persistence, API contracts, and mobile client delivery.

## Why Not Streamlit

Streamlit is the wrong framework for where this product needs to go. Specific reasons:

1. **Android kills Streamlit.** You cannot ship an APK from Streamlit. Even a WebView wrapper gives a poor mobile experience (full-page reruns, no offline, no native nav, no push notifications).

2. **Auth/billing need a real backend.** Streamlit Cloud gives you no webhook endpoints, no database connections, no background jobs. Firebase token verification, Stripe webhooks, and entitlement enforcement all require server-side control. Client-side-only entitlement checks are trivially bypassable.

3. **The cache already outgrew Streamlit.** The `NHLClient -> NHLCache -> diskcache` path exists because `@st.cache_data` was insufficient. There are 38 `@st.cache_data` decorators sitting on top of a perfectly good shared cache. They are redundant on the backend path.

4. **The rerun model fights the UI.** The codebase already has elaborate workarounds: `_dialog_opened_this_run` guards, click nonce tracking, JS bridges via `st.components.v2.component()`, debounced 500ms click handlers. A real SPA eliminates all of this.

Keep Streamlit running during migration as the reference implementation. It costs nothing to maintain and gives users continuity until the new frontend is ready.

### Framework Alternatives Evaluated

| Framework | Pros for Puck Peak | Cons for Puck Peak | Verdict |
|---|---|---|---|
| **Next.js** | Large ecosystem, react-plotly.js works, shared React with Expo RN for Android, strong auth integrations | Requires JS/TS knowledge, heavier tooling, Node.js on server | **Best fit** - chart interactivity demands a real SPA, Android demands React Native, React unifies both |
| **HTMX + FastAPI** | Minimal JS, Python-centric, lightweight | No code sharing with Android, chart click handlers and modals would be clunky, limited interactivity | Wrong for this app's interaction density |
| **Dash (Plotly)** | Same Plotly library, Python-only, callbacks replace reruns | No Android story, limited auth/billing ecosystem, lateral move from Streamlit not a structural upgrade | Solves the wrong problem |
| **Reflex** | Python-only, compiles to React | Immature ecosystem, limited mobile story, risky for production | Too early |
| **PWA instead of native Android** | Single codebase, no app store, no Expo | Unreliable push notifications on Android, no Play Store billing, no offline charts, perceived as "just a website" | Viable fallback if Android is lower priority |

## What Already Exists

This repo already has backend-like foundations.

Strong existing pieces:

- `nhl/api.py` already centralizes outbound NHL HTTP.
- `nhl/cache.py` already defines shared cache tiers and TTL policy.
- `cache_strategy.md` already thinks in backend terms, not just Streamlit hacks.
- `player_pipeline.py`, `team_pipeline.py`, `knn_engine.py`, `win_prob.py`, `era.py`, `baselines.py`, and much of `rarity.py` are mostly reusable business logic.

What does not exist yet:

- a real public API server for web/mobile clients
- authentication and user identity
- a database for users, saved boards, and entitlements
- billing and webhook handling
- server-enforced premium access control
- a non-Streamlit web frontend
- an Android client

So the right framing is:

- keep the Python analytics core
- extract it from Streamlit coupling
- build a real backend around it
- then rebuild the web app and Android app on top of that backend

## Reuse Map

High direct reuse:

- `nhl/api.py`
- `nhl/cache.py`
- `nhl/constants.py`
- `nhl/era.py`
- `nhl/knn_engine.py`
- `nhl/win_prob.py`
- `nhl/player_pipeline.py`
- `nhl/team_pipeline.py`

Medium reuse after refactor:

- `nhl/data_loaders.py`
- `nhl/schedule.py`
- `nhl/baselines.py`
- `nhl/rarity.py`
- `nhl/url_params.py`
- `nhl/cache_warmer.py`

Mostly rewrite:

- `app.py`
- `nhl/sidebar.py`
- `nhl/controls.py`
- `nhl/comparison.py`
- `nhl/chart.py`
- `nhl/dialog.py`
- `nhl/styles.py`
- anything centered on `st.session_state`, `@st.cache_data`, or Streamlit widget flow

## Target Architecture

Recommended target stack:

- Backend API: FastAPI
- Web app: Next.js
- Android app: Expo React Native
- Database: Postgres
- Shared cache: Redis or Valkey
- Auth: Firebase Auth
- Billing and entitlements:
  - Android: Google Play Billing
  - Web: Stripe Checkout
  - Cross-platform entitlement sync: RevenueCat recommended
- Deployment: Hetzner single-box first, split services later if needed

Initial production topology on one Hetzner server:

- `caddy` reverse proxy (automatic HTTPS)
- `nextjs` web app
- `fastapi` API service
- `worker` service for cache warming, async jobs, and webhooks
- `postgres`
- `redis`

You do not need to shut down Streamlit on day 1.

Recommended transition:

1. keep Streamlit as the reference app while extracting backend code
2. build the real API beside it
3. deploy backend to Hetzner early
4. build the new web frontend against the deployed API
5. build Android against the same API
6. cut traffic over
7. retire Streamlit last

## Hetzner Sizing

This repo is not heavy on raw disk. The historical parquet in the repo is under 1 MB on disk and about 10.5 MB in memory once loaded into pandas in the current environment. The bigger cost will come from:

- Python worker memory
- Postgres
- Redis
- Node.js for Next.js
- background jobs
- logs, backups, and snapshots

### Minimum viable one-box server

Use only for staging, private beta, or very light traffic:

- 4 vCPU
- 8 GB RAM
- 80 to 160 GB NVMe SSD

### Recommended first production box

This is the sweet spot if you want one machine running the full stack:

- 8 vCPU
- 16 GB RAM
- 160 to 320 GB NVMe SSD

### More comfortable headroom

If you want fewer resource worries while shipping and debugging:

- 8 to 12 vCPU
- 16 to 24 GB RAM
- 320 GB NVMe SSD

### Practical recommendation

For this project, start with:

- 16 GB RAM
- 160 GB minimum SSD
- 320 GB SSD if the price difference is small

Why:

- FastAPI + pandas + projection workloads can get memory-hungry under concurrency.
- Postgres + Redis + Next.js + worker on the same machine make 8 GB feel tight fast.
- 160 GB is enough for app code, DB, cache, images, logs, and backups for a long while.
- 320 GB buys much nicer operational headroom for snapshots, rollbacks, and mistakes.

### Current Hetzner cloud shapes worth looking at

As of March 29, 2026, Hetzner's regular-performance cloud pages show shapes around:

- 4 GB RAM / 80 GB SSD
- 8 GB RAM / 160 GB SSD
- 16 GB RAM / 320 GB SSD

For this project:

- 4 GB / 80 GB is too tight for the full stack
- 8 GB / 160 GB is okay for low traffic
- 16 GB / 320 GB is the safer first production choice

## Phase Rules

Each phase below is a separate mini-project.

Rules for running phases:

- run one phase per AI context window (sub-phases like 5A/5B/5C can each be a separate window)
- give the AI only that phase prompt plus the repo
- do not ask a phase to do future phases
- each phase should leave the repo in a working, testable state
- if a phase introduces new infra, it must include local dev instructions
- Streamlit should remain runnable until the final cutover phase unless the phase explicitly says otherwise

---

## Phase 1 - Service Core Extraction

### Goal

Create a clean service layer that the future FastAPI backend can call, without touching UI-only modules or requiring the full repo to run without Streamlit.

### Why this phase exists

The business logic needs a Streamlit-free path before building the new backend. But the extraction should be narrow and additive: create parallel service entry points alongside the existing Streamlit-wrapped functions, not replace them.

### Scope

What gets extracted (service-core modules only):

- `nhl/data_loaders.py` - the 25 `@st.cache_data` decorators here wrap functions that already delegate to `NHLClient` with its own diskcache layer. Create parallel Streamlit-free entry points (e.g., `get_player_raw_stats_svc()`) that skip `@st.cache_data` and go straight to `NHLClient`. The `@st.cache_data`-wrapped originals stay intact for the Streamlit app.
- `nhl/schedule.py` - same pattern: 7 `@st.cache_data` decorators. Add `_svc()` variants for the functions the API will need (upcoming games, matchup history, live scores). Leave the originals alone.
- `nhl/baselines.py` - 2 decorators. Add service entry points for baseline builders.
- For permanent-artifact loaders (`load_historical_data`, `load_win_prob_weights`): add `functools.lru_cache` variants alongside the existing `@st.cache_data` ones.

What does NOT get touched:

- `nhl/comparison.py`, `nhl/sidebar.py`, `nhl/controls.py`, `nhl/chart.py`, `nhl/dialog.py`, `nhl/styles.py` - these are UI-only modules. Their `@st.cache_data` and `st.session_state` usage stays exactly as-is. They will be rewritten from scratch when the Next.js frontend is built.
- `nhl/rarity.py` - its 2 decorators are fine for now. If the API needs rarity, it gets extracted in Phase 2 alongside endpoint work.
- `app.py` - untouched.

### New artifacts

- `nhl/services.py` - thin orchestration module that imports the `_svc()` functions and composes them into API-ready operations (e.g., "get player chart payload" = fetch raw stats + run pipeline + format response)
- `nhl/schemas.py` - Pydantic models for the core response shapes (player chart payload, team chart payload, upcoming games, win probability). These become the FastAPI response models in Phase 2.

### Files touched

- `nhl/data_loaders.py` (add `_svc()` variants)
- `nhl/schedule.py` (add `_svc()` variants)
- `nhl/baselines.py` (add `_svc()` variants)
- new `nhl/services.py`
- new `nhl/schemas.py`
- tests for the new service functions

### Definition of done

- `python -c "from nhl.services import get_player_chart_payload"` works in a plain Python environment (no Streamlit import required for this path)
- `streamlit run app.py` still works exactly as before
- New service-layer tests pass
- Existing tests still pass

### CI addition

Add a test job that runs only the service-layer tests without Streamlit installed, verifying the extraction boundary.

### AI prompt for this phase

```text
You are implementing Phase 1 of the Puck Peak platform migration. The goal is to create a Streamlit-free service layer for the core analytics, without changing the existing Streamlit app behavior.

Read these files first:
- platform_migration_plan.md (Phase 1 section and Reuse Map)
- readme.txt (Sections 1-4 for architecture context)
- cache_strategy.md
- nhl/api.py (full file - this is the HTTP client, already Streamlit-free)
- nhl/cache.py (full file - shared cache, already Streamlit-free)
- nhl/data_loaders.py (focus on functions decorated with @st.cache_data)
- nhl/schedule.py (focus on functions decorated with @st.cache_data)
- nhl/baselines.py (focus on the 2 @st.cache_data functions)
- nhl/player_pipeline.py (understand how it calls data_loaders)
- nhl/team_pipeline.py (understand how it calls data_loaders)
- nhl/knn_engine.py (pure Python, no changes needed - understand the interface)
- nhl/win_prob.py (pure Python, no changes needed - understand the interface)

Requirements:
- Create parallel `_svc()` entry points in data_loaders.py, schedule.py, and baselines.py that bypass @st.cache_data and go straight to NHLClient. Keep the originals intact.
- Create nhl/services.py that composes _svc() functions into API-ready operations.
- Create nhl/schemas.py with Pydantic models for response shapes.
- Add functools.lru_cache variants for permanent-artifact loaders (historical parquet, win_prob weights).
- Add tests for the new service layer that run without Streamlit installed.

Do NOT touch:
- UI-only modules (comparison.py, sidebar.py, controls.py, chart.py, dialog.py, styles.py)
- app.py
- rarity.py (extract later if needed)
- Any existing @st.cache_data decorators (leave them, add parallel paths)

Do NOT build FastAPI, Next.js, Docker, Redis, or Android yet.

Success verification:
- `python -c "from nhl.services import get_player_chart_payload"` succeeds without Streamlit
- `streamlit run app.py` still works
- All existing tests pass
- New service-layer tests pass
```

---

## Phase 2 - FastAPI Backend with Redis and Docker Compose

### Goal

Build a complete, locally-runnable backend with all read-only endpoints, Redis shared cache, background worker, and Docker Compose.

### Why this phase exists

Android and a real web app should not call NHL endpoints directly, and they should not depend on Streamlit reruns. The backend also needs a shared cache (Redis) from the start so endpoints behave the same in dev and production.

### Scope

- Create `backend/` package with FastAPI app
- Health, version, and config endpoints
- Read-only endpoints for: player search, player chart/projection payload, team chart payload, upcoming games, matchup history, win probability, standings, season leaderboards
- Wire endpoints to the extracted service core from Phase 1 using Pydantic models from `nhl/schemas.py`
- Add Redis backend adapter to `NHLCache` in `nhl/cache.py` (the interface is already close to Redis GET/SET/DELETE with TTL). Keep diskcache as fallback for local dev without Docker.
- Background worker using existing `cache_warmer.py` logic, running as standalone process instead of Streamlit daemon thread
- Docker Compose: FastAPI (uvicorn), Redis, worker, Postgres (provisioned empty for Phase 4)
- CORS for local frontend dev
- API versioning: all endpoints under `/api/v1/`
- OpenAPI auto-generation from Pydantic models

### Files touched

- new `backend/` directory (app, routes, config)
- `nhl/cache.py` (add Redis adapter alongside existing diskcache)
- `nhl/cache_warmer.py` (add standalone process mode)
- `docker-compose.yml`
- `Dockerfile`
- `.env.example`
- `requirements.txt` or `pyproject.toml` (add fastapi, uvicorn, redis)
- tests for endpoints

### Definition of done

- `docker compose up` starts all services (FastAPI, Redis, worker, Postgres)
- `curl localhost:8000/api/v1/health` returns 200
- `curl localhost:8000/api/v1/players/search?q=mcdavid` returns real data with player names
- Background worker logs cache warming activity
- OpenAPI docs accessible at `localhost:8000/docs`
- Streamlit app still works standalone (outside Docker)

### CI addition

Add a job that builds the Docker image and runs backend endpoint tests.

### AI prompt for this phase

```text
You are implementing Phase 2 of the Puck Peak platform migration. Build a FastAPI backend with Redis cache and Docker Compose on top of the service core from Phase 1.

Read these files first:
- platform_migration_plan.md (Phase 2 section)
- readme.txt (Section 1 for cache architecture, Section 5 for cache tiers)
- cache_strategy.md
- nhl/services.py (the service layer from Phase 1 - this is what endpoints call)
- nhl/schemas.py (Pydantic models from Phase 1 - these are the response shapes)
- nhl/cache.py (full file - you will add a Redis adapter here)
- nhl/api.py (full file - understand NHLClient for cache integration)
- nhl/cache_warmer.py (full file - you will make this runnable as a standalone worker)
- nhl/constants.py (lines 1-50 for URL patterns and stat sets)

Requirements:
- Build a FastAPI app in a new `backend/` package.
- Add `/api/v1/` prefix to all endpoints.
- Implement read-only endpoints:
  - GET /api/v1/health
  - GET /api/v1/version
  - GET /api/v1/players/search?q=<query>
  - GET /api/v1/players/{player_id}/chart?metric=<m>&category=<c>&...
  - GET /api/v1/teams/{team_abbr}/chart?metric=<m>&...
  - GET /api/v1/games/upcoming
  - GET /api/v1/games/matchup-history?team1=<t1>&team2=<t2>
  - GET /api/v1/win-probability?home=<h>&away=<a>
  - GET /api/v1/standings
- Add a Redis backend to NHLCache (keep diskcache as fallback when Redis is unavailable).
- Refactor cache_warmer.py to run as a standalone worker process.
- Create Docker Compose with: FastAPI (uvicorn), Redis, worker, Postgres (empty, provisioned for Phase 4).
- Add CORS middleware allowing localhost origins.
- Use Pydantic models from nhl/schemas.py for all responses.

Do NOT implement:
- Authentication or user accounts
- Billing or premium gates
- Web frontend (Next.js)
- Android app
- Hetzner deployment (that is Phase 3)

Do NOT modify the Streamlit app or UI-only modules.

Success verification:
- `docker compose up` brings up all 4 services
- `curl http://localhost:8000/api/v1/health` returns {"status": "ok"}
- `curl http://localhost:8000/api/v1/players/search?q=mcdavid` returns JSON with Connor McDavid
- `curl http://localhost:8000/docs` shows OpenAPI documentation
- Worker container logs show cache warming messages
- `streamlit run app.py` still works outside Docker
```

---

## Phase 3 - Hetzner Deployment, Monitoring, and Admin

### Goal

Deploy the backend to Hetzner. Add error monitoring, logging, and admin endpoints. The backend must be live on a real URL before any frontend work begins.

### Why this phase exists

The Next.js developer (or AI context) needs a real API URL to build against, not just localhost. Deployment problems surface early when the surface area is small (just the API). SSL, DNS, and reverse proxy configuration are independent of frontend code.

### Scope

- Hetzner server provisioning (16 GB RAM / 160-320 GB SSD)
- Docker Compose deployment with Caddy reverse proxy (automatic HTTPS)
- Environment separation: `.env.staging`, `.env.production`
- Sentry integration for error tracking on FastAPI and worker
- Structured JSON logging with rotation
- Admin/debug endpoints (behind API key): cache stats, cache invalidation, worker status, manual warm trigger
- Postgres backup scripts (empty now, but scripts ready for Phase 4)
- GitHub Actions CD: push to main deploys to staging, manual promotion to production
- Basic uptime monitoring (health check endpoint + external ping)

### Files touched

- `docker-compose.prod.yml` or production overlay
- `Caddyfile` (reverse proxy config)
- `backend/admin.py` (admin endpoints)
- `backend/middleware.py` (Sentry, logging)
- `.github/workflows/deploy.yml`
- `.env.staging`, `.env.production` templates
- `scripts/backup.sh`
- deployment documentation

### Definition of done

- `https://api.yourdomain.com/api/v1/health` returns 200 from Hetzner
- Sentry captures a test error
- GitHub push triggers automatic deploy to staging
- Admin cache-stats endpoint returns cache hit/miss counts
- Structured JSON logs are being written and rotated
- Streamlit app unaffected

### CI addition

Add deploy-to-staging job triggered on merge to main.

### AI prompt for this phase

```text
You are implementing Phase 3 of the Puck Peak platform migration. Deploy the FastAPI backend to a Hetzner server and add production monitoring.

Read these files first:
- platform_migration_plan.md (Phase 3 section and Hetzner Sizing section)
- docker-compose.yml (from Phase 2 - the local dev stack)
- Dockerfile (from Phase 2)
- backend/ directory structure (understand the app layout)
- .github/workflows/ (existing CI config)

Requirements:
- Create a production Docker Compose variant for Hetzner deployment.
- Add Caddy as a reverse proxy with automatic HTTPS (Let's Encrypt).
- Add Sentry SDK integration to FastAPI and the worker process.
- Add structured JSON logging with log rotation.
- Add admin endpoints behind an API key:
  - GET /api/v1/admin/cache-stats (hit/miss counts, size)
  - POST /api/v1/admin/cache-invalidate (selective or full flush)
  - GET /api/v1/admin/worker-status (last warm time, error count)
  - POST /api/v1/admin/warm (manual cache warm trigger)
- Create environment templates (.env.staging, .env.production).
- Create a GitHub Actions deployment workflow (push to main -> deploy to staging).
- Create a Postgres backup script (even though DB is empty now).
- Write deployment documentation.

Do NOT implement:
- Authentication or user accounts (Phase 4)
- Web frontend (Phase 5)
- Billing (Phase 6)
- Android (Phase 7)

Do NOT modify the Streamlit app or nhl/ analytics modules.

Success verification:
- Backend is reachable over HTTPS from a real domain
- Sentry dashboard shows a test error
- `curl https://api.yourdomain.com/api/v1/health` returns 200
- Admin endpoints work with the correct API key
- GitHub push triggers a staging deployment
- Logs are structured JSON
```

---

## Phase 4 - Auth, Users, Saved State, and Persistence

### Goal

Add real user identity and persistence for saved views, favorites, and future entitlements.

### Why this phase exists

Login and monetization require a user model. Saved boards and settings also need a real database.

### Scope

- Firebase Auth ID token verification on backend
- Postgres models via SQLAlchemy + Alembic migrations for: users, saved boards, favorites, entitlements placeholder
- Authenticated CRUD endpoints for saved state
- Rate limiting per authenticated user
- Auth dependency injection middleware in FastAPI
- Free-tier defaults (e.g., max 3 saved boards for free users)

### Files touched

- `backend/auth.py` (Firebase token verification, FastAPI dependency)
- `backend/models/` (SQLAlchemy models)
- `backend/routes/users.py` (user profile, saved boards, favorites)
- `alembic/` (migration config and initial migrations)
- Docker Compose Postgres config (activate the provisioned empty instance)
- `.env` files (Firebase project config, database URL)

### Definition of done

- Backend verifies a Firebase ID token and creates/retrieves the user record
- Authenticated user can POST/GET/DELETE saved boards
- Unauthorized requests to protected endpoints get 401
- Postgres data survives container restart (volume mounted)
- Alembic migrations run cleanly from scratch
- Existing unauthenticated endpoints still work without a token

### CI addition

Add Alembic migration check to CI (verify heads, no conflicts).

### AI prompt for this phase

```text
You are implementing Phase 4 of the Puck Peak platform migration. Add authentication, users, and persistence to the backend.

Read these files first:
- platform_migration_plan.md (Phase 4 section)
- backend/ directory (understand the existing FastAPI app structure, routes, config)
- docker-compose.yml (Postgres is provisioned but empty)
- nhl/url_params.py (understand the current share-link state format - saved boards will store similar data)

Requirements:
- Use Firebase Auth as the identity provider.
- Add a FastAPI dependency that verifies Firebase ID tokens from the Authorization header.
- Add SQLAlchemy models with Alembic migrations for:
  - users (firebase_uid, email, display_name, created_at, tier with default "free")
  - saved_boards (user_id, name, board_state as JSON, created_at, updated_at)
  - favorites (user_id, entity_type, entity_id, created_at)
  - entitlements (user_id, plan, status, provider, expires_at) - placeholder for Phase 6
- Add authenticated API endpoints:
  - GET /api/v1/me (current user profile)
  - GET /api/v1/boards (list saved boards)
  - POST /api/v1/boards (create saved board)
  - GET /api/v1/boards/{id} (get saved board)
  - PUT /api/v1/boards/{id} (update saved board)
  - DELETE /api/v1/boards/{id} (delete saved board)
  - GET /api/v1/favorites (list favorites)
  - POST /api/v1/favorites (add favorite)
  - DELETE /api/v1/favorites/{id} (remove favorite)
- Enforce free-tier limits (e.g., max 3 saved boards).
- Keep existing unauthenticated read-only endpoints working without a token.
- Add tests with mocked Firebase tokens.

Do NOT implement:
- Billing or Stripe integration (Phase 6)
- Web frontend (Phase 5)
- Android (Phase 7)

Success verification:
- `curl -H "Authorization: Bearer <valid_firebase_token>" https://api.yourdomain.com/api/v1/me` returns user profile
- `curl https://api.yourdomain.com/api/v1/players/search?q=mcdavid` still works without auth
- POST /api/v1/boards with auth creates a board; GET returns it
- POST without auth returns 401
- Alembic upgrade from empty DB succeeds
```

---

## Phase 5 - Next.js Web App

### Goal

Replace Streamlit with a Next.js web app covering all current product features.

### Structure

This phase has 3 sub-phases, each a viable standalone AI context window. Each sub-phase leaves the app in a deployable, testable state. Run them in order. If the full phase fits in one context window, run them together.

---

### Phase 5A - App Shell, Search, and Controls

#### Goal

Scaffold the Next.js app with auth, layout, search, and controls. No charts yet.

#### Scope

- Next.js app with TypeScript, project scaffolding, Tailwind CSS
- Firebase Auth integration (sign-in/sign-up/sign-out flows)
- API client layer consuming the FastAPI backend (base URL from env)
- Responsive layout: sidebar + main content area + right rail (matching current 62/38 split)
- Player/team search (consuming `/api/v1/players/search`)
- Add/remove player/team board management (React state)
- Category switching (Skater/Goalie/Team) with metric selector
- Controls panel: era, smoothing, cumulative, baseline, projection toggles
- Deploy to Hetzner alongside backend

#### Files touched

- new `web/` directory (Next.js project)
- Caddy config update (add frontend routing)
- backend CORS update if needed

#### Definition of done

- User can sign in via Firebase
- User can search for players and add them to a board
- Category switching and controls work
- API calls go to the deployed backend
- App is deployed alongside the backend on Hetzner

#### AI prompt for this phase

```text
You are implementing Phase 5A of the Puck Peak platform migration. Build the Next.js app shell with auth, search, and controls.

Read these files first:
- platform_migration_plan.md (Phase 5A section)
- readme.txt (Section 2 for file structure, Section 7 for sidebar/controls behavior)
- backend/routes/ (understand the API endpoints available)
- nhl/schemas.py (response shapes the API returns)
- nhl/sidebar.py (lines 1-100 for search behavior reference, lines 300-500 for board management)
- nhl/controls.py (full file - understand the control options and their states)
- nhl/constants.py (lines 1-80 for stat sets, metric names, category definitions)

Requirements:
- Create a Next.js app in a new `web/` directory using TypeScript and Tailwind CSS.
- Integrate Firebase Auth (email/password and Google sign-in).
- Build a typed API client that calls the FastAPI backend.
- Build the responsive layout:
  - Left sidebar: search input, player/team board with remove buttons
  - Main area: placeholder for chart (Phase 5B)
  - Right rail: placeholder for detail panels (Phase 5B)
- Implement player/team search with debounced API calls.
- Implement board management: add, remove, reorder players/teams.
- Implement category switching (Skater/Goalie/Team) that changes available metrics.
- Implement the controls panel with toggles for: Era Adjust, Smooth, Projection, Cumulative, Baseline, Prime Years.
- Add to Caddy config so the web app is served from the same Hetzner box.

Do NOT implement:
- Charts or Plotly (Phase 5B)
- Detail panels, predictions, or standings (Phase 5B)
- Share URLs or saved boards UI (Phase 5C)
- Billing or premium gates (Phase 6)

Success verification:
- User can sign in, search "McDavid", add him to the board
- Switching to Goalie category changes available metrics
- Controls toggle states are reflected in the URL or component state
- API calls hit the deployed backend
- App loads on the Hetzner domain
```

---

### Phase 5B - Charts and Detail Panels

#### Goal

Add chart rendering and the right-rail detail/prediction panels. This is the point where the app becomes a usable Streamlit alternative.

#### Scope

- Chart rendering using `react-plotly.js` - age curves with projection/baseline overlays
- Chart click handlers: click age point -> season detail modal, click team game -> game detail
- Season picker (all history vs single-season game-log mode)
- Right rail: Overview tab, Current Standings tab, Predictions tab
- Win probability cards (consuming `/api/v1/games/upcoming` + `/api/v1/win-probability`)
- Responsive chart sizing (desktop and mobile breakpoints)

#### Files touched

- `web/` components for chart, modals, detail panels
- Backend: possible new endpoints if current response shapes are insufficient

#### Definition of done

- Chart renders with real data for added players/teams
- Projections and baselines display correctly when toggled
- Clicking a data point opens a season detail modal
- Predictions panel shows upcoming games with win probability
- Chart is responsive on mobile

#### AI prompt for this phase

```text
You are implementing Phase 5B of the Puck Peak platform migration. Add charts and detail panels to the Next.js web app.

Read these files first:
- platform_migration_plan.md (Phase 5B section)
- web/ directory (the app shell from Phase 5A)
- nhl/chart.py (full file - understand trace styling, projection/baseline rendering, click dispatch logic)
- nhl/comparison.py (lines 1-200 for Overview tab, lines 800-1200 for Predictions panel, lines 1200-1700 for Standings tab)
- nhl/dialog.py (lines 1-150 for season detail modal structure)
- nhl/schemas.py (chart payload response shapes)
- backend/routes/ (available chart and game endpoints)

Requirements:
- Install and configure react-plotly.js.
- Build a chart component that:
  - Renders age curves (or Games Played / Season Year x-axis modes) for all board entities
  - Shows projection traces (dotted lines, open markers) when projection toggle is on
  - Shows baseline traces (dashed white lines) when baseline toggle is on
  - Handles click events on data points
- Build a season detail modal that opens when clicking a chart point, showing:
  - Season stats summary
  - Game context where applicable
- Build the right-rail tabs:
  - Overview: entity identity cards with key stats
  - Current Standings: live standings table
  - Predictions: upcoming game cards with win probability percentages
- Make the chart responsive (full width on mobile, 62% on desktop).

Do NOT implement:
- Matchup history modal (Phase 5C)
- Rarity callouts (Phase 5C)
- Share URLs (Phase 5C)
- Saved boards UI (Phase 5C)
- Billing (Phase 6)

Success verification:
- Add McDavid and Crosby, see their age curves on the chart
- Toggle projection on, see dotted projection lines appear
- Click a data point on McDavid's curve, see a season detail modal
- Predictions panel shows today's upcoming games with win probability
- Chart renders correctly on a 375px mobile viewport
```

---

### Phase 5C - Advanced Features and Share URLs

#### Goal

Complete feature parity with the Streamlit app. Add matchup history, rarity, share URLs, and saved boards UI.

#### Scope

- Matchup history modal (click prediction card -> last 10 head-to-head meetings)
- Age rarity callouts in season detail modals
- Games Played x-axis mode, Season Year mode (team)
- League filter with NHLe conversion display
- Compact share URLs: encode board state into query params, restore on load (port `nhl/url_params.py` logic)
- Saved boards: load/save via authenticated API endpoints
- Polish: loading states, error boundaries, empty states

#### Files touched

- `web/` components for matchup history, rarity, share URLs, saved boards
- Backend: rarity endpoint if not yet extracted, league filter support

#### Definition of done

- Clicking a prediction card opens a matchup history modal with last 10 meetings
- Season detail modal shows age rarity percentile
- Share URL encodes full board state; opening the URL restores it
- Authenticated user can save and load boards
- League filter works with NHLe conversion display
- All x-axis modes work (Age, Games Played, Season Year)

#### AI prompt for this phase

```text
You are implementing Phase 5C of the Puck Peak platform migration. Complete feature parity in the Next.js web app.

Read these files first:
- platform_migration_plan.md (Phase 5C section)
- web/ directory (the app from Phases 5A and 5B)
- nhl/url_params.py (full file - understand the compact state encoding/decoding)
- nhl/dialog.py (lines 150-400 for matchup history modal structure)
- nhl/rarity.py (lines 1-100 for ranking logic, lines 200-350 for display formatting)
- nhl/schedule.py (lines 400-700 for matchup history data loading)
- nhl/comparison.py (lines 200-500 for identity card click behavior)
- backend/routes/ (check which endpoints exist, which need to be added)

Requirements:
- Build a matchup history modal:
  - Triggered by clicking a prediction/game card
  - Shows last 10 head-to-head meetings between the two teams
  - Includes win summary and individual game cards
- Add age rarity callouts to the season detail modal:
  - Show percentile rank for the player's stat at that age
  - Show top-5 leaderboard for same age
- Add league filter (NHL + non-NHL leagues with NHLe conversion).
- Port the share URL encoding from nhl/url_params.py:
  - Encode board state (players, teams, metric, toggles) into compact query params
  - Restore full state when loading a shared URL
  - Support the existing param keys (pl, tm, cs, xm, lg, sm, pr, era, cu, bl, pf, etc.)
- Build saved boards UI:
  - Save current board (name + full state)
  - Load saved board from list
  - Delete saved board
- Add loading states, error boundaries, and empty states throughout the app.
- If the backend needs new endpoints (e.g., rarity data, league filter), add them.

Do NOT implement:
- Billing or premium gates (Phase 6)
- Android (Phase 7)

Success verification:
- Click a prediction card -> matchup history modal shows with game data
- Click a chart point -> season detail shows rarity percentile
- Copy a share URL, open in new tab -> board state restores exactly
- Save a board, refresh, load it -> state matches
- All Streamlit features have a working equivalent in the new web app
```

---

## Phase 6 - Premium Billing and Entitlements

### Goal

Monetize premium features with server-enforced access control.

### Why this phase exists

This is the phase that turns projections and premium surfaces into paid product features.

### Scope

- Define free vs premium feature matrix:
  - **Premium:** ML projections, advanced matchup prediction surfaces, saved boards beyond free limit, rarity/history details, future alerts
  - **Free:** Live search, basic stats, career totals, rosters, standings, upcoming games, recent scores
- Stripe Checkout for web subscriptions
- Webhook handlers for subscription lifecycle events
- Entitlement storage in Postgres, enforcement in backend middleware
- Premium gates in both backend responses (omit premium data) and frontend UI (show upgrade prompts)
- RevenueCat integration hooks for future Android compatibility
- Admin endpoint: manually grant/revoke premium for testing

### Files touched

- `backend/billing.py` (Stripe integration)
- `backend/routes/webhooks.py` (Stripe webhook handler)
- `backend/models/` (update entitlements model)
- `backend/middleware.py` (entitlement check dependency)
- `web/` (pricing page, account page, upgrade prompts, premium gates)
- Alembic migration for entitlements activation

### Definition of done

- Free user sees upgrade prompts on premium features (projections, rarity, extra saved boards)
- Paying user gets full access after Stripe checkout
- Subscription state survives page refresh and re-login
- Stripe test-mode webhook fires and updates entitlements in the database
- Admin can manually grant/revoke premium via endpoint
- Unpaid user cannot access premium data even by calling the API directly

### AI prompt for this phase

```text
You are implementing Phase 6 of the Puck Peak platform migration. Add premium billing and entitlements.

Read these files first:
- platform_migration_plan.md (Phase 6 section)
- backend/auth.py (existing Firebase auth integration)
- backend/models/ (existing user and entitlement models)
- backend/routes/ (all existing endpoints - you will gate some of them)
- web/ directory (understand the frontend structure)
- nhl/knn_engine.py (lines 1-50 - this is the projection engine, a premium feature)
- nhl/rarity.py (lines 1-50 - rarity ranking, a premium feature)

Requirements:
- Implement the free vs premium feature matrix as defined in the plan.
- Add Stripe Checkout integration:
  - Pricing page with subscription options
  - Checkout session creation endpoint
  - Success/cancel redirect handling
- Add Stripe webhook handler:
  - checkout.session.completed -> activate entitlement
  - customer.subscription.updated -> update tier/expiry
  - customer.subscription.deleted -> downgrade to free
  - Verify webhook signatures
- Update backend endpoints to check entitlements:
  - Projection data: return empty/teaser for free users
  - Rarity data: return limited data for free users
  - Saved boards: enforce free-tier limit
- Update frontend:
  - Show upgrade prompts where premium content is gated
  - Add account page with subscription management
  - Add pricing page
- Add admin endpoint: POST /api/v1/admin/entitlements (grant/revoke by user ID)
- Structure the entitlement model so Android purchases (Phase 7) can map into the same system later.

Do NOT build:
- Android app or Google Play Billing (Phase 7)
- RevenueCat integration (document the hook points only)

Success verification:
- Free user visits chart with projection toggle -> sees upgrade prompt instead of projection data
- Free user calls projection API directly -> gets 403 or empty response
- User completes Stripe test checkout -> entitlement activates -> projections appear
- Stripe webhook fires on subscription cancel -> user downgrades to free
- Admin grants premium to a user -> user gets premium access immediately
```

---

## Phase 7 - Android App

### Goal

Build the Android client on top of the same backend, auth, and entitlement system.

### Why this phase exists

This is the mobile product delivery layer, not a second backend.

### Scope

- Expo React Native app targeting Android
- Firebase Auth integration
- Backend API consumption (same endpoints as web)
- Core product flows: login, player/team search, charts, predictions, premium gating
- Google Play Billing integration (or RevenueCat for cross-platform entitlement sync)

### PWA alternative

If native Android is not worth the effort, this phase becomes: add a service worker and web app manifest to the Next.js app for installability, skip Play Store billing (use Stripe for everything). This saves roughly half the phase effort but loses Play Store distribution and native billing.

### Files touched

- new `mobile/` directory (Expo React Native project)
- backend: mobile-support endpoints if response shapes need adjustment
- entitlement sync hooks if using RevenueCat

### Definition of done

- Android app installs on a device/emulator
- User can sign in via Firebase
- User can search players, view charts, see predictions
- Premium features are gated for free users
- Subscription via Google Play Billing activates premium access

### AI prompt for this phase

```text
You are implementing Phase 7 of the Puck Peak platform migration. Build the Android app MVP.

Read these files first:
- platform_migration_plan.md (Phase 7 section)
- backend/routes/ (all API endpoints the mobile app will consume)
- nhl/schemas.py (response shapes)
- web/ directory (reference for product flows and component structure)
- backend/billing.py (understand the entitlement model for premium gates)

Requirements:
- Build an Expo React Native app in a new `mobile/` directory targeting Android.
- Integrate Firebase Auth (same project as web).
- Consume the FastAPI backend only (same endpoints as the web app).
- Implement core product flows:
  - Sign-in / sign-up
  - Player and team search with add/remove
  - Chart display (use react-native-plotly or a WebView-based Plotly renderer)
  - Predictions panel with upcoming games and win probability
  - Premium gates matching the web experience
- Integrate subscriptions using Google Play Billing (or RevenueCat if cross-platform sync with Stripe is desired).
- Keep the UX mobile-first: bottom navigation, swipeable panels, touch-friendly chart interactions.

Do NOT:
- Duplicate backend logic in the mobile app
- Build iOS support yet (Android first)
- Modify the backend entitlement model (map Android purchases into the existing structure)

Success verification:
- App installs and runs on an Android emulator
- User can sign in with same Firebase credentials as web
- Charts render with real data from the backend
- Free user sees upgrade prompts on premium features
- Google Play test purchase activates premium access
```

---

## Phase 8 - Streamlit Retirement (Optional)

### Goal

Cut over fully and retire Streamlit.

### Scope

- Remove Streamlit from `requirements.txt`
- Archive `_svc()` compatibility patterns if no longer needed
- Update DNS (point old Streamlit Cloud URL to new web app, or set up redirect)
- Update README.md and readme.txt
- Remove Streamlit-only UI modules if desired (or keep as historical reference)

### Definition of done

- Streamlit Cloud deployment is shut down
- All traffic goes to the new stack
- Old Streamlit URL redirects to the new web app

### Note

This can be combined with Phase 7 if the context window allows, or done as a quick standalone cleanup pass. It is mostly documentation and configuration changes.

---

## Execution Order

1. Phase 1 - Service Core Extraction
2. Phase 2 - FastAPI Backend with Redis and Docker Compose
3. Phase 3 - Hetzner Deployment, Monitoring, and Admin
4. Phase 4 - Auth, Users, Saved State, and Persistence
5. Phase 5A - App Shell, Search, and Controls
6. Phase 5B - Charts and Detail Panels
7. Phase 5C - Advanced Features and Share URLs
8. Phase 6 - Premium Billing and Entitlements
9. Phase 7 - Android App
10. Phase 8 - Streamlit Retirement (optional, can combine with Phase 7)

## Success Metrics By Phase

Phase 1 success: service core runs from plain Python without Streamlit

Phase 2 success: real API returns hockey analytics data, Docker Compose runs locally

Phase 3 success: backend is live on Hetzner with HTTPS, monitoring, and CI/CD

Phase 4 success: users can sign in and save their state

Phase 5A success: web app shell deployed with search and controls

Phase 5B success: charts and predictions work, app is a viable Streamlit alternative

Phase 5C success: full feature parity with Streamlit, share URLs work

Phase 6 success: premium access is monetized and enforced

Phase 7 success: Android app uses same backend and entitlements

Phase 8 success: Streamlit is retired, all traffic on the new stack

## Final Recommendation

If you want the shortest path that still scales:

- keep Python analytics
- build FastAPI around it
- deploy backend to Hetzner early
- build Next.js for web
- build Expo for Android
- use Postgres + Redis
- use Firebase Auth
- use Stripe for web billing
- use Google Play Billing for Android
- use RevenueCat if you want the cleanest shared entitlement model across web and Android
- start on a 16 GB RAM / 160-320 GB SSD Hetzner box

That gets you to a real product without throwing away the best parts of the current repo.
