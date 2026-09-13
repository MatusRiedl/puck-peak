"""Offline trainer and backtester for the pregame win-probability model (artifact v2).

Run ``python train_win_prob.py``. It fetches every regular season and playoff since
2017-18 from the NHL stats API (about 30 requests), then:

1. Runs a rolling-origin backtest. Each test season is predicted by a model whose
   hyperparameters and weights only saw earlier seasons.
2. Compares against the previous feature design (standings form, label fixed), a
   constant home rate, and playoff games scored separately.
3. Fits the overtime model the season simulator needs.
4. Calibrates the simulator's strength noise on historical checkpoints and compares its
   P(make playoffs) against a naive points-pace simulation.
5. Backtests playoff series probabilities (report only; the sample is small).

The artifact is written only if both gates pass. Exit code 1 means a gate failed.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from nhl.constants import (
    STANDINGS_DATE_URL,
    STANDINGS_SEASON_URL,
    TEAM_LIST_URL,
    TEAM_SHOOTING_URL,
    TEAM_STATS_URL,
    previous_season_year,
)
from nhl.goal_model import (
    DEFAULT_GOAL_MODEL,
    GOAL_MODEL_VERSION,
    MAX_GOALS,
    price_games,
    regulation_goals,
    score_grid,
    scoring_environment,
    solve_home_share,
    validate_goal_model,
)
from nhl.season_sim import series_win_probability, simulate_season
from nhl.team_ratings import (
    DEFAULT_RATING_PARAMS,
    LEAGUE_MEAN_ELO,
    MODEL_FEATURES,
    PLAYOFFS,
    REGULAR_SEASON,
    back_to_back_flags,
    build_league_game_table,
    build_model_features,
    canonical_team_abbrev,
    current_team_snapshot,
    run_elo,
)
from nhl.win_prob import (
    WIN_PROB_MODEL_VERSION,
    decompose_linear_model,
    score_home_win_logits,
    validate_model_artifact,
)
from nhl.xg import (
    RESEARCH_FEATURES,
    XG_FEATURES,
    GoalieHistory,
    build_goalie_features,
    parse_play_by_play,
    research_game_features,
    score_expected_goals,
    summarize_games,
    validate_xg_model,
    xg_design_matrix,
)

OUTPUT_PATH = Path("win_prob_weights.json")
REQUEST_TIMEOUT = 90
REQUEST_HEADERS = {"User-Agent": "puck-peak/1.0"}

FIRST_SEASON = 2017
"""Elo warm-up season. Its own games are never used as training rows."""
FIRST_TEST_SEASON = 2021
SIMULATION_SKIPPED_SEASONS = {2019, 2020}
"""2019-20 ended early and used a 24-team bubble; 2020-21 used one-off divisions."""

C_CANDIDATES = (0.05, 0.3, 1.0)
ELO_GRID = {
    "elo_k": (4.0, 6.0, 8.0, 10.0),
    "elo_home_advantage": (15.0, 25.0, 40.0),
    "elo_carryover": (0.6, 0.7, 0.8, 0.9),
}
FORM_PRIOR_WEIGHTS = (8.0, 15.0, 25.0)
FEATURE_SETS = {
    "full": list(MODEL_FEATURES),
    "no_attempt_share": [name for name in MODEL_FEATURES if name != "sat_share_shrunk_diff"],
    "no_shot_share": [name for name in MODEL_FEATURES if name != "sog_share_shrunk_diff"],
    "rating_and_rest": ["elo_diff", "home_back_to_back", "away_back_to_back"],
}
LEGACY_FEATURES = [
    "point_pct_to_date",
    "goal_diff_per_game_to_date",
    "l10_point_pct",
    "l10_goal_diff_per_game",
    "power_play_pct_to_date",
]
LEGACY_C = 0.1
LEGACY_MIN_GAMES = 5

GATE_MIN_MEAN_GAIN = 0.005
"""Required mean log-loss gain over the previous design across test seasons."""

SIM_CHECKPOINTS = ("opening", "12-01", "02-01", "03-15")
SIM_SD_PRESEASON = (0.0, 0.2, 0.4, 0.6, 0.8)
SIM_SD_LATE = (0.0, 0.1, 0.2)
SIM_CALIBRATION_SIMS = 2000
POINTS_PACE_PRIOR_GAMES = 10.0


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _get_json_with_retries(
    url: str,
    params: dict | None = None,
    timeout: int = REQUEST_TIMEOUT,
    max_attempts: int = 5,
    base_sleep: float = 0.75,
) -> dict | list:
    """Fetch one JSON payload with light retry/backoff."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, params=params, timeout=timeout, headers=REQUEST_HEADERS)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    sleep_for = float(retry_after)
                except Exception:
                    sleep_for = base_sleep * (attempt + 1)
                time.sleep(sleep_for)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_sleep * (attempt + 1))

    if last_error is not None:
        raise last_error
    raise ValueError(f"Failed to fetch JSON from {url}")


def _fetch_team_id_to_abbrev_map() -> dict[int, str]:
    """Fetch the NHL team ID -> triCode map once."""
    payload = _get_json_with_retries(TEAM_LIST_URL)
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    mapping: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            team_id = int(row.get("id", 0) or 0)
        except Exception:
            team_id = 0
        tri_code = str(row.get("triCode", "") or "").strip().upper()
        if team_id > 0 and tri_code:
            mapping[team_id] = tri_code
    if not mapping:
        raise ValueError("Could not build team ID map from NHL team-list endpoint.")
    return mapping


def _fetch_team_game_rows(url: str, season_year: int, game_type_id: int) -> list[dict]:
    """Fetch every per-team game row of one report for one season and game type."""
    payload = _get_json_with_retries(
        url,
        params={
            "isGame": "true",
            "limit": -1,
            "start": 0,
            "sort": "gameDate",
            "cayenneExp": f"seasonId={season_year}{season_year + 1} and gameTypeId={game_type_id}",
        },
    )
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def fetch_league_history(first_season: int, last_season: int, team_id_map: dict[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the completed-game table and raw regular-season rows for every season.

    Returns:
        ``(game_table, regular_season_summary_rows)``. The raw rows feed the legacy
        baseline, which needs points and power-play columns the game table drops.
    """
    tables: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    for season_year in range(first_season, last_season + 1):
        span = f"{season_year}-{str(season_year + 1)[2:]}"
        regular = _fetch_team_game_rows(TEAM_STATS_URL, season_year, REGULAR_SEASON)
        shooting = _fetch_team_game_rows(TEAM_SHOOTING_URL, season_year, REGULAR_SEASON)
        playoffs = _fetch_team_game_rows(TEAM_STATS_URL, season_year, PLAYOFFS)
        if not regular:
            raise ValueError(f"No regular-season rows returned for {span}.")
        table = build_league_game_table(regular + playoffs, shooting, team_id_map)
        tables.append(table)
        raw = pd.DataFrame(regular)
        raw["SeasonYear"] = season_year
        raw_frames.append(raw)
        counts = table.groupby("GameTypeId").size().to_dict()
        print(f"  {span}: {counts.get(REGULAR_SEASON, 0)} regular-season and {counts.get(PLAYOFFS, 0)} playoff games")
    return pd.concat(tables, ignore_index=True), pd.concat(raw_frames, ignore_index=True)


def fetch_division_maps(seasons: list[int]) -> dict[int, dict[str, tuple[str, str]]]:
    """Return team -> (conference, division) as of each season's final standings date."""
    payload = _get_json_with_retries(STANDINGS_SEASON_URL)
    end_dates = {
        int(row["id"]) // 10000: str(row.get("standingsEnd") or "")
        for row in (payload.get("seasons", []) if isinstance(payload, dict) else [])
        if isinstance(row, dict) and row.get("id")
    }
    maps: dict[int, dict[str, tuple[str, str]]] = {}
    for season_year in seasons:
        end_date = end_dates.get(season_year)
        if not end_date:
            continue
        standings = _get_json_with_retries(STANDINGS_DATE_URL.format(end_date))
        season_map: dict[str, tuple[str, str]] = {}
        for row in (standings.get("standings", []) if isinstance(standings, dict) else []):
            abbr_payload = row.get("teamAbbrev")
            abbr = canonical_team_abbrev(abbr_payload.get("default") if isinstance(abbr_payload, dict) else abbr_payload)
            if abbr:
                season_map[abbr] = (str(row.get("conferenceName") or ""), str(row.get("divisionName") or ""))
        maps[season_year] = season_map
    return maps


# ---------------------------------------------------------------------------
# Play-by-play history (xG training data), cached locally under .cache/
# ---------------------------------------------------------------------------

PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
PBP_CACHE_DIR = Path(".cache") / "xg_training"
PBP_WORKERS = 6
PBP_REQUEST_INTERVAL = 0.12
"""Seconds between request starts across all workers (about 8 requests per second)."""

_pbp_rate_lock = threading.Lock()
_pbp_next_request = [0.0]


def _fetch_play_by_play(game_id: int) -> dict | None:
    """Fetch one play-by-play payload politely, or ``None`` when the game has none."""
    for attempt in range(5):
        with _pbp_rate_lock:
            wait = _pbp_next_request[0] - time.monotonic()
            _pbp_next_request[0] = max(_pbp_next_request[0], time.monotonic()) + PBP_REQUEST_INTERVAL
        if wait > 0:
            time.sleep(wait)
        try:
            response = requests.get(PBP_URL.format(int(game_id)), timeout=30, headers=REQUEST_HEADERS)
            if response.status_code == 404:
                return None
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def _read_cached_season(shots_path: Path, meta_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Load one season's cached shot rows and game metas."""
    if not shots_path.exists() or not meta_path.exists():
        return pd.DataFrame(), []
    metas = json.loads(meta_path.read_text(encoding="utf-8"))
    for meta in metas:
        meta["goalie_names"] = {int(key): value for key, value in (meta.get("goalie_names") or {}).items()}
    return pd.read_parquet(shots_path), metas


def load_play_by_play_history(games: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return shot rows and game metas for every game, fetching only what the cache lacks.

    Completed seasons never change, so each season/game-type pair is fetched once and kept
    in ``.cache/xg_training`` (gitignored). A game with no play-by-play is recorded as
    missing so reruns do not keep asking for it.

    Args:
        games: Game table with ``GameId``, ``SeasonYear`` and ``GameTypeId``.

    Returns:
        ``(shots, metas)`` across all requested games.
    """
    PBP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_shots: list[pd.DataFrame] = []
    all_metas: list[dict] = []
    for (season_year, game_type), group in games.groupby(["SeasonYear", "GameTypeId"]):
        shots_path = PBP_CACHE_DIR / f"shots_{season_year}_{game_type}.parquet"
        meta_path = PBP_CACHE_DIR / f"meta_{season_year}_{game_type}.json"
        cached_shots, cached_metas = _read_cached_season(shots_path, meta_path)
        known = {int(meta["game_id"]) for meta in cached_metas}
        missing = sorted(int(game_id) for game_id in group["GameId"] if int(game_id) not in known)
        if missing:
            print(f"  play-by-play {season_year}-{str(season_year + 1)[2:]} type {game_type}: fetching {len(missing)} games...")
            new_frames, new_metas = [], []
            started = time.time()
            with ThreadPoolExecutor(max_workers=PBP_WORKERS) as pool:
                futures = {pool.submit(_fetch_play_by_play, game_id): game_id for game_id in missing}
                for done, future in enumerate(as_completed(futures), start=1):
                    game_id = futures[future]
                    payload = future.result()
                    if not payload:
                        new_metas.append({"game_id": game_id, "missing": True, "goalie_names": {}})
                    else:
                        parsed = parse_play_by_play(payload)
                        new_frames.append(parsed["shots"])
                        new_metas.append(parsed["meta"])
                    if done % 250 == 0:
                        print(f"    {done}/{len(missing)} in {time.time() - started:.0f}s")
            frames = [frame for frame in [cached_shots] + new_frames if not frame.empty]
            cached_shots = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            cached_metas = cached_metas + new_metas
            if not cached_shots.empty:
                cached_shots.to_parquet(shots_path, index=False)
            serializable = [dict(meta, goalie_names={str(key): value for key, value in (meta.get("goalie_names") or {}).items()}) for meta in cached_metas]
            meta_path.write_text(json.dumps(serializable), encoding="utf-8")
        all_shots.append(cached_shots)
        all_metas.extend(meta for meta in cached_metas if not meta.get("missing"))
    frames = [frame for frame in all_shots if not frame.empty]
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), all_metas


def backfill_play_by_play_cache() -> None:
    """Fill the local play-by-play cache for every season the trainer uses."""
    team_id_map = _fetch_team_id_to_abbrev_map()
    games, _ = fetch_league_history(FIRST_SEASON, previous_season_year(), team_id_map)
    shots, metas = load_play_by_play_history(games)
    print(f"Cached {len(shots)} unblocked shots from {len(metas)} games.")


XG_C = 1.0


def fit_xg_model(shots: pd.DataFrame, seasons: list[int]) -> dict:
    """Fit the logistic expected-goals model on unblocked shots from ``seasons``.

    Args:
        shots: ``xg.SHOT_COLUMNS`` rows.
        seasons: Season start years whose shots train the model.

    Returns:
        A JSON-ready xG model block (``xg.validate_xg_model`` accepts it) with a content
        version hash and training metadata.
    """
    training = shots[shots["SeasonYear"].isin(seasons)]
    matrix = xg_design_matrix(training)
    labels = training["IsGoal"].astype(int).to_numpy()
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(C=XG_C, max_iter=3000).fit(scaler.transform(matrix), labels)
    payload = {
        "feature_order": list(XG_FEATURES),
        "coefficients": [float(value) for value in model.coef_[0].tolist()],
        "intercept": float(model.intercept_[0]),
        "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
    }
    fingerprint = json.dumps([payload["coefficients"], payload["intercept"], payload["scaler_mean"], payload["scaler_scale"]])
    payload["version"] = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
    payload["training_seasons"] = [int(season) for season in seasons]
    payload["training_shots"] = int(len(training))
    return payload


def xg_metrics_by_season(shots: pd.DataFrame, payload: dict) -> list[dict]:
    """Report xG log loss, AUC and goals-to-xG ratio per season (calibration drift check)."""
    xg_model = validate_xg_model(payload)
    probabilities = score_expected_goals(shots, xg_model)
    frame = pd.DataFrame({"season": shots["SeasonYear"].to_numpy(), "goal": shots["IsGoal"].astype(int).to_numpy(), "xg": probabilities})
    rows = []
    for season, group in frame.groupby("season"):
        rows.append(
            {
                "season": int(season),
                "shots": int(len(group)),
                "log_loss": float(log_loss(group["goal"], np.clip(group["xg"], 1e-6, 1 - 1e-6), labels=[0, 1])),
                "auc": float(roc_auc_score(group["goal"], group["xg"])),
                "goals_per_xg": float(group["goal"].sum() / max(group["xg"].sum(), 1e-9)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Legacy baseline (the design this model replaces, with the shootout label fixed)
# ---------------------------------------------------------------------------

def build_legacy_feature_rows(raw_rows: pd.DataFrame, team_id_map: dict[int, str], games: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the v1 standings-form features so the gate compares like with like."""
    d = raw_rows.copy()
    d["Team"] = pd.to_numeric(d["teamId"], errors="coerce").map(team_id_map).map(canonical_team_abbrev)
    d["GameId"] = pd.to_numeric(d["gameId"], errors="coerce").astype("int64")
    d["HomeRoad"] = d["homeRoad"].astype(str).str.upper()
    d["Points"] = pd.to_numeric(d["points"], errors="coerce").fillna(0.0)
    d["GoalDiff"] = pd.to_numeric(d["goalsFor"], errors="coerce").fillna(0.0) - pd.to_numeric(d["goalsAgainst"], errors="coerce").fillna(0.0)
    power_play = pd.to_numeric(d.get("powerPlayPct"), errors="coerce")
    d["PowerPlay"] = power_play.where(power_play.abs() > 1.5, power_play * 100.0)
    d = d.sort_values(["SeasonYear", "Team", "gameDate", "GameId"], kind="stable").reset_index(drop=True)

    group_keys = [d["SeasonYear"], d["Team"]]
    d["GamesBefore"] = d.groupby(["SeasonYear", "Team"]).cumcount()
    games_before = d["GamesBefore"].replace(0, np.nan)
    d["point_pct_to_date"] = (d["Points"].groupby(group_keys).cumsum() - d["Points"]) / games_before / 2.0
    d["goal_diff_per_game_to_date"] = (d["GoalDiff"].groupby(group_keys).cumsum() - d["GoalDiff"]) / games_before
    previous_points = d.groupby(["SeasonYear", "Team"])["Points"].shift(1)
    previous_goal_diff = d.groupby(["SeasonYear", "Team"])["GoalDiff"].shift(1)
    rolling_games = previous_points.groupby(group_keys).transform(lambda s: s.rolling(10, min_periods=1).count())
    d["l10_point_pct"] = previous_points.groupby(group_keys).transform(lambda s: s.rolling(10, min_periods=1).sum()) / rolling_games.replace(0, np.nan) / 2.0
    d["l10_goal_diff_per_game"] = previous_goal_diff.groupby(group_keys).transform(lambda s: s.rolling(10, min_periods=1).sum()) / rolling_games.replace(0, np.nan)
    pp_valid = d["PowerPlay"].notna().astype(float)
    pp_sum_before = d["PowerPlay"].fillna(0.0).groupby(group_keys).cumsum() - d["PowerPlay"].fillna(0.0)
    pp_count_before = pp_valid.groupby(group_keys).cumsum() - pp_valid
    d["power_play_pct_to_date"] = pp_sum_before / pp_count_before.replace(0, np.nan)

    columns = ["GameId", "GamesBefore"] + LEGACY_FEATURES
    home = d[d["HomeRoad"].eq("H")][columns].set_index("GameId")
    away = d[d["HomeRoad"].eq("R")][columns].set_index("GameId")
    paired = home.join(away, lsuffix="_home", rsuffix="_away", how="inner")
    paired = paired[(paired["GamesBefore_home"] >= LEGACY_MIN_GAMES) & (paired["GamesBefore_away"] >= LEGACY_MIN_GAMES)]
    for feature in LEGACY_FEATURES:
        paired[feature] = paired[f"{feature}_home"] - paired[f"{feature}_away"]
    paired = paired.dropna(subset=LEGACY_FEATURES).reset_index()
    labels = games[games["GameTypeId"].eq(REGULAR_SEASON)][["GameId", "SeasonYear", "HomeWin"]]
    return paired[["GameId"] + LEGACY_FEATURES].merge(labels, on="GameId", how="inner")


# ---------------------------------------------------------------------------
# Model fitting helpers
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Return log loss, Brier score, accuracy and sample size."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
    }


def _fit_logistic(frame: pd.DataFrame, features: list[str], c_value: float, label: str = "HomeWin") -> tuple[StandardScaler, LogisticRegression]:
    """Fit a standardized logistic regression."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(frame[features].to_numpy(dtype=float))
    model = LogisticRegression(C=float(c_value), max_iter=2000, solver="lbfgs")
    model.fit(x_train, frame[label].to_numpy(dtype=int))
    return scaler, model


def _predict(scaler: StandardScaler, model: LogisticRegression, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Return home-win probabilities for every row."""
    return model.predict_proba(scaler.transform(frame[features].to_numpy(dtype=float)))[:, 1]


def precompute_elo_diffs(games: pd.DataFrame) -> dict[tuple[float, float, float], pd.Series]:
    """Run Elo once per grid point. Elo is online, so one pass serves every fold."""
    ordered = games.reset_index(drop=True)
    diffs: dict[tuple[float, float, float], pd.Series] = {}
    for combo in product(*ELO_GRID.values()):
        params = dict(zip(ELO_GRID.keys(), combo))
        run = run_elo(ordered, params)
        diffs[combo] = pd.Series(run.pregame_diff.to_numpy(), index=ordered["GameId"].to_numpy())
    return diffs


def select_elo_params(games: pd.DataFrame, elo_diffs: dict, seasons: list[int]) -> tuple[float, float, float]:
    """Pick the Elo grid point with the best Elo-only log loss on the given seasons."""
    mask = games["GameTypeId"].eq(REGULAR_SEASON) & games["SeasonYear"].isin(seasons)
    game_ids = games.loc[mask, "GameId"].to_numpy()
    labels = games.loc[mask, "HomeWin"].to_numpy(dtype=int)

    def _loss(combo: tuple[float, float, float]) -> float:
        """Elo-only log loss for one grid point."""
        home_advantage = combo[1]
        gap = elo_diffs[combo].loc[game_ids].to_numpy() + home_advantage
        return float(log_loss(labels, 1.0 / (1.0 + 10.0 ** (-gap / 400.0)), labels=[0, 1]))

    return min(elo_diffs, key=_loss)


def _with_elo(form_table: pd.DataFrame, elo_diffs: dict, combo: tuple) -> pd.DataFrame:
    """Swap in the Elo difference computed with one grid point."""
    table = form_table.copy()
    table["elo_diff"] = table["GameId"].map(elo_diffs[combo])
    return table


def select_config(
    train_seasons: list[int],
    combo: tuple,
    form_tables: dict[float, pd.DataFrame],
    elo_diffs: dict,
    feature_sets: dict[str, list[str]] | None = None,
) -> dict:
    """Choose prior weight, feature set and C on the last training season (nested)."""
    feature_sets = feature_sets or FEATURE_SETS
    inner_train, inner_validation = train_seasons[:-1], train_seasons[-1]
    best: dict | None = None
    for prior_weight, form_table in form_tables.items():
        table = _with_elo(form_table, elo_diffs, combo)
        regular = table[table["GameTypeId"].eq(REGULAR_SEASON)]
        train = regular[regular["SeasonYear"].isin(inner_train)]
        validation = regular[regular["SeasonYear"].eq(inner_validation)]
        for name, features in feature_sets.items():
            for c_value in C_CANDIDATES:
                scaler, model = _fit_logistic(train, features, c_value)
                loss = _metrics(validation["HomeWin"], _predict(scaler, model, validation, features))["log_loss"]
                if best is None or loss < best["validation_log_loss"]:
                    best = {
                        "form_prior_weight": float(prior_weight),
                        "feature_set": name,
                        "c": float(c_value),
                        "validation_log_loss": loss,
                    }
    return best or {}


def build_artifact_payload(
    scaler: StandardScaler,
    model: LogisticRegression,
    features: list[str],
    c_value: float,
    rating_params: dict,
    overtime_model: dict | None = None,
    simulation: dict | None = None,
) -> dict:
    """Assemble a version-2 artifact dict from a fitted scaler/model."""
    payload = {
        "model_version": WIN_PROB_MODEL_VERSION,
        "model_type": "logistic_regression",
        "feature_order": list(features),
        "coefficients": [float(value) for value in model.coef_[0].tolist()],
        "intercept": float(model.intercept_[0]),
        "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        "selected_c": float(c_value),
        "rating_params": {key: float(value) for key, value in rating_params.items()},
    }
    if overtime_model:
        payload["overtime_model"] = overtime_model
    if simulation:
        payload["simulation"] = simulation
    return payload


def fit_model(
    train_seasons: list[int],
    games: pd.DataFrame,
    elo_diffs: dict,
    form_tables: dict[float, pd.DataFrame],
    feature_sets: dict[str, list[str]] | None = None,
) -> dict:
    """Select hyperparameters on ``train_seasons`` only and fit the model on them."""
    feature_sets = feature_sets or FEATURE_SETS
    combo = select_elo_params(games, elo_diffs, train_seasons)
    config = select_config(train_seasons, combo, form_tables, elo_diffs, feature_sets)
    features = feature_sets[config["feature_set"]]
    table = _with_elo(form_tables[config["form_prior_weight"]], elo_diffs, combo)
    train = table[table["GameTypeId"].eq(REGULAR_SEASON) & table["SeasonYear"].isin(train_seasons)]
    scaler, model = _fit_logistic(train, features, config["c"])
    rating_params = dict(DEFAULT_RATING_PARAMS)
    rating_params.update(dict(zip(ELO_GRID.keys(), combo)))
    rating_params["form_prior_weight"] = config["form_prior_weight"]
    return {
        "config": config,
        "elo_combo": combo,
        "table": table,
        "features": features,
        "scaler": scaler,
        "model": model,
        "train_rows": int(len(train)),
        "home_win_rate": float(train["HomeWin"].mean()),
        "rating_params": rating_params,
        "payload": build_artifact_payload(scaler, model, features, config["c"], rating_params),
    }


def fit_overtime_model(fitted: dict) -> dict:
    """Fit P(tied after 60) against the absolute home-win logit, plus the shootout share."""
    table = fitted["table"]
    regular = table[table["GameTypeId"].eq(REGULAR_SEASON) & table["SeasonYear"].gt(FIRST_SEASON)]
    artifact = validate_model_artifact(fitted["payload"])
    abs_logits = np.abs(score_home_win_logits(regular, artifact)).reshape(-1, 1)
    past_regulation = regular["ResultType"].ne("REG").to_numpy(dtype=int)
    model = LogisticRegression(C=1e4, max_iter=2000).fit(abs_logits, past_regulation)
    shootouts = int(regular["ResultType"].eq("SO").sum())
    return {
        "intercept": float(model.intercept_[0]),
        "abs_logit_coef": float(model.coef_[0][0]),
        "shootout_share": float(shootouts / max(int(past_regulation.sum()), 1)),
        "overtime_rate": float(past_regulation.mean()),
    }


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------

def evaluate_fold(
    test_season: int,
    games: pd.DataFrame,
    elo_diffs: dict,
    form_tables: dict[float, pd.DataFrame],
    legacy_rows: pd.DataFrame,
    feature_sets: dict[str, list[str]] | None = None,
) -> dict:
    """Score one season with a model fitted only on the seasons before it."""
    train_seasons = list(range(FIRST_SEASON + 1, test_season))
    fitted = fit_model(train_seasons, games, elo_diffs, form_tables, feature_sets)
    table = fitted["table"]
    test = table[table["GameTypeId"].eq(REGULAR_SEASON) & table["SeasonYear"].eq(test_season)]
    probabilities = _predict(fitted["scaler"], fitted["model"], test, fitted["features"])
    model_metrics = _metrics(test["HomeWin"], probabilities)
    constant_metrics = _metrics(test["HomeWin"], np.full(len(test), fitted["home_win_rate"]))

    legacy_train = legacy_rows[legacy_rows["SeasonYear"].isin(train_seasons)]
    legacy_test = legacy_rows[legacy_rows["SeasonYear"].eq(test_season)]
    legacy_scaler, legacy_model = _fit_logistic(legacy_train, LEGACY_FEATURES, LEGACY_C)
    legacy_probabilities = _predict(legacy_scaler, legacy_model, legacy_test, LEGACY_FEATURES)
    by_game = pd.Series(probabilities, index=test["GameId"].to_numpy())
    common = legacy_test["GameId"].isin(by_game.index).to_numpy()
    common_ids = legacy_test.loc[common, "GameId"].to_numpy()
    common_labels = legacy_test.loc[common, "HomeWin"].to_numpy()

    playoff = table[table["GameTypeId"].eq(PLAYOFFS) & table["SeasonYear"].eq(test_season)]
    playoff_probabilities = _predict(fitted["scaler"], fitted["model"], playoff, fitted["features"]) if not playoff.empty else np.array([])

    return {
        "season": test_season,
        "fitted": fitted,
        "probabilities": probabilities,
        "labels": test["HomeWin"].to_numpy(dtype=int),
        "model": model_metrics,
        "constant": constant_metrics,
        "model_common": _metrics(common_labels, by_game.loc[common_ids].to_numpy()),
        "legacy_common": _metrics(common_labels, legacy_probabilities[common]),
        "playoffs": _metrics(playoff["HomeWin"], playoff_probabilities) if not playoff.empty else {},
        "playoff_probabilities": playoff_probabilities,
        "playoff_labels": playoff["HomeWin"].to_numpy(dtype=int) if not playoff.empty else np.array([], dtype=int),
    }


def calibration_table(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict]:
    """Group predictions into probability deciles and compare predicted with observed."""
    frame = pd.DataFrame({"y": labels, "p": probabilities})
    frame["bin"] = pd.qcut(frame["p"], q=bins, labels=False, duplicates="drop")
    rows = []
    for _, group in frame.groupby("bin"):
        rows.append({"mean_predicted": float(group["p"].mean()), "observed": float(group["y"].mean()), "n": int(len(group))})
    return rows


def standings_record(regular_games: pd.DataFrame) -> pd.DataFrame:
    """Build points, regulation wins, regulation+OT wins, wins and games per team."""
    columns = ["Team", "points", "regulation_wins", "regulation_plus_ot_wins", "wins", "games_played"]
    if regular_games.empty:
        return pd.DataFrame({column: [] for column in columns})
    home_won = regular_games["HomeWin"].eq(1)
    winners = np.where(home_won, regular_games["HomeTeam"], regular_games["AwayTeam"])
    losers = np.where(home_won, regular_games["AwayTeam"], regular_games["HomeTeam"])
    result = regular_games["ResultType"].to_numpy()
    winner_rows = pd.DataFrame({
        "Team": winners,
        "points": 2.0,
        "regulation_wins": (result == "REG").astype(float),
        "regulation_plus_ot_wins": (result != "SO").astype(float),
        "wins": 1.0,
        "games_played": 1.0,
    })
    loser_rows = pd.DataFrame({
        "Team": losers,
        "points": (result != "REG").astype(float),
        "regulation_wins": 0.0,
        "regulation_plus_ot_wins": 0.0,
        "wins": 0.0,
        "games_played": 1.0,
    })
    return pd.concat([winner_rows, loser_rows]).groupby("Team", as_index=False).sum()[columns]


def _points_pace_inputs(record: pd.DataFrame, teams: list[str]) -> dict[str, dict[str, float]]:
    """Express each team's regressed points pace as an Elo-equivalent rating."""
    by_team = record.set_index("Team") if not record.empty else pd.DataFrame()
    inputs = {}
    for team in teams:
        points = float(by_team.loc[team, "points"]) if team in by_team.index else 0.0
        games_played = float(by_team.loc[team, "games_played"]) if team in by_team.index else 0.0
        pace = (points + POINTS_PACE_PRIOR_GAMES) / (2.0 * games_played + 2.0 * POINTS_PACE_PRIOR_GAMES)
        pace = min(max(pace, 0.05), 0.95)
        inputs[team] = {"elo": LEAGUE_MEAN_ELO + 400.0 / math.log(10.0) * math.log(pace / (1.0 - pace))}
    return inputs


def _checkpoint_date(season_year: int, checkpoint: str, regular_games: pd.DataFrame) -> str:
    """Resolve a checkpoint label into a ``YYYY-MM-DD`` cut-off date."""
    if checkpoint == "opening":
        return str(regular_games["GameDate"].min())
    month, day = checkpoint.split("-")
    year = season_year if int(month) >= 9 else season_year + 1
    return f"{year}-{month}-{day}"


def calibrate_simulation(
    games: pd.DataFrame,
    folds: list[dict],
    division_maps: dict[int, dict[str, tuple[str, str]]],
    overtime_model: dict,
) -> dict:
    """Tune simulator strength noise on historical checkpoints and test it against points pace."""
    checkpoints = []
    for fold in folds:
        season_year = fold["season"]
        if season_year in SIMULATION_SKIPPED_SEASONS or season_year not in division_maps:
            continue
        season_games = games[games["SeasonYear"].eq(season_year)]
        regular = season_games[season_games["GameTypeId"].eq(REGULAR_SEASON)]
        playoff_teams = set(season_games.loc[season_games["GameTypeId"].eq(PLAYOFFS), "HomeTeam"]) | set(
            season_games.loc[season_games["GameTypeId"].eq(PLAYOFFS), "AwayTeam"]
        )
        final_series = season_games[season_games["GameTypeId"].eq(PLAYOFFS)].sort_values(["GameDate", "GameId"]).tail(1)
        champion = ""
        if not final_series.empty:
            last_game = final_series.iloc[0]
            champion = last_game["HomeTeam"] if int(last_game["HomeWin"]) == 1 else last_game["AwayTeam"]
        flags = back_to_back_flags(regular)
        division_map = division_maps[season_year]
        for checkpoint in SIM_CHECKPOINTS:
            cutoff = _checkpoint_date(season_year, checkpoint, regular)
            completed = regular[regular["GameDate"] < cutoff]
            remaining = regular[regular["GameDate"] >= cutoff].merge(flags, on="GameId", how="left")
            record = standings_record(completed).set_index("Team")
            teams = []
            for team, (conference, division) in sorted(division_map.items()):
                entry = record.loc[team] if team in record.index else None
                teams.append({
                    "team_abbr": team,
                    "conference": conference,
                    "division": division,
                    "points": float(entry["points"]) if entry is not None else 0.0,
                    "regulation_wins": float(entry["regulation_wins"]) if entry is not None else 0.0,
                    "regulation_plus_ot_wins": float(entry["regulation_plus_ot_wins"]) if entry is not None else 0.0,
                    "wins": float(entry["wins"]) if entry is not None else 0.0,
                })
            history = games[games["GameDate"] < cutoff]
            checkpoints.append({
                "season": season_year,
                "checkpoint": checkpoint,
                "teams": teams,
                "remaining": remaining,
                "progress": float(len(completed) / max(len(regular), 1)),
                "inputs": current_team_snapshot(history, season_year, fold["fitted"]["rating_params"]),
                "pace_inputs": _points_pace_inputs(standings_record(completed), [team["team_abbr"] for team in teams]),
                "fold_payload": fold["fitted"]["payload"],
                "home_win_rate": fold["fitted"]["home_win_rate"],
                "made_playoffs": {team["team_abbr"]: int(team["team_abbr"] in playoff_teams) for team in teams},
                "champion": champion,
            })

    def _playoff_log_loss(result: dict, made_playoffs: dict[str, int]) -> tuple[float, int]:
        """Summed log loss of P(make playoffs) over the league at one checkpoint."""
        total = 0.0
        for team, made in made_playoffs.items():
            probability = min(max(result["teams"][team]["make_playoffs"], 1e-4), 1 - 1e-4)
            total -= math.log(probability) if made else math.log(1.0 - probability)
        return total, len(made_playoffs)

    grid_losses: dict[tuple[float, float], list[float]] = {}
    grid_results: dict[tuple[float, float], list[dict]] = {}
    for sd_preseason, sd_late in product(SIM_SD_PRESEASON, SIM_SD_LATE):
        losses, results = [], []
        for number, checkpoint in enumerate(checkpoints):
            payload = dict(checkpoint["fold_payload"])
            payload["overtime_model"] = overtime_model
            payload["simulation"] = {"strength_sd_preseason": sd_preseason, "strength_sd_late": sd_late}
            result = simulate_season(
                checkpoint["teams"], checkpoint["inputs"], checkpoint["remaining"], validate_model_artifact(payload),
                season_progress=checkpoint["progress"], n_sims=SIM_CALIBRATION_SIMS, seed=1000 + number,
            )
            total, count = _playoff_log_loss(result, checkpoint["made_playoffs"])
            losses.append(total / count)
            results.append(result)
        grid_losses[(sd_preseason, sd_late)] = losses
        grid_results[(sd_preseason, sd_late)] = results

    best_grid = min(grid_losses, key=lambda key: float(np.mean(grid_losses[key])))
    pace_losses = []
    for number, checkpoint in enumerate(checkpoints):
        pace_payload = {
            "model_version": WIN_PROB_MODEL_VERSION,
            "feature_order": ["elo_diff"],
            "coefficients": [math.log(10.0) / 400.0],
            "intercept": math.log(checkpoint["home_win_rate"] / (1.0 - checkpoint["home_win_rate"])),
            "scaler_mean": [0.0],
            "scaler_scale": [1.0],
            "overtime_model": overtime_model,
            "simulation": {"strength_sd_preseason": 0.0, "strength_sd_late": 0.0},
        }
        result = simulate_season(
            checkpoint["teams"], checkpoint["pace_inputs"], checkpoint["remaining"], validate_model_artifact(pace_payload),
            season_progress=checkpoint["progress"], n_sims=SIM_CALIBRATION_SIMS, seed=1000 + number,
        )
        total, count = _playoff_log_loss(result, checkpoint["made_playoffs"])
        pace_losses.append(total / count)

    by_checkpoint = {}
    for label in SIM_CHECKPOINTS:
        positions = [i for i, checkpoint in enumerate(checkpoints) if checkpoint["checkpoint"] == label]
        if not positions:
            continue
        champion_ranks = []
        champion_odds = []
        for position in positions:
            checkpoint = checkpoints[position]
            if not checkpoint["champion"]:
                continue
            cup_odds = {team: values["win_cup"] for team, values in grid_results[best_grid][position]["teams"].items()}
            ranking = sorted(cup_odds, key=cup_odds.get, reverse=True)
            champion_ranks.append(ranking.index(checkpoint["champion"]) + 1)
            champion_odds.append(cup_odds[checkpoint["champion"]])
        by_checkpoint[label] = {
            "model_log_loss": float(np.mean([grid_losses[best_grid][i] for i in positions])),
            "points_pace_log_loss": float(np.mean([pace_losses[i] for i in positions])),
            "champion_cup_rank": champion_ranks,
            "champion_cup_odds": [round(value, 4) for value in champion_odds],
        }

    return {
        "checkpoints": len(checkpoints),
        "selected": {"strength_sd_preseason": best_grid[0], "strength_sd_late": best_grid[1]},
        "model_log_loss": float(np.mean(grid_losses[best_grid])),
        "points_pace_log_loss": float(np.mean(pace_losses)),
        "zero_noise_log_loss": float(np.mean(grid_losses[(0.0, 0.0)])),
        "by_checkpoint": by_checkpoint,
    }


def backtest_series(games: pd.DataFrame, folds: list[dict]) -> dict:
    """Score every best-of-7 series with the fold model as of the series' first game."""
    labels, model_probabilities, home_ice_probabilities = [], [], []
    for fold in folds:
        season_year = fold["season"]
        if season_year in SIMULATION_SKIPPED_SEASONS:
            continue
        season_games = games[games["SeasonYear"].eq(season_year)]
        playoff = season_games[season_games["GameTypeId"].eq(PLAYOFFS)].sort_values(["GameDate", "GameId"])
        record = standings_record(season_games[season_games["GameTypeId"].eq(REGULAR_SEASON)]).set_index("Team")
        artifact = validate_model_artifact(fold["fitted"]["payload"])
        constant, weights, _ = decompose_linear_model(artifact)
        home_rate = fold["fitted"]["home_win_rate"]
        pair_key = playoff.apply(lambda row: tuple(sorted((row["HomeTeam"], row["AwayTeam"]))), axis=1)
        for pair, series_games in playoff.groupby(pair_key):
            winners = np.where(series_games["HomeWin"].eq(1), series_games["HomeTeam"], series_games["AwayTeam"])
            win_counts = pd.Series(winners).value_counts()
            if win_counts.max() != 4:
                continue
            first, second = pair
            first_key = tuple(record.loc[first, ["points", "regulation_wins"]]) if first in record.index else (0, 0)
            second_key = tuple(record.loc[second, ["points", "regulation_wins"]]) if second in record.index else (0, 0)
            high, low = (first, second) if first_key >= second_key else (second, first)
            snapshot = current_team_snapshot(games[games["GameDate"] < series_games["GameDate"].min()], season_year, fold["fitted"]["rating_params"])
            strength_high = sum(weight * snapshot[high][attribute] for attribute, weight in weights.items())
            strength_low = sum(weight * snapshot[low][attribute] for attribute, weight in weights.items())
            p_home = 1.0 / (1.0 + math.exp(-(constant + strength_high - strength_low)))
            p_away = 1.0 - 1.0 / (1.0 + math.exp(-(constant + strength_low - strength_high)))
            labels.append(int(win_counts.idxmax() == high))
            model_probabilities.append(float(series_win_probability(p_home, p_away)))
            home_ice_probabilities.append(float(series_win_probability(home_rate, 1.0 - home_rate)))
    if not labels:
        return {}
    return {
        "series": len(labels),
        "model": _metrics(np.array(labels), np.array(model_probabilities)),
        "home_ice_only": _metrics(np.array(labels), np.array(home_ice_probabilities)),
    }


# ---------------------------------------------------------------------------
# Phase 3: 60-minute (1X2) and puck-line markets from one score distribution
# ---------------------------------------------------------------------------

GOAL_MODEL_FIT_GAMES = 6000
GOAL_SHAPE_KEYS = ("rate_scale", "tie_inflation", "lead1_transfer", "lead2_transfer")
GOAL_SHAPE_BOUNDS = ((0.8, 1.2), (0.0, 2.0), (0.0, 0.8), (0.0, 0.8))


def goal_market_frame(fitted: dict, games: pd.DataFrame) -> pd.DataFrame:
    """Return regular-season and playoff games with the fitted win model's probability and market labels."""
    table = fitted["table"]
    probabilities = pd.Series(_predict(fitted["scaler"], fitted["model"], table, fitted["features"]), index=table["GameId"].to_numpy())
    frame = games.copy()
    frame["Environment"] = scoring_environment(frame, DEFAULT_GOAL_MODEL["environment_prior_team_games"])
    frame["RegHome"], frame["RegAway"] = regulation_goals(frame)
    frame["PModel"] = frame["GameId"].map(probabilities)
    return frame.dropna(subset=["PModel", "Environment"]).reset_index(drop=True)


def fit_goal_model(train: pd.DataFrame) -> dict:
    """Fit the overtime tie-break and the score-grid shape on regular-season training games.

    The home share of each game's rate depends on the shape, and the shape is fitted
    given those shares. Two alternating passes settle both.

    Args:
        train: ``goal_market_frame`` rows from the training seasons.

    Returns:
        A goal-model block that ``validate_goal_model`` accepts.
    """
    regular = train[train["GameTypeId"].eq(REGULAR_SEASON)]
    tied = regular[regular["RegHome"] == regular["RegAway"]]
    logits = np.log(tied["PModel"] / (1 - tied["PModel"])).to_numpy().reshape(-1, 1)
    tie_break = LogisticRegression(C=1e4, max_iter=1000).fit(logits, tied["HomeWin"].astype(int))
    model = dict(DEFAULT_GOAL_MODEL)
    model["overtime_intercept"] = float(tie_break.intercept_[0])
    model["overtime_logit_coef"] = float(tie_break.coef_[0][0])

    sample = regular.sample(n=min(GOAL_MODEL_FIT_GAMES, len(regular)), random_state=0)
    p_model = sample["PModel"].to_numpy()
    total = 2.0 * sample["Environment"].to_numpy()
    home_goals = np.minimum(sample["RegHome"].to_numpy(dtype=int), MAX_GOALS - 1)
    away_goals = np.minimum(sample["RegAway"].to_numpy(dtype=int), MAX_GOALS - 1)
    rows = np.arange(len(sample))
    for _ in range(2):
        share = solve_home_share(p_model, total, validate_goal_model(model))

        def _negative_log_likelihood(values: np.ndarray) -> float:
            """Mean negative log-likelihood of the observed regulation scores."""
            candidate = validate_goal_model(dict(model, **dict(zip(GOAL_SHAPE_KEYS, values))))
            if candidate is None:
                return 1e6
            grid = score_grid(total * share, total * (1 - share), candidate)
            return float(-np.mean(np.log(np.clip(grid[rows, home_goals, away_goals], 1e-12, 1.0))))

        start = [model[key] for key in GOAL_SHAPE_KEYS]
        result = minimize(_negative_log_likelihood, x0=start, bounds=GOAL_SHAPE_BOUNDS, method="L-BFGS-B")
        model.update({key: float(value) for key, value in zip(GOAL_SHAPE_KEYS, result.x)})
    model["version"] = GOAL_MODEL_VERSION
    return model


def _multiclass_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean negative log probability of the observed class."""
    return float(-np.mean(np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-9, 1.0))))


def _brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error of binary probabilities."""
    return float(np.mean((np.asarray(probabilities, dtype=float) - np.asarray(labels, dtype=float)) ** 2))


def goal_market_backtest(folds: list[dict], games: pd.DataFrame) -> dict:
    """Backtest the 1X2 and puck-line markets season by season against league-rate baselines.

    Each season is priced with that fold's win model and a goal model fitted only on
    earlier seasons. Totals are reported but not gated or published.

    Returns:
        Per-season metrics, pooled playoff metrics and the gate verdict.
    """
    seasons = []
    playoff_labels, playoff_probabilities = [], []
    for fold in folds:
        season = fold["season"]
        frame = goal_market_frame(fold["fitted"], games)
        train = frame[frame["SeasonYear"].between(FIRST_SEASON + 1, season - 1)]
        test = frame[frame["SeasonYear"].eq(season)]
        model = validate_goal_model(fit_goal_model(train))
        train_regular = train[train["GameTypeId"].eq(REGULAR_SEASON)]

        def _labels(rows: pd.DataFrame) -> dict[str, np.ndarray]:
            """Observed market outcomes."""
            margin = (rows["RegHome"] - rows["RegAway"]).to_numpy()
            graded_total = (rows["HomeGoals"] + rows["AwayGoals"] + rows["ResultType"].eq("SO")).to_numpy()
            return {
                "result": np.where(margin > 0, 0, np.where(margin == 0, 1, 2)),
                "home_minus_1_5": (margin >= 2).astype(int),
                "away_minus_1_5": (margin <= -2).astype(int),
                "over_5_5": (graded_total > 5.5).astype(int),
            }

        for game_type, rows in ((REGULAR_SEASON, test[test["GameTypeId"].eq(REGULAR_SEASON)]), (PLAYOFFS, test[test["GameTypeId"].eq(PLAYOFFS)])):
            if rows.empty:
                continue
            priced = price_games(rows["PModel"].to_numpy(), rows["Environment"].to_numpy(), model)
            predicted = np.column_stack([priced["home_regulation"], priced["draw"], priced["away_regulation"]])
            outcome = _labels(rows)
            if game_type == PLAYOFFS:
                playoff_labels.append(outcome["result"])
                playoff_probabilities.append(predicted)
                continue
            baseline = _labels(train_regular)
            rates = np.bincount(baseline["result"], minlength=3) / len(baseline["result"])
            tie_rate = rates[1]
            tied_train = train_regular[train_regular["RegHome"] == train_regular["RegAway"]]
            home_tie_break = float(tied_train["HomeWin"].mean())
            conversion = np.column_stack([
                rows["PModel"] - tie_rate * home_tie_break,
                np.full(len(rows), tie_rate),
                1 - rows["PModel"] - tie_rate * (1 - home_tie_break),
            ])
            conversion = np.clip(conversion, 0.01, 1.0)
            conversion = conversion / conversion.sum(axis=1, keepdims=True)
            seasons.append({
                "season": season,
                "games": int(len(rows)),
                "model": {
                    "one_x_two_log_loss": _multiclass_log_loss(outcome["result"], predicted),
                    "home_minus_1_5_brier": _brier(outcome["home_minus_1_5"], priced["home_minus_1_5"]),
                    "away_minus_1_5_brier": _brier(outcome["away_minus_1_5"], priced["away_minus_1_5"]),
                    "over_5_5_brier": _brier(outcome["over_5_5"], _over_probability(rows, model, 5.5)),
                },
                "league_rate": {
                    "one_x_two_log_loss": _multiclass_log_loss(outcome["result"], np.tile(rates, (len(rows), 1))),
                    "home_minus_1_5_brier": _brier(outcome["home_minus_1_5"], np.full(len(rows), baseline["home_minus_1_5"].mean())),
                    "away_minus_1_5_brier": _brier(outcome["away_minus_1_5"], np.full(len(rows), baseline["away_minus_1_5"].mean())),
                    "over_5_5_brier": _brier(outcome["over_5_5"], np.full(len(rows), baseline["over_5_5"].mean())),
                },
                "moneyline_with_fixed_draw_rate_log_loss": _multiclass_log_loss(outcome["result"], conversion),
                "predicted_draw_rate": float(priced["draw"].mean()),
                "observed_draw_rate": float((outcome["result"] == 1).mean()),
                "shape": {key: round(float(model[key]), 4) for key in GOAL_SHAPE_KEYS + ("overtime_intercept", "overtime_logit_coef")},
            })

    def _beats(metric: str) -> bool:
        """True when the model beats the league rate on ``metric`` in every season."""
        return all(row["model"][metric] < row["league_rate"][metric] for row in seasons)

    gate = {
        "one_x_two_beats_league_rate_every_season": _beats("one_x_two_log_loss"),
        "home_minus_1_5_beats_league_rate_every_season": _beats("home_minus_1_5_brier"),
        "away_minus_1_5_beats_league_rate_every_season": _beats("away_minus_1_5_brier"),
        "over_5_5_beats_league_rate_every_season": _beats("over_5_5_brier"),
    }
    gate["passed"] = gate["one_x_two_beats_league_rate_every_season"] and gate["home_minus_1_5_beats_league_rate_every_season"] and gate["away_minus_1_5_beats_league_rate_every_season"]
    playoffs = {}
    if playoff_labels:
        labels = np.concatenate(playoff_labels)
        probabilities = np.concatenate(playoff_probabilities)
        playoffs = {
            "games": int(len(labels)),
            "one_x_two_log_loss": _multiclass_log_loss(labels, probabilities),
            "predicted_draw_rate": float(probabilities[:, 1].mean()),
            "observed_draw_rate": float((labels == 1).mean()),
        }
    return {"seasons": seasons, "playoffs": playoffs, "gate": gate}


def _over_probability(rows: pd.DataFrame, model: dict, line: float) -> np.ndarray:
    """P(graded total > line) for report-only totals checks."""
    share = solve_home_share(rows["PModel"].to_numpy(), 2.0 * rows["Environment"].to_numpy(), model)
    total = 2.0 * rows["Environment"].to_numpy()
    grid = score_grid(total * share, total * (1 - share), model)
    goals = np.arange(MAX_GOALS)
    graded = goals[:, None] + goals[None, :] + (goals[:, None] == goals[None, :])
    return (grid * (graded > line)).sum(axis=(1, 2))


def _print_goal_markets(report: dict) -> None:
    """Print the Phase 3 market backtest."""
    print("\n60-minute result (1X2 log loss) and puck line (Brier), each season priced by models fitted on earlier seasons")
    print(f"{'season':<9}{'1X2 model':>10}{'league':>8}{'fixed-draw':>11}   {'home -1.5':>10}{'league':>8}   {'away -1.5':>10}{'league':>8}   {'O/U 5.5':>8}{'league':>8}   draw pred/obs")
    for row in report["seasons"]:
        model, league = row["model"], row["league_rate"]
        print(
            f"{row['season']}-{str(row['season'] + 1)[2:]:<4}{model['one_x_two_log_loss']:>10.4f}{league['one_x_two_log_loss']:>8.4f}{row['moneyline_with_fixed_draw_rate_log_loss']:>11.4f}   "
            f"{model['home_minus_1_5_brier']:>10.4f}{league['home_minus_1_5_brier']:>8.4f}   {model['away_minus_1_5_brier']:>10.4f}{league['away_minus_1_5_brier']:>8.4f}   "
            f"{model['over_5_5_brier']:>8.4f}{league['over_5_5_brier']:>8.4f}   {row['predicted_draw_rate']:.3f}/{row['observed_draw_rate']:.3f}"
        )
    if report["playoffs"]:
        playoffs = report["playoffs"]
        print(f"Playoffs (pooled, {playoffs['games']} games): 1X2 log loss {playoffs['one_x_two_log_loss']:.4f}, draw {playoffs['predicted_draw_rate']:.3f} predicted vs {playoffs['observed_draw_rate']:.3f} observed")
    gate = report["gate"]
    print(
        f"Market gate: 1X2 {'PASS' if gate['one_x_two_beats_league_rate_every_season'] else 'FAIL'}, "
        f"home -1.5 {'PASS' if gate['home_minus_1_5_beats_league_rate_every_season'] else 'FAIL'}, "
        f"away -1.5 {'PASS' if gate['away_minus_1_5_beats_league_rate_every_season'] else 'FAIL'} -> {'PASS' if gate['passed'] else 'FAIL'}; "
        f"totals (not published) {'would pass' if gate['over_5_5_beats_league_rate_every_season'] else 'fail'}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_backtest(folds: list[dict]) -> None:
    """Print the per-season backtest table."""
    print("\nRolling-origin backtest (regular season; each season predicted by a model fitted on earlier seasons)")
    print(f"{'season':<9}{'games':>6}{'const LL':>10}{'model LL':>10}{'brier':>8}{'acc':>7}   {'same games: legacy LL / model LL':>34}   {'playoff LL (n)':>16}   config")
    for fold in folds:
        span = f"{fold['season']}-{str(fold['season'] + 1)[2:]}"
        config = fold["fitted"]["config"]
        playoff = fold["playoffs"]
        playoff_text = f"{playoff['log_loss']:.4f} ({playoff['n']})" if playoff else "-"
        print(
            f"{span:<9}{fold['model']['n']:>6}{fold['constant']['log_loss']:>10.4f}{fold['model']['log_loss']:>10.4f}"
            f"{fold['model']['brier']:>8.4f}{fold['model']['accuracy']:>7.3f}   "
            f"{fold['legacy_common']['log_loss']:>16.4f} / {fold['model_common']['log_loss']:.4f} (n={fold['model_common']['n']})"
            f"   {playoff_text:>16}   {config['feature_set']}, C={config['c']}, k={config['form_prior_weight']:.0f}, elo={fold['fitted']['elo_combo']}"
        )


def main() -> int:
    """Fetch, backtest, gate and export the win-probability artifact."""
    start_time = time.time()
    last_season = previous_season_year()
    print(f"Fetching {FIRST_SEASON}-{str(FIRST_SEASON + 1)[2:]} through {last_season}-{str(last_season + 1)[2:]}...")
    team_id_map = _fetch_team_id_to_abbrev_map()
    games, raw_regular = fetch_league_history(FIRST_SEASON, last_season, team_id_map)

    print("Precomputing Elo grid and team form...")
    elo_diffs = precompute_elo_diffs(games)
    form_tables = {weight: build_model_features(games, {"form_prior_weight": weight}) for weight in FORM_PRIOR_WEIGHTS}
    legacy_rows = build_legacy_feature_rows(raw_regular, team_id_map, games)

    test_seasons = list(range(FIRST_TEST_SEASON, last_season + 1))
    folds = [evaluate_fold(season, games, elo_diffs, form_tables, legacy_rows) for season in test_seasons]
    _print_backtest(folds)

    legacy_mean = float(np.mean([fold["legacy_common"]["log_loss"] for fold in folds]))
    model_mean = float(np.mean([fold["model_common"]["log_loss"] for fold in folds]))
    worse_seasons = [fold["season"] for fold in folds if fold["model_common"]["log_loss"] > fold["legacy_common"]["log_loss"]]
    game_gate = {
        "legacy_mean_log_loss": legacy_mean,
        "model_mean_log_loss": model_mean,
        "mean_gain": legacy_mean - model_mean,
        "required_gain": GATE_MIN_MEAN_GAIN,
        "seasons_worse_than_legacy": worse_seasons,
        "passed": (legacy_mean - model_mean) >= GATE_MIN_MEAN_GAIN and not worse_seasons,
    }
    print(
        f"\nGame-model gate: mean log loss {legacy_mean:.4f} (previous design) -> {model_mean:.4f} "
        f"(gain {legacy_mean - model_mean:+.4f}, need >= {GATE_MIN_MEAN_GAIN}); "
        f"seasons worse: {worse_seasons or 'none'} -> {'PASS' if game_gate['passed'] else 'FAIL'}"
    )

    pooled_labels = np.concatenate([fold["labels"] for fold in folds])
    pooled_probabilities = np.concatenate([fold["probabilities"] for fold in folds])
    calibration = calibration_table(pooled_labels, pooled_probabilities)
    print("\nCalibration (all test seasons, probability deciles): predicted -> observed (n)")
    print("  " + "  ".join(f"{row['mean_predicted']:.3f}->{row['observed']:.3f}({row['n']})" for row in calibration))
    playoff_labels = np.concatenate([fold["playoff_labels"] for fold in folds])
    playoff_probabilities = np.concatenate([fold["playoff_probabilities"] for fold in folds])
    playoff_metrics = _metrics(playoff_labels, playoff_probabilities) if len(playoff_labels) else {}
    if playoff_metrics:
        print(f"Playoff games (all test seasons): log loss {playoff_metrics['log_loss']:.4f}, accuracy {playoff_metrics['accuracy']:.3f}, n={playoff_metrics['n']}, mean predicted {playoff_probabilities.mean():.3f} vs observed {playoff_labels.mean():.3f}")

    print("\nFitting final model on every completed season...")
    final = fit_model(list(range(FIRST_SEASON + 1, last_season + 1)), games, elo_diffs, form_tables)
    overtime_model = fit_overtime_model(final)
    print(f"  config: {final['config']}, elo={final['elo_combo']}, rows={final['train_rows']}")
    print(f"  overtime model: {overtime_model}")

    print("\nCalibrating the season simulator on historical checkpoints...")
    division_maps = fetch_division_maps([season for season in test_seasons if season not in SIMULATION_SKIPPED_SEASONS])
    simulation_report = calibrate_simulation(games, folds, division_maps, overtime_model)
    simulation_gate = simulation_report["model_log_loss"] < simulation_report["points_pace_log_loss"]
    print(
        f"  P(make playoffs) log loss over {simulation_report['checkpoints']} checkpoints: model {simulation_report['model_log_loss']:.4f} "
        f"(no noise {simulation_report['zero_noise_log_loss']:.4f}) vs points pace {simulation_report['points_pace_log_loss']:.4f} "
        f"-> {'PASS' if simulation_gate else 'FAIL'}; selected noise {simulation_report['selected']}"
    )
    for label, values in simulation_report["by_checkpoint"].items():
        print(
            f"    {label:>8}: model {values['model_log_loss']:.4f} vs pace {values['points_pace_log_loss']:.4f}; "
            f"eventual champion's Cup-odds rank {values['champion_cup_rank']} odds {values['champion_cup_odds']}"
        )

    series_report = backtest_series(games, folds)
    if series_report:
        print(
            f"\nPlayoff series ({series_report['series']}): model log loss {series_report['model']['log_loss']:.4f}, "
            f"accuracy {series_report['model']['accuracy']:.3f} vs home-ice only {series_report['home_ice_only']['log_loss']:.4f}"
        )

    goal_report = goal_market_backtest(folds, games)
    _print_goal_markets(goal_report)

    if not game_gate["passed"] or not simulation_gate:
        print("\nGate failed; win_prob_weights.json was NOT written.")
        return 1

    artifact = dict(final["payload"])
    artifact["overtime_model"] = overtime_model
    artifact["simulation"] = simulation_report["selected"]
    if goal_report["gate"]["passed"]:
        final_frame = goal_market_frame(final, games)
        artifact["goal_model"] = fit_goal_model(final_frame[final_frame["SeasonYear"].gt(FIRST_SEASON)])
        print(f"  goal model: { {key: round(value, 4) if isinstance(value, float) else value for key, value in artifact['goal_model'].items()} }")
    else:
        print("  market gate failed: the artifact carries no goal model, so no 1X2 or puck-line markets are published.")
    artifact.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_seasons": list(range(FIRST_SEASON + 1, last_season + 1)),
        "warmup_seasons": [FIRST_SEASON],
        "training_rows": final["train_rows"],
        "home_win_rate": final["home_win_rate"],
        "validation_metrics": {
            "method": "rolling-origin; hyperparameters and weights fitted on seasons before each test season",
            "backtest": [
                {
                    "season": fold["season"],
                    "model": fold["model"],
                    "constant_home_rate": fold["constant"],
                    "same_games_model": fold["model_common"],
                    "same_games_previous_design": fold["legacy_common"],
                    "playoffs": fold["playoffs"],
                    "config": fold["fitted"]["config"],
                }
                for fold in folds
            ],
            "game_gate": game_gate,
            "calibration": calibration,
            "playoff_games": playoff_metrics,
            "simulation": simulation_report,
            "series": series_report,
            "goal_markets": goal_report,
        },
    })
    validate_model_artifact(artifact)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\nSaved {OUTPUT_PATH.name} in {round(time.time() - start_time, 1)}s.")
    return 0


PHASE2_GATE_MIN_MEAN_GAIN = 0.002
PHASE2_GATE_WORST_SEASON = -0.002


def phase2_report() -> int:
    """Re-run the Phase 2 check: do xG shares and projected-goalie ratings beat Phase 1?

    Report only; the artifact is never touched. The first run fetches play-by-play for
    every game since 2017-18 (about 12,000 requests, ~25 min), which is cached in
    ``.cache/xg_training``. Later runs only fetch new seasons. The xG model is trained
    on seasons before ``FIRST_TEST_SEASON``, so no test season leaks into it. Starters
    are projected from earlier starts only, never read from the game itself.

    Returns:
        0 when the Phase 2 gate passes, 1 otherwise.
    """
    last_season = previous_season_year()
    team_id_map = _fetch_team_id_to_abbrev_map()
    print(f"Fetching {FIRST_SEASON}-{str(FIRST_SEASON + 1)[2:]} through {last_season}-{str(last_season + 1)[2:]}...")
    games, raw_regular = fetch_league_history(FIRST_SEASON, last_season, team_id_map)
    shots, metas = load_play_by_play_history(games)
    print(f"{len(shots)} unblocked shots from {len(metas)} games.")

    xg_payload = fit_xg_model(shots, list(range(FIRST_SEASON, FIRST_TEST_SEASON)))
    print("\nxG model (trained on seasons before the first test season):")
    for row in xg_metrics_by_season(shots, xg_payload):
        print(f"  {row['season']}: AUC {row['auc']:.3f}, log loss {row['log_loss']:.4f}, goals per xG {row['goals_per_xg']:.3f}")

    xg_games, goalie_games = summarize_games(shots, score_expected_goals(shots, validate_xg_model(xg_payload)), metas)
    games_with_flags = games.merge(back_to_back_flags(games), on="GameId", how="left")
    goalie_features = build_goalie_features(games_with_flags, GoalieHistory(goalie_games))
    starters = goalie_features.merge(xg_games[["GameId", "SeasonYear", "HomeStarter", "AwayStarter"]], on="GameId")
    accuracy = pd.concat(
        [starters["HomeProjectedStarter"] == starters["HomeStarter"], starters["AwayProjectedStarter"] == starters["AwayStarter"]]
    ).mean()
    print(f"Projected starter matched the actual starter in {accuracy:.1%} of team-games.")

    elo_diffs = precompute_elo_diffs(games)
    legacy_rows = build_legacy_feature_rows(raw_regular, team_id_map, games)
    phase1_tables = {weight: build_model_features(games, {"form_prior_weight": weight}) for weight in FORM_PRIOR_WEIGHTS}
    phase2_tables = {
        weight: table.merge(research_game_features(games, xg_games, goalie_features, prior_weight=weight), on="GameId", how="left")
        for weight, table in phase1_tables.items()
    }
    full = list(FEATURE_SETS["full"])
    phase2_sets = dict(FEATURE_SETS)
    phase2_sets.update({
        "full_xg5": full + ["xg_share_shrunk_diff"],
        "full_xg_all": full + ["xg_all_share_shrunk_diff"],
        "full_goalie": full + ["goalie_gsax_diff"],
        "full_research": full + list(RESEARCH_FEATURES),
        "xg_replaces_shots": ["elo_diff", "goal_diff_shrunk_diff", "xg_share_shrunk_diff", "xg_all_share_shrunk_diff", "home_back_to_back", "away_back_to_back"],
    })

    print("\nRolling-origin backtest, same games (log loss): Phase 1 vs Phase 1 + xG/goalie candidates")
    gains = []
    for season in range(FIRST_TEST_SEASON, last_season + 1):
        phase1 = evaluate_fold(season, games, elo_diffs, phase1_tables, legacy_rows, FEATURE_SETS)
        phase2 = evaluate_fold(season, games, elo_diffs, phase2_tables, legacy_rows, phase2_sets)
        gains.append(phase1["model"]["log_loss"] - phase2["model"]["log_loss"])
        print(
            f"  {season}-{str(season + 1)[2:]}: {phase1['model']['log_loss']:.4f} -> {phase2['model']['log_loss']:.4f} "
            f"(gain {gains[-1]:+.4f}; selected {phase2['fitted']['config']['feature_set']})"
        )
    passed = float(np.mean(gains)) >= PHASE2_GATE_MIN_MEAN_GAIN and min(gains) >= PHASE2_GATE_WORST_SEASON
    print(
        f"\nPhase 2 gate: mean gain {np.mean(gains):+.4f} (need >= {PHASE2_GATE_MIN_MEAN_GAIN}), "
        f"worst season {min(gains):+.4f} (need >= {PHASE2_GATE_WORST_SEASON}) -> {'PASS' if passed else 'FAIL'}"
    )
    if passed:
        print("The runtime does not compute these features yet; wiring them in is a separate change.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(phase2_report() if "--phase2-report" in sys.argv[1:] else main())
