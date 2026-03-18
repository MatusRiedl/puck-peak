"""Offline trainer for pregame NHL win-probability weights."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from nhl.constants import TEAM_LIST_URL, TEAM_STATS_URL
from nhl.win_prob import (
    MIN_GAMES_FOR_ESTIMATE,
    WIN_PROB_FEATURE_ORDER,
    WIN_PROB_MODEL_VERSION,
    build_matchup_feature_rows,
    compute_team_feature_history,
    normalize_team_game_frame,
)

OUTPUT_PATH = Path("win_prob_weights.json")
REQUEST_TIMEOUT = 20
REQUEST_HEADERS = {"User-Agent": "puck-peak/1.0"}
TRAINING_SEASONS = [2020, 2021, 2022, 2023, 2024]
VALIDATION_SEASON = 2024
C_CANDIDATES = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


def _season_id_from_year(season_year: int) -> int:
    """Convert a start year like 2024 into an NHL seasonId like 20242025."""
    return int(f"{int(season_year)}{int(season_year) + 1}")


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
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers=REQUEST_HEADERS,
            )
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


def _fetch_regular_season_team_games(
    season_year: int,
    team_abbrev_map: dict[int, str],
) -> pd.DataFrame:
    """Fetch and normalize all regular-season team-game rows for one season."""
    season_id = _season_id_from_year(season_year)
    payload = _get_json_with_retries(
        TEAM_STATS_URL,
        params={
            "limit": -1,
            "start": 0,
            "sort": "gameDate",
            "isGame": "true",
            "cayenneExp": f"seasonId={season_id} and gameTypeId=2",
        },
    )
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not rows:
        raise ValueError(f"No team-game rows returned for season {season_year}-{str(season_year + 1)[2:]}.")
    return normalize_team_game_frame(
        pd.DataFrame(rows),
        season_year=season_year,
        team_abbrev_map=team_abbrev_map,
    )


def _build_training_dataset(season_years: list[int]) -> pd.DataFrame:
    """Build the full leak-free training table across seasons."""
    team_abbrev_map = _fetch_team_id_to_abbrev_map()
    matchup_frames: list[pd.DataFrame] = []

    for season_year in season_years:
        span = f"{season_year}-{str(season_year + 1)[2:]}"
        print(f"Fetching {span} regular-season team games...")
        normalized_games = _fetch_regular_season_team_games(season_year, team_abbrev_map)
        team_feature_history = compute_team_feature_history(normalized_games)
        season_matchups = build_matchup_feature_rows(
            team_feature_history,
            min_games=MIN_GAMES_FOR_ESTIMATE,
        )
        if season_matchups.empty:
            raise ValueError(f"No training rows survived feature generation for {span}.")
        print(f"  {span}: {len(season_matchups)} training games")
        matchup_frames.append(season_matchups)

    return pd.concat(matchup_frames, ignore_index=True)


def _fit_logistic_model(matchups: pd.DataFrame, c_value: float) -> tuple[StandardScaler, LogisticRegression]:
    """Fit one standardized logistic regression on the provided matchup table."""
    x_train = matchups[WIN_PROB_FEATURE_ORDER].to_numpy(dtype=float)
    y_train = matchups["home_win"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LogisticRegression(
        C=float(c_value),
        max_iter=2000,
        solver="lbfgs",
        random_state=0,
    )
    model.fit(x_train_scaled, y_train)
    return scaler, model


def _evaluate_candidate(
    train_matchups: pd.DataFrame,
    validation_matchups: pd.DataFrame,
    c_value: float,
) -> dict[str, float]:
    """Train one candidate model and evaluate it on the held-out season."""
    scaler, model = _fit_logistic_model(train_matchups, c_value)
    x_val = validation_matchups[WIN_PROB_FEATURE_ORDER].to_numpy(dtype=float)
    y_val = validation_matchups["home_win"].to_numpy(dtype=int)
    x_val_scaled = scaler.transform(x_val)
    probabilities = model.predict_proba(x_val_scaled)[:, 1]
    return {
        "c": float(c_value),
        "log_loss": float(log_loss(y_val, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_val, probabilities)),
    }


def _select_best_c(
    train_matchups: pd.DataFrame,
    validation_matchups: pd.DataFrame,
) -> tuple[float, list[dict[str, float]]]:
    """Choose the best regularization strength by held-out log loss."""
    candidate_results = [
        _evaluate_candidate(train_matchups, validation_matchups, c_value)
        for c_value in C_CANDIDATES
    ]
    candidate_results.sort(key=lambda row: (row["log_loss"], row["brier_score"], row["c"]))
    return candidate_results[0]["c"], candidate_results


def _build_artifact(
    scaler: StandardScaler,
    model: LogisticRegression,
    training_matchups: pd.DataFrame,
    selected_c: float,
    candidate_results: list[dict[str, float]],
) -> dict:
    """Build the exported JSON artifact from the fitted scaler/model."""
    return {
        "model_version": WIN_PROB_MODEL_VERSION,
        "model_type": "logistic_regression",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "feature_order": WIN_PROB_FEATURE_ORDER,
        "coefficients": [float(value) for value in model.coef_[0].tolist()],
        "intercept": float(model.intercept_[0]),
        "scaler_mean": [float(value) for value in scaler.mean_.tolist()],
        "scaler_scale": [float(value) for value in scaler.scale_.tolist()],
        "selected_c": float(selected_c),
        "min_games": int(MIN_GAMES_FOR_ESTIMATE),
        "training_seasons": [int(season) for season in TRAINING_SEASONS],
        "training_rows": int(len(training_matchups)),
        "home_win_rate": float(training_matchups["home_win"].mean()),
        "validation_metrics": {
            "validation_season": VALIDATION_SEASON,
            "candidates": candidate_results,
        },
    }


def main() -> None:
    """Train the offline logistic model and export the JSON weight artifact."""
    start_time = time.time()
    all_matchups = _build_training_dataset(TRAINING_SEASONS)
    training_matchups = all_matchups[all_matchups["SeasonYear"] < VALIDATION_SEASON].copy()
    validation_matchups = all_matchups[all_matchups["SeasonYear"] == VALIDATION_SEASON].copy()
    if training_matchups.empty or validation_matchups.empty:
        raise ValueError("Temporal train/validation split produced an empty dataset.")

    selected_c, candidate_results = _select_best_c(training_matchups, validation_matchups)
    print(f"Selected C={selected_c} from held-out {VALIDATION_SEASON}-{str(VALIDATION_SEASON + 1)[2:]}.")

    final_scaler, final_model = _fit_logistic_model(all_matchups, selected_c)
    artifact = _build_artifact(
        scaler=final_scaler,
        model=final_model,
        training_matchups=all_matchups,
        selected_c=selected_c,
        candidate_results=candidate_results,
    )
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    elapsed = round(time.time() - start_time, 1)
    print(f"Saved {OUTPUT_PATH.name} in {elapsed}s.")
    print(f"Training rows: {artifact['training_rows']}")
    print(f"Feature order: {', '.join(WIN_PROB_FEATURE_ORDER)}")


if __name__ == "__main__":
    main()
