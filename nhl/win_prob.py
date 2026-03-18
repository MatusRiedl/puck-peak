"""Shared pregame win-probability feature engineering and scoring helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

WIN_PROB_FEATURE_ORDER = [
    "point_pct_to_date",
    "goal_diff_per_game_to_date",
    "l10_point_pct",
    "l10_goal_diff_per_game",
    "power_play_pct_to_date",
]
WIN_PROB_FEATURE_LABELS = {
    "point_pct_to_date": "season points %",
    "goal_diff_per_game_to_date": "season goal diff/game",
    "l10_point_pct": "last 10 points %",
    "l10_goal_diff_per_game": "last 10 goal diff/game",
    "power_play_pct_to_date": "season power-play %",
}
MIN_GAMES_FOR_ESTIMATE = 5
WIN_PROB_MODEL_VERSION = 1

_NORMALIZED_GAME_COLUMNS = [
    "SeasonYear",
    "GameDate",
    "GameId",
    "TeamAbbrev",
    "OpponentAbbrev",
    "HomeRoadFlag",
    "Points",
    "Goals",
    "GoalsAgainst",
    "PP%",
]
_TEAM_FEATURE_COLUMNS = {
    "point_pct_to_date": "PointPctToDate",
    "goal_diff_per_game_to_date": "GoalDiffPerGameToDate",
    "l10_point_pct": "L10PointPct",
    "l10_goal_diff_per_game": "L10GoalDiffPerGame",
    "power_play_pct_to_date": "PowerPlayPctToDate",
}


def _coerce_percentage_scale(value: object) -> float:
    """Normalize either decimal-scale or percent-scale values to 0-100 scale."""
    try:
        numeric_value = float(value)
    except Exception:
        return float("nan")

    if math.isnan(numeric_value):
        return float("nan")
    if abs(numeric_value) <= 1.5:
        numeric_value *= 100.0
    return numeric_value


def sigmoid(value: float) -> float:
    """Return a numerically stable sigmoid."""
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def normalize_team_game_frame(
    team_games: pd.DataFrame,
    season_year: int | None = None,
    team_abbrev_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Normalize team-game rows into one shared schema for training and inference."""
    if team_games is None or team_games.empty:
        return pd.DataFrame(columns=_NORMALIZED_GAME_COLUMNS)

    d = team_games.copy()
    rename_map = {
        "gameDate": "GameDate",
        "gameId": "GameId",
        "teamAbbrev": "TeamAbbrev",
        "opponentTeamAbbrev": "OpponentAbbrev",
        "OpponentTeamAbbrev": "OpponentAbbrev",
        "homeRoad": "HomeRoadFlag",
        "HomeRoad": "HomeRoadFlag",
        "points": "Points",
        "goalsFor": "Goals",
        "GoalsFor": "Goals",
        "goalsAgainst": "GoalsAgainst",
        "powerPlayPct": "PP%",
        "PowerPlayPct": "PP%",
    }
    d = d.rename(columns=rename_map)

    if "SeasonYear" not in d.columns:
        if season_year is not None:
            d["SeasonYear"] = int(season_year)
        elif "seasonId" in d.columns:
            season_id_series = d["seasonId"].astype(str).str[:4]
            d["SeasonYear"] = pd.to_numeric(season_id_series, errors="coerce")
        else:
            d["SeasonYear"] = float("nan")

    if "TeamAbbrev" not in d.columns:
        if "teamId" in d.columns and team_abbrev_map:
            d["TeamAbbrev"] = pd.to_numeric(d["teamId"], errors="coerce").map(team_abbrev_map)
        else:
            d["TeamAbbrev"] = ""

    if "OpponentAbbrev" not in d.columns:
        d["OpponentAbbrev"] = ""
    if "HomeRoadFlag" not in d.columns:
        d["HomeRoadFlag"] = ""

    if "GameDate" not in d.columns:
        d["GameDate"] = ""
    if "GameId" not in d.columns:
        d["GameId"] = float("nan")
    if "Points" not in d.columns:
        d["Points"] = 0.0
    if "Goals" not in d.columns:
        d["Goals"] = 0.0
    if "GoalsAgainst" not in d.columns:
        d["GoalsAgainst"] = 0.0
    if "PP%" not in d.columns:
        d["PP%"] = float("nan")

    d["GameDate"] = d["GameDate"].astype(str).str.strip()
    d["GameId"] = pd.to_numeric(d["GameId"], errors="coerce")
    d["SeasonYear"] = pd.to_numeric(d.get("SeasonYear", float("nan")), errors="coerce")
    d["TeamAbbrev"] = d["TeamAbbrev"].astype(str).str.strip().str.upper()
    d["OpponentAbbrev"] = d["OpponentAbbrev"].astype(str).str.strip().str.upper()
    d["HomeRoadFlag"] = d["HomeRoadFlag"].astype(str).str.strip().str.upper()
    d["Points"] = pd.to_numeric(d["Points"], errors="coerce").fillna(0.0)
    d["Goals"] = pd.to_numeric(d["Goals"], errors="coerce").fillna(0.0)
    d["GoalsAgainst"] = pd.to_numeric(d["GoalsAgainst"], errors="coerce").fillna(0.0)
    d["PP%"] = d["PP%"].apply(_coerce_percentage_scale)

    normalized = d[_NORMALIZED_GAME_COLUMNS].dropna(subset=["SeasonYear", "GameId"]).copy()
    normalized["SeasonYear"] = normalized["SeasonYear"].astype(int)
    normalized["GameId"] = normalized["GameId"].astype(int)
    normalized = normalized[
        normalized["GameDate"].ne("")
        & normalized["TeamAbbrev"].ne("")
        & normalized["OpponentAbbrev"].ne("")
        & normalized["HomeRoadFlag"].isin({"H", "R"})
    ]

    if normalized.empty:
        return pd.DataFrame(columns=_NORMALIZED_GAME_COLUMNS)

    return normalized.sort_values(
        ["SeasonYear", "TeamAbbrev", "GameDate", "GameId"],
        kind="stable",
    ).reset_index(drop=True)


def compute_team_feature_history(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add strict pregame lagged features for each team-game row."""
    normalized = normalize_team_game_frame(team_games)
    if normalized.empty:
        return normalized

    def _apply_group_features(group: pd.DataFrame) -> pd.DataFrame:
        """Build leak-safe rolling team features for one season/team slice."""
        g = group.sort_values(["GameDate", "GameId"], kind="stable").reset_index(drop=True).copy()
        goal_diff = g["Goals"] - g["GoalsAgainst"]
        games_before = np.arange(len(g), dtype=float)
        points = g["Points"].astype(float)
        power_play = pd.to_numeric(g["PP%"], errors="coerce")

        points_before = points.cumsum().shift(1)
        goal_diff_before = goal_diff.cumsum().shift(1)
        prev_points = points.shift(1)
        prev_goal_diff = goal_diff.shift(1)
        prev_pp = power_play.shift(1)

        g["GamesBefore"] = games_before
        g["PointPctToDate"] = points_before / pd.Series(games_before, index=g.index).replace(0, np.nan) / 2.0
        g["GoalDiffPerGameToDate"] = goal_diff_before / pd.Series(games_before, index=g.index).replace(0, np.nan)
        rolling_games = prev_points.rolling(10, min_periods=1).count()
        g["L10PointPct"] = prev_points.rolling(10, min_periods=1).sum() / rolling_games.replace(0, np.nan) / 2.0
        g["L10GoalDiffPerGame"] = prev_goal_diff.rolling(10, min_periods=1).sum() / rolling_games.replace(0, np.nan)
        pp_counts = prev_pp.notna().cumsum()
        g["PowerPlayPctToDate"] = prev_pp.fillna(0.0).cumsum() / pp_counts.replace(0, np.nan)
        return g

    featured_groups = [
        _apply_group_features(group)
        for _, group in normalized.groupby(["SeasonYear", "TeamAbbrev"], sort=False)
    ]
    if not featured_groups:
        return pd.DataFrame(columns=list(normalized.columns) + list(_TEAM_FEATURE_COLUMNS.values()) + ["GamesBefore"])
    featured = pd.concat(featured_groups, ignore_index=True)
    featured["GamesBefore"] = pd.to_numeric(featured["GamesBefore"], errors="coerce").fillna(0).astype(int)
    return featured


def build_matchup_feature_rows(
    team_feature_history: pd.DataFrame,
    min_games: int = MIN_GAMES_FOR_ESTIMATE,
) -> pd.DataFrame:
    """Pair home and away team-game rows into one model-ready training table."""
    if team_feature_history is None or team_feature_history.empty:
        return pd.DataFrame()

    d = team_feature_history.copy()
    home_cols = [
        "SeasonYear",
        "GameDate",
        "GameId",
        "TeamAbbrev",
        "OpponentAbbrev",
        "GamesBefore",
        "Goals",
        "GoalsAgainst",
        *_TEAM_FEATURE_COLUMNS.values(),
    ]
    away_cols = home_cols.copy()

    home = d[d["HomeRoadFlag"] == "H"][home_cols].rename(
        columns={
            "TeamAbbrev": "HomeTeamAbbrev",
            "OpponentAbbrev": "AwayTeamAbbrev",
            "GamesBefore": "HomeGamesBefore",
            "Goals": "HomeGoals",
            "GoalsAgainst": "HomeGoalsAgainst",
            "PointPctToDate": "HomePointPctToDate",
            "GoalDiffPerGameToDate": "HomeGoalDiffPerGameToDate",
            "L10PointPct": "HomeL10PointPct",
            "L10GoalDiffPerGame": "HomeL10GoalDiffPerGame",
            "PowerPlayPctToDate": "HomePowerPlayPctToDate",
        }
    )
    away = d[d["HomeRoadFlag"] == "R"][away_cols].rename(
        columns={
            "TeamAbbrev": "AwayTeamAbbrev",
            "OpponentAbbrev": "HomeTeamAbbrev",
            "GamesBefore": "AwayGamesBefore",
            "Goals": "AwayGoals",
            "GoalsAgainst": "AwayGoalsAgainst",
            "PointPctToDate": "AwayPointPctToDate",
            "GoalDiffPerGameToDate": "AwayGoalDiffPerGameToDate",
            "L10PointPct": "AwayL10PointPct",
            "L10GoalDiffPerGame": "AwayL10GoalDiffPerGame",
            "PowerPlayPctToDate": "AwayPowerPlayPctToDate",
        }
    )

    matchups = home.merge(
        away,
        on=["SeasonYear", "GameDate", "GameId", "HomeTeamAbbrev", "AwayTeamAbbrev"],
        how="inner",
    )
    if matchups.empty:
        return pd.DataFrame()

    matchups = matchups[
        (matchups["HomeGamesBefore"] >= int(min_games))
        & (matchups["AwayGamesBefore"] >= int(min_games))
    ].copy()
    if matchups.empty:
        return pd.DataFrame()

    matchups["home_win"] = (matchups["HomeGoals"] > matchups["HomeGoalsAgainst"]).astype(int)
    for feature_name, feature_column in _TEAM_FEATURE_COLUMNS.items():
        home_column = f"Home{feature_column}"
        away_column = f"Away{feature_column}"
        matchups[feature_name] = (
            pd.to_numeric(matchups[home_column], errors="coerce")
            - pd.to_numeric(matchups[away_column], errors="coerce")
        )

    required_columns = ["SeasonYear", "GameDate", "GameId", "HomeTeamAbbrev", "AwayTeamAbbrev", "home_win"]
    required_columns.extend(WIN_PROB_FEATURE_ORDER)
    matchups = matchups.dropna(subset=WIN_PROB_FEATURE_ORDER)
    return matchups[required_columns].reset_index(drop=True)


def build_team_snapshot(
    team_games: pd.DataFrame,
    min_games: int = MIN_GAMES_FOR_ESTIMATE,
) -> dict[str, float] | None:
    """Compute the next-game pregame snapshot for one team from completed games."""
    if team_games is None or team_games.empty:
        return None

    d = team_games.copy()
    rename_map = {
        "gameDate": "GameDate",
        "gameId": "GameId",
        "points": "Points",
        "goalsFor": "Goals",
        "goalsAgainst": "GoalsAgainst",
        "powerPlayPct": "PP%",
    }
    d = d.rename(columns=rename_map)
    if "GameType" in d.columns:
        d = d[d["GameType"].astype(str).str.strip().eq("Regular")]
    if d.empty:
        return None

    if "GameDate" not in d.columns:
        d["GameDate"] = ""
    if "GameId" not in d.columns:
        d["GameId"] = pd.RangeIndex(start=1, stop=len(d) + 1)
    if "Points" not in d.columns:
        d["Points"] = 0.0
    if "Goals" not in d.columns:
        d["Goals"] = 0.0
    if "GoalsAgainst" not in d.columns:
        d["GoalsAgainst"] = 0.0
    if "PP%" not in d.columns:
        d["PP%"] = float("nan")

    d["GameDate"] = d["GameDate"].astype(str).str.strip()
    d["GameId"] = pd.to_numeric(d["GameId"], errors="coerce")
    d["Points"] = pd.to_numeric(d["Points"], errors="coerce").fillna(0.0)
    d["Goals"] = pd.to_numeric(d["Goals"], errors="coerce").fillna(0.0)
    d["GoalsAgainst"] = pd.to_numeric(d["GoalsAgainst"], errors="coerce").fillna(0.0)
    d["PP%"] = d["PP%"].apply(_coerce_percentage_scale)
    d = d.dropna(subset=["GameId"]).sort_values(["GameDate", "GameId"], kind="stable").reset_index(drop=True)
    if d.empty:
        return None

    completed_games = len(d)
    if completed_games < int(min_games):
        return None

    goal_diff = d["Goals"] - d["GoalsAgainst"]
    last_ten = d.tail(10)
    pp_values = pd.to_numeric(d["PP%"], errors="coerce").dropna()

    snapshot = {
        "games_played": float(completed_games),
        "point_pct_to_date": float(d["Points"].sum() / (2.0 * completed_games)),
        "goal_diff_per_game_to_date": float(goal_diff.sum() / completed_games),
        "l10_point_pct": float(last_ten["Points"].sum() / (2.0 * len(last_ten))),
        "l10_goal_diff_per_game": float(((last_ten["Goals"] - last_ten["GoalsAgainst"]).sum()) / len(last_ten)),
        "power_play_pct_to_date": float(pp_values.mean()) if not pp_values.empty else float("nan"),
    }
    if any(math.isnan(float(snapshot[feature_name])) for feature_name in WIN_PROB_FEATURE_ORDER):
        return None
    return snapshot


def build_matchup_snapshot(
    home_team_games: pd.DataFrame,
    away_team_games: pd.DataFrame,
    min_games: int = MIN_GAMES_FOR_ESTIMATE,
) -> dict[str, object] | None:
    """Build one model-ready feature vector for an upcoming matchup."""
    home_snapshot = build_team_snapshot(home_team_games, min_games=min_games)
    away_snapshot = build_team_snapshot(away_team_games, min_games=min_games)
    if home_snapshot is None or away_snapshot is None:
        return None

    feature_values = {
        feature_name: float(home_snapshot[feature_name] - away_snapshot[feature_name])
        for feature_name in WIN_PROB_FEATURE_ORDER
    }
    return {
        "home_games_played": int(home_snapshot["games_played"]),
        "away_games_played": int(away_snapshot["games_played"]),
        "feature_values": feature_values,
    }


def validate_model_artifact(payload: object) -> dict:
    """Validate and normalize the exported JSON model artifact."""
    if not isinstance(payload, dict):
        raise ValueError("Win-probability artifact must be a dict.")

    feature_order = list(payload.get("feature_order") or [])
    if feature_order != WIN_PROB_FEATURE_ORDER:
        raise ValueError("Win-probability artifact feature order does not match runtime expectations.")

    coefficients = [float(value) for value in payload.get("coefficients", [])]
    scaler_mean = [float(value) for value in payload.get("scaler_mean", [])]
    scaler_scale = [float(value) for value in payload.get("scaler_scale", [])]
    if not (len(coefficients) == len(scaler_mean) == len(scaler_scale) == len(WIN_PROB_FEATURE_ORDER)):
        raise ValueError("Win-probability artifact coefficient shape is invalid.")

    cleaned_scale = [value if value != 0 else 1.0 for value in scaler_scale]
    return {
        "model_version": int(payload.get("model_version", WIN_PROB_MODEL_VERSION)),
        "feature_order": feature_order,
        "coefficients": coefficients,
        "intercept": float(payload.get("intercept", 0.0)),
        "scaler_mean": scaler_mean,
        "scaler_scale": cleaned_scale,
        "selected_c": float(payload.get("selected_c", 0.0)),
        "min_games": int(payload.get("min_games", MIN_GAMES_FOR_ESTIMATE)),
        "training_seasons": [int(season) for season in payload.get("training_seasons", [])],
        "validation_metrics": payload.get("validation_metrics", {}),
    }


def score_home_win_probability(
    feature_values: dict[str, float],
    artifact: dict,
) -> dict[str, object]:
    """Score one home-win probability from raw feature deltas and exported weights."""
    validated_artifact = validate_model_artifact(artifact)
    raw_vector = np.array(
        [float(feature_values[feature_name]) for feature_name in validated_artifact["feature_order"]],
        dtype=float,
    )
    mean_vector = np.array(validated_artifact["scaler_mean"], dtype=float)
    scale_vector = np.array(validated_artifact["scaler_scale"], dtype=float)
    standardized_vector = (raw_vector - mean_vector) / scale_vector
    coefficient_vector = np.array(validated_artifact["coefficients"], dtype=float)
    logit = float(validated_artifact["intercept"] + np.dot(coefficient_vector, standardized_vector))
    probability = sigmoid(logit)
    contributions = {
        feature_name: float(value)
        for feature_name, value in zip(
            validated_artifact["feature_order"],
            coefficient_vector * standardized_vector,
        )
    }
    return {
        "home_win_prob": probability,
        "logit": logit,
        "standardized_values": {
            feature_name: float(value)
            for feature_name, value in zip(validated_artifact["feature_order"], standardized_vector)
        },
        "contributions": contributions,
        "feature_values": {feature_name: float(feature_values[feature_name]) for feature_name in validated_artifact["feature_order"]},
    }


def get_top_feature_driver(
    scored_probability: dict[str, object],
) -> tuple[str, float]:
    """Return the strongest single feature contribution from one scored matchup."""
    contributions = scored_probability.get("contributions", {})
    if not isinstance(contributions, dict) or not contributions:
        return ("", 0.0)

    feature_name = max(
        contributions,
        key=lambda current_feature: abs(float(contributions.get(current_feature, 0.0))),
    )
    return feature_name, float(contributions.get(feature_name, 0.0))
