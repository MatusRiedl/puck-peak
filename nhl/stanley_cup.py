"""Build a current-standings board with one model-based Stanley Cup favorite."""

from __future__ import annotations

import math

import pandas as pd

from nhl.constants import ACTIVE_TEAMS
from nhl.win_prob import (
    WIN_PROB_FEATURE_LABELS,
    WIN_PROB_FEATURE_ORDER,
    score_home_win_probability,
    validate_model_artifact,
)

_CONFERENCE_SORT_ORDER = {"Eastern": 0, "Western": 1}
_DIVISION_SORT_ORDER = {
    "Atlantic": 0,
    "Metropolitan": 1,
    "Central": 2,
    "Pacific": 3,
}
_FALLBACK_ARTIFACT = {
    "model_version": 1,
    "feature_order": WIN_PROB_FEATURE_ORDER,
    "coefficients": [1.2, 0.8, 0.6, 0.5, 0.2],
    "intercept": 0.0,
    "scaler_mean": [0.0] * len(WIN_PROB_FEATURE_ORDER),
    "scaler_scale": [1.0] * len(WIN_PROB_FEATURE_ORDER),
    "selected_c": 1.0,
    "min_games": 5,
}


def _safe_float(value: object, default: float = 0.0) -> float:
    """Return one float or a stable default when parsing fails."""
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if math.isnan(numeric):
        return float(default)
    return numeric


def _mean_or_default(series: pd.Series, default: float) -> float:
    """Return the mean of valid numeric values, else the provided default."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[numeric.notna()]
    if numeric.empty:
        return float(default)
    return float(numeric.mean())


def _format_generated_at_label(standings_df: pd.DataFrame) -> str:
    """Format the live-standings timestamp for display."""
    if standings_df.empty or "standingsDateTimeUtc" not in standings_df.columns:
        return ""

    raw_value = str(standings_df["standingsDateTimeUtc"].iloc[0] or "").strip()
    if not raw_value:
        return ""

    parsed = pd.to_datetime(raw_value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return ""
    return f"Current as of {parsed.strftime('%b %d, %Y %H:%M UTC')}"


def _build_feature_frame(
    standings_df: pd.DataFrame,
    goalie_proxy_by_team: dict[str, float | None] | None = None,
) -> pd.DataFrame:
    """Normalize live standings rows into one contender-scoring table."""
    d = standings_df.copy()
    for text_column in ["teamAbbrev", "teamName", "teamCommonName", "conferenceName", "divisionName", "teamLogo"]:
        if text_column not in d.columns:
            d[text_column] = ""
        d[text_column] = d[text_column].fillna("").astype(str).str.strip()
    d["teamAbbrev"] = d["teamAbbrev"].str.upper()
    d["teamName"] = d["teamName"].where(d["teamName"].ne(""), d["teamAbbrev"])

    numeric_columns = [
        "gamesPlayed",
        "wins",
        "losses",
        "otLosses",
        "points",
        "divisionSequence",
        "conferenceSequence",
        "leagueSequence",
        "pointPctg",
        "goalDiffPerGame",
        "l10PointPctg",
        "l10GamesPlayed",
        "l10GoalDifferential",
        "PP%",
    ]
    for column in numeric_columns:
        if column not in d.columns:
            d[column] = float("nan")
        d[column] = pd.to_numeric(d[column], errors="coerce")

    l10_games = d["l10GamesPlayed"].replace(0, pd.NA)
    d["l10GoalDiffPerGame"] = d["l10GoalDifferential"] / l10_games
    d["goalieProxySavePct"] = d["teamAbbrev"].map(goalie_proxy_by_team or {})

    league_pp = _mean_or_default(d["PP%"], 20.0)
    league_goalie_proxy = _mean_or_default(d["goalieProxySavePct"], 0.905)

    d["pp_neutralized"] = d["PP%"].isna()
    d["goalie_neutralized"] = d["goalieProxySavePct"].isna()
    d.loc[d["pp_neutralized"], "PP%"] = league_pp
    d.loc[d["goalie_neutralized"], "goalieProxySavePct"] = league_goalie_proxy

    return d


def _resolve_artifact(artifact: dict | None) -> dict:
    """Use the saved model artifact when possible, else a safe fallback."""
    try:
        return validate_model_artifact(artifact)
    except Exception:
        return validate_model_artifact(dict(_FALLBACK_ARTIFACT))


def _build_top_drivers(contributions: dict[str, float]) -> list[str]:
    """Return the three strongest feature drivers for one contender score."""
    if not isinstance(contributions, dict):
        return []

    ordered_features = sorted(
        contributions.items(),
        key=lambda item: abs(float(item[1] or 0.0)),
        reverse=True,
    )
    top_drivers: list[str] = []
    for feature_name, contribution in ordered_features[:3]:
        direction = "up" if float(contribution) >= 0 else "down"
        feature_label = WIN_PROB_FEATURE_LABELS.get(feature_name, feature_name.replace("_", " "))
        top_drivers.append(f"{feature_label} {direction}")
    return top_drivers


def build_stanley_cup_board(
    standings_df: pd.DataFrame,
    artifact: dict | None,
    goalie_proxy_by_team: dict[str, float | None] | None = None,
) -> dict:
    """Build a four-division board and pick the strongest current contender."""
    if standings_df is None or standings_df.empty:
        return {
            "generated_at_label": "",
            "favorite_team_abbr": "",
            "favorite_team": {},
            "teams": [],
            "divisions": [],
        }

    contender_df = _build_feature_frame(standings_df, goalie_proxy_by_team=goalie_proxy_by_team)
    contender_df = contender_df[contender_df["teamAbbrev"].ne("")].copy()
    if contender_df.empty:
        return {
            "generated_at_label": _format_generated_at_label(standings_df),
            "favorite_team_abbr": "",
            "favorite_team": {},
            "teams": [],
            "divisions": [],
        }

    artifact_payload = _resolve_artifact(artifact)
    league_means = {
        "point_pct_to_date": _mean_or_default(contender_df["pointPctg"], 0.5),
        "goal_diff_per_game_to_date": _mean_or_default(contender_df["goalDiffPerGame"], 0.0),
        "l10_point_pct": _mean_or_default(contender_df["l10PointPctg"], 0.5),
        "l10_goal_diff_per_game": _mean_or_default(contender_df["l10GoalDiffPerGame"], 0.0),
        "power_play_pct_to_date": _mean_or_default(contender_df["PP%"], 20.0),
    }
    league_goalie_proxy = _mean_or_default(contender_df["goalieProxySavePct"], 0.905)

    teams: list[dict] = []
    for _, row in contender_df.iterrows():
        feature_values = {
            "point_pct_to_date": _safe_float(row.get("pointPctg")) - league_means["point_pct_to_date"],
            "goal_diff_per_game_to_date": _safe_float(row.get("goalDiffPerGame")) - league_means["goal_diff_per_game_to_date"],
            "l10_point_pct": _safe_float(row.get("l10PointPctg")) - league_means["l10_point_pct"],
            "l10_goal_diff_per_game": _safe_float(row.get("l10GoalDiffPerGame")) - league_means["l10_goal_diff_per_game"],
            "power_play_pct_to_date": _safe_float(row.get("PP%")) - league_means["power_play_pct_to_date"],
        }
        scored = score_home_win_probability(feature_values, artifact_payload)

        goalie_proxy = _safe_float(row.get("goalieProxySavePct"), league_goalie_proxy)
        goalie_bonus = max(-0.04, min(0.04, (goalie_proxy - league_goalie_proxy) * 4.0))
        contender_score = max(0.0, min(1.0, float(scored["home_win_prob"]) + goalie_bonus))

        neutralized_inputs: list[str] = []
        if bool(row.get("pp_neutralized")):
            neutralized_inputs.append("Power play %")
        if bool(row.get("goalie_neutralized")):
            neutralized_inputs.append("Goalie proxy save %")

        team_payload = {
            "team_abbr": str(row.get("teamAbbrev") or "").strip().upper(),
            "team_name": str(row.get("teamName") or ACTIVE_TEAMS.get(row.get("teamAbbrev", ""), row.get("teamAbbrev", ""))).strip(),
            "team_common_name": str(row.get("teamCommonName") or "").strip(),
            "team_logo": str(row.get("teamLogo") or "").strip(),
            "conference_name": str(row.get("conferenceName") or "").strip(),
            "division_name": str(row.get("divisionName") or "").strip(),
            "games_played": int(round(_safe_float(row.get("gamesPlayed")))),
            "wins": int(round(_safe_float(row.get("wins")))),
            "losses": int(round(_safe_float(row.get("losses")))),
            "ot_losses": int(round(_safe_float(row.get("otLosses")))),
            "points": int(round(_safe_float(row.get("points")))),
            "division_sequence": int(round(_safe_float(row.get("divisionSequence"), 999.0))),
            "conference_sequence": int(round(_safe_float(row.get("conferenceSequence"), 999.0))),
            "league_sequence": int(round(_safe_float(row.get("leagueSequence"), 999.0))),
            "contender_score": float(contender_score),
            "top_drivers": _build_top_drivers(scored.get("contributions", {})),
            "neutralized_inputs": neutralized_inputs,
            "pp_neutralized": bool(row.get("pp_neutralized")),
            "goalie_neutralized": bool(row.get("goalie_neutralized")),
            "summary_text": (
                f"{str(row.get('teamName') or row.get('teamAbbrev') or '').strip()} "
                f"owns a {contender_score * 100.0:.1f}% neutral-opponent contender score."
            ),
            "is_favorite": False,
            "rank": 0,
        }
        teams.append(team_payload)

    teams.sort(
        key=lambda team: (
            -float(team["contender_score"]),
            -int(team["points"]),
            int(team["league_sequence"]),
            team["team_abbr"],
        )
    )
    for rank, team in enumerate(teams, start=1):
        team["rank"] = rank

    favorite_team = teams[0] if teams else {}
    favorite_team_abbr = str(favorite_team.get("team_abbr", "") or "")
    for team in teams:
        team["is_favorite"] = team["team_abbr"] == favorite_team_abbr

    divisions: list[dict] = []
    for (conference_name, division_name), division_df in contender_df.groupby(["conferenceName", "divisionName"], sort=False):
        division_rows: list[dict] = []
        team_lookup = {team["team_abbr"]: team for team in teams}
        sorted_division = division_df.sort_values(
            ["divisionSequence", "leagueSequence", "points", "teamAbbrev"],
            ascending=[True, True, False, True],
            kind="stable",
            na_position="last",
        )
        for _, row in sorted_division.iterrows():
            team_abbr = str(row.get("teamAbbrev") or "").strip().upper()
            team_payload = team_lookup.get(team_abbr)
            if team_payload is not None:
                division_rows.append(team_payload)

        divisions.append(
            {
                "conference_name": str(conference_name or "").strip(),
                "division_name": str(division_name or "").strip(),
                "teams": division_rows,
            }
        )

    divisions.sort(
        key=lambda division: (
            _CONFERENCE_SORT_ORDER.get(division["conference_name"], 999),
            _DIVISION_SORT_ORDER.get(division["division_name"], 999),
            division["division_name"],
        )
    )

    return {
        "generated_at_label": _format_generated_at_label(contender_df),
        "favorite_team_abbr": favorite_team_abbr,
        "favorite_team": favorite_team,
        "teams": teams,
        "divisions": divisions,
    }
