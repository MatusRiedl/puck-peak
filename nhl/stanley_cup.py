"""Build the current-standings board with simulated Stanley Cup odds."""

from __future__ import annotations

import math

import pandas as pd

from nhl.constants import ACTIVE_TEAMS, current_season_year
from nhl.team_ratings import canonical_team_abbrev

_CONFERENCE_SORT_ORDER = {"Eastern": 0, "Western": 1}
_DIVISION_SORT_ORDER = {
    "Atlantic": 0,
    "Metropolitan": 1,
    "Central": 2,
    "Pacific": 3,
}
_CONTENDER_COUNT = 5

BOARD_MODE_ODDS = "odds"
"""In-season or playoffs: real record plus playoff and Cup odds."""
BOARD_MODE_PRESEASON = "preseason"
"""Before opening night: projected points plus odds; last season's record is hidden."""
BOARD_MODE_CHAMPION = "champion"
"""The Cup is decided: final standings plus the champion."""
BOARD_MODE_STANDINGS = "standings"
"""No projection available: standings only."""


def _safe_float(value: object, default: float = 0.0) -> float:
    """Return one float or a stable default when parsing fails."""
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if math.isnan(numeric):
        return float(default)
    return numeric


def _season_span(season_year: int | None) -> str:
    """Format a season start year as ``2026-27``."""
    if not season_year:
        return ""
    return f"{int(season_year)}-{str(int(season_year) + 1)[2:]}"


def _parse_standings_timestamp(standings_df: pd.DataFrame) -> str:
    """Return the standings timestamp formatted for display, or an empty string."""
    if standings_df.empty or "standingsDateTimeUtc" not in standings_df.columns:
        return ""
    raw_value = str(standings_df["standingsDateTimeUtc"].iloc[0] or "").strip()
    if not raw_value:
        return ""
    parsed = pd.to_datetime(raw_value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%b %d, %Y %H:%M UTC")


def _format_generated_at_label(standings_df: pd.DataFrame) -> str:
    """Format the live-standings timestamp for display.

    When the table belongs to a season that has already ended - which is what
    ``/standings/now`` returns for the whole offseason - the label says so instead of
    calling a finished season "current".

    Args:
        standings_df: Normalized standings frame.

    Returns:
        A display label, or an empty string when no timestamp is available.
    """
    timestamp = _parse_standings_timestamp(standings_df)
    if not timestamp:
        return ""

    season_label = _format_standings_season_label(standings_df)
    if season_label:
        return f"Final {season_label} standings — as of {timestamp}"
    return f"Current as of {timestamp}"


def _format_standings_season_label(standings_df: pd.DataFrame) -> str:
    """Return a ``2025-26`` style label when the standings are not the live season.

    Args:
        standings_df: Normalized standings frame carrying a ``seasonId`` column.

    Returns:
        Season label for a completed season, or an empty string when the table
        describes the season currently in progress.
    """
    if standings_df.empty or "seasonId" not in standings_df.columns:
        return ""

    try:
        season_id = int(standings_df["seasonId"].iloc[0] or 0)
    except (TypeError, ValueError):
        return ""
    if not season_id:
        return ""

    season_year = season_id // 10000
    if season_year >= current_season_year():
        return ""
    return _season_span(season_year)


def _resolve_board_mode(projection: dict) -> str:
    """Map the season projection state onto a board display mode."""
    state = str(projection.get("state", "") or "")
    if state == "projection" and projection.get("teams"):
        return BOARD_MODE_PRESEASON if projection.get("phase") == "preseason" else BOARD_MODE_ODDS
    if state == "champion" and projection.get("champion"):
        return BOARD_MODE_CHAMPION
    return BOARD_MODE_STANDINGS


def _build_team_payload(row: pd.Series, odds: dict, include_record: bool) -> dict:
    """Merge one standings row with its simulated odds into the board's team shape."""
    team_abbr = str(row.get("teamAbbrev") or "").strip().upper()
    team_name = str(row.get("teamName") or ACTIVE_TEAMS.get(team_abbr, team_abbr)).strip()

    def _record(column: str, default: float = 0.0) -> int:
        """Return one record value, zeroed when the record belongs to another season."""
        return int(round(_safe_float(row.get(column), default))) if include_record else 0

    return {
        "team_abbr": team_abbr,
        "team_name": team_name,
        "team_common_name": str(row.get("teamCommonName") or "").strip(),
        "team_logo": str(row.get("teamLogo") or "").strip(),
        "conference_name": str(row.get("conferenceName") or "").strip(),
        "division_name": str(row.get("divisionName") or "").strip(),
        "games_played": _record("gamesPlayed"),
        "wins": _record("wins"),
        "losses": _record("losses"),
        "ot_losses": _record("otLosses"),
        "points": _record("points"),
        "division_sequence": int(round(_safe_float(row.get("divisionSequence"), 999.0))),
        "conference_sequence": int(round(_safe_float(row.get("conferenceSequence"), 999.0))),
        "league_sequence": int(round(_safe_float(row.get("leagueSequence"), 999.0))),
        "projected_points": float(odds.get("projected_points", 0.0)) if odds else None,
        "points_p10": float(odds.get("points_p10", 0.0)) if odds else None,
        "points_p90": float(odds.get("points_p90", 0.0)) if odds else None,
        "playoff_pct": float(odds.get("make_playoffs", 0.0)) if odds else None,
        "division_pct": float(odds.get("win_division", 0.0)) if odds else None,
        "final_pct": float(odds.get("win_conference", 0.0)) if odds else None,
        "cup_pct": float(odds.get("win_cup", 0.0)) if odds else None,
        "is_favorite": False,
        "is_champion": False,
        "rank": 0,
    }


def build_stanley_cup_board(
    standings_df: pd.DataFrame,
    projection: dict | None = None,
) -> dict:
    """Build the four-division board and attach the season projection.

    Args:
        standings_df: Output of ``data_loaders.get_current_nhl_standings``. Before
            opening night this still holds last season's final table, and only its
            division membership is used.
        projection: Output of ``schedule.get_season_projection``.

    Returns:
        Board payload: display ``mode``, labels, favorite or champion, top contenders,
        all teams (most likely champion first) and the four division tables.
    """
    empty_board = {
        "generated_at_label": "",
        "mode": BOARD_MODE_STANDINGS,
        "season_label": "",
        "simulation_count": 0,
        "favorite_team_abbr": "",
        "favorite_team": {},
        "champion_team": {},
        "contenders": [],
        "summary_text": "",
        "teams": [],
        "divisions": [],
    }
    if standings_df is None or standings_df.empty or "teamAbbrev" not in standings_df.columns:
        return empty_board

    projection = projection or {}
    mode = _resolve_board_mode(projection)
    season_year = int(projection.get("season_year") or current_season_year())
    projected_teams = projection.get("teams", {}) if mode in (BOARD_MODE_ODDS, BOARD_MODE_PRESEASON) else {}
    include_record = mode != BOARD_MODE_PRESEASON

    standings = standings_df.copy()
    standings["teamAbbrev"] = standings["teamAbbrev"].fillna("").astype(str).str.strip().str.upper()
    standings = standings[standings["teamAbbrev"].ne("")]
    for text_column in ("conferenceName", "divisionName"):
        if text_column not in standings.columns:
            standings[text_column] = ""
        standings[text_column] = standings[text_column].fillna("").astype(str).str.strip()

    teams: list[dict] = []
    for _, row in standings.iterrows():
        odds = projected_teams.get(canonical_team_abbrev(row.get("teamAbbrev")), {})
        teams.append(_build_team_payload(row, odds, include_record))
    if not teams:
        return empty_board

    if projected_teams:
        teams.sort(key=lambda team: (-(team["cup_pct"] or 0.0), -(team["projected_points"] or 0.0), team["team_abbr"]))
    else:
        teams.sort(key=lambda team: (team["league_sequence"], -team["points"], team["team_abbr"]))
    for rank, team in enumerate(teams, start=1):
        team["rank"] = rank

    favorite_team: dict = {}
    if projected_teams and (teams[0]["cup_pct"] or 0.0) > 0:
        favorite_team = teams[0]
        favorite_team["is_favorite"] = True

    champion_team: dict = {}
    if mode == BOARD_MODE_CHAMPION:
        champion_abbr = canonical_team_abbrev(projection.get("champion"))
        champion_team = next((team for team in teams if canonical_team_abbrev(team["team_abbr"]) == champion_abbr), {})
        if not champion_team and champion_abbr:
            champion_team = {"team_abbr": champion_abbr, "team_name": ACTIVE_TEAMS.get(champion_abbr, champion_abbr)}
        if champion_team:
            champion_team["is_champion"] = True

    season_label = _season_span(season_year)
    simulation_count = int(projection.get("n_sims", 0) or 0) if projected_teams else 0
    contenders = [
        {"team_abbr": team["team_abbr"], "team_name": team["team_name"], "team_common_name": team["team_common_name"], "cup_pct": team["cup_pct"]}
        for team in teams[:_CONTENDER_COUNT]
    ] if projected_teams else []

    summary_text = ""
    if favorite_team:
        summary_text = (
            f"{favorite_team['team_name']}: {favorite_team['cup_pct'] * 100.0:.1f}% to win the "
            f"{season_label} Stanley Cup, from {simulation_count:,} simulations of the remaining "
            "schedule and playoff bracket."
        )
    elif champion_team:
        summary_text = f"{champion_team['team_name']} won the {season_label} Stanley Cup."

    timestamp = _parse_standings_timestamp(standings)
    if mode == BOARD_MODE_PRESEASON:
        generated_at_label = f"{season_label} preseason projection — updated {timestamp}" if timestamp else f"{season_label} preseason projection"
    elif mode == BOARD_MODE_CHAMPION:
        generated_at_label = f"Final {season_label} standings — as of {timestamp}" if timestamp else f"Final {season_label} standings"
    else:
        generated_at_label = _format_generated_at_label(standings)

    team_lookup = {team["team_abbr"]: team for team in teams}
    divisions: list[dict] = []
    for (conference_name, division_name), division_df in standings.groupby(["conferenceName", "divisionName"], sort=False):
        division_rows = [team_lookup[abbr] for abbr in division_df["teamAbbrev"] if abbr in team_lookup]
        if mode == BOARD_MODE_PRESEASON:
            division_rows.sort(key=lambda team: (-(team["projected_points"] or 0.0), team["team_abbr"]))
        else:
            division_rows.sort(key=lambda team: (team["division_sequence"], team["league_sequence"], -team["points"], team["team_abbr"]))
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
        "generated_at_label": generated_at_label,
        "mode": mode,
        "season_label": season_label,
        "simulation_count": simulation_count,
        "favorite_team_abbr": str(favorite_team.get("team_abbr", "") or ""),
        "favorite_team": favorite_team,
        "champion_team": champion_team,
        "contenders": contenders,
        "summary_text": summary_text,
        "teams": teams,
        "divisions": divisions,
    }
