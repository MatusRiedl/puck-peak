"""League-wide team strength ratings shared by the win-probability trainer and runtime.

Everything here is plain pandas/numpy, so the offline trainer, the Streamlit runtime and
the season simulator build identical features from identical inputs. Two rules keep the
model honest and are pinned by tests:

- The outcome label is the home team's ``wins`` flag, never a goal comparison. The stats
  API leaves the shootout goal out of ``goalsFor``, so comparing goals records every
  shootout as a tie, and the old trainer counted each one as an away win.
- Every pregame feature only sees games that finished before that game started.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nhl.constants import TEAM_LINEAGES

LEAGUE_MEAN_ELO = 1505.0
"""Rating every team starts from, and the value ratings regress toward each season."""

DEFAULT_RATING_PARAMS: dict[str, float] = {
    "elo_k": 6.0,
    "elo_home_advantage": 25.0,
    "elo_carryover": 0.8,
    "form_prior_weight": 15.0,
    "goal_diff_prior_keep": 0.5,
    "share_prior_keep": 0.6,
}
"""Fallback rating hyperparameters. The trained artifact carries the tuned values."""

MODEL_FEATURES: tuple[str, ...] = (
    "elo_diff",
    "goal_diff_shrunk_diff",
    "sat_share_shrunk_diff",
    "sog_share_shrunk_diff",
    "home_back_to_back",
    "away_back_to_back",
)
"""Every pregame feature the runtime can build. An artifact scores an ordered subset."""

DIFF_FEATURE_ATTRIBUTES: dict[str, str] = {
    "elo_diff": "elo",
    "goal_diff_shrunk_diff": "goal_diff_shrunk",
    "sat_share_shrunk_diff": "sat_share_shrunk",
    "sog_share_shrunk_diff": "sog_share_shrunk",
}
"""Home-minus-away features and the per-team snapshot attribute each one differences."""

FLAG_FEATURE_SIDES: dict[str, str] = {
    "home_back_to_back": "home",
    "away_back_to_back": "away",
}
"""Schedule flags and the side of the matchup each one describes."""

GAME_TABLE_COLUMNS = [
    "SeasonYear",
    "GameTypeId",
    "GameId",
    "GameDate",
    "HomeTeam",
    "AwayTeam",
    "HomeWin",
    "ResultType",
    "HomeGoals",
    "AwayGoals",
    "HomeShots",
    "AwayShots",
    "HomeSatFor",
    "HomeSatAgainst",
]
"""One row per completed game. ``ResultType`` is ``REG``, ``OT`` or ``SO``."""

SCHEDULE_COLUMNS = ["SeasonYear", "GameTypeId", "GameId", "GameDate", "GameStateId", "HomeTeam", "AwayTeam"]
"""One row per scheduled game, completed or not."""

REGULAR_SEASON = 2
PLAYOFFS = 3

_FORM_METRICS = (
    # (per-game metric, shrunk snapshot attribute, neutral league value, prior-keep param)
    ("goal_diff", "goal_diff_shrunk", 0.0, "goal_diff_prior_keep"),
    ("sat_share", "sat_share_shrunk", 0.5, "share_prior_keep"),
    ("sog_share", "sog_share_shrunk", 0.5, "share_prior_keep"),
)
_TEAM_ALIAS_TO_ACTIVE = {
    alias: active_abbr
    for active_abbr, aliases in TEAM_LINEAGES.items()
    for alias in aliases
}


def canonical_team_abbrev(team_abbr: object) -> str:
    """Map a historical abbreviation onto the active franchise (``ARI`` -> ``UTA``).

    Ratings carry over through relocations, so every table in this module keys teams
    by franchise rather than by the abbreviation printed in that season's data.

    Args:
        team_abbr: Raw abbreviation from any NHL payload.

    Returns:
        Upper-cased active franchise abbreviation, or an empty string.
    """
    clean_abbr = str(team_abbr or "").strip().upper()
    if not clean_abbr or clean_abbr == "NAN":
        return ""
    return _TEAM_ALIAS_TO_ACTIVE.get(clean_abbr, clean_abbr)


def resolve_rating_params(params: dict | None) -> dict[str, float]:
    """Return rating hyperparameters with defaults filled in and values coerced to float.

    Args:
        params: Partial or complete parameter mapping, usually from the artifact.

    Returns:
        A complete parameter dict.
    """
    resolved = dict(DEFAULT_RATING_PARAMS)
    for key, value in (params or {}).items():
        if key not in resolved:
            continue
        try:
            resolved[key] = float(value)
        except (TypeError, ValueError):
            continue
    return resolved


def season_year_from_game_id(game_id: object) -> int:
    """Return the season start year encoded in an NHL game id (2025020001 -> 2025)."""
    try:
        return int(game_id) // 1_000_000
    except (TypeError, ValueError):
        return 0


def game_type_from_game_id(game_id: object) -> int:
    """Return the game type encoded in an NHL game id (2025020001 -> 2)."""
    try:
        return (int(game_id) // 10_000) % 100
    except (TypeError, ValueError):
        return 0


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    """Return an empty frame with a fixed column order."""
    return pd.DataFrame({column: [] for column in columns})


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one column as floats, or an all-NaN series when the payload lacks it."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def build_league_game_table(
    summary_rows: list[dict] | pd.DataFrame | None,
    shooting_rows: list[dict] | pd.DataFrame | None = None,
    team_id_to_abbrev: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Pair per-team game rows from the stats API into one row per completed game.

    Args:
        summary_rows: ``team/summary?isGame=true`` rows, any seasons and game types.
        shooting_rows: Matching ``team/summaryshooting?isGame=true`` rows. Optional;
            without them the shot-attempt columns stay NaN.
        team_id_to_abbrev: NHL team id -> tri-code map from the team-list endpoint.

    Returns:
        A frame with ``GAME_TABLE_COLUMNS``, sorted by date then game id.
    """
    summary = pd.DataFrame(summary_rows) if not isinstance(summary_rows, pd.DataFrame) else summary_rows.copy()
    required = {"gameId", "gameDate", "homeRoad", "wins"}
    if summary.empty or not required.issubset(summary.columns):
        return _empty_frame(GAME_TABLE_COLUMNS)

    d = summary.copy()
    d["GameId"] = pd.to_numeric(d["gameId"], errors="coerce")
    d = d.dropna(subset=["GameId"])
    d["GameId"] = d["GameId"].astype("int64")
    d["HomeRoad"] = d["homeRoad"].astype(str).str.strip().str.upper()

    id_map = {int(key): str(value) for key, value in (team_id_to_abbrev or {}).items()}
    if "teamId" in d.columns and id_map:
        d["Team"] = pd.to_numeric(d["teamId"], errors="coerce").map(id_map).map(canonical_team_abbrev)
    elif "teamAbbrev" in d.columns:
        d["Team"] = d["teamAbbrev"].map(canonical_team_abbrev)
    else:
        return _empty_frame(GAME_TABLE_COLUMNS)

    for column in ("wins", "winsInRegulation", "winsInShootout", "otLosses", "goalsFor", "goalsAgainst", "shotsForPerGame"):
        d[column] = _numeric_column(d, column)

    d["satFor"] = np.nan
    d["satAgainst"] = np.nan
    shooting = pd.DataFrame(shooting_rows) if not isinstance(shooting_rows, pd.DataFrame) else shooting_rows.copy()
    if not shooting.empty and {"gameId", "teamId", "satFor", "satAgainst"}.issubset(shooting.columns) and "teamId" in d.columns:
        shooting = shooting[["gameId", "teamId", "satFor", "satAgainst"]].copy()
        shooting["gameId"] = pd.to_numeric(shooting["gameId"], errors="coerce")
        shooting["teamId"] = pd.to_numeric(shooting["teamId"], errors="coerce")
        shooting = shooting.dropna(subset=["gameId", "teamId"]).drop_duplicates(["gameId", "teamId"])
        d["_team_id"] = pd.to_numeric(d["teamId"], errors="coerce")
        d = d.drop(columns=["satFor", "satAgainst"]).merge(
            shooting.rename(columns={"gameId": "GameId", "teamId": "_team_id"}),
            on=["GameId", "_team_id"],
            how="left",
        )
        d["satFor"] = pd.to_numeric(d["satFor"], errors="coerce")
        d["satAgainst"] = pd.to_numeric(d["satAgainst"], errors="coerce")

    side_columns = ["GameId", "Team", "wins", "winsInRegulation", "winsInShootout", "otLosses", "goalsFor", "goalsAgainst", "shotsForPerGame", "satFor", "satAgainst"]
    home = d[d["HomeRoad"].eq("H")][side_columns + ["gameDate"]].drop_duplicates("GameId")
    away = d[d["HomeRoad"].eq("R")][side_columns].drop_duplicates("GameId")
    games = home.merge(away, on="GameId", suffixes=("_home", "_away"))
    games = games[games["Team_home"].ne("") & games["Team_away"].ne("")]
    # Exactly one winner per game. Anything else is a half-published or corrupt row.
    games = games[(games["wins_home"].fillna(0) + games["wins_away"].fillna(0)).eq(1)]
    if games.empty:
        return _empty_frame(GAME_TABLE_COLUMNS)

    home_win = games["wins_home"].eq(1)
    winner_regulation = np.where(home_win, games["winsInRegulation_home"], games["winsInRegulation_away"])
    winner_shootout = np.where(home_win, games["winsInShootout_home"], games["winsInShootout_away"])
    loser_ot_losses = np.where(home_win, games["otLosses_away"], games["otLosses_home"])
    goals_level = games["goalsFor_home"].eq(games["goalsFor_away"]).to_numpy()
    regulation_known = ~np.isnan(winner_regulation.astype(float))

    is_shootout = (np.nan_to_num(winner_shootout.astype(float)) >= 1) | goals_level
    went_past_regulation = np.where(
        regulation_known,
        np.nan_to_num(winner_regulation.astype(float)) < 1,
        np.nan_to_num(loser_ot_losses.astype(float)) >= 1,
    )
    result_type = np.where(is_shootout, "SO", np.where(went_past_regulation, "OT", "REG"))

    home_sat_for = games["satFor_home"].where(games["satFor_home"].notna(), games["satAgainst_away"])
    home_sat_against = games["satAgainst_home"].where(games["satAgainst_home"].notna(), games["satFor_away"])

    table = pd.DataFrame(
        {
            "SeasonYear": games["GameId"].map(season_year_from_game_id).astype(int),
            "GameTypeId": games["GameId"].map(game_type_from_game_id).astype(int),
            "GameId": games["GameId"].astype("int64"),
            "GameDate": games["gameDate"].astype(str).str.strip().str[:10],
            "HomeTeam": games["Team_home"],
            "AwayTeam": games["Team_away"],
            "HomeWin": home_win.astype(int),
            "ResultType": result_type,
            "HomeGoals": games["goalsFor_home"].astype(float),
            "AwayGoals": games["goalsFor_away"].astype(float),
            "HomeShots": games["shotsForPerGame_home"].astype(float),
            "AwayShots": games["shotsForPerGame_away"].astype(float),
            "HomeSatFor": home_sat_for.astype(float),
            "HomeSatAgainst": home_sat_against.astype(float),
        }
    )
    return table.sort_values(["GameDate", "GameId"], kind="stable").reset_index(drop=True)[GAME_TABLE_COLUMNS]


def build_league_schedule(
    game_rows: list[dict] | pd.DataFrame | None,
    team_id_to_abbrev: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Normalize stats-API ``game`` rows into one regular-season or playoff row per game.

    Args:
        game_rows: Rows from ``stats/rest/en/game``.
        team_id_to_abbrev: NHL team id -> tri-code map.

    Returns:
        A frame with ``SCHEDULE_COLUMNS``, sorted by date then game id. Exhibition game
        types (preseason, all-star, international) are dropped.
    """
    frame = pd.DataFrame(game_rows) if not isinstance(game_rows, pd.DataFrame) else game_rows.copy()
    required = {"id", "gameDate", "gameType", "homeTeamId", "visitingTeamId"}
    if frame.empty or not required.issubset(frame.columns):
        return _empty_frame(SCHEDULE_COLUMNS)

    id_map = {int(key): canonical_team_abbrev(value) for key, value in (team_id_to_abbrev or {}).items()}
    d = frame.copy()
    d["GameId"] = pd.to_numeric(d["id"], errors="coerce")
    d["GameTypeId"] = pd.to_numeric(d["gameType"], errors="coerce")
    d = d.dropna(subset=["GameId", "GameTypeId"])
    d = d[d["GameTypeId"].isin([REGULAR_SEASON, PLAYOFFS])]
    if d.empty:
        return _empty_frame(SCHEDULE_COLUMNS)

    schedule = pd.DataFrame(
        {
            "SeasonYear": d["GameId"].map(season_year_from_game_id).astype(int),
            "GameTypeId": d["GameTypeId"].astype(int),
            "GameId": d["GameId"].astype("int64"),
            "GameDate": d["gameDate"].astype(str).str.strip().str[:10],
            "GameStateId": _numeric_column(d, "gameStateId").fillna(0).astype(int),
            "HomeTeam": pd.to_numeric(d["homeTeamId"], errors="coerce").map(id_map).fillna(""),
            "AwayTeam": pd.to_numeric(d["visitingTeamId"], errors="coerce").map(id_map).fillna(""),
        }
    )
    schedule = schedule[schedule["HomeTeam"].ne("") & schedule["AwayTeam"].ne("")]
    return schedule.drop_duplicates("GameId").sort_values(["GameDate", "GameId"], kind="stable").reset_index(drop=True)


@dataclass
class EloRun:
    """Output of one pass of the Elo rating system over a game table.

    Attributes:
        pregame_diff: Home rating minus away rating before each game, aligned to the
            input frame's index. Home advantage is not included.
        ratings: Rating per team after the last game.
        last_season: Season of the last game processed, or ``None`` for no games.
    """

    pregame_diff: pd.Series
    ratings: dict[str, float]
    last_season: int | None


def regress_ratings(ratings: dict[str, float], seasons: int, carryover: float) -> dict[str, float]:
    """Pull ratings toward the league mean once per elapsed season boundary.

    Args:
        ratings: Current rating per team.
        seasons: Number of season boundaries to apply (0 returns a copy).
        carryover: Share of each team's distance from the mean that survives a boundary.

    Returns:
        A new rating dict.
    """
    factor = float(carryover) ** max(int(seasons), 0)
    return {team: LEAGUE_MEAN_ELO + (rating - LEAGUE_MEAN_ELO) * factor for team, rating in ratings.items()}


def run_elo(games: pd.DataFrame, params: dict | None = None) -> EloRun:
    """Run a margin-of-victory Elo over every game in date order.

    Shootout wins count as one-goal wins. Ratings regress toward the mean at every
    season boundary. The autocorrelation term damps updates for favourites, which
    otherwise inflate ratings through the margin multiplier.

    Args:
        games: Game table (``GAME_TABLE_COLUMNS``), any seasons and game types.
        params: Rating hyperparameters; see ``DEFAULT_RATING_PARAMS``.

    Returns:
        The pregame rating differences plus the final ratings.
    """
    resolved = resolve_rating_params(params)
    if games is None or games.empty:
        return EloRun(pregame_diff=pd.Series(dtype=float), ratings={}, last_season=None)

    k_factor = resolved["elo_k"]
    home_advantage = resolved["elo_home_advantage"]
    carryover = resolved["elo_carryover"]

    ordered = games.sort_values(["GameDate", "GameId"], kind="stable")
    seasons = ordered["SeasonYear"].to_numpy(dtype=int)
    home_teams = ordered["HomeTeam"].to_numpy()
    away_teams = ordered["AwayTeam"].to_numpy()
    home_wins = ordered["HomeWin"].to_numpy(dtype=float)
    margins = np.abs(np.nan_to_num(ordered["HomeGoals"].to_numpy(dtype=float) - ordered["AwayGoals"].to_numpy(dtype=float)))

    ratings: dict[str, float] = {}
    pregame = np.zeros(len(ordered), dtype=float)
    current_season: int | None = None
    for position in range(len(ordered)):
        season = int(seasons[position])
        if current_season is not None and season > current_season:
            ratings = regress_ratings(ratings, season - current_season, carryover)
        current_season = season if current_season is None else max(current_season, season)

        home_team = home_teams[position]
        away_team = away_teams[position]
        home_rating = ratings.get(home_team, LEAGUE_MEAN_ELO)
        away_rating = ratings.get(away_team, LEAGUE_MEAN_ELO)
        pregame[position] = home_rating - away_rating

        gap = home_rating + home_advantage - away_rating
        expected_home = 1.0 / (1.0 + 10.0 ** (-gap / 400.0))
        home_won = home_wins[position]
        winner_gap = max(-1000.0, min(1000.0, gap if home_won >= 0.5 else -gap))
        margin = max(float(margins[position]), 1.0)
        multiplier = (0.6686 * math.log(margin) + 0.8048) * (2.05 / (winner_gap * 0.001 + 2.05))
        delta = k_factor * multiplier * (home_won - expected_home)
        ratings[home_team] = home_rating + delta
        ratings[away_team] = away_rating - delta

    return EloRun(
        pregame_diff=pd.Series(pregame, index=ordered.index).reindex(games.index),
        ratings=ratings,
        last_season=current_season,
    )


def _team_game_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Split regular-season games into one row per team with that team's per-game metrics."""
    regular = games[games["GameTypeId"].eq(REGULAR_SEASON)]
    if regular.empty:
        return _empty_frame(["GameId", "SeasonYear", "GameDate", "Team", "goal_diff", "sat_share", "sog_share"])

    def _side(team_col: str, goals_for: str, goals_against: str, shots_for: str, shots_against: str, sat_for: str, sat_against: str) -> pd.DataFrame:
        sat_total = regular[sat_for] + regular[sat_against]
        shots_total = regular[shots_for] + regular[shots_against]
        return pd.DataFrame(
            {
                "GameId": regular["GameId"],
                "SeasonYear": regular["SeasonYear"],
                "GameDate": regular["GameDate"],
                "Team": regular[team_col],
                "goal_diff": regular[goals_for] - regular[goals_against],
                "sat_share": (regular[sat_for] / sat_total).where(sat_total > 0),
                "sog_share": (regular[shots_for] / shots_total).where(shots_total > 0),
            }
        )

    home = _side("HomeTeam", "HomeGoals", "AwayGoals", "HomeShots", "AwayShots", "HomeSatFor", "HomeSatAgainst")
    away = _side("AwayTeam", "AwayGoals", "HomeGoals", "AwayShots", "HomeShots", "HomeSatAgainst", "HomeSatFor")
    rows = pd.concat([home, away], ignore_index=True)
    return rows.sort_values(["Team", "SeasonYear", "GameDate", "GameId"], kind="stable").reset_index(drop=True)


def _season_form_priors(team_rows: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    """Return the regressed prior for every team-season that has a previous season on file."""
    season_means = team_rows.groupby(["Team", "SeasonYear"])[[metric for metric, *_ in _FORM_METRICS]].mean().reset_index()
    priors = season_means.copy()
    priors["SeasonYear"] = priors["SeasonYear"] + 1
    for metric, _, neutral, keep_param in _FORM_METRICS:
        keep = params[keep_param]
        priors[f"{metric}_prior"] = neutral + (priors[metric] - neutral) * keep
    return priors[["Team", "SeasonYear"] + [f"{metric}_prior" for metric, *_ in _FORM_METRICS]]


def compute_team_form(games: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Return pregame shrunk goal differential and shot shares for every team-game.

    Each value blends the team's season-to-date average with a prior worth
    ``form_prior_weight`` games: last season's average pulled toward the league mean, or
    the league mean itself for a team with no previous season on file. Only regular
    season games feed the form. A missing shot-attempt count leaves that game out of
    the average rather than counting as zero.

    Args:
        games: Game table, any seasons.
        params: Rating hyperparameters.

    Returns:
        One row per regular-season team-game with ``GamesBefore`` and the three
        ``*_shrunk`` pregame values.
    """
    resolved = resolve_rating_params(params)
    columns = ["GameId", "SeasonYear", "Team", "GamesBefore"] + [attribute for _, attribute, *_ in _FORM_METRICS]
    team_rows = _team_game_rows(games) if games is not None and not games.empty else _empty_frame([])
    if team_rows.empty:
        return _empty_frame(columns)

    prior_weight = resolved["form_prior_weight"]
    team_rows = team_rows.merge(_season_form_priors(team_rows, resolved), on=["Team", "SeasonYear"], how="left")
    grouped = team_rows.groupby(["Team", "SeasonYear"], sort=False)
    team_rows["GamesBefore"] = grouped.cumcount()
    for metric, attribute, neutral, _ in _FORM_METRICS:
        prior = team_rows[f"{metric}_prior"].fillna(neutral)
        values = team_rows[metric]
        valid = values.notna().astype(float)
        filled = values.fillna(0.0)
        sum_before = filled.groupby([team_rows["Team"], team_rows["SeasonYear"]]).cumsum() - filled
        count_before = valid.groupby([team_rows["Team"], team_rows["SeasonYear"]]).cumsum() - valid
        team_rows[attribute] = (prior_weight * prior + sum_before) / (prior_weight + count_before)
    return team_rows[columns]


def back_to_back_flags(schedule: pd.DataFrame) -> pd.DataFrame:
    """Flag each side of each game that played the calendar day before.

    Works on the completed-game table (training) and on the full schedule (runtime);
    only regular-season and playoff games count as a previous game.

    Args:
        schedule: Frame with ``GameId``, ``GameDate``, ``HomeTeam`` and ``AwayTeam``.

    Returns:
        ``GameId``, ``HomeBackToBack`` and ``AwayBackToBack`` (0/1 ints).
    """
    columns = ["GameId", "HomeBackToBack", "AwayBackToBack"]
    if schedule is None or schedule.empty:
        return _empty_frame(columns)

    appearances = pd.concat(
        [
            pd.DataFrame({"GameId": schedule["GameId"], "GameDate": schedule["GameDate"], "Team": schedule["HomeTeam"], "Side": "Home"}),
            pd.DataFrame({"GameId": schedule["GameId"], "GameDate": schedule["GameDate"], "Team": schedule["AwayTeam"], "Side": "Away"}),
        ],
        ignore_index=True,
    )
    appearances["Date"] = pd.to_datetime(appearances["GameDate"], errors="coerce")
    appearances = appearances.dropna(subset=["Date"]).sort_values(["Team", "Date", "GameId"], kind="stable")
    previous_date = appearances.groupby("Team")["Date"].shift(1)
    appearances["BackToBack"] = ((appearances["Date"] - previous_date).dt.days == 1).astype(int)
    flags = appearances.pivot_table(index="GameId", columns="Side", values="BackToBack", aggfunc="max").reset_index()
    flags = flags.rename(columns={"Home": "HomeBackToBack", "Away": "AwayBackToBack"})
    for column in ("HomeBackToBack", "AwayBackToBack"):
        if column not in flags.columns:
            flags[column] = 0
        flags[column] = flags[column].fillna(0).astype(int)
    return flags[columns]


def _end_of_season_form(games: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    """Return each team's shrunk form after its last regular-season game of each season."""
    team_rows = _team_game_rows(games)
    if team_rows.empty:
        return _empty_frame(["Team", "SeasonYear"] + [attribute for _, attribute, *_ in _FORM_METRICS])
    team_rows = team_rows.merge(_season_form_priors(team_rows, params), on=["Team", "SeasonYear"], how="left")
    prior_weight = params["form_prior_weight"]
    aggregations = {}
    for metric, attribute, neutral, _ in _FORM_METRICS:
        aggregations[f"{metric}_sum"] = (metric, "sum")
        aggregations[f"{metric}_count"] = (metric, "count")
        aggregations[f"{metric}_prior"] = (f"{metric}_prior", "first")
    totals = team_rows.groupby(["Team", "SeasonYear"]).agg(**aggregations).reset_index()
    for metric, attribute, neutral, _ in _FORM_METRICS:
        prior = totals[f"{metric}_prior"].fillna(neutral)
        totals[attribute] = (prior_weight * prior + totals[f"{metric}_sum"]) / (prior_weight + totals[f"{metric}_count"])
    return totals[["Team", "SeasonYear"] + [attribute for _, attribute, *_ in _FORM_METRICS]]


def build_model_features(games: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Build the leak-safe training table: one row per game with every pregame feature.

    Playoff games take each team's end-of-regular-season form. Elo runs through every
    game, playoffs included.

    Args:
        games: Game table covering the seasons to train on plus warm-up seasons.
        params: Rating hyperparameters.

    Returns:
        Game identity columns, ``HomeWin``, ``ResultType``, games-before counts and
        every feature in ``MODEL_FEATURES``.
    """
    resolved = resolve_rating_params(params)
    identity = ["SeasonYear", "GameTypeId", "GameId", "GameDate", "HomeTeam", "AwayTeam", "HomeWin", "ResultType"]
    output_columns = identity + ["HomeGamesBefore", "AwayGamesBefore"] + list(MODEL_FEATURES)
    if games is None or games.empty:
        return _empty_frame(output_columns)

    table = games.sort_values(["GameDate", "GameId"], kind="stable").reset_index(drop=True)
    table["elo_diff"] = run_elo(table, resolved).pregame_diff.to_numpy()

    attributes = [attribute for _, attribute, *_ in _FORM_METRICS]
    form = compute_team_form(table, resolved)
    end_of_season = _end_of_season_form(table, resolved)
    for side, team_col in (("Home", "HomeTeam"), ("Away", "AwayTeam")):
        side_form = form.rename(columns={"Team": team_col, "GamesBefore": f"{side}GamesBefore", **{a: f"{side}_{a}" for a in attributes}})
        table = table.merge(side_form.drop(columns=["SeasonYear"]), on=["GameId", team_col], how="left")
        season_form = end_of_season.rename(columns={"Team": team_col, **{a: f"{side}_{a}_season" for a in attributes}})
        table = table.merge(season_form, on=[team_col, "SeasonYear"], how="left")
        for attribute in attributes:
            table[f"{side}_{attribute}"] = table[f"{side}_{attribute}"].fillna(table[f"{side}_{attribute}_season"])
        table[f"{side}GamesBefore"] = table[f"{side}GamesBefore"].fillna(-1).astype(int)

    for feature, attribute in DIFF_FEATURE_ATTRIBUTES.items():
        if attribute == "elo":
            continue
        table[feature] = table[f"Home_{attribute}"] - table[f"Away_{attribute}"]

    flags = back_to_back_flags(table)
    table = table.merge(flags, on="GameId", how="left")
    table["home_back_to_back"] = table["HomeBackToBack"].fillna(0).astype(float)
    table["away_back_to_back"] = table["AwayBackToBack"].fillna(0).astype(float)
    table = table.dropna(subset=list(MODEL_FEATURES))
    return table.sort_values(["GameDate", "GameId"], kind="stable").reset_index(drop=True)[output_columns]


def current_team_snapshot(
    games: pd.DataFrame,
    target_season: int,
    params: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Return every team's current rating inputs for scoring games in ``target_season``.

    Before a team's first game of the season, its form is the regressed prior and its
    Elo is last season's rating pulled toward the mean. This is what lets opening-night
    games get an estimate without falling back to a different season's features.

    Args:
        games: Game table covering ``target_season`` and the warm-up seasons before it.
        target_season: Season start year being predicted.
        params: Rating hyperparameters.

    Returns:
        Team abbreviation -> ``elo``, ``goal_diff_shrunk``, ``sat_share_shrunk``,
        ``sog_share_shrunk`` and ``games_played`` (regular season, target season).
    """
    resolved = resolve_rating_params(params)
    if games is None or games.empty:
        return {}

    played = games[games["SeasonYear"] <= int(target_season)]
    elo = run_elo(played, resolved)
    ratings = elo.ratings
    if elo.last_season is not None and elo.last_season < int(target_season):
        ratings = regress_ratings(ratings, int(target_season) - elo.last_season, resolved["elo_carryover"])

    team_rows = _team_game_rows(played)
    teams = set(
        played.loc[played["SeasonYear"] >= int(target_season) - 1, "HomeTeam"]
    ) | set(played.loc[played["SeasonYear"] >= int(target_season) - 1, "AwayTeam"])
    snapshot: dict[str, dict[str, float]] = {}
    if team_rows.empty:
        for team in sorted(teams):
            snapshot[team] = {"elo": ratings.get(team, LEAGUE_MEAN_ELO), "goal_diff_shrunk": 0.0, "sat_share_shrunk": 0.5, "sog_share_shrunk": 0.5, "games_played": 0}
        return snapshot

    priors = _season_form_priors(team_rows, resolved)
    current_rows = team_rows[team_rows["SeasonYear"].eq(int(target_season))]
    prior_weight = resolved["form_prior_weight"]
    for team in sorted(teams):
        team_prior = priors[(priors["Team"] == team) & (priors["SeasonYear"] == int(target_season))]
        team_current = current_rows[current_rows["Team"] == team]
        entry: dict[str, float] = {"elo": float(ratings.get(team, LEAGUE_MEAN_ELO)), "games_played": int(len(team_current))}
        for metric, attribute, neutral, _ in _FORM_METRICS:
            prior = float(team_prior[f"{metric}_prior"].iloc[0]) if not team_prior.empty and pd.notna(team_prior[f"{metric}_prior"].iloc[0]) else neutral
            values = team_current[metric].dropna()
            entry[attribute] = float((prior_weight * prior + values.sum()) / (prior_weight + len(values)))
        snapshot[team] = entry
    return snapshot


def matchup_feature_values(
    home: dict[str, float],
    away: dict[str, float],
    home_back_to_back: bool = False,
    away_back_to_back: bool = False,
) -> dict[str, float]:
    """Build the feature dict for one matchup from two team snapshots.

    Args:
        home: Home team entry from ``current_team_snapshot``.
        away: Away team entry from ``current_team_snapshot``.
        home_back_to_back: Whether the home team played the day before.
        away_back_to_back: Whether the away team played the day before.

    Returns:
        Every feature in ``MODEL_FEATURES``.
    """
    features = {
        feature: float(home[attribute]) - float(away[attribute])
        for feature, attribute in DIFF_FEATURE_ATTRIBUTES.items()
    }
    features["home_back_to_back"] = 1.0 if home_back_to_back else 0.0
    features["away_back_to_back"] = 1.0 if away_back_to_back else 0.0
    return features
