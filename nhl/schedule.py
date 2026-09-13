"""Live schedule helpers: defaults, upcoming games, featured players and model inference."""

import logging
import zlib
from contextlib import closing
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from nhl.api import get_client
from nhl.cache import T1_TTL, T2_DEFAULT_TTL, T3_DEFAULT_TTL
from nhl.constants import (
    ACTIVE_TEAMS,
    PARTNER_ODDS_URL,
    TEAM_LINEAGES,
    current_season_year,
)
from nhl.data_loaders import (
    get_current_nhl_standings,
    get_league_game_table,
    get_league_schedule,
    get_playoff_bracket,
    get_team_available_nhl_seasons,
    get_team_season_game_log,
    load_win_prob_weights,
)
from nhl.goal_model import DEFAULT_GOAL_MODEL, current_scoring_environment, game_markets
from nhl.ledger import (
    MIN_GRADED_GAMES_FOR_RECORD,
    connect_ledger,
    grade_predictions,
    load_ledger,
    parse_partner_odds,
    prediction_row,
    record_market_odds,
    record_prediction,
    track_record,
)
from nhl.season_sim import (
    parse_playoff_bracket,
    remaining_regular_season_games,
    simulate_season,
    teams_from_standings,
)
from nhl.team_ratings import (
    REGULAR_SEASON,
    back_to_back_flags,
    current_team_snapshot,
    matchup_feature_values,
    season_year_from_game_id,
)
from nhl.win_prob import (
    WIN_PROB_FEATURE_LABELS,
    get_top_feature_driver,
    score_home_win_probability,
)

log = logging.getLogger("nhl.schedule")

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_SCOREBOARD_URL  = "https://api-web.nhle.com/v1/scoreboard/now"
_SCORE_DATE_URL  = "https://api-web.nhle.com/v1/score/{date}"
_CLUB_STATS_URL  = "https://api-web.nhle.com/v1/club-stats/{}/now"

_LIVE_STATES        = {"LIVE", "CRIT"}
_FINAL_STATES       = {"FINAL", "OVER", "OFF"}
_FUTURE_STATES      = {"FUT", "PRE"}
_VALID_GAME_TYPES   = {2, 3}   # 2 = regular season, 3 = playoffs
# Preseason (1) counts as "something to show". Excluded from _VALID_GAME_TYPES because
# preseason results must not feed standings or win-probability math, but the matchups
# are real and are the only NHL games on the calendar for most of September.
_UPCOMING_GAME_TYPES = {1, 2, 3}
_CENTRAL_EUROPE_TZ  = ZoneInfo("Europe/Prague")
_WEEKDAY_ABBR       = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTH_ABBR         = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
_TEAM_ALIAS_TO_ACTIVE = {
    alias: active_abbr
    for active_abbr, aliases in TEAM_LINEAGES.items()
    for alias in aliases
}
_RATING_WARMUP_SEASONS = 3
"""Completed seasons replayed before the target season so Elo ratings have settled."""
_EARLY_SEASON_GAMES = 10
"""Below this many games played, a team's rating still leans mostly on last season."""
_SIMULATION_COUNT = 10_000
_LEDGER_CAPTURE_HOURS = 36
"""Games starting within this window get their prediction logged (and refreshed until puck drop)."""
_PARTNER_ODDS_COUNTRIES = ("CA", "US", "SE", "FI", "CZ")
"""NHL.com betting-partner feeds read for the market benchmark (free; no key, no payment)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def get_live_or_recent_game() -> tuple[str, str] | None:
    """Return the current or most recent NHL matchup, or ``None`` on failure."""
    try:
        client = get_client()

        # Scoreboard is more reliable and covers ~11 days in one request.
        # reverse_dates=True so we find the most recent FINAL game first.
        scoreboard = client.get(
            url=_SCOREBOARD_URL,
            cache_key="scoreboard",
            ttl=T3_DEFAULT_TTL,
            timeout=5,
        )
        if scoreboard:
            result = _find_game_from_data(scoreboard, reverse_dates=True)
            if result:
                return result

        # Fallback: walk back day-by-day.
        for days_back in range(0, 8):
            date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            score_data = client.get(
                url=_SCORE_DATE_URL.format(date=date_str),
                cache_key=f"score:{date_str}",
                ttl=T3_DEFAULT_TTL,
                timeout=5,
            )
            if score_data:
                result = _find_game_from_data(score_data)
                if result:
                    return result

        return None
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_upcoming_games(limit: int = 6, days_ahead: int = 60) -> list[dict]:
    """Return the next few upcoming games for the Live games tab.

    Reads the multi-day scoreboard first, which covers roughly 11 days in a single
    request and, in the offseason, rolls its own window forward to the next games on
    the calendar. Only if that comes up short does this walk individual dates.

    The default window is wide enough to span the September gap between the last
    preseason slate and opening night. A 14-day window returned nothing at all in
    early September: the only games inside it were preseason, and the first regular
    season game sat just beyond the edge.

    Args:
        limit: Maximum number of games to return.
        days_ahead: How many days forward the per-date fallback may walk.

    Returns:
        A list of normalized upcoming-game dicts, soonest first.
    """
    if limit <= 0:
        return []

    try:
        now_utc = datetime.now(timezone.utc)
        upcoming_games: list[dict] = []
        seen_game_ids: set[int] = set()

        client = get_client()

        def _absorb(games: list[dict]) -> None:
            """Add newly seen upcoming games from one payload, de-duplicated by id."""
            for game in _extract_upcoming_games(games, now_utc):
                game_id = game.get("game_id", 0)
                if game_id and game_id in seen_game_ids:
                    continue
                seen_game_ids.add(game_id)
                upcoming_games.append(game)

        scoreboard = client.get(
            url=_SCOREBOARD_URL,
            cache_key="scoreboard",
            ttl=T3_DEFAULT_TTL,
            timeout=5,
        )
        if scoreboard:
            for day in scoreboard.get("gamesByDate", []) or []:
                _absorb(day.get("games", []) or [])

        # Per-date fallback: only runs when the scoreboard window was empty or thin,
        # so the common case costs one request rather than fifteen.
        if len(upcoming_games) < limit:
            for day_offset in range(max(days_ahead, 0) + 1):
                date_str = (now_utc + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                data = client.get(
                    url=_SCORE_DATE_URL.format(date=date_str),
                    cache_key=f"score:{date_str}",
                    ttl=T3_DEFAULT_TTL,
                    timeout=5,
                )
                if data is None:
                    continue

                _absorb(data.get("games", []) or [])

                if len(upcoming_games) >= limit:
                    break

        upcoming_games.sort(key=lambda game: game["sort_ts"])
        trimmed_games = upcoming_games[:limit]
        for game in trimmed_games:
            game["pregame_win_prob"] = get_game_win_probabilities(
                game["away_abbr"],
                game["home_abbr"],
                game.get("game_id", 0),
                game.get("game_type", 2),
            )
        return trimmed_games
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_game_details(game_date: str, game_id: int) -> dict:
    """Return normalized score details for one exact NHL game.

    Args:
        game_date: Date string in ``YYYY-MM-DD`` form.
        game_id: Exact NHL game identifier.

    Returns:
        Normalized game detail dict, or ``{}`` if lookup fails.
    """
    clean_date = str(game_date or '').strip()
    if not clean_date or not game_id:
        return {}

    try:
        from datetime import date as _date
        today = _date.today().isoformat()
        ttl = T3_DEFAULT_TTL if clean_date >= today else T1_TTL
        data = get_client().get(
            url=_SCORE_DATE_URL.format(date=clean_date),
            cache_key=f"score:{clean_date}",
            ttl=ttl,
            timeout=5,
        )
        if data is None:
            return {}
        return _extract_game_details_from_payload(data, int(game_id), clean_date)
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def get_matchup_history(away_abbr: str, home_abbr: str, limit: int = 10) -> list[dict]:
    """Return the most recent completed meetings for one away/home franchise pair."""
    clean_away_abbr = _canonical_team_abbr(away_abbr)
    clean_home_abbr = _canonical_team_abbr(home_abbr)
    try:
        limit_int = max(0, int(limit))
    except Exception:
        limit_int = 10

    if (
        limit_int <= 0
        or not clean_away_abbr
        or not clean_home_abbr
        or clean_away_abbr == clean_home_abbr
    ):
        return []

    base_team_abbr = clean_away_abbr
    opponent_team_abbr = clean_home_abbr
    seasons = get_team_available_nhl_seasons(base_team_abbr)
    if not seasons:
        base_team_abbr = clean_home_abbr
        opponent_team_abbr = clean_away_abbr
        seasons = get_team_available_nhl_seasons(base_team_abbr)

    history_rows: list[dict] = []
    for season_year in seasons:
        season_games = get_team_season_game_log(base_team_abbr, int(season_year))
        if season_games.empty or "OpponentAbbrev" not in season_games.columns:
            continue

        season_games = season_games.copy()
        season_games["_OpponentCanonical"] = season_games["OpponentAbbrev"].map(_canonical_team_abbr)
        filtered_games = season_games[season_games["_OpponentCanonical"] == opponent_team_abbr]
        if filtered_games.empty:
            continue

        history_rows.extend(filtered_games.to_dict("records"))
        history_rows.sort(
            key=lambda row: (
                str(row.get("GameDate", "") or ""),
                int(row.get("GameId", 0) or 0),
            ),
            reverse=True,
        )
        if len(history_rows) >= limit_int:
            history_rows = history_rows[:limit_int]
            break

    history_games: list[dict] = []
    for row in history_rows[:limit_int]:
        score_details = get_game_details(
            str(row.get("GameDate", "") or ""),
            int(row.get("GameId", 0) or 0),
        )
        history_games.append(_build_matchup_history_game(row, score_details))
    return history_games


@st.cache_data(ttl=3600)
def get_featured_players(home_abbr: str, away_abbr: str) -> dict:
    """Return featured skaters, goalies, and team names for a matchup pair."""
    try:
        players: dict[int, str] = {}
        teams:   dict[str, str] = {}

        for abbr in (home_abbr, away_abbr):
            if abbr not in ACTIVE_TEAMS:
                continue

            stats = _get_cached_club_stats(abbr)
            if not stats:
                continue

            teams[abbr] = ACTIVE_TEAMS[abbr]

            skaters = stats["skaters"]
            goalies = stats["goalies"]

            best = _select_best_skater(skaters)
            if best:
                players[best["playerId"]] = best["name"]

            best_goalie = _select_best_goalie(goalies)
            if best_goalie:
                players[best_goalie["playerId"]] = best_goalie["name"]

        return {"players": players, "teams": teams}

    except Exception:
        return {"players": {}, "teams": {}}


@st.cache_data(ttl=1800)
def get_current_team_ratings(season_year: int | None = None) -> dict:
    """Return one rating snapshot for every team, shared by predictions and the Cup board.

    Elo needs history to settle, so the snapshot replays ``_RATING_WARMUP_SEASONS``
    completed seasons before the target season. Those come from the 24-hour disk cache,
    so a warm call only refetches the current season's three league-wide reports.

    Args:
        season_year: Season to rate. Defaults to the current season.

    Returns:
        ``{"season_year": int, "teams": {abbr: snapshot}, "scoring_environment": float | None}``,
        or ``{}`` when the model artifact or the game data is unavailable.
        ``scoring_environment`` is the league's regulation goals per team-game that the
        goal model prices markets with.
    """
    artifact = load_win_prob_weights()
    if not artifact:
        return {}

    target_season = int(season_year) if season_year else current_season_year()
    frames = [
        get_league_game_table(year)
        for year in range(target_season - _RATING_WARMUP_SEASONS, target_season + 1)
    ]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return {}

    games = pd.concat(frames, ignore_index=True)
    teams = current_team_snapshot(games, target_season, artifact["rating_params"])
    if not teams:
        return {}
    goal_model = artifact.get("goal_model") or DEFAULT_GOAL_MODEL
    return {
        "season_year": target_season,
        "teams": teams,
        "scoring_environment": current_scoring_environment(games, target_season, goal_model["environment_prior_team_games"]),
    }


@st.cache_data(ttl=3600)
def _get_schedule_back_to_back_flags(season_year: int) -> dict[int, tuple[bool, bool]]:
    """Return ``game_id -> (home on back-to-back, away on back-to-back)`` for one season."""
    flags = back_to_back_flags(get_league_schedule(int(season_year)))
    return {
        int(row.GameId): (bool(row.HomeBackToBack), bool(row.AwayBackToBack))
        for row in flags.itertuples(index=False)
    }


@st.cache_data(ttl=300)
def get_game_win_probabilities(
    away_abbr: str,
    home_abbr: str,
    game_id: int = 0,
    game_type: int = REGULAR_SEASON,
) -> dict | None:
    """Return one pregame win-probability estimate for a regular-season or playoff game.

    Preseason and other exhibition games get ``None``. Their lineups are mostly
    prospects, so a regular-season model has nothing honest to say about them.

    Args:
        away_abbr: Away team abbreviation.
        home_abbr: Home team abbreviation.
        game_id: NHL game id. It supplies the season and the back-to-back lookup.
        game_type: NHL game type (1 preseason, 2 regular season, 3 playoffs).

    Returns:
        Percentages, raw probabilities, fair decimal odds, a one-line model driver
        label, games played and back-to-back flags. When the artifact carries a goal
        model it also returns ``markets``: the 60-minute result and the puck line. Totals
        are never included; they failed their backtest. ``None`` when no estimate can
        be made.
    """
    clean_away_abbr = _canonical_team_abbr(away_abbr)
    clean_home_abbr = _canonical_team_abbr(home_abbr)
    if not clean_away_abbr or not clean_home_abbr or clean_away_abbr == clean_home_abbr:
        return None
    try:
        if int(game_type or 0) not in _VALID_GAME_TYPES:
            return None
    except (TypeError, ValueError):
        return None

    artifact = load_win_prob_weights()
    if not artifact:
        return None

    try:
        season_year = season_year_from_game_id(game_id) or current_season_year()
        ratings = get_current_team_ratings(season_year)
        teams = ratings.get("teams", {}) if ratings else {}
        home_entry = teams.get(clean_home_abbr)
        away_entry = teams.get(clean_away_abbr)
        if home_entry is None or away_entry is None:
            return None

        home_back_to_back, away_back_to_back = (
            _get_schedule_back_to_back_flags(season_year).get(int(game_id), (False, False))
            if game_id
            else (False, False)
        )
        scored_probability = score_home_win_probability(
            matchup_feature_values(home_entry, away_entry, home_back_to_back, away_back_to_back),
            artifact,
        )
        home_prob = min(max(float(scored_probability["home_win_prob"]), 0.001), 0.999)
        home_pct = min(max(int(round(home_prob * 100.0)), 0), 100)
        home_games = int(home_entry.get("games_played", 0) or 0)
        away_games = int(away_entry.get("games_played", 0) or 0)

        return {
            "away_pct": 100 - home_pct,
            "home_pct": home_pct,
            "home_win_prob": home_prob,
            "away_win_prob": 1.0 - home_prob,
            "fair_odds_home": round(1.0 / home_prob, 2),
            "fair_odds_away": round(1.0 / (1.0 - home_prob), 2),
            "model_label": _build_model_label(clean_away_abbr, clean_home_abbr, scored_probability),
            "home_games_played": home_games,
            "away_games_played": away_games,
            "early_season": int(game_type) == REGULAR_SEASON and min(home_games, away_games) < _EARLY_SEASON_GAMES,
            "home_back_to_back": bool(home_back_to_back),
            "away_back_to_back": bool(away_back_to_back),
            "season_used": season_year,
            "model_version": int(artifact.get("model_version", 0) or 0),
            "markets": game_markets(home_prob, ratings.get("scoring_environment"), artifact.get("goal_model")),
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_season_projection() -> dict:
    """Simulate the rest of the current season and playoffs for the Cup odds board.

    Returns:
        A dict whose ``state`` is one of:

        - ``projection``: ``teams`` maps each team to its projected points and playoff
          milestone odds. ``phase`` is ``preseason``, ``regular_season`` or ``playoffs``.
        - ``champion``: the Cup is decided and ``champion`` names the winner of
          ``season_year``. This covers the summer before the rollover.
        - ``unavailable``: the model, standings or schedule could not be loaded.
    """
    season_year = current_season_year()
    unavailable = {"state": "unavailable", "season_year": season_year, "teams": {}}
    artifact = load_win_prob_weights()
    if not artifact:
        return unavailable

    try:
        bracket = parse_playoff_bracket(get_playoff_bracket(season_year))
        if bracket["champion"]:
            return {"state": "champion", "season_year": season_year, "champion": bracket["champion"], "teams": {}}

        standings_df = get_current_nhl_standings()
        schedule = get_league_schedule(season_year)
        if standings_df.empty or schedule.empty:
            previous = parse_playoff_bracket(get_playoff_bracket(season_year - 1))
            if previous["champion"]:
                return {"state": "champion", "season_year": season_year - 1, "champion": previous["champion"], "teams": {}}
            return unavailable

        standings_season = 0
        if "seasonId" in standings_df.columns:
            standings_season = int(pd.to_numeric(standings_df["seasonId"], errors="coerce").fillna(0).iloc[0]) // 10000
        in_season = standings_season == season_year
        teams = teams_from_standings(standings_df, include_record=in_season)
        ratings = get_current_team_ratings(season_year)
        if not teams or not ratings:
            return unavailable

        completed = get_league_game_table(season_year)
        completed_ids = (
            set(completed.loc[completed["GameTypeId"].eq(REGULAR_SEASON), "GameId"].astype(int))
            if not completed.empty
            else set()
        )
        remaining = remaining_regular_season_games(schedule, completed_ids)
        regular_total = int(schedule["GameTypeId"].eq(REGULAR_SEASON).sum())
        progress = 1.0 - len(remaining) / regular_total if regular_total else 0.0
        standings_stamp = str(standings_df["standingsDateTimeUtc"].iloc[0]) if "standingsDateTimeUtc" in standings_df.columns else ""
        # Seeded from the data it simulates, so reruns show identical numbers until
        # a game finishes.
        seed = zlib.crc32(f"{season_year}:{standings_stamp}:{len(remaining)}".encode("utf-8"))

        result = simulate_season(
            teams,
            ratings.get("teams", {}),
            remaining,
            artifact,
            bracket=bracket,
            season_progress=progress,
            n_sims=_SIMULATION_COUNT,
            seed=seed,
        )
        if not in_season:
            phase = "preseason"
        elif bracket["round_one_complete"]:
            phase = "playoffs"
        else:
            phase = "regular_season"
        return {
            "state": "projection",
            "phase": phase,
            "season_year": season_year,
            "n_sims": result["n_sims"],
            "teams": result["teams"],
        }
    except Exception:
        log.exception("Season projection failed")
        return unavailable


def capture_prediction_ledger(now_utc: datetime | None = None) -> dict:
    """Log upcoming predictions and market prices until puck drop, then grade finished games.

    Runs from the cache warmer's live cycle. Each pass refreshes the rows of games that
    have not started, so the ledger keeps the last pregame numbers and the market's
    near-closing prices; rows are frozen at puck drop (see ``nhl.ledger``). Market prices
    come from the NHL API's free partner-odds feeds and are never displayed.

    Args:
        now_utc: Capture time; defaults to now.

    Returns:
        Counts of predictions written, market rows written and predictions graded.
    """
    now = now_utc or datetime.now(timezone.utc)
    summary = {"predictions": 0, "market_rows": 0, "graded": 0}
    artifact = load_win_prob_weights()
    if not artifact:
        return summary

    client = get_client()
    scoreboard = client.get(url=_SCOREBOARD_URL, cache_key="scoreboard", ttl=T3_DEFAULT_TTL, timeout=5)
    upcoming: list[dict] = []
    for day in (scoreboard or {}).get("gamesByDate", []) or []:
        upcoming.extend(_extract_upcoming_games(day.get("games", []) or [], now))
    horizon = now + timedelta(hours=_LEDGER_CAPTURE_HOURS)

    with closing(connect_ledger()) as connection:
        for game in upcoming:
            start = _parse_utc_timestamp(game.get("start_time_utc"))
            if game.get("game_type") not in _VALID_GAME_TYPES or start is None or start > horizon:
                continue
            probability = get_game_win_probabilities(game["away_abbr"], game["home_abbr"], game["game_id"], game["game_type"])
            if not probability:
                continue
            row = prediction_row(game, probability, artifact.get("generated_at_utc", ""))
            summary["predictions"] += int(record_prediction(connection, row, now))

        odds_rows: list[dict] = []
        for country in _PARTNER_ODDS_COUNTRIES:
            payload = client.get(
                url=PARTNER_ODDS_URL.format(country),
                cache_key=f"partner_odds:{country}",
                ttl=T3_DEFAULT_TTL,
                timeout=10,
            )
            odds_rows.extend(row for row in parse_partner_odds(payload) if row.get("game_type") in _VALID_GAME_TYPES)
        summary["market_rows"] = record_market_odds(connection, odds_rows, now)

        season = current_season_year()
        tables = [get_league_game_table(year) for year in (season - 1, season)]
        tables = [table for table in tables if table is not None and not table.empty]
        if tables:
            summary["graded"] = grade_predictions(connection, pd.concat(tables, ignore_index=True), now)
    return summary


@st.cache_data(ttl=600)
def get_track_record() -> dict:
    """Return this season's live ledger record and the model's backtest summary.

    Returns:
        ``season_year``, ``live`` (``nhl.ledger.track_record`` output), ``backtest``
        (games, accuracy and log loss pooled over the artifact's test seasons, or
        ``None``) and ``min_games``, the graded-game count before the live record is shown.
    """
    season = current_season_year()
    try:
        predictions, market_odds = load_ledger()
        live = track_record(predictions, market_odds, season)
    except Exception:
        log.exception("Reading the prediction ledger failed")
        live = {"logged": 0, "games": 0}

    backtest = None
    artifact = load_win_prob_weights()
    folds = ((artifact or {}).get("validation_metrics") or {}).get("backtest") or []
    folds = [fold for fold in folds if isinstance(fold, dict) and isinstance(fold.get("model"), dict)]
    if folds:
        games = sum(int(fold["model"]["n"]) for fold in folds)
        backtest = {
            "first_season": min(int(fold["season"]) for fold in folds),
            "last_season": max(int(fold["season"]) for fold in folds),
            "games": games,
            "accuracy": sum(float(fold["model"]["accuracy"]) * int(fold["model"]["n"]) for fold in folds) / games,
            "log_loss": sum(float(fold["model"]["log_loss"]) * int(fold["model"]["n"]) for fold in folds) / games,
        }
    return {"season_year": season, "live": live, "backtest": backtest, "min_games": MIN_GRADED_GAMES_FOR_RECORD}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _canonical_team_abbr(team_abbr: str | None) -> str:
    """Return the active-team abbreviation for a historical franchise alias."""
    clean_abbr = str(team_abbr or "").strip().upper()
    if not clean_abbr:
        return ""
    return _TEAM_ALIAS_TO_ACTIVE.get(clean_abbr, clean_abbr)


def _coerce_optional_int(value) -> int | None:
    """Return an integer when possible, else ``None``."""
    try:
        return int(round(float(value)))
    except Exception:
        return None


def _coalesce_non_empty(*values):
    """Return the first value that is neither ``None`` nor an empty string."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _build_matchup_history_game(game_row: dict, score_details: dict) -> dict:
    """Merge season-log metadata with score-endpoint details for history cards."""
    team_abbr = _canonical_team_abbr(game_row.get("TeamAbbrev", ""))
    opponent_abbr = _canonical_team_abbr(game_row.get("OpponentAbbrev", ""))
    home_road_flag = str(game_row.get("HomeRoadFlag", "") or "").strip().upper()
    team_name = ACTIVE_TEAMS.get(team_abbr, str(game_row.get("TeamName", "") or team_abbr).strip())
    opponent_name = ACTIVE_TEAMS.get(opponent_abbr, str(game_row.get("OpponentName", "") or opponent_abbr).strip())
    goals_for = _coerce_optional_int(game_row.get("Goals"))
    goals_against = _coerce_optional_int(game_row.get("GoalsAgainst"))

    if home_road_flag == "H":
        fallback_away_abbr, fallback_away_name = opponent_abbr, opponent_name
        fallback_home_abbr, fallback_home_name = team_abbr, team_name
        fallback_away_score, fallback_home_score = goals_against, goals_for
    else:
        fallback_away_abbr, fallback_away_name = team_abbr, team_name
        fallback_home_abbr, fallback_home_name = opponent_abbr, opponent_name
        fallback_away_score, fallback_home_score = goals_for, goals_against

    game_type_value = _coalesce_non_empty(score_details.get("game_type"), game_row.get("gameTypeId"))
    if game_type_value is None:
        game_type_label = str(game_row.get("GameType", "") or "").strip().lower()
        game_type_value = 3 if game_type_label == "playoffs" else 2 if game_type_label == "regular" else 0

    return {
        "game_id": _coalesce_non_empty(score_details.get("game_id"), _coerce_optional_int(game_row.get("GameId"))) or 0,
        "game_date": str(_coalesce_non_empty(score_details.get("game_date"), game_row.get("GameDate")) or ""),
        "game_type": int(game_type_value or 0),
        "away_abbr": str(_coalesce_non_empty(score_details.get("away_abbr"), fallback_away_abbr) or ""),
        "away_name": str(_coalesce_non_empty(score_details.get("away_name"), fallback_away_name) or ""),
        "away_score": _coalesce_non_empty(score_details.get("away_score"), fallback_away_score),
        "home_abbr": str(_coalesce_non_empty(score_details.get("home_abbr"), fallback_home_abbr) or ""),
        "home_name": str(_coalesce_non_empty(score_details.get("home_name"), fallback_home_name) or ""),
        "home_score": _coalesce_non_empty(score_details.get("home_score"), fallback_home_score),
        "venue": str(_coalesce_non_empty(score_details.get("venue"), "") or ""),
        "start_label_cest": str(
            _coalesce_non_empty(
                score_details.get("start_label_cest"),
                game_row.get("GameDate"),
            ) or ""
        ),
        "status_label": str(_coalesce_non_empty(score_details.get("status_label"), "Final") or ""),
    }


def _find_game_from_data(data: dict, reverse_dates: bool = False) -> tuple[str, str] | None:
    """Parse a NHL score payload and return (home_abbr, away_abbr) for one game.

    Preference order is live, then most recently finished, then the soonest
    upcoming game. The upcoming pass is what keeps the app populated through the
    offseason: between the Cup final and opening night there is no live or finished
    game anywhere in the payload, and without it the board seeds nothing and the
    landing page renders empty.

    Args:
        data: Pre-fetched score endpoint JSON payload.
        reverse_dates: If True, reverses the gamesByDate list before scanning so
            the most recent date is searched first (used for multi-day endpoints).

    Returns:
        A (home_abbr, away_abbr) string tuple, or None.
    """
    games_by_date = data.get("gamesByDate", [])
    if not games_by_date:
        return None

    if reverse_dates:
        games_by_date = list(reversed(games_by_date))

    all_games: list[dict] = []
    for day in games_by_date:
        all_games.extend(day.get("games", []))

    valid = [g for g in all_games if g.get("gameType") in _VALID_GAME_TYPES]

    # Sort games by start time (most recent first) to ensure we pick the latest game
    # even if multiple games are live or finished on the same day
    def _get_start_time(game: dict) -> datetime:
        """Return one game's parsed UTC start time for stable descending sorting."""
        start_time_utc = game.get("startTimeUTC")
        if start_time_utc:
            try:
                return datetime.fromisoformat(start_time_utc.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        return datetime.min.replace(tzinfo=timezone.utc)

    valid.sort(key=_get_start_time, reverse=True)

    # Priority: live first, finished second
    for state_set in (_LIVE_STATES, _FINAL_STATES):
        for game in valid:
            if game.get("gameState") in state_set:
                home = game.get("homeTeam", {}).get("abbrev", "")
                away = game.get("awayTeam", {}).get("abbrev", "")
                if home and away:
                    return (home, away)

    # Nothing live or finished — fall back to the soonest upcoming game, preseason
    # included. Re-filtered and re-sorted ascending: `valid` above is regular/playoff
    # only and ordered most-recent-first, which is the wrong end for future games.
    upcoming = [
        g for g in all_games
        if g.get("gameType") in _UPCOMING_GAME_TYPES
        and g.get("gameState") in _FUTURE_STATES
    ]
    upcoming.sort(key=_get_start_time)
    for game in upcoming:
        home = game.get("homeTeam", {}).get("abbrev", "")
        away = game.get("awayTeam", {}).get("abbrev", "")
        if home and away:
            return (home, away)

    return None


def _extract_game_details_from_payload(payload: dict, game_id: int, fallback_date: str = '') -> dict:
    """Normalize one exact game from an NHL score payload.

    Args:
        payload: Raw score endpoint JSON payload.
        game_id: Exact NHL game identifier to match.
        fallback_date: Date string to keep when the payload omits it.

    Returns:
        Normalized detail dict, or ``{}`` when the game is not found.
    """
    games = payload.get('games', []) if isinstance(payload, dict) else []
    if not games and isinstance(payload, dict):
        for day in payload.get('gamesByDate', []) or []:
            games.extend(day.get('games', []))

    for game in games:
        try:
            current_game_id = int(game.get('id', 0) or 0)
        except Exception:
            continue
        if current_game_id != int(game_id):
            continue

        away_team = game.get('awayTeam', {})
        home_team = game.get('homeTeam', {})
        away_abbr = str(away_team.get('abbrev', '') or '').strip().upper()
        home_abbr = str(home_team.get('abbrev', '') or '').strip().upper()
        away_name = ACTIVE_TEAMS.get(away_abbr) or str(away_team.get('name', {}).get('default', away_abbr)).strip()
        home_name = ACTIVE_TEAMS.get(home_abbr) or str(home_team.get('name', {}).get('default', home_abbr)).strip()
        venue_name = str(game.get('venue', {}).get('default', '') or '').strip()
        start_time_utc = str(game.get('startTimeUTC', '') or '')
        game_state = str(game.get('gameState', '') or '').strip().upper()
        period_type = str(game.get('periodDescriptor', {}).get('periodType', '') or '').strip().upper()

        try:
            away_score = int(away_team.get('score')) if away_team.get('score') is not None else None
        except Exception:
            away_score = None
        try:
            home_score = int(home_team.get('score')) if home_team.get('score') is not None else None
        except Exception:
            home_score = None

        if game_state in _FINAL_STATES:
            if period_type == 'SO':
                status_label = 'Final/SO'
            elif period_type == 'OT':
                status_label = 'Final/OT'
            else:
                status_label = 'Final'
        elif game_state in _LIVE_STATES:
            status_label = 'Live'
        elif game_state == 'FUT':
            status_label = 'Scheduled'
        else:
            status_label = game_state.title() if game_state else ''

        return {
            'game_id': current_game_id,
            'game_date': str(game.get('gameDate', '') or fallback_date),
            'game_type': int(game.get('gameType', 0) or 0),
            'away_abbr': away_abbr,
            'away_name': away_name,
            'away_score': away_score,
            'home_abbr': home_abbr,
            'home_name': home_name,
            'home_score': home_score,
            'matchup': f'{away_name} at {home_name}',
            'venue': venue_name,
            'start_time_utc': start_time_utc,
            'start_label_cest': _format_game_time_cest(start_time_utc),
            'status_label': status_label,
        }

    return {}


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    """Parse an NHL API UTC timestamp string into an aware datetime.

    Args:
        value: Timestamp string such as ``2026-03-07T17:30:00Z``.

    Returns:
        A timezone-aware UTC datetime, or None if parsing fails.
    """
    if not value:
        return None

    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _format_game_time_cest(start_time_utc: str | None) -> str:
    """Format a UTC puck-drop timestamp in Central European local time.

    Args:
        start_time_utc: UTC timestamp string from the NHL score API.

    Returns:
        A deterministic display string such as ``Sat 07 Mar, 18:30 CET``.
        Returns ``Time TBD`` if the timestamp is missing or invalid.
    """
    start_dt_utc = _parse_utc_timestamp(start_time_utc)
    if start_dt_utc is None:
        return "Time TBD"

    local_dt = start_dt_utc.astimezone(_CENTRAL_EUROPE_TZ)
    weekday = _WEEKDAY_ABBR[local_dt.weekday()]
    month = _MONTH_ABBR[local_dt.month - 1]
    tz_label = local_dt.tzname() or "CET"
    return f"{weekday} {local_dt.day:02d} {month}, {local_dt:%H:%M} {tz_label}"


def _extract_upcoming_games(games: list[dict], now_utc: datetime) -> list[dict]:
    """Filter one score payload down to valid future games.

    Args:
        games: Raw ``games`` list from the NHL score endpoint.
        now_utc: Current time used to discard stale future-state rows.

    Returns:
        A list of normalized game dicts sorted by start time.
    """
    upcoming_games: list[dict] = []

    for game in games:
        if game.get("gameType") not in _UPCOMING_GAME_TYPES:
            continue
        if game.get("gameState") not in _FUTURE_STATES:
            continue

        start_dt_utc = _parse_utc_timestamp(game.get("startTimeUTC"))
        if start_dt_utc is None or start_dt_utc < now_utc:
            continue

        away_team = game.get("awayTeam", {})
        home_team = game.get("homeTeam", {})
        away_abbr = str(away_team.get("abbrev", "")).strip().upper()
        home_abbr = str(home_team.get("abbrev", "")).strip().upper()
        if not away_abbr or not home_abbr:
            continue

        away_name = ACTIVE_TEAMS.get(away_abbr) or str(away_team.get("name", {}).get("default", away_abbr)).strip()
        home_name = ACTIVE_TEAMS.get(home_abbr) or str(home_team.get("name", {}).get("default", home_abbr)).strip()
        venue_name = str(game.get("venue", {}).get("default", "")).strip()

        upcoming_games.append(
            {
                "game_id": int(game.get("id", 0) or 0),
                "game_type": int(game.get("gameType", 0) or 0),
                "away_abbr": away_abbr,
                "away_name": away_name,
                "home_abbr": home_abbr,
                "home_name": home_name,
                "matchup": f"{away_name} at {home_name}",
                "venue": venue_name,
                "start_time_utc": game.get("startTimeUTC", ""),
                "start_label_cest": _format_game_time_cest(game.get("startTimeUTC")),
                "sort_ts": start_dt_utc.timestamp(),
            }
        )

    upcoming_games.sort(key=lambda game: game["sort_ts"])
    return upcoming_games


def _select_best_skater(skaters: list[dict]) -> dict | None:
    """Pick the current-season points leader from a team skater list.

    Args:
        skaters: List of normalized skater rows from ``_fetch_club_stats()``.

    Returns:
        The selected skater dict, or None if no valid skaters exist.
    """
    if not skaters:
        return None

    return max(
        skaters,
        key=lambda player: (
            int(player.get("points", 0)),
            int(player.get("playerId", 0)),
        ),
    )


def _select_best_goalie(goalies: list[dict]) -> dict | None:
    """Pick the franchise starter from a team goalie list.

    Sorting priority: minimum 6 GP (filters tourist call-ups), then wins
    (identifies the workhorse starter), then save percentage as a tiebreaker,
    then playerId for a stable final ordering.

    Args:
        goalies: List of normalized goalie rows from ``_fetch_club_stats()``.

    Returns:
        The selected goalie dict, or None if no valid goalies exist.
    """
    if not goalies:
        return None

    return max(
        goalies,
        key=lambda goalie: (
            int(goalie.get("gamesPlayed", 0)) > 5,
            int(goalie.get("wins", 0)),
            _coerce_save_percentage(goalie.get("savePercentage", 0.0)),
            int(goalie.get("playerId", 0)),
        ),
    )


def _build_model_label(away_abbr: str, home_abbr: str, scored_probability: dict) -> str:
    """Build one short runtime label from the strongest model contribution."""
    base_home_prob = float(scored_probability.get("home_win_prob", 0.5))
    if abs(base_home_prob - 0.5) < 0.02:
        return "Model: near toss-up."

    top_feature, contribution = get_top_feature_driver(scored_probability)
    if not top_feature or abs(contribution) < 0.01:
        return "Model: modest edge from team form."

    if top_feature in ("home_back_to_back", "away_back_to_back"):
        tired_abbr = home_abbr if top_feature == "home_back_to_back" else away_abbr
        return f"Model: {tired_abbr} on the second night of a back-to-back."

    feature_label = WIN_PROB_FEATURE_LABELS.get(top_feature, top_feature.replace("_", " "))
    edge_abbr = home_abbr if contribution >= 0 else away_abbr
    return f"Model: {edge_abbr} edge from {feature_label}."


def _coerce_save_percentage(value: object) -> float:
    """Normalize save percentage values to 0-1 scale."""
    try:
        numeric_value = float(value or 0.0)
    except Exception:
        return 0.0
    if numeric_value > 1.5:
        numeric_value = numeric_value / 100.0
    return max(0.0, min(numeric_value, 1.0))


@st.cache_data(ttl=3600)
def _get_cached_club_stats(abbr: str) -> dict | None:
    """Cached wrapper around the current club-stats endpoint."""
    return _fetch_club_stats(abbr)


def _fetch_club_stats(abbr: str) -> dict | None:
    """Fetches current-season stats for all players on a team.

    Args:
        abbr: Three-letter team abbreviation (e.g. 'PIT').

    Returns:
        A dict with 'skaters' and 'goalies' lists, each containing dicts with
        keys 'playerId' (int), 'name' (str), 'points' (int, skaters only),
        'gamesPlayed' (int), 'wins' (int), and 'savePercentage' (float for
        goalies). Returns None on network or parse error.
    """
    try:
        data = get_client().get(
            url=_CLUB_STATS_URL.format(abbr),
            cache_key=f"club_stats:{abbr}",
            ttl=T2_DEFAULT_TTL,
            timeout=5,
        )
        if data is None:
            return None
    except Exception:
        return None

    skaters: list[dict] = []
    goalies:  list[dict] = []

    for raw in data.get("skaters", []):
        pid  = int(raw.get("playerId", 0))
        name = (
            f"{raw.get('firstName', {}).get('default', '')}"
            f" {raw.get('lastName', {}).get('default', '')}"
        ).strip()
        if pid and name:
            skaters.append({
                "playerId": pid,
                "name":     name,
                "points":   int(raw.get("points", 0)),
            })

    for raw in data.get("goalies", []):
        pid  = int(raw.get("playerId", 0))
        name = (
            f"{raw.get('firstName', {}).get('default', '')}"
            f" {raw.get('lastName', {}).get('default', '')}"
        ).strip()
        if pid and name:
            goalies.append({
                "playerId":    pid,
                "name":        name,
                "gamesPlayed": int(raw.get("gamesPlayed", 0)),
                "wins":        int(raw.get("wins", 0)),
                "savePercentage": _coerce_save_percentage(raw.get("savePercentage", 0.0)),
                "saves":       float(raw.get("saves", 0.0) or 0.0),
                "shotsAgainst": float(raw.get("shotsAgainst", 0.0) or 0.0),
            })

    return {"skaters": skaters, "goalies": goalies}
