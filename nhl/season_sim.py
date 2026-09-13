"""Monte Carlo season and playoff simulator behind the Stanley Cup odds board.

The simulator plays out the rest of the regular season game by game with the trained
win-probability model, seeds the playoffs with the NHL division/wild-card format and
resolves every series with an exact best-of-7 calculation. Each simulation also draws a
random strength offset per team. Without that noise the model treats its own ratings as
exact, and the favourite's Cup odds come out far too high.

Pure numpy/pandas. The Streamlit wiring lives in ``nhl.schedule.get_season_projection``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nhl.team_ratings import (
    REGULAR_SEASON,
    back_to_back_flags,
    canonical_team_abbrev,
)
from nhl.win_prob import decompose_linear_model, overtime_probability

WINS_TO_TAKE_SERIES = 4
HIGH_SEED_HOME_GAMES = (True, True, False, False, True, False, True)
"""2-2-1-1-1 format: the team with home ice hosts games 1, 2, 5 and 7."""

ROUND_ONE_LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H")
LATER_ROUND_FEEDERS = {
    "I": ("A", "B"),
    "J": ("C", "D"),
    "K": ("E", "F"),
    "L": ("G", "H"),
    "M": ("I", "J"),
    "N": ("K", "L"),
    "O": ("M", "N"),
}
"""NHL bracket letters. A-D are the first conference alphabetically, A/B and C/D its divisions."""

SERIES_WIN_KEY = {
    **{letter: "win_round_1" for letter in ROUND_ONE_LETTERS},
    "I": "win_round_2",
    "J": "win_round_2",
    "K": "win_round_2",
    "L": "win_round_2",
    "M": "win_conference",
    "N": "win_conference",
    "O": "win_cup",
}
PROJECTION_KEYS = ("make_playoffs", "win_division", "win_round_1", "win_round_2", "win_conference", "win_cup")
FINAL_GAME_STATE_IDS = frozenset({6, 7})
"""Stats-API ``gameStateId`` values that mean the game is over."""


def series_win_probability(
    p_home: float | np.ndarray,
    p_away: float | np.ndarray,
    wins_high: int = 0,
    wins_low: int = 0,
) -> np.ndarray:
    """Return the exact chance the team with home ice wins a best-of-7 series.

    Args:
        p_home: Chance the home-ice team wins a game it hosts (scalar or array).
        p_away: Chance the home-ice team wins a road game (same shape).
        wins_high: Games the home-ice team has already won.
        wins_low: Games the other team has already won.

    Returns:
        Series win probability with the broadcast shape of the inputs.
    """
    home = np.asarray(p_home, dtype=float)
    away = np.asarray(p_away, dtype=float)
    home, away = np.broadcast_arrays(home, away)
    high = max(int(wins_high), 0)
    low = max(int(wins_low), 0)
    if high >= WINS_TO_TAKE_SERIES:
        return np.ones_like(home)
    if low >= WINS_TO_TAKE_SERIES:
        return np.zeros_like(home)

    memo: dict[tuple[int, int], np.ndarray] = {}

    def _solve(high_wins: int, low_wins: int) -> np.ndarray:
        """Return the win probability from one series state."""
        if high_wins >= WINS_TO_TAKE_SERIES:
            return np.ones_like(home)
        if low_wins >= WINS_TO_TAKE_SERIES:
            return np.zeros_like(home)
        state = (high_wins, low_wins)
        if state not in memo:
            game_probability = home if HIGH_SEED_HOME_GAMES[high_wins + low_wins] else away
            memo[state] = (
                game_probability * _solve(high_wins + 1, low_wins)
                + (1.0 - game_probability) * _solve(high_wins, low_wins + 1)
            )
        return memo[state]

    return _solve(high, low)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Vectorized logistic function."""
    return 1.0 / (1.0 + np.exp(-values))


def _conference_layout(teams: list[dict], index: dict[str, int]) -> list[tuple[str, list[tuple[str, np.ndarray]]]]:
    """Group team indices by conference and division, both in alphabetical order.

    Alphabetical order matches the NHL bracket letters (Atlantic before Metropolitan,
    Central before Pacific, Eastern before Western).

    Raises:
        ValueError: The league does not have two conferences of two divisions.
    """
    grouped: dict[str, dict[str, list[int]]] = {}
    for team in teams:
        grouped.setdefault(str(team["conference"]), {}).setdefault(str(team["division"]), []).append(index[team["team_abbr"]])

    layout: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
    for conference in sorted(grouped):
        divisions = grouped[conference]
        if len(divisions) != 2 or any(len(members) < 3 for members in divisions.values()):
            raise ValueError(f"Conference {conference!r} does not fit the division/wild-card format.")
        if sum(len(members) for members in divisions.values()) < 8:
            raise ValueError(f"Conference {conference!r} has too few teams for an 8-team bracket.")
        layout.append(
            (conference, [(division, np.array(sorted(divisions[division]), dtype=int)) for division in sorted(divisions)])
        )
    if len(layout) != 2:
        raise ValueError("The simulator expects exactly two conferences.")
    return layout


def _division_seeds(order_key: np.ndarray, division_idx: np.ndarray) -> np.ndarray:
    """Return the top three team indices of one division per simulation, best first."""
    order = np.argsort(-order_key[:, division_idx], axis=1, kind="stable")[:, :3]
    return division_idx[order]


def _wild_cards(order_key: np.ndarray, conference_idx: np.ndarray, qualified: np.ndarray, team_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the two wild-card team indices per simulation, first wild card first."""
    position = np.full(team_count, -1, dtype=int)
    position[conference_idx] = np.arange(len(conference_idx))
    conference_key = order_key[:, conference_idx].copy()
    np.put_along_axis(conference_key, position[qualified], -np.inf, axis=1)
    order = np.argsort(-conference_key, axis=1, kind="stable")[:, :2]
    cards = conference_idx[order]
    return cards[:, 0], cards[:, 1]


def _play_series(
    team_x: np.ndarray,
    team_y: np.ndarray,
    order_key: np.ndarray,
    strength: np.ndarray,
    constant: float,
    rng: np.random.Generator,
    wins_x: int = 0,
    wins_y: int = 0,
    x_has_home_ice: bool | None = None,
) -> np.ndarray:
    """Resolve one playoff series in every simulation and return the winner's index."""
    rows = np.arange(len(team_x))
    if x_has_home_ice is None:
        x_home = order_key[rows, team_x] >= order_key[rows, team_y]
    else:
        x_home = np.full(len(team_x), bool(x_has_home_ice))
    high = np.where(x_home, team_x, team_y)
    low = np.where(x_home, team_y, team_x)
    strength_high = strength[rows, high]
    strength_low = strength[rows, low]
    p_high_home = _sigmoid(constant + strength_high - strength_low)
    p_high_away = 1.0 - _sigmoid(constant + strength_low - strength_high)
    if x_has_home_ice is None:
        series_probability = series_win_probability(p_high_home, p_high_away)
    else:
        high_wins, low_wins = (wins_x, wins_y) if x_has_home_ice else (wins_y, wins_x)
        series_probability = series_win_probability(p_high_home, p_high_away, high_wins, low_wins)
    high_takes_it = rng.random(len(team_x)) < series_probability
    return np.where(high_takes_it, high, low)


def _count(counts: dict[str, np.ndarray], key: str, team_idx: np.ndarray) -> None:
    """Add one to ``counts[key]`` for every team index in ``team_idx``."""
    counts[key] += np.bincount(np.asarray(team_idx, dtype=int).ravel(), minlength=len(counts[key]))


def _simulate_playoffs_from_standings(
    order_key: np.ndarray,
    strength: np.ndarray,
    constant: float,
    layout: list[tuple[str, list[tuple[str, np.ndarray]]]],
    rng: np.random.Generator,
    counts: dict[str, np.ndarray],
) -> None:
    """Seed the bracket from simulated final standings and play every round."""
    team_count = order_key.shape[1]
    rows = np.arange(order_key.shape[0])
    matchups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for conference_number, (_, divisions) in enumerate(layout):
        (_, first_idx), (_, second_idx) = divisions
        first_seeds = _division_seeds(order_key, first_idx)
        second_seeds = _division_seeds(order_key, second_idx)
        qualified = np.concatenate([first_seeds, second_seeds], axis=1)
        first_card, second_card = _wild_cards(order_key, np.concatenate([first_idx, second_idx]), qualified, team_count)
        _count(counts, "make_playoffs", qualified)
        _count(counts, "make_playoffs", first_card)
        _count(counts, "make_playoffs", second_card)
        _count(counts, "win_division", first_seeds[:, 0])
        _count(counts, "win_division", second_seeds[:, 0])

        # The division winner with the better record draws the second wild card.
        first_is_better = order_key[rows, first_seeds[:, 0]] > order_key[rows, second_seeds[:, 0]]
        letters = ROUND_ONE_LETTERS[conference_number * 4: conference_number * 4 + 4]
        matchups[letters[0]] = (first_seeds[:, 0], np.where(first_is_better, second_card, first_card))
        matchups[letters[1]] = (first_seeds[:, 1], first_seeds[:, 2])
        matchups[letters[2]] = (second_seeds[:, 0], np.where(first_is_better, first_card, second_card))
        matchups[letters[3]] = (second_seeds[:, 1], second_seeds[:, 2])

    winners: dict[str, np.ndarray] = {}
    for letter in ROUND_ONE_LETTERS:
        team_x, team_y = matchups[letter]
        winners[letter] = _play_series(team_x, team_y, order_key, strength, constant, rng)
        _count(counts, SERIES_WIN_KEY[letter], winners[letter])
    for letter, (feeder_x, feeder_y) in LATER_ROUND_FEEDERS.items():
        winners[letter] = _play_series(winners[feeder_x], winners[feeder_y], order_key, strength, constant, rng)
        _count(counts, SERIES_WIN_KEY[letter], winners[letter])


def _simulate_playoffs_from_bracket(
    bracket: dict,
    order_key: np.ndarray,
    strength: np.ndarray,
    constant: float,
    index: dict[str, int],
    rng: np.random.Generator,
    counts: dict[str, np.ndarray],
) -> None:
    """Play the remaining series of a live bracket, keeping decided series fixed."""
    batch = order_key.shape[0]
    series = bracket.get("series", {})
    winners: dict[str, np.ndarray] = {}
    for letter in ROUND_ONE_LETTERS:
        info = series[letter]
        for team in (info["top"], info["bottom"]):
            _count(counts, "make_playoffs", np.full(batch, index[team]))
        if letter in ("A", "C", "E", "G"):
            _count(counts, "win_division", np.full(batch, index[info["top"]]))

    for letter in ROUND_ONE_LETTERS + tuple(LATER_ROUND_FEEDERS):
        info = series.get(letter) or {}
        winner = info.get("winner", "")
        top = info.get("top", "")
        bottom = info.get("bottom", "")
        if winner in index:
            winners[letter] = np.full(batch, index[winner])
        elif top in index and bottom in index:
            winners[letter] = _play_series(
                np.full(batch, index[top]),
                np.full(batch, index[bottom]),
                order_key,
                strength,
                constant,
                rng,
                wins_x=int(info.get("top_wins", 0) or 0),
                wins_y=int(info.get("bottom_wins", 0) or 0),
                x_has_home_ice=True,
            )
        else:
            feeder_x, feeder_y = LATER_ROUND_FEEDERS[letter]
            winners[letter] = _play_series(winners[feeder_x], winners[feeder_y], order_key, strength, constant, rng)
        _count(counts, SERIES_WIN_KEY[letter], winners[letter])


def simulate_season(
    teams: list[dict],
    team_inputs: dict[str, dict[str, float]],
    remaining_games: pd.DataFrame | None,
    artifact: dict,
    bracket: dict | None = None,
    season_progress: float = 0.0,
    n_sims: int = 10_000,
    seed: int = 0,
    batch_size: int = 1_000,
) -> dict:
    """Simulate the rest of the season and the playoffs.

    Args:
        teams: One dict per team with ``team_abbr``, ``conference``, ``division``,
            ``points``, ``regulation_wins``, ``regulation_plus_ot_wins`` and ``wins``
            (the record so far; zeros before opening night).
        team_inputs: ``current_team_snapshot`` output keyed by team abbreviation.
        remaining_games: Unplayed regular-season games with ``HomeTeam``, ``AwayTeam``,
            ``HomeBackToBack`` and ``AwayBackToBack``. Empty once the season is over.
        artifact: Validated win-probability artifact.
        bracket: ``parse_playoff_bracket`` output. Used only once every first-round
            series has both teams, which means the regular season is final.
        season_progress: Share of the regular season already played, 0 to 1. It sets how
            much strength noise each simulation draws.
        n_sims: Number of simulated seasons.
        seed: Random seed. The same inputs and seed give identical output.
        batch_size: Simulations per vectorized batch, which bounds memory use.

    Returns:
        ``n_sims``, ``strength_sd`` and ``teams``. Each team entry holds the projected
        points (mean, p10, p90) and the probability of each playoff milestone in
        ``PROJECTION_KEYS``.

    Raises:
        ValueError: The team list does not fit the division/wild-card format.
    """
    abbrs = [team["team_abbr"] for team in teams]
    index = {abbr: position for position, abbr in enumerate(abbrs)}
    team_count = len(abbrs)
    layout = _conference_layout(teams, index)

    constant, attribute_weights, flag_weights = decompose_linear_model(artifact)
    attribute_means = {
        attribute: float(np.mean([float(entry[attribute]) for entry in team_inputs.values() if attribute in entry]))
        for attribute in attribute_weights
        if any(attribute in entry for entry in team_inputs.values())
    }
    base_strength = np.zeros(team_count, dtype=float)
    for position, abbr in enumerate(abbrs):
        entry = team_inputs.get(abbr, {})
        for attribute, weight in attribute_weights.items():
            base_strength[position] += weight * float(entry.get(attribute, attribute_means.get(attribute, 0.0)))

    progress = min(max(float(season_progress), 0.0), 1.0)
    simulation = artifact["simulation"]
    strength_sd = simulation["strength_sd_preseason"] * (1.0 - progress) + simulation["strength_sd_late"] * progress

    record = {
        field: np.array([float(team.get(field, 0) or 0) for team in teams], dtype=np.float64)
        for field in ("points", "regulation_wins", "regulation_plus_ot_wins", "wins")
    }

    games = remaining_games if remaining_games is not None else pd.DataFrame()
    if not games.empty:
        games = games[games["HomeTeam"].isin(index) & games["AwayTeam"].isin(index)]
    game_count = len(games)
    if game_count:
        home_idx = games["HomeTeam"].map(index).to_numpy(dtype=int)
        away_idx = games["AwayTeam"].map(index).to_numpy(dtype=int)
        flag_term = (
            flag_weights.get("home_back_to_back", 0.0) * games["HomeBackToBack"].fillna(0).to_numpy(dtype=float)
            + flag_weights.get("away_back_to_back", 0.0) * games["AwayBackToBack"].fillna(0).to_numpy(dtype=float)
        )
        base_logit = constant + base_strength[home_idx] - base_strength[away_idx] + flag_term
        p_overtime = overtime_probability(base_logit, artifact).astype(np.float32)
        shootout_share = float(artifact["overtime_model"]["shootout_share"])
        home_incidence = np.zeros((game_count, team_count), dtype=np.float32)
        away_incidence = np.zeros((game_count, team_count), dtype=np.float32)
        home_incidence[np.arange(game_count), home_idx] = 1.0
        away_incidence[np.arange(game_count), away_idx] = 1.0

    use_bracket = bool(bracket) and bool(bracket.get("round_one_complete")) and all(
        (bracket["series"].get(letter) or {}).get(side) in index
        for letter in ROUND_ONE_LETTERS
        for side in ("top", "bottom")
    )

    rng = np.random.default_rng(int(seed))
    counts = {key: np.zeros(team_count, dtype=np.int64) for key in PROJECTION_KEYS}
    final_points = np.zeros((n_sims, team_count), dtype=np.float32)
    for start in range(0, n_sims, batch_size):
        batch = min(batch_size, n_sims - start)
        offsets = rng.normal(0.0, strength_sd, size=(batch, team_count)) if strength_sd > 0 else np.zeros((batch, team_count))
        strength = base_strength[None, :] + offsets
        points = np.tile(record["points"], (batch, 1))
        regulation_wins = np.tile(record["regulation_wins"], (batch, 1))
        row_wins = np.tile(record["regulation_plus_ot_wins"], (batch, 1))
        wins = np.tile(record["wins"], (batch, 1))

        if game_count:
            logits = base_logit[None, :] + offsets[:, home_idx] - offsets[:, away_idx]
            home_win = rng.random((batch, game_count), dtype=np.float32) < _sigmoid(logits)
            overtime = rng.random((batch, game_count), dtype=np.float32) < p_overtime[None, :]
            shootout = overtime & (rng.random((batch, game_count), dtype=np.float32) < shootout_share)
            away_win = ~home_win
            home_stats = np.stack([
                2 * home_win + (away_win & overtime),
                home_win & ~overtime,
                home_win & ~shootout,
                home_win,
            ]).astype(np.float32)
            away_stats = np.stack([
                2 * away_win + (home_win & overtime),
                away_win & ~overtime,
                away_win & ~shootout,
                away_win,
            ]).astype(np.float32)
            team_stats = home_stats @ home_incidence + away_stats @ away_incidence
            points += team_stats[0]
            regulation_wins += team_stats[1]
            row_wins += team_stats[2]
            wins += team_stats[3]

        final_points[start:start + batch] = points
        # NHL tiebreakers after points: regulation wins, then regulation + OT wins,
        # then total wins. Head-to-head and goal differential are left to chance.
        order_key = (
            points * 1e6 + regulation_wins * 1e4 + row_wins * 1e2 + wins
            + rng.random((batch, team_count)) * 0.5
        )
        if use_bracket:
            _simulate_playoffs_from_bracket(bracket, order_key, strength, constant, index, rng, counts)
        else:
            _simulate_playoffs_from_standings(order_key, strength, constant, layout, rng, counts)

    team_results: dict[str, dict[str, float]] = {}
    for position, abbr in enumerate(abbrs):
        column = final_points[:, position]
        team_results[abbr] = {
            "projected_points": float(column.mean()),
            "points_p10": float(np.percentile(column, 10)),
            "points_p90": float(np.percentile(column, 90)),
            **{key: float(counts[key][position]) / n_sims for key in PROJECTION_KEYS},
        }
    return {"n_sims": int(n_sims), "strength_sd": float(strength_sd), "teams": team_results}


# ---------------------------------------------------------------------------
# Input adapters: NHL payloads -> simulator inputs
# ---------------------------------------------------------------------------

def teams_from_standings(standings_df: pd.DataFrame, include_record: bool = True) -> list[dict]:
    """Convert the normalized standings frame into simulator team dicts.

    Args:
        standings_df: Output of ``data_loaders.get_current_nhl_standings``.
        include_record: ``False`` before opening night, when the table still holds last
            season's final record and only division membership should carry over.

    Returns:
        One team dict per standings row.
    """
    if standings_df is None or standings_df.empty:
        return []

    teams: list[dict] = []
    for _, row in standings_df.iterrows():
        abbr = canonical_team_abbrev(row.get("teamAbbrev"))
        if not abbr:
            continue

        def _record(column: str) -> float:
            """Return one record field, or zero when the record is excluded."""
            if not include_record:
                return 0.0
            value = pd.to_numeric(row.get(column), errors="coerce")
            return 0.0 if pd.isna(value) else float(value)

        teams.append(
            {
                "team_abbr": abbr,
                "conference": str(row.get("conferenceName") or "").strip(),
                "division": str(row.get("divisionName") or "").strip(),
                "points": _record("points"),
                "regulation_wins": _record("regulationWins"),
                "regulation_plus_ot_wins": _record("regulationPlusOtWins"),
                "wins": _record("wins"),
                "games_played": _record("gamesPlayed"),
            }
        )
    return teams


def parse_playoff_bracket(payload: dict | None) -> dict:
    """Normalize ``/v1/playoff-bracket/{year}`` into series keyed by bracket letter.

    Args:
        payload: Raw bracket payload, or ``None``.

    Returns:
        ``series`` (letter -> top/bottom abbreviations, wins, winner and round),
        ``round_one_complete`` and ``champion`` (empty until the Final is decided).
    """
    parsed = {"series": {}, "round_one_complete": False, "champion": ""}
    if not isinstance(payload, dict):
        return parsed

    for raw in payload.get("series", []) or []:
        if not isinstance(raw, dict):
            continue
        letter = str(raw.get("seriesLetter") or "").strip().upper()
        if not letter:
            continue
        top_team = raw.get("topSeedTeam") or {}
        bottom_team = raw.get("bottomSeedTeam") or {}
        top_abbr = canonical_team_abbrev(top_team.get("abbrev"))
        bottom_abbr = canonical_team_abbrev(bottom_team.get("abbrev"))
        winner_id = raw.get("winningTeamId")
        winner = ""
        if winner_id and winner_id == top_team.get("id"):
            winner = top_abbr
        elif winner_id and winner_id == bottom_team.get("id"):
            winner = bottom_abbr
        parsed["series"][letter] = {
            "round": int(raw.get("playoffRound", 0) or 0),
            "top": top_abbr,
            "bottom": bottom_abbr,
            "top_wins": int(raw.get("topSeedWins", 0) or 0),
            "bottom_wins": int(raw.get("bottomSeedWins", 0) or 0),
            "winner": winner,
        }

    series = parsed["series"]
    parsed["round_one_complete"] = all(
        series.get(letter, {}).get("top") and series.get(letter, {}).get("bottom")
        for letter in ROUND_ONE_LETTERS
    )
    parsed["champion"] = (series.get("O") or {}).get("winner", "")
    return parsed


def remaining_regular_season_games(schedule: pd.DataFrame, completed_game_ids: set[int] | None = None) -> pd.DataFrame:
    """Return unplayed regular-season games with back-to-back flags from the full schedule.

    Args:
        schedule: ``team_ratings.build_league_schedule`` output for one season.
        completed_game_ids: Ids already present in the completed-game table. A game
            counts as played if it is in this set or its state is final.

    Returns:
        ``GameId``, ``GameDate``, ``HomeTeam``, ``AwayTeam``, ``HomeBackToBack`` and
        ``AwayBackToBack`` for every regular-season game still to be played.
    """
    columns = ["GameId", "GameDate", "HomeTeam", "AwayTeam", "HomeBackToBack", "AwayBackToBack"]
    if schedule is None or schedule.empty:
        return pd.DataFrame({column: [] for column in columns})

    flags = back_to_back_flags(schedule)
    completed = set(int(game_id) for game_id in (completed_game_ids or set()))
    remaining = schedule[
        schedule["GameTypeId"].eq(REGULAR_SEASON)
        & ~schedule["GameId"].isin(completed)
        & ~schedule["GameStateId"].isin(FINAL_GAME_STATE_IDS)
    ]
    remaining = remaining.merge(flags, on="GameId", how="left")
    for column in ("HomeBackToBack", "AwayBackToBack"):
        remaining[column] = remaining[column].fillna(0).astype(int)
    return remaining[columns].reset_index(drop=True)
