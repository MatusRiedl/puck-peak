"""Score distribution behind the 60-minute (1X2) and puck-line markets.

A game's regulation score is modelled as two Poisson goal counts. Their combined rate
follows the league's current scoring level. Their split is solved so the distribution's
moneyline, including overtime and shootouts, equals the win-probability model. Every
market is then read off one grid, so the numbers cannot contradict each other.

Plain Poisson gets hockey badly wrong, so three corrections are fitted by maximum
likelihood on historical regulation scores in ``train_win_prob.py``:

- ``tie_inflation``: regulation ties are ~22% of games; independent Poisson says ~17%.
- ``lead1_transfer`` / ``lead2_transfer``: late empty-net goals turn a share of one- and
  two-goal leads into bigger wins. One-goal regulation games are ~18% of games, not the
  ~30% Poisson implies.
- ``rate_scale``: those transfers add goals, so the base rate shrinks to keep the mean.

Over/under totals are NOT published. They failed the backtest gate: no better than
the league's plain over-rate. Per-team scoring rates added nothing over the league level
for these markets, so the model does not use them.

Pure numpy/pandas, no scipy.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MAX_GOALS = 16
GOAL_MODEL_VERSION = 1
_GOALS = np.arange(MAX_GOALS)
_DIFF = _GOALS[:, None] - _GOALS[None, :]
_LOG_FACTORIAL = np.array([math.lgamma(k + 1.0) for k in range(MAX_GOALS)])
_TOTAL_WITH_DECIDER = _GOALS[:, None] + _GOALS[None, :] + (_DIFF == 0)
"""Final goal total as sportsbooks grade it: an overtime goal or shootout win adds one."""

DEFAULT_GOAL_MODEL: dict[str, float] = {
    "rate_scale": 0.973,
    "tie_inflation": 0.40,
    "lead1_transfer": 0.32,
    "lead2_transfer": 0.49,
    "overtime_intercept": -0.1,
    "overtime_logit_coef": 0.43,
    "environment_prior_team_games": 400.0,
    "fallback_goals_per_team_game": 3.0,
}
"""Fallback shape. The trained artifact carries fitted values."""


def validate_goal_model(payload: object) -> dict | None:
    """Return a normalized goal-model block, or ``None`` when it is absent or malformed.

    Args:
        payload: ``goal_model`` object from the artifact.

    Returns:
        Complete parameter dict with ``version``, or ``None``.
    """
    if not isinstance(payload, dict):
        return None
    resolved: dict[str, float] = dict(DEFAULT_GOAL_MODEL)
    for key in DEFAULT_GOAL_MODEL:
        if key not in payload:
            continue
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        resolved[key] = value
    if resolved["rate_scale"] <= 0 or resolved["tie_inflation"] < 0:
        return None
    if not (0.0 <= resolved["lead1_transfer"] < 1.0 and 0.0 <= resolved["lead2_transfer"] < 1.0):
        return None
    resolved["version"] = int(payload.get("version", GOAL_MODEL_VERSION) or GOAL_MODEL_VERSION)
    return resolved


def regulation_goals(games: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return each side's regulation goals (the stats API includes the overtime goal, never the shootout).

    Args:
        games: Game table with ``HomeGoals``, ``AwayGoals``, ``ResultType`` and ``HomeWin``.

    Returns:
        ``(home_regulation_goals, away_regulation_goals)`` aligned with ``games``.
    """
    overtime = games["ResultType"].eq("OT")
    home_won = games["HomeWin"].eq(1)
    return games["HomeGoals"] - (overtime & home_won), games["AwayGoals"] - (overtime & ~home_won)


def scoring_environment(games: pd.DataFrame, prior_team_games: float = 400.0) -> pd.Series:
    """Return the pregame league scoring level (regulation goals per team-game) for every game.

    The season-to-date average before each game date is blended with the previous
    season's average, worth ``prior_team_games``. Playoff games use the end of that
    regular season.

    Args:
        games: Game table (any seasons).
        prior_team_games: Weight of last season's level, in team-games.

    Returns:
        Float series aligned with ``games``; NaN for seasons with no regular-season data.
    """
    regular = games[games["GameTypeId"].eq(2)]
    if regular.empty:
        return pd.Series(np.nan, index=games.index, dtype=float)
    home_goals, away_goals = regulation_goals(regular)
    per_game = pd.DataFrame({"SeasonYear": regular["SeasonYear"], "GameDate": regular["GameDate"], "goals": home_goals + away_goals})
    season_level = per_game.groupby("SeasonYear")["goals"].mean() / 2.0
    fallback = float(season_level.mean())

    by_date = per_game.groupby(["SeasonYear", "GameDate"]).agg(goals=("goals", "sum"), games=("goals", "size")).reset_index()
    by_date = by_date.sort_values(["SeasonYear", "GameDate"], kind="stable")
    by_date["goals_before"] = by_date.groupby("SeasonYear")["goals"].cumsum() - by_date["goals"]
    by_date["games_before"] = by_date.groupby("SeasonYear")["games"].cumsum() - by_date["games"]
    prior = by_date["SeasonYear"].map(lambda season: season_level.get(season - 1, fallback))
    by_date["environment"] = (prior_team_games * prior + by_date["goals_before"]) / (prior_team_games + 2.0 * by_date["games_before"])

    end_prior = pd.Series({season: season_level.get(season - 1, fallback) for season in season_level.index})
    season_totals = per_game.groupby("SeasonYear").agg(goals=("goals", "sum"), games=("goals", "size"))
    season_end = (prior_team_games * end_prior + season_totals["goals"]) / (prior_team_games + 2.0 * season_totals["games"])

    lookup = by_date.set_index(["SeasonYear", "GameDate"])["environment"]
    keys = list(zip(games["SeasonYear"], games["GameDate"]))
    values = [lookup.get(key, np.nan) for key in keys]
    environment = pd.Series(values, index=games.index, dtype=float)
    playoff = games["GameTypeId"].ne(2)
    environment[playoff] = games.loc[playoff, "SeasonYear"].map(season_end)
    return environment


def current_scoring_environment(games: pd.DataFrame, target_season: int, prior_team_games: float = 400.0) -> float | None:
    """Return the league scoring level to price games in ``target_season`` right now.

    Args:
        games: Game table covering ``target_season`` and the season before it.
        target_season: Season start year.
        prior_team_games: Weight of last season's level, in team-games.

    Returns:
        Regulation goals per team-game, or ``None`` without enough data.
    """
    if games is None or games.empty:
        return None
    regular = games[games["GameTypeId"].eq(2) & games["SeasonYear"].between(int(target_season) - 1, int(target_season))]
    if regular.empty:
        return None
    home_goals, away_goals = regulation_goals(regular)
    goals = home_goals + away_goals
    previous = goals[regular["SeasonYear"].eq(int(target_season) - 1)]
    current = goals[regular["SeasonYear"].eq(int(target_season))]
    if previous.empty and current.empty:
        return None
    prior = float(previous.mean()) / 2.0 if not previous.empty else float(current.mean()) / 2.0
    return float((prior_team_games * prior + current.sum()) / (prior_team_games + 2.0 * len(current)))


def _poisson_pmf(rates: np.ndarray) -> np.ndarray:
    """Poisson probabilities for 0..MAX_GOALS-1 goals, one row per rate."""
    safe = np.maximum(rates, 1e-9)[:, None]
    return np.exp(_GOALS[None, :] * np.log(safe) - safe - _LOG_FACTORIAL[None, :])


def score_grid(home_rate: float | np.ndarray, away_rate: float | np.ndarray, model: dict) -> np.ndarray:
    """Return regulation score probabilities, ``grid[g, home_goals, away_goals]``.

    Args:
        home_rate: Expected home regulation goals before the corrections (scalar or array).
        away_rate: Same for the away side.
        model: ``validate_goal_model`` output.

    Returns:
        Array of shape ``(games, MAX_GOALS, MAX_GOALS)`` whose slices each sum to 1.
    """
    home = np.atleast_1d(np.asarray(home_rate, dtype=float)) * model["rate_scale"]
    away = np.atleast_1d(np.asarray(away_rate, dtype=float)) * model["rate_scale"]
    home, away = np.broadcast_arrays(home, away)
    grid = _poisson_pmf(home)[:, :, None] * _poisson_pmf(away)[:, None, :]
    grid = grid * (1.0 + model["tie_inflation"] * (_DIFF == 0))[None]
    grid = grid / grid.sum(axis=(1, 2), keepdims=True)
    # Two-goal leads first, so a one-goal lead that becomes two is not moved again.
    for margin, share in ((2, model["lead2_transfer"]), (1, model["lead1_transfer"])):
        moved = grid * share * (_DIFF == margin)[None]
        grid = grid - moved
        grid[:, 1:, :] += moved[:, :-1, :]
        moved = grid * share * (_DIFF == -margin)[None]
        grid = grid - moved
        grid[:, :, 1:] += moved[:, :, :-1]
    return grid / grid.sum(axis=(1, 2), keepdims=True)


def overtime_home_probability(p_home: float | np.ndarray, model: dict) -> np.ndarray:
    """Chance the home team wins a game that is tied after 60 minutes."""
    p = np.clip(np.atleast_1d(np.asarray(p_home, dtype=float)), 1e-6, 1 - 1e-6)
    linear = model["overtime_intercept"] + model["overtime_logit_coef"] * np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-linear))


def market_probabilities(grid: np.ndarray) -> dict[str, np.ndarray]:
    """Read every market off a score grid.

    Args:
        grid: ``score_grid`` output.

    Returns:
        ``home_regulation``, ``draw``, ``away_regulation``, ``home_minus_1_5``,
        ``away_minus_1_5`` and ``expected_total`` (graded total, one array per key).
    """
    return {
        "home_regulation": (grid * (_DIFF > 0)).sum(axis=(1, 2)),
        "draw": (grid * (_DIFF == 0)).sum(axis=(1, 2)),
        "away_regulation": (grid * (_DIFF < 0)).sum(axis=(1, 2)),
        "home_minus_1_5": (grid * (_DIFF >= 2)).sum(axis=(1, 2)),
        "away_minus_1_5": (grid * (_DIFF <= -2)).sum(axis=(1, 2)),
        "expected_total": (grid * _TOTAL_WITH_DECIDER).sum(axis=(1, 2)),
    }


def solve_home_share(p_home: float | np.ndarray, total_rate: float | np.ndarray, model: dict, iterations: int = 40) -> np.ndarray:
    """Find the home share of the total rate whose moneyline equals ``p_home``.

    Args:
        p_home: Win-model home probability, overtime and shootout included.
        total_rate: Expected regulation goals for both teams before the corrections.
        model: ``validate_goal_model`` output.
        iterations: Bisection steps (40 gives about 1e-12 precision on the share).

    Returns:
        Home share of the total rate, one per game.
    """
    target = np.atleast_1d(np.asarray(p_home, dtype=float))
    total = np.broadcast_to(np.atleast_1d(np.asarray(total_rate, dtype=float)), target.shape)
    tie_break = overtime_home_probability(target, model)
    low = np.full(target.shape, 0.02)
    high = np.full(target.shape, 0.98)
    for _ in range(iterations):
        middle = (low + high) / 2.0
        markets = market_probabilities(score_grid(total * middle, total * (1.0 - middle), model))
        implied = markets["home_regulation"] + markets["draw"] * tie_break
        low = np.where(implied < target, middle, low)
        high = np.where(implied >= target, middle, high)
    return (low + high) / 2.0


def price_games(p_home: float | np.ndarray, goals_per_team_game: float | np.ndarray, model: dict) -> dict[str, np.ndarray]:
    """Price the 60-minute and puck-line markets for one or many games.

    Args:
        p_home: Win-model home probabilities.
        goals_per_team_game: League scoring level per game (see ``scoring_environment``).
        model: ``validate_goal_model`` output.

    Returns:
        ``market_probabilities`` keys plus ``home_share``.
    """
    target = np.atleast_1d(np.asarray(p_home, dtype=float))
    total = 2.0 * np.broadcast_to(np.atleast_1d(np.asarray(goals_per_team_game, dtype=float)), target.shape)
    share = solve_home_share(target, total, model)
    markets = market_probabilities(score_grid(total * share, total * (1.0 - share), model))
    markets["home_share"] = share
    return markets


def game_markets(p_home: float, goals_per_team_game: float | None, model: dict | None) -> dict | None:
    """Price one game for the prediction card.

    Args:
        p_home: Win-model home probability.
        goals_per_team_game: Current league scoring level; ``None`` falls back to the model's default.
        model: Validated goal model, or ``None`` when the artifact has none.

    Returns:
        ``regulation`` (home/draw/away), ``puck_line`` (each side's -1.5 chance) and the
        graded ``expected_total`` (internal only; totals are not published). ``None``
        without a goal model.
    """
    if not model:
        return None
    level = goals_per_team_game if goals_per_team_game and goals_per_team_game > 0 else model["fallback_goals_per_team_game"]
    priced = price_games(float(p_home), float(level), model)
    return {
        "regulation": {
            "home": float(priced["home_regulation"][0]),
            "draw": float(priced["draw"][0]),
            "away": float(priced["away_regulation"][0]),
        },
        "puck_line": {
            "home_minus_1_5": float(priced["home_minus_1_5"][0]),
            "away_minus_1_5": float(priced["away_minus_1_5"][0]),
        },
        "expected_total": float(priced["expected_total"][0]),
    }
