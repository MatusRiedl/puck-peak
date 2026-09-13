"""Prediction ledger: every published pregame prediction, frozen at puck drop and graded after.

A paid prediction product needs a record nobody can edit in hindsight. The cache
warmer writes each card's numbers into this ledger, and keeps updating them only until the
game starts. It records the betting market's prices for the same game next to them, from
the NHL API's free partner-odds feed. Once results are in, every row is graded. The track
record shown in the app is computed only from these rows. The backtest shown next to it
is labelled as such.

Storage is a single SQLite file in ``PUCKPEAK_DATA_DIR`` (default ``.data/`` in the repo).
Production must mount that directory as a Docker volume, or the record resets on every
deploy.

Market prices are stored for measurement only. The app never shows bookmaker names, odds
or links: that would be gambling advertising.
"""

from __future__ import annotations

import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nhl.goal_model import regulation_goals
from nhl.team_ratings import canonical_team_abbrev, season_year_from_game_id

DATA_DIR_ENV = "PUCKPEAK_DATA_DIR"
LEDGER_FILENAME = "prediction_ledger.sqlite3"
MIN_GRADED_GAMES_FOR_RECORD = 20
"""Below this many graded games the app says the live record is still building."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    game_id INTEGER PRIMARY KEY,
    season_year INTEGER NOT NULL,
    game_type INTEGER NOT NULL,
    start_time_utc TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    model_version TEXT NOT NULL,
    first_captured_utc TEXT NOT NULL,
    captured_utc TEXT NOT NULL,
    home_win_prob REAL NOT NULL,
    regulation_home REAL,
    regulation_draw REAL,
    regulation_away REAL,
    home_minus_1_5 REAL,
    away_minus_1_5 REAL,
    early_season INTEGER NOT NULL DEFAULT 0,
    graded_utc TEXT,
    home_win INTEGER,
    result_type TEXT,
    home_goals INTEGER,
    away_goals INTEGER,
    regulation_home_goals INTEGER,
    regulation_away_goals INTEGER
);
CREATE TABLE IF NOT EXISTS market_odds (
    game_id INTEGER NOT NULL,
    partner TEXT NOT NULL,
    start_time_utc TEXT NOT NULL,
    captured_utc TEXT NOT NULL,
    feed_updated_utc TEXT,
    home_team TEXT,
    away_team TEXT,
    home_moneyline REAL,
    away_moneyline REAL,
    home_regulation REAL,
    draw_regulation REAL,
    away_regulation REAL,
    home_puck_line REAL,
    home_puck_line_handicap REAL,
    away_puck_line REAL,
    total_line REAL,
    over_odds REAL,
    under_odds REAL,
    PRIMARY KEY (game_id, partner)
);
"""

_PREDICTION_FIELDS = (
    "game_id", "season_year", "game_type", "start_time_utc", "home_team", "away_team", "model_version",
    "home_win_prob", "regulation_home", "regulation_draw", "regulation_away", "home_minus_1_5", "away_minus_1_5",
    "early_season",
)
_MARKET_FIELDS = (
    "game_id", "partner", "start_time_utc", "feed_updated_utc", "home_team", "away_team", "home_moneyline", "away_moneyline",
    "home_regulation", "draw_regulation", "away_regulation", "home_puck_line", "home_puck_line_handicap", "away_puck_line",
    "total_line", "over_odds", "under_odds",
)


def ledger_path() -> Path:
    """Return the ledger file location from ``PUCKPEAK_DATA_DIR`` or the repo ``.data`` folder."""
    directory = os.environ.get(DATA_DIR_ENV) or str(Path(__file__).resolve().parent.parent / ".data")
    return Path(directory) / LEDGER_FILENAME


def connect_ledger(path: str | Path | None = None) -> sqlite3.Connection:
    """Open (and if needed create) the ledger database.

    Args:
        path: Database file; defaults to ``ledger_path()``.

    Returns:
        An open connection with the schema in place.
    """
    target = Path(path) if path is not None else ledger_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target), timeout=10)
    connection.executescript(_SCHEMA)
    return connection


def _utc_text(moment: datetime) -> str:
    """Format an aware datetime as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(value: object) -> datetime | None:
    """Parse an ISO UTC timestamp, or ``None``."""
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def record_prediction(connection: sqlite3.Connection, prediction: dict, now_utc: datetime) -> bool:
    """Insert or refresh one game's prediction, but never once the game has started.

    Args:
        connection: Ledger connection.
        prediction: Values for ``_PREDICTION_FIELDS``.
        now_utc: Capture time (aware).

    Returns:
        True when the row was written.
    """
    start = _parse_utc(prediction.get("start_time_utc"))
    if start is None or now_utc >= start:
        return False
    values = [prediction.get(field) for field in _PREDICTION_FIELDS]
    stamp = _utc_text(now_utc)
    placeholders = ", ".join("?" for _ in _PREDICTION_FIELDS)
    updates = ", ".join(f"{field} = excluded.{field}" for field in _PREDICTION_FIELDS if field != "game_id")
    cursor = connection.execute(
        f"INSERT INTO predictions ({', '.join(_PREDICTION_FIELDS)}, first_captured_utc, captured_utc) "
        f"VALUES ({placeholders}, ?, ?) "
        f"ON CONFLICT(game_id) DO UPDATE SET {updates}, captured_utc = excluded.captured_utc "
        "WHERE predictions.graded_utc IS NULL AND excluded.captured_utc < predictions.start_time_utc",
        values + [stamp, stamp],
    )
    connection.commit()
    return cursor.rowcount > 0


def record_market_odds(connection: sqlite3.Connection, rows: list[dict], now_utc: datetime) -> int:
    """Insert or refresh partner odds rows for games that have not started.

    Args:
        connection: Ledger connection.
        rows: ``parse_partner_odds`` output.
        now_utc: Capture time (aware).

    Returns:
        Number of rows written.
    """
    written = 0
    stamp = _utc_text(now_utc)
    placeholders = ", ".join("?" for _ in _MARKET_FIELDS)
    updates = ", ".join(f"{field} = excluded.{field}" for field in _MARKET_FIELDS if field not in ("game_id", "partner"))
    for row in rows:
        start = _parse_utc(row.get("start_time_utc"))
        if start is None or now_utc >= start:
            continue
        cursor = connection.execute(
            f"INSERT INTO market_odds ({', '.join(_MARKET_FIELDS)}, captured_utc) VALUES ({placeholders}, ?) "
            f"ON CONFLICT(game_id, partner) DO UPDATE SET {updates}, captured_utc = excluded.captured_utc "
            "WHERE excluded.captured_utc < market_odds.start_time_utc",
            [row.get(field) for field in _MARKET_FIELDS] + [stamp],
        )
        written += cursor.rowcount
    connection.commit()
    return written


def grade_predictions(connection: sqlite3.Connection, games: pd.DataFrame, now_utc: datetime) -> int:
    """Fill outcomes for every ungraded prediction whose game is in the completed-game table.

    Args:
        connection: Ledger connection.
        games: ``team_ratings.GAME_TABLE_COLUMNS`` rows of completed games.
        now_utc: Grading time (aware).

    Returns:
        Number of predictions graded.
    """
    if games is None or games.empty:
        return 0
    pending = {row[0] for row in connection.execute("SELECT game_id FROM predictions WHERE graded_utc IS NULL")}
    finished = games[games["GameId"].isin(pending)]
    if finished.empty:
        return 0
    home_regulation, away_regulation = regulation_goals(finished)
    stamp = _utc_text(now_utc)
    for row, home_reg, away_reg in zip(finished.itertuples(index=False), home_regulation, away_regulation):
        connection.execute(
            "UPDATE predictions SET graded_utc = ?, home_win = ?, result_type = ?, home_goals = ?, away_goals = ?, "
            "regulation_home_goals = ?, regulation_away_goals = ? WHERE game_id = ? AND graded_utc IS NULL",
            (stamp, int(row.HomeWin), str(row.ResultType), int(row.HomeGoals), int(row.AwayGoals), int(home_reg), int(away_reg), int(row.GameId)),
        )
    connection.commit()
    return int(len(finished))


def load_ledger(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read predictions and market odds without creating the database when it is missing.

    Args:
        path: Database file; defaults to ``ledger_path()``.

    Returns:
        ``(predictions, market_odds)`` frames, empty when there is no ledger yet.
    """
    target = Path(path) if path is not None else ledger_path()
    if not target.exists():
        return pd.DataFrame(), pd.DataFrame()
    connection = sqlite3.connect(f"{target.resolve().as_uri()}?mode=ro", uri=True, timeout=10)
    try:
        return pd.read_sql_query("SELECT * FROM predictions", connection), pd.read_sql_query("SELECT * FROM market_odds", connection)
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Market odds
# ---------------------------------------------------------------------------

def odds_to_probability(odds: object) -> float | None:
    """Convert one price to its implied probability (bookmaker margin included).

    The partner feeds mix formats: North American partners quote American odds
    (-125, +104), European partners quote decimal odds (1.80, 4.00). American odds are
    never between -100 and +100, and decimal odds for NHL games are never 100 or more,
    so the value itself tells the format.

    Args:
        odds: American or decimal price.

    Returns:
        Implied probability, or ``None`` for a missing or impossible price.
    """
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value >= 100.0:
        return 100.0 / (value + 100.0)
    if value <= -100.0:
        return -value / (-value + 100.0)
    if 1.0 < value < 100.0:
        return 1.0 / value
    return None


def remove_margin(*odds: object) -> list[float] | None:
    """Turn one market's prices into fair probabilities by proportional normalization.

    Args:
        *odds: American or decimal prices of every outcome of one market.

    Returns:
        Probabilities summing to 1, or ``None`` if any price is missing.
    """
    implied = [odds_to_probability(value) for value in odds]
    if any(value is None for value in implied):
        return None
    total = sum(implied)
    return [value / total for value in implied] if total > 0 else None


def parse_partner_odds(payload: object) -> list[dict]:
    """Normalize ``/v1/partner-game/{country}/now`` into one row per game for that partner.

    Args:
        payload: Raw partner-odds payload.

    Returns:
        Rows with the partner's prices (American or decimal, as quoted) for the moneyline,
        60-minute 3-way, puck line and totals.
    """
    if not isinstance(payload, dict):
        return []
    partner = payload.get("bettingPartner") or {}
    partner_name = str(partner.get("name") or "").strip()
    if not partner_name:
        return []
    label = f"{partner_name} ({partner.get('country', '')})"
    rows = []
    for game in payload.get("games", []) or []:
        if not isinstance(game, dict) or not game.get("gameId"):
            continue
        row = {
            "game_id": int(game["gameId"]),
            "game_type": int(game.get("gameType", 0) or 0),
            "partner": label,
            "start_time_utc": str(game.get("startTimeUTC") or ""),
            "feed_updated_utc": str(payload.get("lastUpdatedUTC") or ""),
            "home_team": canonical_team_abbrev((game.get("homeTeam") or {}).get("abbrev")),
            "away_team": canonical_team_abbrev((game.get("awayTeam") or {}).get("abbrev")),
        }
        for side in ("home", "away"):
            for entry in (game.get(f"{side}Team") or {}).get("odds", []) or []:
                description = str(entry.get("description") or "").upper()
                qualifier = str(entry.get("qualifier") or "").strip()
                value = entry.get("value")
                if description == "MONEY_LINE_2_WAY":
                    row[f"{side}_moneyline"] = value
                elif description == "MONEY_LINE_3_WAY":
                    if qualifier.lower() == "draw":
                        row["draw_regulation"] = value
                    else:
                        row[f"{side}_regulation"] = value
                elif description == "PUCK_LINE":
                    row[f"{side}_puck_line"] = value
                    if side == "home":
                        try:
                            row["home_puck_line_handicap"] = float(qualifier)
                        except ValueError:
                            row["home_puck_line_handicap"] = None
                elif description == "OVER_UNDER" and qualifier[:1].upper() in ("O", "U"):
                    try:
                        row["total_line"] = float(qualifier[1:])
                    except ValueError:
                        continue
                    row["over_odds" if qualifier[:1].upper() == "O" else "under_odds"] = value
        rows.append(row)
    return rows


def market_consensus(market_odds: pd.DataFrame) -> pd.DataFrame:
    """Average every partner's margin-free probabilities per game.

    Args:
        market_odds: ``market_odds`` table rows.

    Returns:
        One row per game: ``market_home_win``, ``market_regulation_home``, ``market_regulation_draw``,
        ``market_regulation_away`` and ``market_home_minus_1_5``, plus how many partners fed it.
    """
    columns = ["game_id", "market_home_win", "market_regulation_home", "market_regulation_draw", "market_regulation_away", "market_home_minus_1_5", "market_partners"]
    if market_odds is None or market_odds.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in market_odds.itertuples(index=False):
        moneyline = remove_margin(row.home_moneyline, row.away_moneyline)
        regulation = remove_margin(row.home_regulation, row.draw_regulation, row.away_regulation)
        puck = remove_margin(row.home_puck_line, row.away_puck_line)
        home_minus = None
        if puck is not None and row.home_puck_line_handicap is not None and not pd.isna(row.home_puck_line_handicap):
            home_minus = puck[0] if float(row.home_puck_line_handicap) < 0 else None
        rows.append({
            "game_id": int(row.game_id),
            "market_home_win": moneyline[0] if moneyline else np.nan,
            "market_regulation_home": regulation[0] if regulation else np.nan,
            "market_regulation_draw": regulation[1] if regulation else np.nan,
            "market_regulation_away": regulation[2] if regulation else np.nan,
            "market_home_minus_1_5": home_minus if home_minus is not None else np.nan,
        })
    frame = pd.DataFrame(rows)
    consensus = frame.groupby("game_id").mean(numeric_only=True).reset_index()
    consensus["market_partners"] = frame.groupby("game_id").size().to_numpy()
    return consensus[columns]


# ---------------------------------------------------------------------------
# Track record
# ---------------------------------------------------------------------------

def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean binary log loss."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def track_record(predictions: pd.DataFrame, market_odds: pd.DataFrame | None = None, season_year: int | None = None) -> dict:
    """Score graded ledger predictions, and the betting market on the same games.

    Args:
        predictions: ``predictions`` table rows.
        market_odds: ``market_odds`` table rows, optional.
        season_year: Only games of this season, when given.

    Returns:
        ``logged`` and ``games`` (graded) counts, moneyline accuracy / log loss / Brier,
        60-minute log loss, and, where market prices exist, the model's and the market's
        log loss on exactly the same games.
    """
    empty = {"logged": 0, "games": 0}
    if predictions is None or predictions.empty:
        return empty
    frame = predictions.copy()
    if season_year is not None:
        frame = frame[frame["season_year"].eq(int(season_year))]
    logged = int(len(frame))
    graded = frame[frame["graded_utc"].notna()].copy()
    if graded.empty:
        return {"logged": logged, "games": 0}

    labels = graded["home_win"].astype(int).to_numpy()
    probabilities = graded["home_win_prob"].astype(float).to_numpy()
    record = {
        "logged": logged,
        "games": int(len(graded)),
        "accuracy": float(np.mean((probabilities >= 0.5) == (labels == 1))),
        "log_loss": _binary_log_loss(labels, probabilities),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "coin_flip_log_loss": math.log(2.0),
        "first_game_utc": str(graded["start_time_utc"].min()),
        "last_game_utc": str(graded["start_time_utc"].max()),
    }

    margin = (graded["regulation_home_goals"] - graded["regulation_away_goals"]).to_numpy()
    result = np.where(margin > 0, 0, np.where(margin == 0, 1, 2))
    regulation = graded[["regulation_home", "regulation_draw", "regulation_away"]].to_numpy(dtype=float)
    has_regulation = ~np.isnan(regulation).any(axis=1)
    if has_regulation.any():
        chosen = regulation[has_regulation][np.arange(int(has_regulation.sum())), result[has_regulation]]
        record["regulation_log_loss"] = float(-np.mean(np.log(np.clip(chosen, 1e-9, 1.0))))

    consensus = market_consensus(market_odds)
    if not consensus.empty:
        joined = graded.merge(consensus, on="game_id", how="inner").dropna(subset=["market_home_win"])
        if not joined.empty:
            joined_labels = joined["home_win"].astype(int).to_numpy()
            record["market_games"] = int(len(joined))
            record["market_log_loss"] = _binary_log_loss(joined_labels, joined["market_home_win"].to_numpy())
            record["model_log_loss_on_market_games"] = _binary_log_loss(joined_labels, joined["home_win_prob"].to_numpy())
            record["market_accuracy"] = float(np.mean((joined["market_home_win"].to_numpy() >= 0.5) == (joined_labels == 1)))
    return record


def prediction_row(game: dict, probability: dict, model_version: str) -> dict:
    """Build a ledger row from an upcoming-game dict and its ``get_game_win_probabilities`` output.

    Args:
        game: ``schedule.get_upcoming_games`` entry.
        probability: The game's pregame probability dict.
        model_version: Artifact identifier (its ``generated_at_utc``).

    Returns:
        Values for ``record_prediction``.
    """
    markets = probability.get("markets") or {}
    regulation = markets.get("regulation") or {}
    puck_line = markets.get("puck_line") or {}
    game_id = int(game.get("game_id", 0) or 0)
    return {
        "game_id": game_id,
        "season_year": season_year_from_game_id(game_id),
        "game_type": int(game.get("game_type", 0) or 0),
        "start_time_utc": str(game.get("start_time_utc") or ""),
        "home_team": canonical_team_abbrev(game.get("home_abbr")),
        "away_team": canonical_team_abbrev(game.get("away_abbr")),
        "model_version": str(model_version or "unknown"),
        "home_win_prob": float(probability["home_win_prob"]),
        "regulation_home": regulation.get("home"),
        "regulation_draw": regulation.get("draw"),
        "regulation_away": regulation.get("away"),
        "home_minus_1_5": puck_line.get("home_minus_1_5"),
        "away_minus_1_5": puck_line.get("away_minus_1_5"),
        "early_season": int(bool(probability.get("early_season"))),
    }
