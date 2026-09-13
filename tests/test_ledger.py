import math
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nhl import ledger

PUCK_DROP = datetime(2026, 10, 8, 23, 0, tzinfo=timezone.utc)


def _at(minutes: int) -> datetime:
    """A capture time relative to puck drop."""
    return PUCK_DROP + timedelta(minutes=minutes)


def _prediction(**overrides):
    """One ledger prediction row."""
    row = {
        "game_id": 2026020050, "season_year": 2026, "game_type": 2, "start_time_utc": "2026-10-08T23:00:00Z",
        "home_team": "TOR", "away_team": "MTL", "model_version": "2026-09-13T00:00:00Z", "home_win_prob": 0.60,
        "regulation_home": 0.47, "regulation_draw": 0.22, "regulation_away": 0.31, "home_minus_1_5": 0.30,
        "away_minus_1_5": 0.15, "early_season": 1,
    }
    row.update(overrides)
    return row


FANDUEL = {
    "lastUpdatedUTC": "2026-09-13T18:50:01Z",
    "bettingPartner": {"partnerId": 7, "country": "CAN", "name": "FanDuel"},
    "games": [
        {
            "gameId": 2026020001,
            "gameType": 2,
            "startTimeUTC": "2026-09-29T21:00:00Z",
            "homeTeam": {"abbrev": "CAR", "odds": [
                {"description": "MONEY_LINE_2_WAY", "value": -125.0, "qualifier": ""},
                {"description": "PUCK_LINE", "value": 176.0, "qualifier": "-1.5"},
                {"description": "MONEY_LINE_3_WAY", "value": 125.0, "qualifier": ""},
                {"description": "MONEY_LINE_3_WAY", "value": 340.0, "qualifier": "Draw"},
                {"description": "OVER_UNDER", "value": 106.0, "qualifier": "O6.5"},
            ]},
            "awayTeam": {"abbrev": "FLA", "odds": [
                {"description": "MONEY_LINE_2_WAY", "value": 104.0, "qualifier": ""},
                {"description": "PUCK_LINE", "value": -225.0, "qualifier": "+1.5"},
                {"description": "MONEY_LINE_3_WAY", "value": 160.0, "qualifier": ""},
                {"description": "MONEY_LINE_3_WAY", "value": 340.0, "qualifier": "Draw"},
                {"description": "OVER_UNDER", "value": -130.0, "qualifier": "U6.5"},
            ]},
        }
    ],
}


class LedgerStorageTests(unittest.TestCase):
    """Cover writing, freezing and grading ledger rows."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = Path(self.directory) / "ledger.sqlite3"
        self.connection = ledger.connect_ledger(self.path)

    def tearDown(self):
        self.connection.close()
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_prediction_refreshes_until_puck_drop_and_is_frozen_after(self):
        """The ledger keeps the last pregame numbers; nothing written after puck drop sticks."""
        self.assertTrue(ledger.record_prediction(self.connection, _prediction(home_win_prob=0.58), _at(-60)))
        self.assertTrue(ledger.record_prediction(self.connection, _prediction(home_win_prob=0.62), _at(-5)))
        self.assertFalse(ledger.record_prediction(self.connection, _prediction(home_win_prob=0.95), _at(1)))

        predictions, _ = ledger.load_ledger(self.path)
        row = predictions.iloc[0]
        self.assertAlmostEqual(float(row["home_win_prob"]), 0.62)
        self.assertEqual(row["first_captured_utc"], "2026-10-08T22:00:00Z")
        self.assertEqual(row["captured_utc"], "2026-10-08T22:55:00Z")

    def test_grading_records_the_result_and_the_regulation_score(self):
        """An overtime win is a regulation tie; graded rows are not graded twice."""
        ledger.record_prediction(self.connection, _prediction(), _at(-30))
        games = pd.DataFrame({"GameId": [2026020050], "HomeGoals": [3.0], "AwayGoals": [2.0], "ResultType": ["OT"], "HomeWin": [1]})

        self.assertEqual(ledger.grade_predictions(self.connection, games, _at(200)), 1)
        self.assertEqual(ledger.grade_predictions(self.connection, games, _at(300)), 0)
        row = ledger.load_ledger(self.path)[0].iloc[0]
        self.assertEqual(int(row["home_win"]), 1)
        self.assertEqual(row["result_type"], "OT")
        self.assertEqual((int(row["regulation_home_goals"]), int(row["regulation_away_goals"])), (2, 2))

    def test_market_odds_also_freeze_at_puck_drop(self):
        """Market prices captured after the game starts are ignored."""
        rows = ledger.parse_partner_odds(FANDUEL)
        start = datetime(2026, 9, 29, 21, 0, tzinfo=timezone.utc)

        self.assertEqual(ledger.record_market_odds(self.connection, rows, start - timedelta(minutes=10)), 1)
        changed = [dict(rows[0], home_moneyline=-300.0)]
        self.assertEqual(ledger.record_market_odds(self.connection, changed, start + timedelta(minutes=2)), 0)
        market = ledger.load_ledger(self.path)[1]
        self.assertEqual(float(market.iloc[0]["home_moneyline"]), -125.0)

    def test_reading_a_missing_ledger_creates_nothing(self):
        """Page renders read the ledger; they must not create database files."""
        missing = Path(self.directory) / "nested" / "absent.sqlite3"
        predictions, market = ledger.load_ledger(missing)

        self.assertTrue(predictions.empty and market.empty)
        self.assertFalse(missing.exists())


class MarketOddsTests(unittest.TestCase):
    """Cover partner-odds parsing and margin removal."""

    def test_american_and_decimal_odds_convert_to_implied_probability(self):
        """North American partners quote American odds, European partners decimal odds."""
        self.assertAlmostEqual(ledger.odds_to_probability(104), 100 / 204)
        self.assertAlmostEqual(ledger.odds_to_probability(-125), 125 / 225)
        self.assertAlmostEqual(ledger.odds_to_probability(100), 0.5)
        self.assertAlmostEqual(ledger.odds_to_probability(1.8), 1 / 1.8)
        self.assertAlmostEqual(ledger.odds_to_probability(4.15), 1 / 4.15)
        self.assertIsNone(ledger.odds_to_probability(None))
        self.assertIsNone(ledger.odds_to_probability(0))
        self.assertIsNone(ledger.odds_to_probability(-50))
        self.assertIsNone(ledger.odds_to_probability(1.0))

    def test_removing_the_margin_normalizes_implied_probabilities(self):
        fair = ledger.remove_margin(-125, 104)
        self.assertAlmostEqual(sum(fair), 1.0)
        self.assertAlmostEqual(fair[0], (125 / 225) / (125 / 225 + 100 / 204))
        decimal = ledger.remove_margin(1.8, 2.0)
        self.assertAlmostEqual(decimal[0], (1 / 1.8) / (1 / 1.8 + 1 / 2.0))
        self.assertIsNone(ledger.remove_margin(-125, None))

    def test_partner_feed_parses_every_market(self):
        """Moneyline, 60-minute 3-way, puck line with handicap, and totals with the line."""
        row = ledger.parse_partner_odds(FANDUEL)[0]

        self.assertEqual(row["partner"], "FanDuel (CAN)")
        self.assertEqual((row["home_team"], row["away_team"]), ("CAR", "FLA"))
        self.assertEqual((row["home_moneyline"], row["away_moneyline"]), (-125.0, 104.0))
        self.assertEqual((row["home_regulation"], row["draw_regulation"], row["away_regulation"]), (125.0, 340.0, 160.0))
        self.assertEqual((row["home_puck_line"], row["home_puck_line_handicap"], row["away_puck_line"]), (176.0, -1.5, -225.0))
        self.assertEqual((row["total_line"], row["over_odds"], row["under_odds"]), (6.5, 106.0, -130.0))
        self.assertEqual(ledger.parse_partner_odds({"bettingPartner": {}, "games": []}), [])

    def test_consensus_averages_partners_per_game(self):
        """Two partners' fair moneyline probabilities are averaged."""
        first = ledger.parse_partner_odds(FANDUEL)[0]
        second = dict(first, partner="Other (USA)", home_moneyline=-150.0, away_moneyline=130.0)
        consensus = ledger.market_consensus(pd.DataFrame([first, second]))

        expected = (ledger.remove_margin(-125, 104)[0] + ledger.remove_margin(-150, 130)[0]) / 2
        self.assertAlmostEqual(float(consensus.iloc[0]["market_home_win"]), expected)
        self.assertEqual(int(consensus.iloc[0]["market_partners"]), 2)


class TrackRecordTests(unittest.TestCase):
    """Cover the live track record metrics."""

    def test_record_scores_graded_games_and_compares_the_market_on_the_same_games(self):
        predictions = pd.DataFrame([
            _prediction(game_id=1, home_win_prob=0.7) | {"graded_utc": "x", "home_win": 1, "regulation_home_goals": 3, "regulation_away_goals": 1},
            _prediction(game_id=2, home_win_prob=0.4) | {"graded_utc": "x", "home_win": 1, "regulation_home_goals": 2, "regulation_away_goals": 2},
            _prediction(game_id=3, home_win_prob=0.55) | {"graded_utc": "x", "home_win": 0, "regulation_home_goals": 1, "regulation_away_goals": 4},
            _prediction(game_id=4, home_win_prob=0.5) | {"graded_utc": None, "home_win": None, "regulation_home_goals": None, "regulation_away_goals": None},
            _prediction(game_id=5, season_year=2025, home_win_prob=0.9) | {"graded_utc": "x", "home_win": 0, "regulation_home_goals": 0, "regulation_away_goals": 1},
        ])
        market = pd.DataFrame([
            {"game_id": 1, "home_moneyline": -150.0, "away_moneyline": 130.0, "home_regulation": None, "draw_regulation": None, "away_regulation": None, "home_puck_line": None, "home_puck_line_handicap": None, "away_puck_line": None},
            {"game_id": 3, "home_moneyline": 110.0, "away_moneyline": -130.0, "home_regulation": None, "draw_regulation": None, "away_regulation": None, "home_puck_line": None, "home_puck_line_handicap": None, "away_puck_line": None},
        ])

        record = ledger.track_record(predictions, market, season_year=2026)

        self.assertEqual((record["logged"], record["games"]), (4, 3))
        self.assertAlmostEqual(record["accuracy"], 1 / 3)
        expected_log_loss = -(math.log(0.7) + math.log(0.4) + math.log(0.45)) / 3
        self.assertAlmostEqual(record["log_loss"], expected_log_loss)
        expected_regulation = -(math.log(0.47) + math.log(0.22) + math.log(0.31)) / 3
        self.assertAlmostEqual(record["regulation_log_loss"], expected_regulation)
        self.assertEqual(record["market_games"], 2)
        market_home = [ledger.remove_margin(-150, 130)[0], ledger.remove_margin(110, -130)[0]]
        self.assertAlmostEqual(record["market_log_loss"], -(math.log(market_home[0]) + math.log(1 - market_home[1])) / 2)
        self.assertAlmostEqual(record["model_log_loss_on_market_games"], -(math.log(0.7) + math.log(0.45)) / 2)

    def test_empty_ledger_reports_nothing(self):
        self.assertEqual(ledger.track_record(pd.DataFrame()), {"logged": 0, "games": 0})


if __name__ == "__main__":
    unittest.main()
