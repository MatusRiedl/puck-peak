import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import train_win_prob
from nhl import ledger

PUCK_DROP = datetime(2026, 10, 8, 23, 0, tzinfo=timezone.utc)


class LedgerReportTests(unittest.TestCase):
    """Cover ``train_win_prob.py --ledger-report``: report only, never the artifact."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = Path(self.directory) / "ledger.sqlite3"

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _build_ledger(self) -> None:
        """Three graded games; two have a fresh market price, one only a stale one."""
        connection = ledger.connect_ledger(self.path)
        results = []
        for number, (model, home_odds, away_odds, home_win, fresh) in enumerate(
            ((0.60, 1.60, 2.40, 1, True), (0.45, 2.40, 1.60, 0, True), (0.55, 1.90, 1.90, 1, False))
        ):
            game_id = 2026020200 + number
            ledger.record_prediction(connection, {
                "game_id": game_id, "season_year": 2026, "game_type": 2, "start_time_utc": "2026-10-08T23:00:00Z",
                "home_team": "TOR", "away_team": "MTL", "model_version": "test", "home_win_prob": model,
                "regulation_home": 0.45, "regulation_draw": 0.22, "regulation_away": 0.33, "home_minus_1_5": 0.3,
                "away_minus_1_5": 0.2, "early_season": 1,
            }, PUCK_DROP - timedelta(minutes=30))
            updated = PUCK_DROP - (timedelta(minutes=20) if fresh else timedelta(days=6))
            ledger.record_market_odds(connection, [{
                "game_id": game_id, "partner": "Partner (USA)", "start_time_utc": "2026-10-08T23:00:00Z",
                "feed_updated_utc": updated.strftime("%Y-%m-%dT%H:%M:%SZ"), "home_team": "TOR", "away_team": "MTL",
                "home_moneyline": home_odds, "away_moneyline": away_odds,
            }], PUCK_DROP - timedelta(minutes=10))
            results.append({"GameId": game_id, "HomeGoals": 3.0 if home_win else 1.0, "AwayGoals": 1.0 if home_win else 3.0, "ResultType": "REG", "HomeWin": home_win})
        ledger.grade_predictions(connection, pd.DataFrame(results), PUCK_DROP + timedelta(hours=4))
        connection.close()

    def test_report_prints_the_market_comparison_and_leaves_the_artifact_alone(self):
        """The stale-only game is graded but takes no part in the market comparison."""
        self._build_ledger()
        artifact_before = train_win_prob.OUTPUT_PATH.read_bytes() if train_win_prob.OUTPUT_PATH.exists() else None

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = train_win_prob.ledger_report(str(self.path))
        text = output.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertIn("2026-27: 3 predictions logged, 3 graded", text)
        self.assertIn("market: 2 of those games have a pregame market price", text)
        self.assertIn("log loss on those games: model", text)
        self.assertIn("best mix fitted on these same games:", text)
        self.assertIn(f"2 games is too few to act on; wait for {ledger.MARKET_DECISION_GAMES}.", text)
        artifact_after = train_win_prob.OUTPUT_PATH.read_bytes() if train_win_prob.OUTPUT_PATH.exists() else None
        self.assertEqual(artifact_before, artifact_after)

    def test_report_on_a_missing_ledger_says_so_and_creates_nothing(self):
        missing = Path(self.directory) / "absent" / "ledger.sqlite3"

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = train_win_prob.ledger_report(str(missing))

        self.assertEqual(exit_code, 0)
        self.assertIn("No predictions in", output.getvalue())
        self.assertFalse(missing.parent.exists())

    def test_the_optional_path_follows_the_flag(self):
        self.assertEqual(train_win_prob._ledger_report_path(["--ledger-report", "copy.sqlite3"]), "copy.sqlite3")
        self.assertIsNone(train_win_prob._ledger_report_path(["--ledger-report"]))
        self.assertIsNone(train_win_prob._ledger_report_path(["--ledger-report", "--phase2-report"]))


if __name__ == "__main__":
    unittest.main()
