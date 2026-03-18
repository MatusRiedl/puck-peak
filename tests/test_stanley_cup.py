import unittest

import pandas as pd

from nhl.stanley_cup import build_stanley_cup_board
from nhl.win_prob import WIN_PROB_FEATURE_ORDER


class StanleyCupBoardTests(unittest.TestCase):
    """Cover Stanley Cup board scoring and fallback behavior."""

    def test_build_stanley_cup_board_picks_the_strongest_team_and_keeps_top_drivers(self):
        """Rank teams by the model-backed contender score instead of raw points alone."""
        standings_df = pd.DataFrame(
            [
                {
                    "teamAbbrev": "COL",
                    "teamName": "Colorado Avalanche",
                    "teamCommonName": "Avalanche",
                    "teamLogo": "https://assets.nhle.com/logos/nhl/svg/COL_light.svg",
                    "conferenceName": "Western",
                    "divisionName": "Central",
                    "divisionSequence": 1,
                    "conferenceSequence": 1,
                    "leagueSequence": 1,
                    "gamesPlayed": 63,
                    "wins": 43,
                    "losses": 11,
                    "otLosses": 9,
                    "points": 95,
                    "goalDifferential": 82,
                    "pointPctg": 0.754,
                    "goalDiffPerGame": 1.302,
                    "l10PointPctg": 0.700,
                    "l10GoalDiffPerGame": 0.800,
                    "recordLabel": "43-11-9",
                    "l10RecordLabel": "7-3-0",
                    "PP%": 16.0,
                    "standingsDateTimeUtc": "2026-03-12T20:28:00Z",
                },
                {
                    "teamAbbrev": "DAL",
                    "teamName": "Dallas Stars",
                    "teamCommonName": "Stars",
                    "teamLogo": "https://assets.nhle.com/logos/nhl/svg/DAL_light.svg",
                    "conferenceName": "Western",
                    "divisionName": "Central",
                    "divisionSequence": 2,
                    "conferenceSequence": 2,
                    "leagueSequence": 3,
                    "gamesPlayed": 64,
                    "wins": 40,
                    "losses": 14,
                    "otLosses": 10,
                    "points": 90,
                    "goalDifferential": 48,
                    "pointPctg": 0.703,
                    "goalDiffPerGame": 0.750,
                    "l10PointPctg": 0.950,
                    "l10GoalDiffPerGame": 1.800,
                    "recordLabel": "40-14-10",
                    "l10RecordLabel": "9-1-0",
                    "PP%": 29.7,
                    "standingsDateTimeUtc": "2026-03-12T20:28:00Z",
                },
                {
                    "teamAbbrev": "EDM",
                    "teamName": "Edmonton Oilers",
                    "teamCommonName": "Oilers",
                    "teamLogo": "https://assets.nhle.com/logos/nhl/svg/EDM_light.svg",
                    "conferenceName": "Western",
                    "divisionName": "Pacific",
                    "divisionSequence": 2,
                    "conferenceSequence": 5,
                    "leagueSequence": 9,
                    "gamesPlayed": 65,
                    "wins": 32,
                    "losses": 25,
                    "otLosses": 8,
                    "points": 72,
                    "goalDifferential": 13,
                    "pointPctg": 0.554,
                    "goalDiffPerGame": 0.200,
                    "l10PointPctg": 0.400,
                    "l10GoalDiffPerGame": -0.200,
                    "recordLabel": "32-25-8",
                    "l10RecordLabel": "4-6-0",
                    "PP%": float("nan"),
                    "standingsDateTimeUtc": "2026-03-12T20:28:00Z",
                },
                {
                    "teamAbbrev": "CGY",
                    "teamName": "Calgary Flames",
                    "teamCommonName": "Flames",
                    "teamLogo": "https://assets.nhle.com/logos/nhl/svg/CGY_light.svg",
                    "conferenceName": "Western",
                    "divisionName": "Pacific",
                    "divisionSequence": 7,
                    "conferenceSequence": 14,
                    "leagueSequence": 27,
                    "gamesPlayed": 64,
                    "wins": 25,
                    "losses": 32,
                    "otLosses": 7,
                    "points": 57,
                    "goalDifferential": -31,
                    "pointPctg": 0.445,
                    "goalDiffPerGame": -0.484,
                    "l10PointPctg": 0.300,
                    "l10GoalDiffPerGame": -0.900,
                    "recordLabel": "25-32-7",
                    "l10RecordLabel": "3-7-0",
                    "PP%": 18.0,
                    "standingsDateTimeUtc": "2026-03-12T20:28:00Z",
                },
            ]
        )
        artifact = {
            "model_version": 1,
            "feature_order": WIN_PROB_FEATURE_ORDER,
            "coefficients": [1.2, 0.8, 0.6, 0.5, 0.2],
            "intercept": 0.0,
            "scaler_mean": [0.0] * len(WIN_PROB_FEATURE_ORDER),
            "scaler_scale": [1.0] * len(WIN_PROB_FEATURE_ORDER),
            "selected_c": 1.0,
            "min_games": 5,
        }
        goalie_proxy_by_team = {
            "COL": 0.916,
            "DAL": 0.899,
            "EDM": None,
            "CGY": 0.895,
        }

        board = build_stanley_cup_board(standings_df, artifact, goalie_proxy_by_team)

        self.assertEqual(board["favorite_team_abbr"], "DAL")
        self.assertEqual(board["favorite_team"]["team_name"], "Dallas Stars")
        self.assertTrue(board["favorite_team"]["is_favorite"])
        self.assertEqual(board["favorite_team"]["rank"], 1)
        self.assertEqual(len(board["favorite_team"]["top_drivers"]), 3)
        self.assertIn("neutral-opponent contender score", board["favorite_team"]["summary_text"])
        self.assertEqual([division["division_name"] for division in board["divisions"]], ["Central", "Pacific"])

    def test_build_stanley_cup_board_neutralizes_missing_pp_and_goalie_inputs(self):
        """Fill missing power-play and goalie inputs with league-average values."""
        standings_df = pd.DataFrame(
            [
                {
                    "teamAbbrev": "COL",
                    "teamName": "Colorado Avalanche",
                    "teamCommonName": "Avalanche",
                    "conferenceName": "Western",
                    "divisionName": "Central",
                    "divisionSequence": 1,
                    "conferenceSequence": 1,
                    "leagueSequence": 1,
                    "gamesPlayed": 63,
                    "wins": 43,
                    "losses": 11,
                    "otLosses": 9,
                    "points": 95,
                    "goalDifferential": 82,
                    "pointPctg": 0.754,
                    "goalDiffPerGame": 1.302,
                    "l10PointPctg": 0.700,
                    "l10GoalDiffPerGame": 0.800,
                    "recordLabel": "43-11-9",
                    "l10RecordLabel": "7-3-0",
                    "PP%": 16.0,
                    "standingsDateTimeUtc": "2026-03-12T20:28:00Z",
                },
                {
                    "teamAbbrev": "EDM",
                    "teamName": "Edmonton Oilers",
                    "teamCommonName": "Oilers",
                    "conferenceName": "Western",
                    "divisionName": "Pacific",
                    "divisionSequence": 2,
                    "conferenceSequence": 5,
                    "leagueSequence": 9,
                    "gamesPlayed": 65,
                    "wins": 32,
                    "losses": 25,
                    "otLosses": 8,
                    "points": 72,
                    "goalDifferential": 13,
                    "pointPctg": 0.554,
                    "goalDiffPerGame": 0.200,
                    "l10PointPctg": 0.400,
                    "l10GoalDiffPerGame": -0.200,
                    "recordLabel": "32-25-8",
                    "l10RecordLabel": "4-6-0",
                    "PP%": float("nan"),
                    "standingsDateTimeUtc": "2026-03-12T20:28:00Z",
                },
            ]
        )
        artifact = {
            "model_version": 1,
            "feature_order": WIN_PROB_FEATURE_ORDER,
            "coefficients": [0.8, 0.8, 0.5, 0.4, 0.1],
            "intercept": 0.0,
            "scaler_mean": [0.0] * len(WIN_PROB_FEATURE_ORDER),
            "scaler_scale": [1.0] * len(WIN_PROB_FEATURE_ORDER),
            "selected_c": 1.0,
            "min_games": 5,
        }

        board = build_stanley_cup_board(
            standings_df=standings_df,
            artifact=artifact,
            goalie_proxy_by_team={"COL": 0.916, "EDM": None},
        )

        edm = next(team for team in board["teams"] if team["team_abbr"] == "EDM")
        self.assertTrue(edm["pp_neutralized"])
        self.assertTrue(edm["goalie_neutralized"])
        self.assertIn("Power play %", edm["neutralized_inputs"])
        self.assertIn("Goalie proxy save %", edm["neutralized_inputs"])
        self.assertIsInstance(edm["contender_score"], float)
        self.assertTrue(edm["contender_score"] >= 0.0)


if __name__ == "__main__":
    unittest.main()
