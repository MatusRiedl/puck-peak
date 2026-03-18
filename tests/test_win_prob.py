import math
import unittest

import pandas as pd

import nhl.win_prob as win_prob


class WinProbFeatureTests(unittest.TestCase):
    """Cover the shared training/runtime win-probability helpers."""

    def test_compute_team_feature_history_uses_only_prior_games(self):
        """Pregame features for one row must exclude that row's result."""
        team_games = pd.DataFrame(
            {
                "SeasonYear": [2025] * 6,
                "GameDate": [f"2025-10-{day:02d}" for day in range(1, 7)],
                "GameId": list(range(1, 7)),
                "TeamAbbrev": ["DAL"] * 6,
                "OpponentAbbrev": ["EDM", "COL", "WPG", "VGK", "NSH", "NYR"],
                "HomeRoadFlag": ["H", "R", "H", "R", "H", "R"],
                "Points": [2, 0, 2, 2, 0, 2],
                "Goals": [4, 1, 5, 3, 2, 6],
                "GoalsAgainst": [2, 4, 2, 1, 3, 2],
                "PP%": [20.0, 10.0, 30.0, 25.0, 15.0, 40.0],
            }
        )

        featured = win_prob.compute_team_feature_history(team_games)
        sixth_game = featured.iloc[5]

        self.assertEqual(int(sixth_game["GamesBefore"]), 5)
        self.assertAlmostEqual(float(sixth_game["PointPctToDate"]), 0.6)
        self.assertAlmostEqual(float(sixth_game["L10PointPct"]), 0.6)
        self.assertAlmostEqual(float(sixth_game["GoalDiffPerGameToDate"]), 0.6)

    def test_score_home_win_probability_matches_exported_weights(self):
        """Runtime dot-product scoring must honor exported means/scales/weights exactly."""
        artifact = {
            "model_version": 1,
            "feature_order": win_prob.WIN_PROB_FEATURE_ORDER,
            "coefficients": [0.4, 0.2, -0.1, 0.05, 0.03],
            "intercept": 0.1,
            "scaler_mean": [0.0] * len(win_prob.WIN_PROB_FEATURE_ORDER),
            "scaler_scale": [1.0] * len(win_prob.WIN_PROB_FEATURE_ORDER),
            "selected_c": 1.0,
            "min_games": win_prob.MIN_GAMES_FOR_ESTIMATE,
        }
        feature_values = {
            "point_pct_to_date": 0.20,
            "goal_diff_per_game_to_date": 0.60,
            "l10_point_pct": 0.15,
            "l10_goal_diff_per_game": 0.25,
            "power_play_pct_to_date": 3.50,
        }

        scored = win_prob.score_home_win_probability(feature_values, artifact)
        linear_term = 0.1 + (0.4 * 0.20) + (0.2 * 0.60) - (0.1 * 0.15) + (0.05 * 0.25) + (0.03 * 3.50)
        expected_probability = 1.0 / (1.0 + math.exp(-linear_term))

        self.assertAlmostEqual(float(scored["home_win_prob"]), expected_probability)
        self.assertAlmostEqual(float(scored["contributions"]["power_play_pct_to_date"]), 0.105)

    def test_build_matchup_snapshot_returns_none_when_team_sample_is_too_small(self):
        """Do not manufacture a pregame estimate before the minimum-games threshold."""
        home_games = pd.DataFrame(
            {
                "SeasonYear": [2025] * 5,
                "GameDate": [f"2025-10-{day:02d}" for day in range(1, 6)],
                "GameId": list(range(1, 6)),
                "TeamAbbrev": ["TOR"] * 5,
                "OpponentAbbrev": ["MTL", "OTT", "BUF", "BOS", "DET"],
                "HomeRoadFlag": ["H", "R", "H", "R", "H"],
                "Points": [2, 2, 0, 2, 2],
                "Goals": [4, 5, 2, 4, 3],
                "GoalsAgainst": [1, 2, 3, 2, 2],
                "PP%": [25.0, 20.0, 15.0, 22.0, 24.0],
            }
        )
        away_games = pd.DataFrame(
            {
                "SeasonYear": [2025] * 4,
                "GameDate": [f"2025-10-{day:02d}" for day in range(1, 5)],
                "GameId": list(range(11, 15)),
                "TeamAbbrev": ["MTL"] * 4,
                "OpponentAbbrev": ["TOR", "OTT", "BUF", "BOS"],
                "HomeRoadFlag": ["R", "H", "R", "H"],
                "Points": [0, 2, 0, 2],
                "Goals": [2, 3, 1, 4],
                "GoalsAgainst": [4, 2, 3, 3],
                "PP%": [16.0, 18.0, 14.0, 19.0],
            }
        )

        snapshot = win_prob.build_matchup_snapshot(home_games, away_games, min_games=5)

        self.assertIsNone(snapshot)


if __name__ == "__main__":
    unittest.main()
