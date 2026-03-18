import unittest
from unittest.mock import patch

import pandas as pd

from nhl.player_pipeline import process_players


class PlayerPipelineLeagueNormalizationTests(unittest.TestCase):
    """Lock the Era toggle's effect on mixed-league skater scoring."""

    def test_non_nhl_scoring_stays_raw_until_era_is_on(self):
        """Keep mixed-league skater points raw unless Era is enabled.

        Args:
            None.

        Returns:
            None.
        """
        raw_df = pd.DataFrame(
            {
                "League": ["AHL", "NHL"],
                "Age": [18, 19],
                "SeasonYear": [2020, 1985],
                "GameType": ["Regular", "Regular"],
                "GP": [50, 80],
                "Points": [100.0, 100.0],
                "Goals": [40.0, 40.0],
                "Assists": [60.0, 60.0],
                "PIM": [10.0, 20.0],
                "+/-": [5.0, 10.0],
                "Shots": [200.0, 250.0],
                "TotalTOIMins": [500.0, 1200.0],
                "Wins": [0.0, 0.0],
                "Shutouts": [0.0, 0.0],
                "Saves": [0.0, 0.0],
                "WeightedSV": [0.0, 0.0],
                "WeightedGAA": [0.0, 0.0],
                "NHLeMultiplier": [0.39, 1.0],
            }
        )

        with patch("nhl.player_pipeline.get_player_raw_stats", return_value=(raw_df, "Test Skater", "C")):
            raw_result, *_ = process_players(
                players={"1": "Test Skater"},
                metric="Points",
                hist_df=pd.DataFrame(),
                id_to_name_map={},
                clone_details_map={},
                season_type="Regular",
                stat_category="Skater",
                do_era=False,
                do_predict=False,
                do_smooth=False,
                do_cumul=False,
                games_mode=False,
                league_filter=["NHL", "AHL"],
            )
            era_result, *_ = process_players(
                players={"1": "Test Skater"},
                metric="Points",
                hist_df=pd.DataFrame(),
                id_to_name_map={},
                clone_details_map={},
                season_type="Regular",
                stat_category="Skater",
                do_era=True,
                do_predict=False,
                do_smooth=False,
                do_cumul=False,
                games_mode=False,
                league_filter=["NHL", "AHL"],
            )

        raw_points = raw_result[0].set_index("Age")["Points"]
        era_points = era_result[0].set_index("Age")["Points"]

        self.assertAlmostEqual(float(raw_points.loc[18]), 100.0)
        self.assertAlmostEqual(float(raw_points.loc[19]), 100.0)
        self.assertAlmostEqual(float(era_points.loc[18]), 39.0)
        self.assertAlmostEqual(float(era_points.loc[19]), 80.0)

    def test_raw_cache_keeps_player_identity_columns_for_dialogs(self):
        """Carry PlayerID and PositionCode into the raw cache used by Season Snapshot."""
        raw_df = pd.DataFrame(
            {
                "League": ["NHL"],
                "Age": [24],
                "SeasonYear": [2024],
                "GameType": ["Regular"],
                "GP": [82],
                "Points": [120.0],
                "Goals": [50.0],
                "Assists": [70.0],
                "PIM": [20.0],
                "+/-": [15.0],
                "Shots": [250.0],
                "TotalTOIMins": [1600.0],
                "Wins": [0.0],
                "Shutouts": [0.0],
                "Saves": [0.0],
                "WeightedSV": [0.0],
                "WeightedGAA": [0.0],
                "NHLeMultiplier": [1.0],
            }
        )

        with patch("nhl.player_pipeline.get_player_raw_stats", return_value=(raw_df, "Test Skater", "C")):
            _, raw_cache, *_ = process_players(
                players={"99": "Test Skater"},
                metric="Points",
                hist_df=pd.DataFrame(),
                id_to_name_map={},
                clone_details_map={},
                season_type="Regular",
                stat_category="Skater",
                do_era=False,
                do_predict=False,
                do_smooth=False,
                do_cumul=False,
                games_mode=False,
                league_filter=["NHL"],
            )

        self.assertEqual(len(raw_cache), 1)
        self.assertEqual(int(raw_cache[0].iloc[0]["PlayerID"]), 99)
        self.assertEqual(str(raw_cache[0].iloc[0]["PositionCode"]), "C")

    def test_toi_projection_uses_knn_only_when_player_has_modern_toi_history(self):
        """Project TOI only through KNN when the skater has enough 1997+ TOI seasons."""
        raw_df = pd.DataFrame(
            {
                "League": ["NHL", "NHL", "NHL"],
                "Age": [18, 19, 20],
                "SeasonYear": [1997, 1998, 1999],
                "GameType": ["Regular", "Regular", "Regular"],
                "GP": [40.0, 40.0, 45.0],
                "Points": [30.0, 35.0, 40.0],
                "Goals": [10.0, 12.0, 15.0],
                "Assists": [20.0, 23.0, 25.0],
                "PIM": [10.0, 12.0, 14.0],
                "+/-": [1.0, 3.0, 5.0],
                "Shots": [100.0, 110.0, 120.0],
                "TotalTOIMins": [600.0, 680.0, 810.0],
                "Wins": [0.0, 0.0, 0.0],
                "Shutouts": [0.0, 0.0, 0.0],
                "Saves": [0.0, 0.0, 0.0],
                "WeightedSV": [0.0, 0.0, 0.0],
                "WeightedGAA": [0.0, 0.0, 0.0],
                "NHLeMultiplier": [1.0, 1.0, 1.0],
            }
        )
        hist_df = pd.DataFrame(
            {
                "PlayerID": [1],
                "SeasonYear": [1999],
                "Age": [20],
                "Position": ["C"],
                "GP": [45.0],
                "TotalTOIMins": [810.0],
                "TOI": [18.0],
            }
        )

        with patch("nhl.player_pipeline.get_player_raw_stats", return_value=(raw_df, "Test Skater", "C")), patch(
            "nhl.player_pipeline.run_knn_projection",
            return_value=([{"Age": 21, "TOI": 18.5, "Player": "Test Skater", "BaseName": "Test Skater"}], []),
        ) as mock_knn, patch(
            "nhl.player_pipeline.run_linear_fallback",
        ) as mock_fallback:
            processed, *_ = process_players(
                players={"99": "Test Skater"},
                metric="TOI",
                hist_df=hist_df,
                id_to_name_map={},
                clone_details_map={},
                season_type="Regular",
                stat_category="Skater",
                do_era=False,
                do_predict=True,
                do_smooth=False,
                do_cumul=False,
                games_mode=False,
                league_filter=["NHL"],
            )

        mock_knn.assert_called_once()
        mock_fallback.assert_not_called()
        self.assertIn("Test Skater (Proj)", processed[0]["Player"].tolist())
        self.assertAlmostEqual(
            float(processed[0].loc[processed[0]["Age"] == 21, "TOI"].iloc[0]),
            18.5,
        )

    def test_toi_projection_stays_hidden_without_enough_modern_toi_history(self):
        """Hide TOI projection when the skater lacks three 1997+ TOI-bearing seasons."""
        raw_df = pd.DataFrame(
            {
                "League": ["NHL", "NHL", "NHL"],
                "Age": [18, 19, 20],
                "SeasonYear": [1995, 1996, 1997],
                "GameType": ["Regular", "Regular", "Regular"],
                "GP": [40.0, 40.0, 40.0],
                "Points": [20.0, 25.0, 30.0],
                "Goals": [8.0, 10.0, 12.0],
                "Assists": [12.0, 15.0, 18.0],
                "PIM": [10.0, 11.0, 12.0],
                "+/-": [0.0, 1.0, 2.0],
                "Shots": [90.0, 100.0, 110.0],
                "TotalTOIMins": [0.0, 0.0, 650.0],
                "Wins": [0.0, 0.0, 0.0],
                "Shutouts": [0.0, 0.0, 0.0],
                "Saves": [0.0, 0.0, 0.0],
                "WeightedSV": [0.0, 0.0, 0.0],
                "WeightedGAA": [0.0, 0.0, 0.0],
                "NHLeMultiplier": [1.0, 1.0, 1.0],
            }
        )
        hist_df = pd.DataFrame(
            {
                "PlayerID": [1],
                "SeasonYear": [1999],
                "Age": [20],
                "Position": ["C"],
                "GP": [45.0],
                "TotalTOIMins": [810.0],
                "TOI": [18.0],
            }
        )

        with patch("nhl.player_pipeline.get_player_raw_stats", return_value=(raw_df, "Test Skater", "C")), patch(
            "nhl.player_pipeline.run_knn_projection",
        ) as mock_knn, patch(
            "nhl.player_pipeline.run_linear_fallback",
        ) as mock_fallback:
            processed, *_ = process_players(
                players={"99": "Test Skater"},
                metric="TOI",
                hist_df=hist_df,
                id_to_name_map={},
                clone_details_map={},
                season_type="Regular",
                stat_category="Skater",
                do_era=False,
                do_predict=True,
                do_smooth=False,
                do_cumul=False,
                games_mode=False,
                league_filter=["NHL"],
            )

        mock_knn.assert_not_called()
        mock_fallback.assert_not_called()
        self.assertEqual(processed[0]["Player"].tolist(), ["Test Skater", "Test Skater", "Test Skater"])

    def test_non_toi_metrics_still_use_linear_fallback_when_knn_is_unavailable(self):
        """Keep legacy fallback behavior for non-KNN-only metrics like Points."""
        raw_df = pd.DataFrame(
            {
                "League": ["NHL", "NHL"],
                "Age": [18, 19],
                "SeasonYear": [2022, 2023],
                "GameType": ["Regular", "Regular"],
                "GP": [41.0, 41.0],
                "Points": [50.0, 60.0],
                "Goals": [20.0, 25.0],
                "Assists": [30.0, 35.0],
                "PIM": [10.0, 12.0],
                "+/-": [5.0, 7.0],
                "Shots": [150.0, 160.0],
                "TotalTOIMins": [700.0, 760.0],
                "Wins": [0.0, 0.0],
                "Shutouts": [0.0, 0.0],
                "Saves": [0.0, 0.0],
                "WeightedSV": [0.0, 0.0],
                "WeightedGAA": [0.0, 0.0],
                "NHLeMultiplier": [1.0, 1.0],
            }
        )

        with patch("nhl.player_pipeline.get_player_raw_stats", return_value=(raw_df, "Test Skater", "C")), patch(
            "nhl.player_pipeline.run_knn_projection",
        ) as mock_knn, patch(
            "nhl.player_pipeline.run_linear_fallback",
            return_value=[{"Age": 20, "Points": 65.0, "Player": "Test Skater", "BaseName": "Test Skater"}],
        ) as mock_fallback:
            processed, *_ = process_players(
                players={"99": "Test Skater"},
                metric="Points",
                hist_df=pd.DataFrame(),
                id_to_name_map={},
                clone_details_map={},
                season_type="Regular",
                stat_category="Skater",
                do_era=False,
                do_predict=True,
                do_smooth=False,
                do_cumul=False,
                games_mode=False,
                league_filter=["NHL"],
            )

        mock_knn.assert_not_called()
        mock_fallback.assert_called_once()
        self.assertIn("Test Skater (Proj)", processed[0]["Player"].tolist())


if __name__ == "__main__":
    unittest.main()
