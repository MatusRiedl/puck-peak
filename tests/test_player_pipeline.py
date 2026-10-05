import unittest
from unittest.mock import patch

import pandas as pd

from nhl.constants import LIVE_SEASON_SUFFIX, current_season_year
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


class PlayerPipelineAgeModeZeroGPTests(unittest.TestCase):
    """Guard the Age-mode rate-stat denominators against GP == 0 buckets."""

    def test_age_bucket_with_zero_gp_yields_nan_not_inf(self):
        """An age whose GP sums to zero must produce NaN rate stats, never inf.

        The Age-mode branch divides counting stats by the summed GP per age. A
        bucket with GP == 0 but non-zero counting stats previously produced inf;
        the zero-guard turns the denominator into NaN so the rate is NaN instead.
        """
        raw_df = pd.DataFrame(
            {
                "League": ["NHL", "NHL"],
                "Age": [20, 21],
                "SeasonYear": [2021, 2022],
                "GameType": ["Regular", "Regular"],
                "GP": [0, 80],          # age 20 sums to GP == 0
                "Points": [5.0, 80.0],  # but still has points -> would be inf pre-fix
                "Goals": [2.0, 40.0],
                "Assists": [3.0, 40.0],
                "PIM": [0.0, 10.0],
                "+/-": [0.0, 5.0],
                "Shots": [10.0, 200.0],
                "TotalTOIMins": [0.0, 1200.0],
                "Wins": [0.0, 0.0],
                "Shutouts": [0.0, 0.0],
                "Saves": [0.0, 0.0],
                "WeightedSV": [0.0, 0.0],
                "WeightedGAA": [0.0, 0.0],
                "NHLeMultiplier": [1.0, 1.0],
            }
        )

        with patch("nhl.player_pipeline.get_player_raw_stats", return_value=(raw_df, "Test Skater", "C")):
            result, *_ = process_players(
                players={"1": "Test Skater"},
                metric="PPG",
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

        ppg = result[0].set_index("Age")["PPG"]
        self.assertTrue(pd.isna(ppg.loc[20]))           # guarded: NaN, not inf
        self.assertAlmostEqual(float(ppg.loc[21]), 1.0)  # normal bucket unaffected


class PlayerPipelineInProgressSeasonTests(unittest.TestCase):
    """Keep a few-games-old season off the real line and out of the forecast."""

    @staticmethod
    def _raw_df(points: list, gp: list) -> pd.DataFrame:
        """Build NHL regular-season rows ending in the current season.

        Args:
            points: Points per season, oldest first.
            gp: Games played per season, oldest first.

        Returns:
            Raw player frame shaped like `get_player_raw_stats()` output.
        """
        n = len(points)
        last_year = current_season_year()
        zeros = [0.0] * n
        return pd.DataFrame(
            {
                "League": ["NHL"] * n,
                "Age": list(range(27 - n + 1, 28)),
                "SeasonYear": list(range(last_year - n + 1, last_year + 1)),
                "GameType": ["Regular"] * n,
                "GP": gp,
                "Points": points,
                "Goals": [p / 2 for p in points],
                "Assists": [p / 2 for p in points],
                "PIM": zeros, "+/-": zeros, "Shots": zeros, "TotalTOIMins": zeros,
                "Wins": zeros, "Shutouts": zeros, "Saves": zeros,
                "WeightedSV": zeros, "WeightedGAA": zeros,
                "NHLeMultiplier": [1.0] * n,
            }
        )

    def _run(self, raw_df: pd.DataFrame, in_progress: bool = True, do_cumul: bool = False):
        """Run the age-mode pipeline with the linear fallback mocked.

        Args:
            raw_df: Raw player rows.
            in_progress: What `regular_season_in_progress` reports.
            do_cumul: Whether cumulative mode is on.

        Returns:
            Tuple of (processed frame, peak_info, fallback mock).
        """
        with patch("nhl.player_pipeline.get_player_raw_stats", return_value=(raw_df, "Test Skater", "C")), patch(
            "nhl.player_pipeline.regular_season_in_progress", return_value=in_progress,
        ), patch(
            "nhl.player_pipeline.run_linear_fallback",
            side_effect=lambda career_df, metric, max_age, stat_category: [
                {"Age": age, metric: 20.0, "Player": "Test Skater", "BaseName": "Test Skater"}
                for age in range(max_age + 1, 41)
            ],
        ) as mock_fallback:
            processed, _, _, peak_info = process_players(
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
                do_cumul=do_cumul,
                games_mode=False,
                league_filter=["NHL"],
            )
        return processed[0], peak_info, mock_fallback

    def test_partial_season_becomes_a_standalone_point(self):
        """Split the current season off the real line and project from the last full one."""
        frame, peak_info, mock_fallback = self._run(self._raw_df([18, 25, 15, 6], [55, 76, 79, 4]))

        real = frame[frame["Player"] == "Test Skater"]
        live = frame[frame["Player"] == f"Test Skater{LIVE_SEASON_SUFFIX}"]
        proj = frame[frame["Player"] == "Test Skater (Proj)"]

        self.assertEqual(real["Age"].tolist(), [24, 25, 26])
        self.assertEqual(live["Age"].tolist(), [27])
        self.assertAlmostEqual(float(live["Points"].iloc[0]), 6.0)
        self.assertEqual(int(live["GP"].iloc[0]), 4)
        # The forecast starts at the last full season, not at the 4-game one.
        self.assertEqual(int(proj["Age"].min()), 26)
        self.assertAlmostEqual(float(proj.loc[proj["Age"] == 26, "Points"].iloc[0]), 15.0)
        career_df = mock_fallback.call_args.kwargs["career_df"]
        self.assertEqual(int(career_df["Age"].max()), 26)
        self.assertEqual(peak_info["Test Skater"]["age"], 25)

    def test_cumulative_point_is_the_career_total_to_date(self):
        """Plot the in-progress point at the real career total, not a pace."""
        frame, _, _ = self._run(self._raw_df([18, 25, 15, 6], [55, 76, 79, 4]), do_cumul=True)

        live = frame[frame["Player"] == f"Test Skater{LIVE_SEASON_SUFFIX}"]
        real = frame[frame["Player"] == "Test Skater"]
        self.assertAlmostEqual(float(real["Points"].iloc[-1]), 58.0)
        self.assertAlmostEqual(float(live["Points"].iloc[0]), 64.0)

    def test_finished_season_stays_on_the_real_line(self):
        """Leave the latest season alone once the regular season is over."""
        frame, _, _ = self._run(self._raw_df([18, 25, 15, 70], [55, 76, 79, 82]), in_progress=False)

        self.assertFalse(frame["Player"].str.endswith(LIVE_SEASON_SUFFIX).any())
        self.assertEqual(frame.loc[frame["Player"] == "Test Skater", "Age"].tolist(), [24, 25, 26, 27])

    def test_single_partial_season_stays_a_real_point(self):
        """Keep a first-year player's only season on the real trace."""
        frame, _, _ = self._run(self._raw_df([6], [4]))

        self.assertFalse(frame["Player"].str.endswith(LIVE_SEASON_SUFFIX).any())
        self.assertEqual(frame["Age"].tolist(), [27])


if __name__ == "__main__":
    unittest.main()
