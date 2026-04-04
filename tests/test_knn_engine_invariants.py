import unittest

import pandas as pd

from nhl.constants import CURRENT_SEASON_YEAR
from nhl.knn_engine import _apply_stat_cap, run_knn_projection, run_linear_fallback


class KNNEngineInvariantTests(unittest.TestCase):
    """Cover the highest-risk KNN and fallback projection rules."""

    def test_apply_stat_cap_respects_goalie_caps_and_floors(self):
        """Apply per-metric caps and floors exactly as documented.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(_apply_stat_cap(70.0, "GP", "Goalie"), 65)
        self.assertEqual(_apply_stat_cap(90.0, "GP", "Skater"), 82)
        self.assertEqual(_apply_stat_cap(1.2, "GAA", "Goalie"), 1.8)
        self.assertEqual(_apply_stat_cap(-90.0, "+/-", "Skater"), -60)

    def test_run_linear_fallback_uses_goalie_gp_durability_curve(self):
        """Project goalie GP with the staged durability decay bands.

        Args:
            None.

        Returns:
            None.
        """
        career_df = pd.DataFrame(
            {
                "Age": [32, 33, 34],
                "GP": [60.0, 60.0, 60.0],
                "BaseName": ["Test Goalie"] * 3,
            }
        )

        proj_rows = run_linear_fallback(career_df, "GP", max_age=34, stat_category="Goalie")

        self.assertAlmostEqual(float(proj_rows[0]["GP"]), 57.9)
        self.assertAlmostEqual(float(proj_rows[3]["GP"]), 50.143672575000004)
        self.assertTrue(all(float(row["GP"]) <= 65 for row in proj_rows))

    def test_run_knn_projection_clamps_counting_growth_to_25_percent(self):
        """Clamp large counting-stat growth even when clone averages spike.

        Args:
            None.

        Returns:
            None.
        """
        career_df = pd.DataFrame(
            {
                "Age": [30, 31],
                "Points": [10.0, 20.0],
                "SeasonYear": [2019, 2020],
                "GP": [82.0, 82.0],
                "BaseName": ["Test Skater", "Test Skater"],
            }
        )
        hist_rows = []
        for player_id in range(1, 11):
            hist_rows.extend(
                [
                    {"PlayerID": player_id, "Position": "C", "Age": 30, "SeasonYear": 2010, "Points": 10.0},
                    {"PlayerID": player_id, "Position": "C", "Age": 31, "SeasonYear": 2011, "Points": 20.0},
                    {"PlayerID": player_id, "Position": "C", "Age": 32, "SeasonYear": 2012, "Points": 40.0},
                ]
            )

        proj_rows, clone_names = run_knn_projection(
            career_df=career_df,
            metric="Points",
            hist_df=pd.DataFrame(hist_rows),
            is_goalie=False,
            pos_code="C",
            do_era=False,
            season_type="Regular",
            stat_category="Skater",
            id_to_name_map={player_id: f"Clone {player_id}" for player_id in range(1, 11)},
            clone_details_map={},
        )

        self.assertEqual(len(clone_names), 10)
        self.assertEqual(proj_rows[0]["Age"], 32)
        self.assertAlmostEqual(float(proj_rows[0]["Points"]), 25.0)

    def test_run_knn_projection_uses_80_20_clone_prior_blend(self):
        """Blend clone targets with an 80/20 clone-prior split before deltas.

        Args:
            None.

        Returns:
            None.
        """
        career_df = pd.DataFrame(
            {
                "Age": [30, 31],
                "Points": [10.0, 20.0],
                "SeasonYear": [2019, 2020],
                "GP": [82.0, 82.0],
                "BaseName": ["Blend Test", "Blend Test"],
            }
        )
        hist_rows = []
        for player_id in range(1, 11):
            hist_rows.extend(
                [
                    {"PlayerID": player_id, "Position": "C", "Age": 30, "SeasonYear": 2010, "Points": 10.0},
                    {"PlayerID": player_id, "Position": "C", "Age": 31, "SeasonYear": 2011, "Points": 20.0},
                    {"PlayerID": player_id, "Position": "C", "Age": 32, "SeasonYear": 2012, "Points": 22.0},
                ]
            )

        proj_rows, clone_names = run_knn_projection(
            career_df=career_df,
            metric="Points",
            hist_df=pd.DataFrame(hist_rows),
            is_goalie=False,
            pos_code="C",
            do_era=False,
            season_type="Regular",
            stat_category="Skater",
            id_to_name_map={player_id: f"Clone {player_id}" for player_id in range(1, 11)},
            clone_details_map={},
        )

        self.assertEqual(len(clone_names), 10)
        self.assertEqual(proj_rows[0]["Age"], 32)
        self.assertAlmostEqual(float(proj_rows[0]["Points"]), 21.6)

    def test_run_knn_projection_uses_mean_for_goalie_rate_stats(self):
        """Aggregate goalie rate stats by mean rather than sum during pivoting.

        Args:
            None.

        Returns:
            None.
        """
        career_df = pd.DataFrame(
            {
                "Age": [30, 31],
                "Save %": [90.0, 91.0],
                "SeasonYear": [2019, 2020],
                "GP": [50.0, 50.0],
                "BaseName": ["Test Goalie", "Test Goalie"],
            }
        )
        hist_rows = []
        for player_id in range(1, 11):
            hist_rows.extend(
                [
                    {"PlayerID": player_id, "Position": "G", "Age": 30, "SeasonYear": 2010, "Save %": 90.0},
                    {"PlayerID": player_id, "Position": "G", "Age": 31, "SeasonYear": 2011, "Save %": 90.0},
                    {"PlayerID": player_id, "Position": "G", "Age": 31, "SeasonYear": 2011, "Save %": 92.0},
                    {"PlayerID": player_id, "Position": "G", "Age": 32, "SeasonYear": 2012, "Save %": 95.0},
                ]
            )

        proj_rows, clone_names = run_knn_projection(
            career_df=career_df,
            metric="Save %",
            hist_df=pd.DataFrame(hist_rows),
            is_goalie=True,
            pos_code="G",
            do_era=False,
            season_type="Regular",
            stat_category="Goalie",
            id_to_name_map={player_id: f"Goalie Clone {player_id}" for player_id in range(1, 11)},
            clone_details_map={},
        )

        self.assertEqual(len(clone_names), 10)
        self.assertEqual(proj_rows[0]["Age"], 32)
        self.assertAlmostEqual(float(proj_rows[0]["Save %"]), 91.6)

    def test_run_knn_projection_allows_small_dense_skater_late_uptick(self):
        """Allow a modest age-36+ skater bump when at least four clones support it.

        Args:
            None.

        Returns:
            None.
        """
        career_df = pd.DataFrame(
            {
                "Age": [34, 35],
                "Points": [18.0, 20.0],
                "SeasonYear": [2019, 2020],
                "GP": [82.0, 82.0],
                "BaseName": ["Test Skater", "Test Skater"],
            }
        )
        hist_rows = []
        for player_id in range(1, 11):
            hist_rows.extend(
                [
                    {"PlayerID": player_id, "Position": "C", "Age": 34, "SeasonYear": 2010, "Points": 18.0},
                    {"PlayerID": player_id, "Position": "C", "Age": 35, "SeasonYear": 2011, "Points": 20.0},
                ]
            )
            if player_id <= 4:
                hist_rows.append(
                    {"PlayerID": player_id, "Position": "C", "Age": 36, "SeasonYear": 2012, "Points": 30.0}
                )

        proj_rows, clone_names = run_knn_projection(
            career_df=career_df,
            metric="Points",
            hist_df=pd.DataFrame(hist_rows),
            is_goalie=False,
            pos_code="C",
            do_era=False,
            season_type="Regular",
            stat_category="Skater",
            id_to_name_map={player_id: f"Clone {player_id}" for player_id in range(1, 11)},
            clone_details_map={},
        )

        self.assertEqual(len(clone_names), 10)
        self.assertEqual(proj_rows[0]["Age"], 36)
        self.assertAlmostEqual(float(proj_rows[0]["Points"]), 25.0)

    def test_run_knn_projection_paces_integer_current_season_without_dtype_error(self):
        """Allow partial-season pacing on integer counting stats without crashing."""
        career_df = pd.DataFrame(
            {
                "Age": [30, 31],
                "Points": [20, 25],
                "SeasonYear": [CURRENT_SEASON_YEAR - 1, CURRENT_SEASON_YEAR],
                "GP": [82, 41],
                "BaseName": ["Landing Page Skater", "Landing Page Skater"],
            }
        )
        hist_rows = []
        for player_id in range(1, 11):
            hist_rows.extend(
                [
                    {"PlayerID": player_id, "Position": "C", "Age": 30, "SeasonYear": 2010, "Points": 20.0},
                    {"PlayerID": player_id, "Position": "C", "Age": 31, "SeasonYear": 2011, "Points": 50.0},
                    {"PlayerID": player_id, "Position": "C", "Age": 32, "SeasonYear": 2012, "Points": 60.0},
                ]
            )

        proj_rows, clone_names = run_knn_projection(
            career_df=career_df,
            metric="Points",
            hist_df=pd.DataFrame(hist_rows),
            is_goalie=False,
            pos_code="C",
            do_era=False,
            season_type="Regular",
            stat_category="Skater",
            id_to_name_map={player_id: f"Clone {player_id}" for player_id in range(1, 11)},
            clone_details_map={},
        )

        self.assertEqual(len(clone_names), 10)
        self.assertEqual(proj_rows[0]["Age"], 32)
        self.assertGreater(float(proj_rows[0]["Points"]), 0.0)


if __name__ == "__main__":
    unittest.main()
