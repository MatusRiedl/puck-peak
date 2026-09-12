import unittest
from unittest.mock import patch

import pandas as pd

from nhl import knn_engine
from nhl.constants import current_season_year, season_games
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
        # The skater GP cap tracks the current schedule length (84 from 2026-27, 82
        # before) rather than a frozen 82, so a legitimate 83rd or 84th game is not
        # clipped away.
        self.assertEqual(_apply_stat_cap(90.0, "GP", "Skater"), season_games())
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
                "SeasonYear": [current_season_year() - 1, current_season_year()],
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


class KNNProjectionMemoTests(unittest.TestCase):
    """Cover the run_knn_projection memo, where a wrong hit is a correctness bug."""

    def setUp(self):
        """Start every case from an empty memo."""
        knn_engine._KNN_PROJECTION_MEMO.clear()

    @staticmethod
    def _fixture(points=(10.0, 20.0)):
        """Build a minimal career frame, clone pool, and call kwargs."""
        career_df = pd.DataFrame(
            {
                "Age": [30, 31],
                "Points": list(points),
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
        return career_df, dict(
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

    def test_repeat_call_hits_the_memo_without_recomputing(self):
        """A second identical call must reuse the cached projection."""
        career_df, kwargs = self._fixture()

        first = run_knn_projection(career_df=career_df, **kwargs)
        self.assertEqual(len(knn_engine._KNN_PROJECTION_MEMO), 1)

        with patch.object(knn_engine, "_run_knn_projection_uncached") as mock_uncached:
            second = run_knn_projection(career_df=career_df, **kwargs)

        mock_uncached.assert_not_called()
        self.assertEqual(first, second)

    def test_cached_result_is_copied_so_callers_cannot_poison_the_memo(self):
        """Callers append proj_rows into frames and stash clone_names; hand them copies."""
        career_df, kwargs = self._fixture()

        first_rows, first_clones = run_knn_projection(career_df=career_df, **kwargs)
        first_rows.append({"Age": 99, "Points": -1.0})
        first_rows[0]["Points"] = -12345.0
        first_clones.append("bogus clone")

        second_rows, second_clones = run_knn_projection(career_df=career_df, **kwargs)

        self.assertNotIn({"Age": 99, "Points": -1.0}, second_rows)
        self.assertNotIn("bogus clone", second_clones)
        self.assertNotEqual(float(second_rows[0]["Points"]), -12345.0)

    def test_a_single_changed_value_misses_the_memo(self):
        """The fingerprint hashes cell values, so near-identical careers differ.

        A summary fingerprint such as (name, row count, sum) would collide here and
        silently serve the wrong player's projection.
        """
        career_a, kwargs = self._fixture(points=(10.0, 20.0))
        career_b, _ = self._fixture(points=(10.0, 20.5))

        rows_a, _ = run_knn_projection(career_df=career_a, **kwargs)
        rows_b, _ = run_knn_projection(career_df=career_b, **kwargs)

        self.assertEqual(len(knn_engine._KNN_PROJECTION_MEMO), 2)
        self.assertNotEqual(float(rows_a[0]["Points"]), float(rows_b[0]["Points"]))

    def test_changing_any_scalar_argument_misses_the_memo(self):
        """Metric, season type, era flag and friends are all part of the key."""
        career_df, kwargs = self._fixture()
        run_knn_projection(career_df=career_df, **kwargs)

        for override in ({"do_era": True}, {"season_type": "Playoffs"}, {"pos_code": "D"}):
            before = len(knn_engine._KNN_PROJECTION_MEMO)
            run_knn_projection(career_df=career_df, **{**kwargs, **override})
            self.assertEqual(
                len(knn_engine._KNN_PROJECTION_MEMO), before + 1, f"{override} must miss the memo"
            )

    def test_memo_is_bounded_and_evicts_oldest_entries(self):
        """Entries accumulate per player/metric/toggle combo, so the cap must hold."""
        _, kwargs = self._fixture()

        for index in range(knn_engine._KNN_PROJECTION_MEMO_MAX_ENTRIES + 10):
            career_df, _ = self._fixture(points=(10.0, 20.0 + index))
            run_knn_projection(career_df=career_df, **kwargs)

        self.assertLessEqual(
            len(knn_engine._KNN_PROJECTION_MEMO),
            knn_engine._KNN_PROJECTION_MEMO_MAX_ENTRIES,
        )


if __name__ == "__main__":
    unittest.main()
