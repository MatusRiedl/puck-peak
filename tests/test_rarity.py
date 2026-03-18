import unittest
from unittest.mock import patch

import pandas as pd

import nhl.rarity as rarity


class RarityTests(unittest.TestCase):
    """Cover age-rarity ranking math and snapshot-row collapsing."""

    def setUp(self):
        rarity.get_age_rarity_summary.clear()

    def tearDown(self):
        rarity.get_age_rarity_summary.clear()

    def test_skater_points_returns_overall_and_role_split_rarity(self):
        hist_df = pd.DataFrame(
            {
                "PlayerID": [1, 2, 3, 4],
                "SeasonYear": [2018, 2019, 2020, 2021],
                "Age": [25, 25, 25, 25],
                "Position": ["C", "L", "D", "C"],
                "GP": [82, 82, 82, 82],
                "Points": [80.0, 100.0, 120.0, 110.0],
                "Goals": [30.0, 35.0, 20.0, 40.0],
                "Assists": [50.0, 65.0, 100.0, 70.0],
                "PIM": [20.0, 10.0, 40.0, 25.0],
                "+/-": [5.0, 10.0, 15.0, 12.0],
                "Shots": [200.0, 220.0, 150.0, 250.0],
                "TotalTOIMins": [1400.0, 1450.0, 1800.0, 1500.0],
                "Wins": [0.0, 0.0, 0.0, 0.0],
                "Shutouts": [0.0, 0.0, 0.0, 0.0],
                "Saves": [0.0, 0.0, 0.0, 0.0],
                "SavePct": [0.0, 0.0, 0.0, 0.0],
                "GAA": [0.0, 0.0, 0.0, 0.0],
            }
        )
        season_row = {
            "SeasonYear": 2022,
            "League": "NHL",
            "GameType": "Regular",
            "Age": 25,
            "PlayerID": 97,
            "PositionCode": "C",
            "GP": 82.0,
            "Points": 110.0,
            "Goals": 40.0,
            "Assists": 70.0,
            "PIM": 18.0,
            "+/-": 14.0,
            "Shots": 250.0,
            "TotalTOIMins": 1600.0,
            "Wins": 0.0,
            "Shutouts": 0.0,
            "Saves": 0.0,
            "WeightedSV": 0.0,
            "WeightedGAA": 0.0,
        }

        with patch.object(rarity, "load_historical_data", return_value=hist_df), patch.object(
            rarity,
            "_resolve_player_name",
            side_effect=lambda player_id: {
                1: "Player One",
                2: "Player Two",
                3: "Player Three",
                4: "Player Four",
            }.get(int(player_id), f"Player {player_id}"),
        ):
            summary = rarity.get_age_rarity_summary(season_row, "Points", "Skater", do_era=False)

        self.assertEqual(summary["rank"], 2)
        self.assertAlmostEqual(float(summary["percentile"]), 62.5)
        self.assertEqual(summary["sample_size"], 4)
        self.assertEqual(summary["role_label"], "forwards")
        self.assertEqual(summary["role_rank"], 1)
        self.assertAlmostEqual(float(summary["role_percentile"]), (2.5 / 3.0) * 100.0)
        self.assertEqual(len(summary["top_seasons"]), 4)
        self.assertEqual(summary["top_seasons"][0]["player_name"], "Player Three")
        self.assertEqual(summary["top_seasons"][0]["display_rank"], 1)
        self.assertEqual(summary["top_seasons"][0]["value"], 120.0)

    def test_goalie_gaa_ranks_lower_values_as_better(self):
        hist_df = pd.DataFrame(
            {
                "PlayerID": [10, 11, 12, 13],
                "SeasonYear": [2018, 2019, 2020, 2021],
                "Age": [28, 28, 28, 28],
                "Position": ["G", "G", "G", "G"],
                "GP": [40, 40, 40, 40],
                "Points": [0.0, 0.0, 0.0, 0.0],
                "Goals": [0.0, 0.0, 0.0, 0.0],
                "Assists": [0.0, 0.0, 0.0, 0.0],
                "PIM": [0.0, 0.0, 0.0, 0.0],
                "+/-": [0.0, 0.0, 0.0, 0.0],
                "Shots": [0.0, 0.0, 0.0, 0.0],
                "TotalTOIMins": [0.0, 0.0, 0.0, 0.0],
                "Wins": [30.0, 28.0, 27.0, 24.0],
                "Shutouts": [5.0, 4.0, 3.0, 2.0],
                "Saves": [1200.0, 1180.0, 1190.0, 1100.0],
                "SavePct": [0.920, 0.915, 0.918, 0.905],
                "GAA": [2.0, 2.5, 2.5, 3.0],
            }
        )
        season_row = {
            "SeasonYear": 2022,
            "League": "NHL",
            "GameType": "Regular",
            "Age": 28,
            "PlayerID": 31,
            "PositionCode": "G",
            "GP": 45.0,
            "Points": 0.0,
            "Goals": 0.0,
            "Assists": 0.0,
            "PIM": 0.0,
            "+/-": 0.0,
            "Shots": 0.0,
            "TotalTOIMins": 0.0,
            "Wins": 32.0,
            "Shutouts": 4.0,
            "Saves": 1300.0,
            "WeightedSV": 91.5 * 45.0,
            "WeightedGAA": 2.5 * 45.0,
        }

        with patch.object(rarity, "load_historical_data", return_value=hist_df), patch.object(
            rarity,
            "_resolve_player_name",
            side_effect=lambda player_id: f"Goalie {player_id}",
        ):
            summary = rarity.get_age_rarity_summary(season_row, "GAA", "Goalie", do_era=False)

        self.assertEqual(summary["rank"], 2)
        self.assertAlmostEqual(float(summary["percentile"]), 50.0)
        self.assertEqual(summary["sample_size"], 4)

    def test_era_adjusted_skater_rarity_changes_percentile(self):
        hist_df = pd.DataFrame(
            {
                "PlayerID": [1, 2],
                "SeasonYear": [1985, 2022],
                "Age": [25, 25],
                "Position": ["C", "C"],
                "GP": [80, 82],
                "Points": [100.0, 95.0],
                "Goals": [45.0, 35.0],
                "Assists": [55.0, 60.0],
                "PIM": [20.0, 12.0],
                "+/-": [8.0, 9.0],
                "Shots": [250.0, 230.0],
                "TotalTOIMins": [1500.0, 1480.0],
                "Wins": [0.0, 0.0],
                "Shutouts": [0.0, 0.0],
                "Saves": [0.0, 0.0],
                "SavePct": [0.0, 0.0],
                "GAA": [0.0, 0.0],
            }
        )
        season_row = {
            "SeasonYear": 1985,
            "League": "NHL",
            "GameType": "Regular",
            "Age": 25,
            "PlayerID": 9,
            "PositionCode": "C",
            "GP": 80.0,
            "Points": 100.0,
            "Goals": 45.0,
            "Assists": 55.0,
            "PIM": 18.0,
            "+/-": 7.0,
            "Shots": 250.0,
            "TotalTOIMins": 1500.0,
            "Wins": 0.0,
            "Shutouts": 0.0,
            "Saves": 0.0,
            "WeightedSV": 0.0,
            "WeightedGAA": 0.0,
        }

        with patch.object(rarity, "load_historical_data", return_value=hist_df), patch.object(
            rarity,
            "_resolve_player_name",
            side_effect=lambda player_id: f"Player {player_id}",
        ):
            raw_summary = rarity.get_age_rarity_summary(season_row, "Points", "Skater", do_era=False)
            rarity.get_age_rarity_summary.clear()
            era_summary = rarity.get_age_rarity_summary(season_row, "Points", "Skater", do_era=True)

        self.assertTrue(era_summary["is_era_adjusted"])
        self.assertGreater(float(raw_summary["percentile"]), float(era_summary["percentile"]))

    def test_collapse_player_snapshot_rows_merges_traded_season_stints(self):
        season_rows = pd.DataFrame(
            {
                "SeasonYear": [2022, 2022],
                "League": ["NHL", "NHL"],
                "GameType": ["Regular", "Regular"],
                "Age": [25, 25],
                "PlayerID": [97, 97],
                "PositionCode": ["C", "C"],
                "GP": [40.0, 42.0],
                "Points": [70.0, 83.0],
                "Goals": [30.0, 34.0],
                "Assists": [40.0, 49.0],
                "PIM": [8.0, 10.0],
                "+/-": [10.0, 12.0],
                "Shots": [120.0, 130.0],
                "TotalTOIMins": [780.0, 820.0],
                "Wins": [0.0, 0.0],
                "Shutouts": [0.0, 0.0],
                "Saves": [0.0, 0.0],
                "WeightedSV": [0.0, 0.0],
                "WeightedGAA": [0.0, 0.0],
            }
        )

        collapsed = rarity.collapse_player_snapshot_rows(season_rows)

        self.assertEqual(len(collapsed), 1)
        self.assertEqual(int(collapsed.iloc[0]["Points"]), 153)
        self.assertEqual(int(collapsed.iloc[0]["GP"]), 82)
        self.assertAlmostEqual(float(collapsed.iloc[0]["PPG"]), 153.0 / 82.0)
        self.assertAlmostEqual(float(collapsed.iloc[0]["SH%"]), (64.0 / 250.0) * 100.0)
        self.assertAlmostEqual(float(collapsed.iloc[0]["TOI"]), 1600.0 / 82.0)

    def test_non_nhl_or_playoff_rows_return_unavailable_reason(self):
        hist_df = pd.DataFrame(
            {
                "PlayerID": [1],
                "SeasonYear": [2022],
                "Age": [25],
                "Position": ["C"],
                "GP": [82],
                "Points": [100.0],
                "Goals": [40.0],
                "Assists": [60.0],
                "PIM": [20.0],
                "+/-": [10.0],
                "Shots": [200.0],
                "TotalTOIMins": [1400.0],
                "Wins": [0.0],
                "Shutouts": [0.0],
                "Saves": [0.0],
                "SavePct": [0.0],
                "GAA": [0.0],
            }
        )

        with patch.object(rarity, "load_historical_data", return_value=hist_df):
            non_nhl = rarity.get_age_rarity_summary(
                {
                    "SeasonYear": 2022,
                    "League": "AHL",
                    "GameType": "Regular",
                    "Age": 25,
                    "PlayerID": 9,
                    "PositionCode": "C",
                    "GP": 70.0,
                    "Points": 90.0,
                    "Goals": 35.0,
                    "Assists": 55.0,
                    "Shots": 180.0,
                    "TotalTOIMins": 1200.0,
                    "WeightedSV": 0.0,
                    "WeightedGAA": 0.0,
                },
                "Points",
                "Skater",
                do_era=False,
            )
            rarity.get_age_rarity_summary.clear()
            playoff = rarity.get_age_rarity_summary(
                {
                    "SeasonYear": 2022,
                    "League": "NHL",
                    "GameType": "Playoffs",
                    "Age": 25,
                    "PlayerID": 9,
                    "PositionCode": "C",
                    "GP": 18.0,
                    "Points": 24.0,
                    "Goals": 10.0,
                    "Assists": 14.0,
                    "Shots": 50.0,
                    "TotalTOIMins": 350.0,
                    "WeightedSV": 0.0,
                    "WeightedGAA": 0.0,
                },
                "Points",
                "Skater",
                do_era=False,
            )

        self.assertIn("NHL regular-season", non_nhl["unavailable_reason"])
        self.assertIn("NHL regular-season", playoff["unavailable_reason"])


if __name__ == "__main__":
    unittest.main()
