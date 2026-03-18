import unittest
from unittest.mock import patch

import pandas as pd

from nhl import baselines


class BaselineCacheBoundaryTests(unittest.TestCase):
    """Cover the zero-arg cached baseline wrappers.

    Args:
        None.

    Returns:
        None.
    """

    def setUp(self):
        """Clear baseline caches before each test.

        Args:
            None.

        Returns:
            None.
        """
        baselines.get_historical_baselines.clear()
        baselines.get_team_baselines.clear()

    def tearDown(self):
        """Clear baseline caches after each test.

        Args:
            None.

        Returns:
            None.
        """
        baselines.get_historical_baselines.clear()
        baselines.get_team_baselines.clear()

    def test_historical_baselines_wrapper_uses_internal_loader_once(self):
        """Cache historical baselines behind a zero-arg wrapper.

        Args:
            None.

        Returns:
            None.
        """
        hist_df = pd.DataFrame(
            {
                "Position": ["S", "S", "G", "G"],
                "GP": [82, 60, 30, 28],
                "Age": [24, 25, 24, 25],
                "Points": [95.0, 90.0, 0.0, 0.0],
                "Save %": [0.0, 0.0, 91.2, 91.0],
                "GAA": [0.0, 0.0, 2.5, 2.6],
            }
        )

        with patch("nhl.baselines.load_historical_data", return_value=hist_df) as mock_load:
            first = baselines.get_historical_baselines()
            second = baselines.get_historical_baselines()

        self.assertEqual(mock_load.call_count, 1)
        self.assertEqual(set(first.keys()), {"Skater", "Goalie"})
        self.assertEqual(set(second.keys()), {"Skater", "Goalie"})

    def test_team_baselines_wrapper_uses_internal_loader_once(self):
        """Cache team baselines behind a zero-arg wrapper.

        Args:
            None.

        Returns:
            None.
        """
        team_df = pd.DataFrame(
            {
                "gameTypeId": [2, 2, 2, 2],
                "SeasonYear": [2023, 2023, 2024, 2024],
                "Wins": [52, 48, 50, 46],
                "Points": [112, 104, 108, 100],
                "GF/G": [3.5, 3.2, 3.4, 3.1],
                "GA/G": [2.7, 2.9, 2.8, 3.0],
                "Goals": [287, 262, 279, 254],
                "PPG": [0.82, 0.74, 0.80, 0.71],
                "PP%": [25.0, 22.4, 24.6, 21.8],
                "GP": [82, 82, 82, 82],
                "Win%": [68.3, 63.4, 65.9, 61.0],
            }
        )

        with patch("nhl.baselines.load_all_team_seasons", return_value=team_df) as mock_load:
            first = baselines.get_team_baselines()
            second = baselines.get_team_baselines()

        self.assertEqual(mock_load.call_count, 1)
        self.assertIn(2023, first)
        self.assertIn(2024, second)


if __name__ == "__main__":
    unittest.main()