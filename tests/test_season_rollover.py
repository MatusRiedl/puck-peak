"""Cover the date-derived NHL season helpers in nhl.constants.

These are the single source of truth for "which season is it", and they are called
rather than read as a constant so a long-lived container cannot freeze the season at
its start date and serve the wrong year for months.
"""

import unittest
from datetime import date

from nhl.constants import (
    current_season_id,
    current_season_year,
    previous_season_year,
    season_year_to_id,
)


class SeasonHelperTests(unittest.TestCase):
    """Verify season math across the rollover and the full season cycle."""

    def test_season_year_across_the_cycle(self):
        """Every phase of 2026-27 resolves to the same season start year.

        Args:
            None.

        Returns:
            None.
        """
        cases = [
            (date(2026, 9, 6), 2026),    # offseason gap: season named, no games yet
            (date(2026, 9, 20), 2026),   # preseason
            (date(2026, 10, 1), 2026),   # regular season
            (date(2027, 1, 15), 2026),   # new calendar year, same season
            (date(2027, 5, 1), 2026),    # playoffs
        ]
        for day, expected in cases:
            with self.subTest(day=day):
                self.assertEqual(current_season_year(day), expected)

    def test_rollover_boundary(self):
        """The season year advances on the first day of the rollover month.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(current_season_year(date(2026, 8, 31)), 2025)
        self.assertEqual(current_season_year(date(2026, 9, 1)), 2026)

    def test_season_year_to_id(self):
        """Start years convert to eight-digit NHL season ids.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(season_year_to_id(2026), 20262027)
        self.assertEqual(season_year_to_id(1999), 19992000)
        self.assertEqual(season_year_to_id(2009), 20092010)

    def test_current_and_previous_season_id(self):
        """Current and previous season helpers agree with the start year.

        Args:
            None.

        Returns:
            None.
        """
        day = date(2026, 10, 1)
        self.assertEqual(current_season_id(day), 20262027)
        self.assertEqual(previous_season_year(day), 2025)

    def test_helpers_are_evaluated_per_call(self):
        """The value tracks the date argument rather than import time.

        This is the regression guard for the original bug: a module-level constant
        computed at import froze the season for the life of the process.

        Args:
            None.

        Returns:
            None.
        """
        self.assertNotEqual(
            current_season_year(date(2025, 10, 1)),
            current_season_year(date(2026, 10, 1)),
        )


if __name__ == "__main__":
    unittest.main()
