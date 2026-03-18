import unittest

from nhl.controls import _CONTROL_PILL_SPECS, _get_control_pill_groups


class ControlsTests(unittest.TestCase):
    """Validate compact control-toolbar behavior."""

    def test_control_pill_labels_stay_short_and_ordered(self):
        """Keep the toolbar labels compact and predictable.

        Args:
            None.

        Returns:
            None.
        """
        labels = [spec["label"] for spec in _CONTROL_PILL_SPECS]

        self.assertEqual(labels, ["Smooth", "Proj", "Era", "Cumul", "Base", "Prime"])

    def test_games_mode_hides_projection_and_baseline_pills(self):
        """Keep Games Played mode from offering unavailable player options.

        Args:
            None.

        Returns:
            None.
        """
        available, unavailable = _get_control_pill_groups(
            team_mode=False,
            games_mode=True,
            season_mode=False,
        )

        self.assertEqual([spec["label"] for spec in available], ["Smooth", "Era", "Cumul", "Prime"])
        self.assertEqual(unavailable, ["Proj", "Base"])

    def test_team_mode_keeps_only_smooth_and_cumul_available(self):
        """Keep Team mode focused on the options that actually work there.

        Args:
            None.

        Returns:
            None.
        """
        available, unavailable = _get_control_pill_groups(
            team_mode=True,
            games_mode=False,
            season_mode=False,
        )

        self.assertEqual([spec["label"] for spec in available], ["Smooth", "Cumul"])
        self.assertEqual(unavailable, ["Proj", "Era", "Base", "Prime"])

    def test_team_selected_season_hides_cumul_with_other_unavailable_options(self):
        """Keep team season-progress mode on a fixed running-value interpretation."""
        available, unavailable = _get_control_pill_groups(
            team_mode=True,
            games_mode=True,
            season_mode=True,
        )

        self.assertEqual([spec["label"] for spec in available], ["Smooth"])
        self.assertEqual(unavailable, ["Proj", "Era", "Cumul", "Base", "Prime"])


if __name__ == "__main__":
    unittest.main()
