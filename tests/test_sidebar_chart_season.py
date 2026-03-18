import unittest
from pathlib import Path

from nhl.comparison import _format_chart_season_label


class ChartSeasonPickerTests(unittest.TestCase):
    """Cover the current chart-season picker wiring."""

    def test_chart_season_picker_is_rendered_above_the_main_chart(self):
        """Keep the selector out of the sidebar and above the chart in the left column."""
        repo_root = Path(__file__).resolve().parents[1]
        app_text = (repo_root / "app.py").read_text(encoding="utf-8")
        comparison_text = (repo_root / "nhl" / "comparison.py").read_text(encoding="utf-8")
        sidebar_text = (repo_root / "nhl" / "sidebar.py").read_text(encoding="utf-8")

        self.assertIn("with sub_col1:", app_text)
        self.assertIn("render_chart_season_picker(chart_season_options)", app_text)
        self.assertIn("comparison-season-filter", comparison_text)
        self.assertIn('key="_chart_season_picker"', comparison_text)
        self.assertNotIn("_render_chart_season_selector", sidebar_text)

    def test_format_chart_season_label_uses_current_ui_copy(self):
        """Format raw season years into normal NHL season labels."""
        self.assertEqual(_format_chart_season_label("All"), "Whole career")
        self.assertEqual(_format_chart_season_label(2024), "2024-25")
        self.assertEqual(_format_chart_season_label("2025"), "2025-26")


if __name__ == "__main__":
    unittest.main()
