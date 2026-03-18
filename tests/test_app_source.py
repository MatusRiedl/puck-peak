import unittest
from pathlib import Path


class AppSourceTests(unittest.TestCase):
    """Cover app-level state restoration and cleanup invariants."""

    def test_app_restores_pre_season_state_through_shared_helper(self):
        """Keep the season-mode override restoration centralized."""
        app_text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

        self.assertIn("def _restore_pre_season_state() -> None:", app_text)
        self.assertIn('saved_do_era = st.session_state.get("_pre_season_do_era")', app_text)
        self.assertIn("st.session_state._pre_season_do_era = None", app_text)
        self.assertEqual(app_text.count("\n    _restore_pre_season_state()"), 2)

    def test_invalid_chart_season_path_uses_the_same_restore_helper(self):
        """Restore `do_era` as well as x-axis and league state when season selection is invalid."""
        app_text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

        invalid_block_start = app_text.index("if st.session_state.chart_season not in chart_season_options:")
        invalid_block_text = app_text[invalid_block_start: invalid_block_start + 220]

        self.assertIn('_restore_pre_season_state()', invalid_block_text)
        self.assertNotIn("team_sel_abbr", app_text)

    def test_app_resets_dialog_guard_and_pre_mounts_matchup_trigger_before_chart(self):
        """Wire the chart and predictions rail through one shared matchup trigger."""
        app_text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

        self.assertIn('st.session_state["_dialog_opened_this_run"] = False', app_text)
        self.assertIn("has_pending_matchup_history_dialog_request", app_text)
        self.assertIn("_mount_matchup_history_click_bridge", app_text)

        mount_idx = app_text.index("matchup_history_trigger_value = _mount_matchup_history_click_bridge()")
        chart_idx = app_text.index("render_chart(")
        predictions_idx = app_text.index("render_predictions_panel(")

        self.assertLess(mount_idx, chart_idx)
        self.assertIn("suppress_dialogs", app_text[chart_idx: chart_idx + 1200])
        self.assertIn(
            "has_pending_matchup_history_dialog_request(matchup_history_trigger_value)",
            app_text[chart_idx: chart_idx + 1200],
        )
        self.assertIn(
            "matchup_history_trigger_value=matchup_history_trigger_value",
            app_text[predictions_idx: predictions_idx + 400],
        )
        self.assertIn(
            "matchup_history_bridge_mounted=True",
            app_text[predictions_idx: predictions_idx + 400],
        )

    def test_app_imports_and_starts_background_cache_warmer(self):
        """Start the optional cache warmer during app startup."""
        app_text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

        self.assertIn("from nhl.cache_warmer import start_background_warmer", app_text)
        self.assertIn("start_background_warmer()", app_text)


if __name__ == "__main__":
    unittest.main()
