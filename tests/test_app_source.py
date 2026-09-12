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

    def test_app_opens_each_run_through_the_dialog_run_helper(self):
        """`app.py` must reset the dialog slot via begin_script_run(), not by hand.

        The raw assignment it replaced lived only at top-level script scope, so a
        fragment rerun never cleared the flag and every dialog after the first was
        silently swallowed.
        """
        app_text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

        self.assertIn("from nhl.ui_state import begin_script_run", app_text)
        self.assertIn("begin_script_run()", app_text)
        self.assertNotIn('st.session_state["_dialog_opened_this_run"] = False', app_text)

    def test_matchup_bridge_is_not_mounted_at_top_level_script_scope(self):
        """The matchup bridge must live inside a fragment, not the main script body.

        Mounted at top level its trigger belonged to the main run, so every
        prediction-card click forced a full script rerun — the only click on the page
        that did — while the chart and identity bridges were fragment-scoped.
        """
        repo_root = Path(__file__).resolve().parents[1]
        app_text = (repo_root / "app.py").read_text(encoding="utf-8")
        comparison_text = (repo_root / "nhl" / "comparison.py").read_text(encoding="utf-8")

        self.assertNotIn("_mount_matchup_history_click_bridge", app_text)
        self.assertNotIn("bridge_slot", app_text)
        # It is still mounted exactly once, from inside the predictions panel.
        self.assertIn("_mount_matchup_history_click_bridge()", comparison_text)

        chart_idx = app_text.index("chart_fragment(")
        self.assertIn("suppress_dialogs", app_text[chart_idx: chart_idx + 1600])
        self.assertIn(
            "has_pending_matchup_history_dialog_request()",
            app_text[chart_idx: chart_idx + 1600],
        )

    def test_every_fragment_opens_its_own_dialog_run_scope(self):
        """Each @st.fragment wrapper must call begin_dialog_run with a unique scope."""
        fragments_text = (
            Path(__file__).resolve().parents[1] / "nhl" / "fragments.py"
        ).read_text(encoding="utf-8")

        self.assertIn("begin_dialog_run", fragments_text)
        for scope in ("chart", "detail_tabs", "predictions", "faq"):
            self.assertIn(f'begin_dialog_run("{scope}")', fragments_text)

    def test_faq_button_is_fragment_scoped_and_takes_the_dialog_slot(self):
        """The FAQ button must not rerun the whole script to open a modal.

        At top-level sidebar scope its click cost a full pipeline run and a rebuilt
        Plotly figure for a dialog that reads none of it. It was also the only one of
        the six st.dialog call sites that never reserved the one-dialog slot.
        """
        repo_root = Path(__file__).resolve().parents[1]
        fragments_text = (repo_root / "nhl" / "fragments.py").read_text(encoding="utf-8")
        sidebar_text = (repo_root / "nhl" / "sidebar.py").read_text(encoding="utf-8")

        faq_index = fragments_text.index("def faq_button_fragment()")
        self.assertIn("@st.fragment", fragments_text[faq_index - 60: faq_index])

        faq_body = fragments_text[faq_index:]
        self.assertIn("dialog_slot_available()", faq_body)
        self.assertIn("mark_dialog_opened_this_run()", faq_body)
        self.assertIn("show_app_guide()", faq_body)

        # The sidebar only calls the wrapper; it no longer owns the button or dialog.
        self.assertIn("faq_button_fragment()", sidebar_text)
        self.assertNotIn("show_app_guide", sidebar_text)
        self.assertNotIn('key="open_app_guide_sidebar"', sidebar_text)

    def test_app_imports_and_starts_background_cache_warmer(self):
        """Start the optional cache warmer during app startup."""
        app_text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

        self.assertIn("from nhl.cache_warmer import start_background_warmer", app_text)
        self.assertIn("start_background_warmer()", app_text)


if __name__ == "__main__":
    unittest.main()
