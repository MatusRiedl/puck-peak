import unittest
from pathlib import Path

from nhl.sidebar import (
    _STAT_CATEGORY_LABELS,
    _SUPPORT_EMOJI,
    _SUPPORT_LABEL,
    _SUPPORT_SUBLABEL,
    _SUPPORT_URL,
    _build_support_button_markup,
    _format_stat_category_label,
    _resolve_stat_category_selection,
    _sanitize_stat_category,
)


class SidebarTests(unittest.TestCase):
    """Cover current sidebar invariants."""

    def test_sanitize_stat_category_preserves_valid_values(self):
        """Keep valid category values unchanged."""
        self.assertEqual(_sanitize_stat_category("Skater"), "Skater")
        self.assertEqual(_sanitize_stat_category("Goalie"), "Goalie")
        self.assertEqual(_sanitize_stat_category("Team"), "Team")

    def test_format_stat_category_label_uses_the_shared_display_labels(self):
        """Keep emoji-decorated display labels out of the canonical state values."""
        self.assertEqual(_format_stat_category_label("Skater"), _STAT_CATEGORY_LABELS["Skater"])
        self.assertEqual(_format_stat_category_label("Goalie"), _STAT_CATEGORY_LABELS["Goalie"])
        self.assertEqual(_format_stat_category_label("Team"), _STAT_CATEGORY_LABELS["Team"])
        self.assertEqual(_format_stat_category_label("nonsense"), _STAT_CATEGORY_LABELS["Skater"])

    def test_resolve_stat_category_selection_keeps_current_category_on_deselect(self):
        """Prevent a second click from clearing the active category."""
        self.assertEqual(_resolve_stat_category_selection(None, "Skater"), "Skater")
        self.assertEqual(_resolve_stat_category_selection(None, "Goalie"), "Goalie")
        self.assertEqual(_resolve_stat_category_selection(None, "Team"), "Team")

    def test_resolve_stat_category_selection_uses_skater_when_both_values_are_invalid(self):
        """Fall back to Skater when the widget and stored state are both corrupt."""
        self.assertEqual(_resolve_stat_category_selection(None, None), "Skater")
        self.assertEqual(_resolve_stat_category_selection("Bad", "Worse"), "Skater")

    def test_app_guards_against_invalid_category_state(self):
        """Keep the app-level invalid-category guard in place."""
        repo_root = Path(__file__).resolve().parents[1]
        app_text = (repo_root / "app.py").read_text(encoding="utf-8")
        sidebar_text = (repo_root / "nhl" / "sidebar.py").read_text(encoding="utf-8")

        self.assertIn('if st.session_state.stat_category not in {"Skater", "Goalie", "Team"}:', app_text)
        self.assertIn('key="_stat_category_picker"', sidebar_text)
        self.assertIn('on_change=_sync_stat_category_selection', sidebar_text)
        self.assertIn('format_func=_format_stat_category_label', sidebar_text)

    def test_support_button_markup_targets_kofi_in_new_tab(self):
        """Keep the support CTA pointed at Ko-fi with the intended chrome."""
        markup = _build_support_button_markup()

        self.assertIn(_SUPPORT_URL, markup)
        self.assertIn(_SUPPORT_LABEL, markup)
        self.assertIn(_SUPPORT_SUBLABEL, markup)
        self.assertIn(_SUPPORT_EMOJI, markup)
        self.assertIn("sidebar-support-link", markup)
        self.assertIn("sidebar-support-link__label", markup)
        self.assertIn("sidebar-support-link__sublabel", markup)
        self.assertIn("target='_blank'", markup)
        self.assertIn("rel='noopener noreferrer'", markup)

    def test_faq_button_keeps_its_blue_tint_hook(self):
        """Keep the FAQ button tied to its dedicated sidebar tint styling."""
        repo_root = Path(__file__).resolve().parents[1]
        sidebar_text = (repo_root / "nhl" / "sidebar.py").read_text(encoding="utf-8")
        styles_text = (repo_root / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn("faq-btn-anchor", sidebar_text)
        self.assertIn("faq-btn-anchor", styles_text)
        self.assertIn("rgba(43, 113, 199, 0.16)", styles_text)

    def test_sidebar_renders_brand_logo_above_the_faq_button(self):
        """Keep the sidebar logo above the FAQ button without extra legacy title text."""
        repo_root = Path(__file__).resolve().parents[1]
        sidebar_text = (repo_root / "nhl" / "sidebar.py").read_text(encoding="utf-8")
        styles_text = (repo_root / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn("from nhl.styles import get_header_logo_data_uri", sidebar_text)
        self.assertIn("class='sidebar-brand__image'", sidebar_text)
        self.assertIn("get_header_logo_data_uri()", sidebar_text)
        self.assertLess(
            sidebar_text.index("class='sidebar-brand__image'"),
            sidebar_text.index('st.markdown("<div class=\'faq-btn-anchor\'></div>", unsafe_allow_html=True)'),
        )
        self.assertIn(".sidebar-brand", styles_text)
        self.assertIn(".sidebar-brand__image", styles_text)
        self.assertNotIn("class='sidebar-brand__title'", sidebar_text)
        self.assertNotIn("class='sidebar-brand__subtitle'", sidebar_text)

    def test_app_injects_base_css_after_page_config(self):
        """Keep base CSS injection wired immediately after page config."""
        app_text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

        page_config_index = app_text.index("st.set_page_config(")
        inject_index = app_text.index("inject_css()")
        mobile_fix_index = app_text.index("inject_mobile_dropdown_fix()")

        self.assertLess(page_config_index, inject_index)
        self.assertLess(inject_index, mobile_fix_index)

    def test_sidebar_team_remove_mutates_session_state_inline(self):
        """Keep team removal on the current inline session-state path."""
        sidebar_text = (Path(__file__).resolve().parents[1] / "nhl" / "sidebar.py").read_text(encoding="utf-8")

        self.assertIn('del st.session_state.teams[_abbr]', sidebar_text)
        self.assertIn("st.rerun()", sidebar_text)
        self.assertNotIn("from nhl.selection import remove_selected_player, remove_selected_team", sidebar_text)

    def test_sidebar_escapes_shared_link_names_before_injecting_html(self):
        """Keep shared-link display names inert even though the card shell uses raw HTML."""
        sidebar_text = (Path(__file__).resolve().parents[1] / "nhl" / "sidebar.py").read_text(encoding="utf-8")

        self.assertIn('safe_name = escape(str(name or ""))', sidebar_text)
        self.assertIn("safe_team_name = escape(str(_name or \"\"))", sidebar_text)
        self.assertIn("f\"<div class='player-name'>{safe_name}</div>\"", sidebar_text)
        self.assertIn("f\"<div class='player-name'>{safe_team_name}</div>\"", sidebar_text)


if __name__ == "__main__":
    unittest.main()
