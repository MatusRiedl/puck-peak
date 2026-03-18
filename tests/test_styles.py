import unittest

from nhl.styles import get_favicon_path, get_header_logo_data_uri, get_header_logo_path


class StylesTests(unittest.TestCase):
    """Validate style-adjacent helpers and assets."""

    def test_get_favicon_path_points_to_custom_svg_asset(self):
        """Resolve the custom favicon asset path.

        Args:
            None.

        Returns:
            None.
        """
        favicon_path = get_favicon_path()

        self.assertTrue(favicon_path.is_absolute())
        self.assertTrue(favicon_path.exists())
        self.assertEqual(favicon_path.name, "favicon.svg")

        svg = favicon_path.read_text(encoding="utf-8")
        self.assertIn("<svg", svg)
        self.assertIn("#003459", svg)
        self.assertIn("#00a8e8", svg)
        self.assertIn("#ffffff", svg)

    def test_app_page_config_uses_custom_favicon_path(self):
        """Keep app page config wired to the favicon helper.

        Args:
            None.

        Returns:
            None.
        """
        app_text = (get_favicon_path().parent.parent / "app.py").read_text(encoding="utf-8")

        self.assertIn("from nhl.styles import get_favicon_path, inject_css, inject_mobile_dropdown_fix", app_text)
        self.assertIn("page_icon=get_favicon_path().as_posix(),", app_text)
        self.assertIn('initial_sidebar_state="expanded"', app_text)

    def test_get_header_logo_path_points_to_assets_png_location(self):
        """Resolve the preferred assets-folder brand logo location.

        Args:
            None.

        Returns:
            None.
        """
        header_logo_path = get_header_logo_path()

        self.assertTrue(header_logo_path.is_absolute())
        self.assertEqual(header_logo_path.name, "PP.png")
        self.assertEqual(header_logo_path, get_favicon_path().parent / "PP.png")

    def test_get_header_logo_data_uri_returns_embeddable_png_uri(self):
        """Return an embeddable PNG brand image URI.

        Args:
            None.

        Returns:
            None.
        """
        data_uri = get_header_logo_data_uri()

        self.assertTrue(data_uri.startswith("data:image/png;base64,"))
        self.assertIn(";base64,", data_uri)
        self.assertNotIn("data:image/svg+xml", data_uri)

    def test_project_streamlit_config_defaults_to_dark_theme(self):
        """Keep the project-level Streamlit theme defaulted to dark.

        Args:
            None.

        Returns:
            None.
        """
        config_text = (get_favicon_path().parent.parent / ".streamlit" / "config.toml").read_text(encoding="utf-8")

        self.assertIn("[theme]", config_text)
        self.assertRegex(config_text, r'base\s*=\s*"dark"')

    def test_app_uses_descriptive_share_title_without_a_top_header(self):
        """Keep the app title branded without rendering a separate top header.

        Args:
            None.

        Returns:
            None.
        """
        app_text = (get_favicon_path().parent.parent / "app.py").read_text(encoding="utf-8")

        self.assertIn(
            'page_title="Puck Peak"',
            app_text,
        )
        self.assertNotIn("get_header_logo_data_uri()", app_text)
        self.assertNotIn("page-header-logo", app_text)
        self.assertNotIn("page-hero", app_text)
        self.assertNotIn("animated-title", app_text)
        self.assertNotIn("page-subtitle", app_text)
        self.assertNotIn("https://assets.nhle.com/logos/nhl/svg/NHL_light.svg", app_text)
        self.assertNotIn("<hr class='header-divider'>", app_text)

    def test_layout_uses_compact_spacing_classes(self):
        """Keep the top layout compact so controls stay above the fold.

        Args:
            None.

        Returns:
            None.
        """
        repo_root = get_favicon_path().parent.parent
        styles_text = (repo_root / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn(".sidebar-brand", styles_text)
        self.assertIn(".sidebar-brand__image", styles_text)
        self.assertIn('padding-top: 2.65rem !important;', styles_text)
        self.assertNotIn(".nhl-logo", styles_text)
        self.assertNotIn(".header-divider", styles_text)
        self.assertIn(".nhl-chart-toolbar", styles_text)
        self.assertIn('[data-testid="stExpander"] {', styles_text)
        self.assertIn("#main-chart-layout", styles_text)
        self.assertIn('div:has(> #main-chart-layout) + div {', styles_text)
        self.assertIn('margin-top: -0.45rem !important;', styles_text)
        self.assertIn(".comparison-trace-toggle-row", styles_text)
        self.assertIn(".comparison-trace-toggle", styles_text)
        self.assertIn(".comparison-trace-toggle--icon-only", styles_text)
        self.assertIn(".comparison-card-stats", styles_text)
        self.assertIn(".comparison-card-stats__item", styles_text)
        self.assertIn(".comparison-card-stats__label", styles_text)
        self.assertIn(".comparison-card-stats__value", styles_text)
        self.assertIn(".comparison-card-shell--clickable", styles_text)
        self.assertIn(".comparison-card-shell--clickable:hover .comparison-player-card", styles_text)
        self.assertIn(".comparison-trace-toggle__line::after", styles_text)
        self.assertIn('div.element-container:has(#comparison-tabs) + div.element-container {', styles_text)
        self.assertIn('margin-top: -0.2rem !important;', styles_text)
        self.assertIn('div.element-container:has(#comparison-tabs) + div.element-container [data-testid="stTabs"] {', styles_text)
        self.assertIn('padding-top: 0 !important;', styles_text)
        self.assertIn('min-height: 40px !important;', styles_text)
        self.assertIn('margin-bottom: 0.22rem !important;', styles_text)
        self.assertNotIn(".comparison-card-actions-anchor", styles_text)
        self.assertNotIn('div.element-container:has(.comparison-card-actions-anchor) + div.element-container {', styles_text)
        self.assertNotIn(".comparison-player-toggle-row", styles_text)
        self.assertNotIn(".comparison-trace-toggle--remove", styles_text)
        self.assertNotIn("comparison-player-remove-anchor", styles_text)

    def test_mobile_chart_layout_trims_padding_and_chart_header_chrome(self):
        """Keep the mobile chart layout dense enough to use the available width.

        Args:
            None.

        Returns:
            None.
        """
        styles_text = (get_favicon_path().parent.parent / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn('padding-left: 0.5rem !important;', styles_text)
        self.assertIn('padding-right: 0.5rem !important;', styles_text)
        self.assertIn('min-height: 32px !important;', styles_text)
        self.assertIn('font-size: 0.84rem;', styles_text)
        self.assertIn('gap: 0.28rem;', styles_text)
        self.assertIn('padding: 0.24rem 0.52rem;', styles_text)
        self.assertIn('width: 13px;', styles_text)
        self.assertIn('height: 13px;', styles_text)
        self.assertIn('top: 4px !important;', styles_text)
        self.assertIn('right: 4px !important;', styles_text)
        self.assertIn('.comparison-trace-toggle--compact {', styles_text)
        self.assertIn('padding: 0.26rem 0.56rem;', styles_text)
        self.assertIn('font-size: 0.7rem;', styles_text)
        self.assertIn('.comparison-card-shell--clickable .comparison-player-card {', styles_text)
        self.assertIn('transform: none !important;', styles_text)

    def test_chart_hover_labels_keep_shadow_effect_while_predictions_use_popovers(self):
        """Keep the deliberate chart hover shadow and the rail's soft meta popovers."""
        repo_root = get_favicon_path().parent.parent
        chart_text = (repo_root / "nhl" / "chart.py").read_text(encoding="utf-8")
        styles_text = (repo_root / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn('.js-plotly-plot .hoverlayer .hovertext', chart_text)
        self.assertIn('drop-shadow(0 6px 18px rgba(0, 0, 0, 0.62))', chart_text)
        self.assertIn('.lgc-meta-popover {', styles_text)
        self.assertIn('backdrop-filter: blur(10px);', styles_text)
        self.assertIn('transition: opacity 140ms ease, visibility 140ms ease, transform 140ms ease;', styles_text)

    def test_sidebar_help_button_and_app_guide_dialog_exist(self):
        """Keep the sidebar guide affordance and modal explanation wired in.

        Args:
            None.

        Returns:
            None.
        """
        repo_root = get_favicon_path().parent.parent
        app_text = (repo_root / "app.py").read_text(encoding="utf-8")
        sidebar_text = (repo_root / "nhl" / "sidebar.py").read_text(encoding="utf-8")
        dialog_text = (repo_root / "nhl" / "dialog.py").read_text(encoding="utf-8")

        self.assertNotIn('from nhl.dialog import show_app_guide', app_text)
        self.assertIn('from nhl.dialog import show_app_guide', sidebar_text)
        self.assertIn('st.button(', sidebar_text)
        self.assertIn('"FAQ"', sidebar_text)
        self.assertIn('key="open_app_guide_sidebar"', sidebar_text)
        self.assertIn('type="secondary"', sidebar_text)
        self.assertIn('help="How this app works"', sidebar_text)
        self.assertIn('@st.dialog("How This App Works")', dialog_text)
        self.assertIn("**What is ML-ish**", dialog_text)
        self.assertIn("**Skater baseline**", dialog_text)
        self.assertIn("**Goalie baseline**", dialog_text)
        self.assertNotIn("hand over the sauce", dialog_text)

    def test_sidebar_secondary_button_css_is_scoped_to_remove_rows(self):
        """Keep sidebar secondary-button styling from hijacking normal buttons.

        Args:
            None.

        Returns:
            None.
        """
        styles_text = (get_favicon_path().parent.parent / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn(
            '[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button[kind="secondary"][data-testid="stBaseButton-secondary"]',
            styles_text,
        )
        self.assertNotIn(
            '[data-testid="stSidebar"] button[kind="secondary"][data-testid="stBaseButton-secondary"] {',
            styles_text,
        )

    def test_sidebar_support_link_is_styled_and_rendered_above_status(self):
        """Keep the sidebar support CTA visible above the status expander.

        Args:
            None.

        Returns:
            None.
        """
        repo_root = get_favicon_path().parent.parent
        sidebar_text = (repo_root / "nhl" / "sidebar.py").read_text(encoding="utf-8")
        styles_text = (repo_root / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn('_SUPPORT_URL = "https://ko-fi.com/iksperial"', sidebar_text)
        self.assertIn('_SUPPORT_LABEL = "Buy me a coffee"', sidebar_text)
        self.assertIn('_SUPPORT_SUBLABEL = "Support the development"', sidebar_text)
        self.assertIn("_render_support_button()", sidebar_text)
        self.assertLess(
            sidebar_text.index("_render_support_button()"),
            sidebar_text.index('with st.expander("App status", expanded=False):'),
        )
        self.assertIn('[data-testid="stSidebar"] .sidebar-support-link {', styles_text)
        self.assertIn('.sidebar-support-link__emoji {', styles_text)
        self.assertIn('.sidebar-support-link__text {', styles_text)
        self.assertIn('.sidebar-support-link__label {', styles_text)
        self.assertIn('.sidebar-support-link__sublabel {', styles_text)
        self.assertIn('linear-gradient(135deg, #9d6535 0%, #bf7a3f 100%)', styles_text)
        self.assertIn('background: rgba(255, 248, 240, 0.18);', styles_text)
        self.assertIn('font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;', styles_text)
        self.assertIn('font-size: 0.73rem;', styles_text)

    def test_predictions_panel_css_uses_dedicated_card_layout_without_legacy_button_hooks(self):
        """Keep the predictions rail on the dedicated card-only layout without old button hooks.

        Args:
            None.

        Returns:
            None.
        """
        styles_text = (get_favicon_path().parent.parent / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn(".live-game-card-link {", styles_text)
        self.assertIn(".live-game-card {", styles_text)
        self.assertIn(".live-games-probability__bar {", styles_text)
        self.assertIn('div.element-container:has(.live-game-card) {', styles_text)
        self.assertIn('div.element-container:has(#comparison-predictions-panel) {', styles_text)
        self.assertIn('div.element-container:has(#comparison-predictions-panel) + div.element-container {', styles_text)
        self.assertIn(".comparison-panel-heading--predictions", styles_text)
        self.assertNotIn("#comparison-rail-tabs", styles_text)
        self.assertNotIn('live-games-btn-anchor', styles_text)
        self.assertNotIn('live-games-divider-anchor', styles_text)

    def test_stanley_cup_board_css_uses_scoped_comparison_styles(self):
        """Keep the Stanley Cup board visually integrated with the comparison panel."""
        styles_text = (get_favicon_path().parent.parent / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn(".stanley-cup-board-meta {", styles_text)
        self.assertIn(".stanley-cup-division-heading {", styles_text)
        self.assertIn(".stanley-cup-team-logo {", styles_text)
        self.assertIn(".stanley-cup-row-value--pts {", styles_text)
        self.assertIn("stanley-cup-favorite-button-anchor", styles_text)
        self.assertIn('content: "Cup pick";', styles_text)

    def test_readme_uses_updated_short_description(self):
        """Keep the repository description aligned with the new branding.

        Args:
            None.

        Returns:
            None.
        """
        readme_text = (get_favicon_path().parent.parent / "README.md").read_text(encoding="utf-8")

        self.assertIn("Hockey analytics & projections", readme_text)
        self.assertNotIn("An interactive analytics dashboard", readme_text)

    def test_share_link_generation_uses_chart_copy_control(self):
        """Keep compact share-link copying attached to the chart control.

        Args:
            None.

        Returns:
            None.
        """
        app_text = (get_favicon_path().parent.parent / "app.py").read_text(encoding="utf-8")
        chart_text = (get_favicon_path().parent.parent / "nhl" / "chart.py").read_text(encoding="utf-8")

        self.assertNotIn('st.button("Generate share link", use_container_width=True)', app_text)
        self.assertNotIn("Share link ready. Copy it from the browser address bar.", app_text)
        self.assertNotIn("# Sync current state to URL", app_text)
        self.assertIn("nhl-chart-share-btn", chart_text)
        self.assertIn("var SHARE_BUTTON_ID = ", chart_text)
        self.assertIn("var SHARE_PARAMS = ", chart_text)
        self.assertIn("var url = buildShareUrl(parent);", chart_text)
        self.assertIn("parent.history.replaceState(null, '', url);", chart_text)

    def test_plotly_modebar_has_no_capsule_background(self):
        """Keep chart controls visually unobtrusive inside the plot.

        Args:
            None.

        Returns:
            None.
        """
        styles_text = (get_favicon_path().parent.parent / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn('.js-plotly-plot .plotly .modebar {', styles_text)
        self.assertIn('background: transparent !important;', styles_text)
        self.assertIn('border: none !important;', styles_text)
        self.assertIn('box-shadow: none !important;', styles_text)
        self.assertIn('padding: 0 !important;', styles_text)

    def test_chart_toolbar_uses_theme_variables_for_light_and_dark_modes(self):
        """Keep chart toolbar copy readable across Streamlit theme switches.

        Args:
            None.

        Returns:
            None.
        """
        styles_text = (get_favicon_path().parent.parent / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn('.nhl-chart-toolbar__title {', styles_text)
        self.assertIn('color: rgba(255, 255, 255, 0.90);', styles_text)
        self.assertIn('font-weight: 400;', styles_text)
        self.assertIn('.nhl-chart-share-btn {', styles_text)
        self.assertIn('background: rgba(15, 23, 42, 0.72);', styles_text)
        self.assertIn('color: #dbe4f0;', styles_text)

    def test_team_dropdown_uses_native_label_and_first_selectbox_spacing_rule(self):
        """Keep the Team selector driven by the custom heading and first-selectbox spacing rule.

        Args:
            None.

        Returns:
            None.
        """
        repo_root = get_favicon_path().parent.parent
        sidebar_text = (repo_root / "nhl" / "sidebar.py").read_text(encoding="utf-8")
        styles_text = (repo_root / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn('"Team Comparison"', sidebar_text)
        self.assertIn("comparison-panel-heading--rail-title", sidebar_text)
        self.assertIn('label_visibility="collapsed"', sidebar_text)
        self.assertNotIn('label_visibility="visible"', sidebar_text)
        self.assertNotIn('st.subheader("Team Comparison")', sidebar_text)
        self.assertNotIn("team-comparison-anchor", sidebar_text)
        self.assertIn("normalizes first Team dropdown", styles_text)

    def test_controls_dropdowns_use_two_column_mobile_grid(self):
        """Keep the compact controls stacked one per row on small screens.

        Args:
            None.

        Returns:
            None.
        """
        styles_text = (get_favicon_path().parent.parent / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn('div:has(> #controls-dropdowns) + div [data-testid="column"]', styles_text)
        self.assertIn('div:has(> #controls-dropdowns) + div [data-testid="stHorizontalBlock"]', styles_text)
        self.assertIn('flex-wrap: wrap !important;', styles_text)
        self.assertIn('min-width: 100% !important;', styles_text)
        self.assertIn('flex: 1 1 100% !important;', styles_text)
        self.assertIn("hr + .element-container .stSelectbox label", styles_text)

    def test_controls_toolbar_uses_pills_and_team_baseline_stays_off(self):
        """Keep the compact controls toolbar and Team baseline guard in place.

        Args:
            None.

        Returns:
            None.
        """
        repo_root = get_favicon_path().parent.parent
        controls_text = (repo_root / "nhl" / "controls.py").read_text(encoding="utf-8")
        app_text = (repo_root / "app.py").read_text(encoding="utf-8")
        styles_text = (repo_root / "nhl" / "styles.py").read_text(encoding="utf-8")

        self.assertIn('st.pills(', controls_text)
        self.assertIn('label_visibility="collapsed"', controls_text)
        self.assertIn('{"label": "Base", "state_key": "do_base"}', controls_text)
        self.assertIn("controls-toolbar-muted__label'>Unavailable:</span>", controls_text)
        self.assertIn("controls-pill--disabled", controls_text)
        self.assertNotIn("controls-toolbar-label'>View options</div>", controls_text)
        self.assertIn('do_base    = st.session_state.do_base and not team_mode', app_text)
        self.assertIn('do_base              = do_base,', app_text)
        self.assertIn('.controls-pill--disabled {', styles_text)
        self.assertNotIn('.controls-toolbar-label {', styles_text)
        self.assertNotIn('[data-testid="stExpander"] [data-testid="stToggle"] label {', styles_text)


if __name__ == "__main__":
    unittest.main()
