import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import nhl.chart as chart_module
from nhl.chart import (
    _build_chart_axis_cue_annotations,
    _build_chart_header,
    _build_chart_glow_style,
    _build_chart_toolbar_markup,
    _get_chart_context_label,
    _get_chart_era_label,
    _get_chart_season_label,
    _is_baseline_trace,
    _slugify_chart_export_name,
)


def _build_chart_click_trigger(
    chart_instance_id: str,
    *,
    nonce: str,
    trace_name: str,
    x,
    y,
    customdata: list,
    curve_number: int = 0,
    point_number: int = 0,
) -> str:
    """Return one serialized chart-click bridge payload for tests."""
    return json.dumps(
        {
            "nonce": nonce,
            "chart_instance_id": chart_instance_id,
            "trace_name": trace_name,
            "x": x,
            "y": y,
            "customdata": customdata,
            "curve_number": curve_number,
            "point_number": point_number,
        }
    )


class ChartTests(unittest.TestCase):
    """Cover chart-toolbar formatting and share-link wiring."""

    def test_get_chart_context_label_uses_expected_labels(self):
        """Return the right x-context label.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(_get_chart_context_label(team_mode=False, games_mode=False), "Age")
        self.assertEqual(_get_chart_context_label(team_mode=False, games_mode=True), "Games Played")
        self.assertEqual(
            _get_chart_context_label(team_mode=False, games_mode=True, selected_season=2024),
            "Game Number",
        )
        self.assertEqual(_get_chart_context_label(team_mode=True, games_mode=False), "Season")
        self.assertEqual(_get_chart_context_label(team_mode=True, games_mode=True), "Games Played")
        self.assertEqual(
            _get_chart_context_label(team_mode=True, games_mode=True, selected_season=2024),
            "Game Number",
        )

    def test_get_chart_season_label_uses_professional_copy(self):
        """Return the right season wording.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(_get_chart_season_label("Regular"), "Regular season")
        self.assertEqual(_get_chart_season_label("Playoffs"), "Playoffs")
        self.assertEqual(_get_chart_season_label("Both"), "Regular + playoffs")

    def test_get_chart_era_label_reflects_actual_visible_adjustment(self):
        """Return an era label that matches the rendered metric, not just the toggle.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(_get_chart_era_label("Points", "Skater", do_era=True, team_mode=False), "Era adjusted")
        self.assertEqual(_get_chart_era_label("PIM", "Skater", do_era=True, team_mode=False), "")
        self.assertEqual(_get_chart_era_label("Save %", "Goalie", do_era=True, team_mode=False), "Era adjusted")
        self.assertEqual(_get_chart_era_label("Wins", "Goalie", do_era=True, team_mode=False), "")
        self.assertEqual(_get_chart_era_label("Points", "Skater", do_era=False, team_mode=False), "")
        self.assertEqual(_get_chart_era_label("Points", "Team", do_era=True, team_mode=True), "")

    def test_build_chart_header_matches_requested_examples(self):
        """Build the toolbar title text.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(
            _build_chart_header(
                "Points",
                team_mode=False,
                games_mode=False,
                season_type="Regular",
                stat_category="Skater",
                do_era=True,
            ),
            "Points by Age · Regular season · Era adjusted",
        )
        self.assertEqual(
            _build_chart_header(
                "Goals",
                team_mode=False,
                games_mode=False,
                season_type="Playoffs",
                stat_category="Skater",
                do_era=False,
            ),
            "Goals by Age · Playoffs",
        )
        self.assertEqual(
            _build_chart_header(
                "GAA",
                team_mode=False,
                games_mode=True,
                season_type="Both",
                stat_category="Goalie",
                do_era=True,
            ),
            "GAA by Games Played · Regular + playoffs · Era adjusted",
        )
        self.assertEqual(
            _build_chart_header(
                "Points",
                team_mode=True,
                games_mode=False,
                season_type="Regular",
                stat_category="Team",
                do_era=True,
            ),
            "Points by Season · Regular season",
        )

    def test_build_chart_header_includes_selected_chart_season(self):
        """Include the explicit chart-season label in single-season mode.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(
            _build_chart_header(
                "Points",
                team_mode=False,
                games_mode=True,
                season_type="Regular",
                stat_category="Skater",
                do_era=False,
                selected_season=2024,
            ),
            "Points at Game Number · 2024-25 · Regular season",
        )
        self.assertEqual(
            _build_chart_header(
                "Points",
                team_mode=True,
                games_mode=True,
                season_type="Regular",
                stat_category="Team",
                do_era=False,
                selected_season=2024,
            ),
            "Points by Game Number · 2024-25 · Regular season",
        )

    def test_build_chart_toolbar_markup_includes_title_and_button(self):
        """Render one toolbar row with the title and copy-link button.

        Args:
            None.

        Returns:
            None.
        """
        markup = _build_chart_toolbar_markup(
            "Points by Age · Regular season",
            "share-btn-123",
            "toolbar-123",
        )

        self.assertIn("nhl-chart-toolbar", markup)
        self.assertIn("id='toolbar-123'", markup)
        self.assertIn("Points by Age · Regular season", markup)
        self.assertIn("id='share-btn-123'", markup)
        self.assertIn("Copy link", markup)
        self.assertIn("aria-label='Copy share link'", markup)

    def test_build_chart_toolbar_markup_escapes_title_html(self):
        """Escape title text before writing toolbar markup.

        Args:
            None.

        Returns:
            None.
        """
        markup = _build_chart_toolbar_markup(
            "Points <script>alert(1)</script>",
            "share-btn-123",
            "toolbar-123",
        )

        self.assertIn("Points &lt;script&gt;alert(1)&lt;/script&gt;", markup)
        self.assertNotIn("<script>alert(1)</script>", markup)

    def test_build_chart_axis_cue_annotations_use_white_25pct_opacity_labels(self):
        """Render subtle in-chart cues for both axes plus the click affordance.

        Args:
            None.

        Returns:
            None.
        """
        annotations = _build_chart_axis_cue_annotations("Points", team_mode=False, games_mode=False)

        self.assertEqual(len(annotations), 3)
        self.assertEqual(annotations[0]["text"], "Points")
        self.assertEqual(annotations[1]["text"], "Age")
        self.assertEqual(annotations[2]["text"], "Click on chart for details")
        self.assertEqual(annotations[0]["x"], 0.07)
        self.assertEqual(annotations[0]["y"], 1.017)
        self.assertEqual(annotations[1]["x"], 0.988)
        self.assertEqual(annotations[1]["y"], 0.042)
        self.assertEqual(annotations[2]["x"], 0.5)
        self.assertEqual(annotations[2]["y"], 1.017)
        self.assertEqual(annotations[0]["font"]["color"], "rgba(255, 255, 255, 0.25)")
        self.assertEqual(annotations[1]["font"]["color"], "rgba(255, 255, 255, 0.25)")
        self.assertEqual(annotations[2]["font"]["color"], "rgba(255, 255, 255, 0.25)")
        self.assertEqual(annotations[0]["xref"], "paper")
        self.assertEqual(annotations[1]["yref"], "paper")

    def test_build_chart_glow_style_returns_empty_without_colors(self):
        """Skip chart glow CSS when no player colors are available.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(_build_chart_glow_style({}), "")

    def test_build_chart_glow_style_uses_wider_center_glow_and_layered_edge_shadow(self):
        """Use the widened center glow and softer two-layer outer spread.

        Args:
            None.

        Returns:
            None.
        """
        style = _build_chart_glow_style(
            {
                "A": "#112233",
                "B": "rgb(44, 55, 66)",
                "C": "#abcdef",
            }
        )

        self.assertIn("ellipse 54% 42% at 50% 52%", style)
        self.assertIn("rgba(17, 34, 51, 0.070) 0%, rgba(17, 34, 51, 0.000) 78%", style)
        self.assertIn("rgba(44, 55, 66, 0.070) 0%, rgba(44, 55, 66, 0.000) 78%", style)
        self.assertIn("0 0 32px rgba(17, 34, 51, 0.040)", style)
        self.assertIn("0 0 64px rgba(17, 34, 51, 0.020)", style)
        self.assertIn("0 0 32px rgba(44, 55, 66, 0.040)", style)
        self.assertIn("0 0 64px rgba(44, 55, 66, 0.020)", style)

    def test_build_chart_axis_cue_annotations_use_game_number_for_single_season_mode(self):
        """Use Game Number as the x cue for selected-season game-log charts.

        Args:
            None.

        Returns:
            None.
        """
        annotations = _build_chart_axis_cue_annotations(
            "Points",
            team_mode=False,
            games_mode=True,
            selected_season=2024,
        )

        self.assertEqual(annotations[1]["text"], "Game Number")

    def test_slugify_chart_export_name_builds_clean_filename(self):
        """Convert chart titles into sane download filenames.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(
            _slugify_chart_export_name("Points by Age · Regular season"),
            "points_by_age_regular_season",
        )
        self.assertEqual(
            _slugify_chart_export_name("GAA by Games Played · Regular + playoffs"),
            "gaa_by_games_played_regular_playoffs",
        )

    def test_is_baseline_trace_is_case_insensitive(self):
        """Recognize baseline traces even when the label casing changes.

        Args:
            None.

        Returns:
            None.
        """
        self.assertTrue(_is_baseline_trace("Reference baseline"))
        self.assertTrue(_is_baseline_trace("Skater 75th Percentile Baseline"))
        self.assertFalse(_is_baseline_trace("Artemi Panarin"))

    def test_build_trace_color_map_uses_stable_distinct_skater_colors(self):
        """Assign skater colors with stable full-spectrum separation."""
        final_df = pd.DataFrame(
            {
                "Player": [
                    "Connor McDavid",
                    "Connor McDavid",
                    "Leon Draisaitl",
                    "Leon Draisaitl",
                    "Connor McDavid (Proj)",
                    "Reference baseline",
                ],
                "BaseName": [
                    "Connor McDavid",
                    "Connor McDavid",
                    "Leon Draisaitl",
                    "Leon Draisaitl",
                    "Connor McDavid",
                    "Baseline",
                ],
            }
        )

        color_map = chart_module._build_trace_color_map(final_df, stat_category="Skater", team_mode=False)
        second_color_map = chart_module._build_trace_color_map(final_df, stat_category="Skater", team_mode=False)

        self.assertEqual(color_map, second_color_map)
        self.assertEqual(color_map["Connor McDavid"], chart_module.CATEGORY_TRACE_STARTERS["Skater"][0])
        self.assertEqual(color_map["Leon Draisaitl"], chart_module.CATEGORY_TRACE_STARTERS["Skater"][1])
        self.assertNotEqual(color_map["Connor McDavid"], color_map["Leon Draisaitl"])
        self.assertGreaterEqual(
            chart_module._color_distance(color_map["Connor McDavid"], color_map["Leon Draisaitl"]),
            chart_module.TRACE_COLOR_MIN_DISTANCE["Skater"],
        )
        self.assertNotIn("Connor McDavid (Proj)", color_map)
        self.assertNotIn("Reference baseline", color_map)

    def test_build_trace_color_map_uses_stable_distinct_team_colors(self):
        """Assign team colors with stable full-spectrum separation."""
        final_df = pd.DataFrame(
            {
                "Player": ["Toronto Maple Leafs", "Colorado Avalanche", "Edmonton Oilers"],
                "BaseName": ["TOR", "COL", "EDM"],
            }
        )

        color_map = chart_module._build_trace_color_map(final_df, stat_category="Team", team_mode=True)
        second_color_map = chart_module._build_trace_color_map(final_df, stat_category="Team", team_mode=True)

        self.assertEqual(color_map, second_color_map)
        self.assertEqual(color_map["Toronto Maple Leafs"], chart_module.CATEGORY_TRACE_STARTERS["Team"][0])
        self.assertEqual(color_map["Colorado Avalanche"], chart_module.CATEGORY_TRACE_STARTERS["Team"][1])
        self.assertRegex(color_map["Edmonton Oilers"], r"^#[0-9A-F]{6}$")
        self.assertGreaterEqual(
            chart_module._color_distance(color_map["Toronto Maple Leafs"], color_map["Colorado Avalanche"]),
            chart_module.TRACE_COLOR_MIN_DISTANCE["Team"],
        )
        self.assertGreaterEqual(
            chart_module._color_distance(color_map["Colorado Avalanche"], color_map["Edmonton Oilers"]),
            chart_module.TRACE_COLOR_MIN_DISTANCE["Team"],
        )

    def test_render_chart_links_projection_to_player_legend_toggle(self):
        """Hide the separate projection legend item and group it with the player.

        Args:
            None.

        Returns:
            None.
        """
        processed_df = pd.DataFrame(
            {
                "Age": [30, 31, 32, 33],
                "Points": [92, 88, 61, 55],
                "Player": [
                    "Artemi Panarin",
                    "Artemi Panarin",
                    "Artemi Panarin (Proj)",
                    "Artemi Panarin (Proj)",
                ],
                "BaseName": ["Artemi Panarin"] * 4,
            }
        )
        captured = {}
        fake_session_state = SimpleNamespace(do_predict=True, do_smooth=False, x_axis_mode="Age")

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            fake_session_state,
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Points",
                team_mode=False,
                games_mode=False,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Skater",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                do_era=False,
            )

        fig = captured["fig"]
        expected_color_map = chart_module._build_trace_color_map(
            processed_df,
            stat_category="Skater",
            team_mode=False,
        )
        real_trace = next(trace for trace in fig.data if trace.name == "Artemi Panarin")
        proj_trace = next(trace for trace in fig.data if trace.name == "Artemi Panarin (Proj)")
        proj_glows = [trace for trace in fig.data if trace.name.startswith("_proj_glow_")]
        age_marker_glows = [trace for trace in fig.data if trace.name == "_age_marker_glow"]

        self.assertEqual(fig.layout.legend.groupclick, "togglegroup")
        self.assertFalse(fig.layout.showlegend)
        self.assertTrue(all(trace.showlegend is False for trace in fig.data))
        self.assertEqual(real_trace.legendgroup, "Artemi Panarin")
        self.assertEqual(proj_trace.legendgroup, "Artemi Panarin")
        self.assertEqual(real_trace.line.color, chart_module.CATEGORY_TRACE_STARTERS["Skater"][0])
        self.assertEqual(real_trace.line.color, expected_color_map["Artemi Panarin"])
        self.assertNotEqual(real_trace.line.color, "#636efa")
        self.assertFalse(proj_trace.showlegend)
        self.assertEqual(proj_trace.line.dash, "dot")
        self.assertEqual(
            proj_trace.line.color,
            chart_module._with_alpha(real_trace.line.color, chart_module.PROJECTION_LINE_OPACITY),
        )
        self.assertEqual(proj_trace.marker.color, real_trace.line.color)
        self.assertEqual(proj_trace.marker.line.width, 0)
        self.assertEqual(proj_trace.marker.symbol, "circle")
        self.assertEqual(
            fake_session_state.player_chart_colors["Artemi Panarin"],
            real_trace.line.color,
        )
        self.assertTrue(proj_glows)
        self.assertEqual({trace.name for trace in proj_glows}, {"_proj_glow_outer", "_proj_glow_inner"})
        self.assertEqual(len(age_marker_glows), 1)
        self.assertTrue(all(trace.legendgroup == "Artemi Panarin" for trace in proj_glows))
        self.assertTrue(all(trace.showlegend is False for trace in proj_glows))

    def test_render_chart_uses_distinct_skater_colors_for_multiple_players(self):
        """Give adjacent skater traces clearly different colors."""
        processed_df = pd.DataFrame(
            {
                "Age": [24, 25, 24, 25],
                "Points": [100, 112, 22, 31],
                "Player": [
                    "Connor McDavid",
                    "Connor McDavid",
                    "Ryan Poehling",
                    "Ryan Poehling",
                ],
                "BaseName": [
                    "Connor McDavid",
                    "Connor McDavid",
                    "Ryan Poehling",
                    "Ryan Poehling",
                ],
            }
        )
        captured = {}

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age"),
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Points",
                team_mode=False,
                games_mode=False,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Skater",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                do_era=False,
            )

        fig = captured["fig"]
        expected_color_map = chart_module._build_trace_color_map(
            processed_df,
            stat_category="Skater",
            team_mode=False,
        )
        mcdavid_trace = next(trace for trace in fig.data if trace.name == "Connor McDavid")
        poehling_trace = next(trace for trace in fig.data if trace.name == "Ryan Poehling")

        self.assertEqual(mcdavid_trace.line.color, chart_module.CATEGORY_TRACE_STARTERS["Skater"][0])
        self.assertEqual(poehling_trace.line.color, chart_module.CATEGORY_TRACE_STARTERS["Skater"][1])
        self.assertEqual(mcdavid_trace.line.color, expected_color_map["Connor McDavid"])
        self.assertEqual(poehling_trace.line.color, expected_color_map["Ryan Poehling"])
        self.assertNotEqual(mcdavid_trace.line.color, poehling_trace.line.color)
        self.assertGreaterEqual(
            chart_module._color_distance(mcdavid_trace.line.color, poehling_trace.line.color),
            chart_module.TRACE_COLOR_MIN_DISTANCE["Skater"],
        )

    def test_render_chart_uses_transparent_background_layout(self):
        """Keep the chart canvas and Plotly modebar background transparent.

        Args:
            None.

        Returns:
            None.
        """
        processed_df = pd.DataFrame(
            {
                "Age": [24, 25, 26],
                "Wins": [18, 24, 27],
                "Player": ["Igor Shesterkin"] * 3,
                "BaseName": ["Igor Shesterkin"] * 3,
            }
        )
        captured = {}

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age"),
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Wins",
                team_mode=False,
                games_mode=False,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Goalie",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                do_era=False,
            )

        fig = captured["fig"]

        self.assertEqual(fig.layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(fig.layout.plot_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(fig.layout.modebar.bgcolor, "rgba(0,0,0,0)")

    def test_render_chart_uses_seeded_goalie_color_for_first_trace(self):
        """Assign a seeded goalie color instead of Plotly's default colors."""
        processed_df = pd.DataFrame(
            {
                "Age": [24, 25, 26],
                "Wins": [18, 24, 27],
                "Player": ["Igor Shesterkin"] * 3,
                "BaseName": ["Igor Shesterkin"] * 3,
            }
        )
        captured = {}
        fake_session_state = SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age")

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            fake_session_state,
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Wins",
                team_mode=False,
                games_mode=False,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Goalie",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                do_era=False,
            )

        fig = captured["fig"]
        expected_color_map = chart_module._build_trace_color_map(
            processed_df,
            stat_category="Goalie",
            team_mode=False,
        )
        real_trace = next(trace for trace in fig.data if trace.name == "Igor Shesterkin")

        self.assertEqual(real_trace.line.color, chart_module.CATEGORY_TRACE_STARTERS["Goalie"][0])
        self.assertEqual(real_trace.line.color, expected_color_map["Igor Shesterkin"])
        self.assertNotEqual(real_trace.line.color, "#636efa")
        self.assertEqual(
            fake_session_state.player_chart_colors["Igor Shesterkin"],
            expected_color_map["Igor Shesterkin"],
        )

    def test_render_chart_uses_distinct_goalie_colors_for_multiple_players(self):
        """Give adjacent goalie traces clearly different colors."""
        processed_df = pd.DataFrame(
            {
                "Age": [24, 25, 24, 25],
                "Wins": [18, 24, 26, 28],
                "Player": [
                    "Igor Shesterkin",
                    "Igor Shesterkin",
                    "Tristan Jarry",
                    "Tristan Jarry",
                ],
                "BaseName": [
                    "Igor Shesterkin",
                    "Igor Shesterkin",
                    "Tristan Jarry",
                    "Tristan Jarry",
                ],
            }
        )
        captured = {}

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age"),
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Wins",
                team_mode=False,
                games_mode=False,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Goalie",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                do_era=False,
            )

        fig = captured["fig"]
        expected_color_map = chart_module._build_trace_color_map(
            processed_df,
            stat_category="Goalie",
            team_mode=False,
        )
        shesterkin_trace = next(trace for trace in fig.data if trace.name == "Igor Shesterkin")
        jarry_trace = next(trace for trace in fig.data if trace.name == "Tristan Jarry")

        self.assertEqual(shesterkin_trace.line.color, chart_module.CATEGORY_TRACE_STARTERS["Goalie"][0])
        self.assertEqual(jarry_trace.line.color, chart_module.CATEGORY_TRACE_STARTERS["Goalie"][1])
        self.assertEqual(shesterkin_trace.line.color, expected_color_map["Igor Shesterkin"])
        self.assertEqual(jarry_trace.line.color, expected_color_map["Tristan Jarry"])
        self.assertNotEqual(shesterkin_trace.line.color, jarry_trace.line.color)
        self.assertGreaterEqual(
            chart_module._color_distance(shesterkin_trace.line.color, jarry_trace.line.color),
            chart_module.TRACE_COLOR_MIN_DISTANCE["Goalie"],
        )

    def test_render_chart_uses_distinct_team_colors_for_multiple_teams(self):
        """Use distinct team colors instead of franchise brand colors."""
        processed_df = pd.DataFrame(
            {
                "SeasonYear": [2023, 2024, 2023, 2024],
                "Points": [107, 111, 109, 103],
                "Player": [
                    "Colorado Avalanche",
                    "Colorado Avalanche",
                    "Edmonton Oilers",
                    "Edmonton Oilers",
                ],
                "BaseName": ["COL", "COL", "EDM", "EDM"],
            }
        )
        captured = {}

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            SimpleNamespace(
                do_predict=False,
                do_smooth=False,
                x_axis_mode="Season Year",
                teams={"COL": "Colorado Avalanche", "EDM": "Edmonton Oilers"},
            ),
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Points",
                team_mode=True,
                games_mode=False,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Team",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                do_era=False,
            )

        fig = captured["fig"]
        expected_color_map = chart_module._build_trace_color_map(
            processed_df,
            stat_category="Team",
            team_mode=True,
        )
        colorado_trace = next(trace for trace in fig.data if trace.name == "Colorado Avalanche")
        edmonton_trace = next(trace for trace in fig.data if trace.name == "Edmonton Oilers")

        self.assertEqual(colorado_trace.line.color, chart_module.CATEGORY_TRACE_STARTERS["Team"][0])
        self.assertEqual(edmonton_trace.line.color, chart_module.CATEGORY_TRACE_STARTERS["Team"][1])
        self.assertEqual(colorado_trace.line.color, expected_color_map["Colorado Avalanche"])
        self.assertEqual(edmonton_trace.line.color, expected_color_map["Edmonton Oilers"])
        self.assertNotEqual(colorado_trace.line.color, edmonton_trace.line.color)
        self.assertGreaterEqual(
            chart_module._color_distance(colorado_trace.line.color, edmonton_trace.line.color),
            chart_module.TRACE_COLOR_MIN_DISTANCE["Team"],
        )

    def test_render_chart_uses_vrect_peak_highlight_in_player_mode(self):
        """Render the peak highlight as one local Plotly vrect in player mode.

        Args:
            None.

        Returns:
            None.
        """
        processed_df = pd.DataFrame(
            {
                "Age": [24, 25, 26, 27, 28, 29, 30, 31, 32],
                "Points": [54, 67, 80, 92, 101, 107, 103, 95, 82],
                "Player": ["Artemi Panarin"] * 9,
                "BaseName": ["Artemi Panarin"] * 9,
            }
        )
        captured = {}

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age"),
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Points",
                team_mode=False,
                games_mode=False,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Skater",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                peak_info={
                    "Artemi Panarin": {"x": 29.0, "y": 107.0, "age": 29, "season_year": 2021, "pid": 8478550}
                },
                do_prime=True,
                do_era=False,
            )

        fig = captured["fig"]
        self.assertEqual(len(fig.layout.shapes), 1)
        highlight = fig.layout.shapes[0]

        self.assertEqual(highlight.type, "rect")
        self.assertEqual(highlight.x0, 28.5)
        self.assertEqual(highlight.x1, 29.5)
        self.assertEqual(highlight.opacity, 0.10)
        self.assertEqual(highlight.layer, "below")
        self.assertEqual(highlight.line.width, 0)

    def test_render_chart_uses_single_season_games_hover_and_peak_x(self):
        """Use lollipop game-log rendering and peak x-position highlight in season mode.

        Args:
            None.

        Returns:
            None.
        """
        processed_df = pd.DataFrame(
            {
                "CumGP": [1, 2, 3, 4],
                "Age": [28, 28, 28, 28],
                "Points": [0, 1, 3, 1],
                "GameId": [1001, 1002, 1003, 1004],
                "GameDate": ["2024-10-09", "2024-10-12", "2024-10-15", "2024-10-18"],
                "GameType": ["Regular"] * 4,
                "Player": ["Connor McDavid"] * 4,
                "BaseName": ["Connor McDavid"] * 4,
            }
        )
        captured = {}

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.st, "caption"), patch.object(
            chart_module.components,
            "html",
        ), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Games Played"),
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Points",
                team_mode=False,
                games_mode=True,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Skater",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                peak_info={
                    "Connor McDavid": {
                        "x": 3,
                        "y": 3,
                        "age": 28,
                        "season_year": 2024,
                        "game_number": 3,
                    }
                },
                do_prime=True,
                do_era=False,
                selected_season=2024,
            )

        fig = captured["fig"]
        player_trace = next(trace for trace in fig.data if trace.name == "Connor McDavid")
        stem_trace = next(trace for trace in fig.data if trace.name == "_season_lollipop_stems")
        glow_trace = next(trace for trace in fig.data if trace.name == "_season_marker_glow")
        highlight = fig.layout.shapes[0]

        self.assertEqual(player_trace.mode, "markers")
        self.assertEqual(player_trace.marker.size, chart_module.SEASON_MARKER_SIZE)
        self.assertEqual(
            player_trace.hovertemplate,
            "<b>Click for details</b><br><br><b>%{customdata[1]}</b><br>Game %{x}<br>%{y:.0f} Points<extra></extra>",
        )
        self.assertEqual(
            list(player_trace.customdata[0]),
            ["Connor McDavid", "Connor McDavid", 28, 1001, "2024-10-09", "Regular"],
        )
        self.assertEqual(stem_trace.mode, "lines")
        self.assertEqual(glow_trace.mode, "markers")
        self.assertEqual(glow_trace.marker.size, chart_module.SEASON_MARKER_GLOW_SIZE)
        self.assertEqual(
            glow_trace.marker.color,
            chart_module._with_alpha(player_trace.marker.color, chart_module.SEASON_MARKER_GLOW_OPACITY),
        )
        self.assertEqual(int(player_trace.customdata[0][3]), 1001)
        self.assertEqual(highlight.x0, 2.5)
        self.assertEqual(highlight.x1, 3.5)

    def test_render_chart_uses_clickable_age_markers_in_player_mode(self):
        """Make player age-chart points look obviously clickable.

        Args:
            None.

        Returns:
            None.
        """
        processed_df = pd.DataFrame(
            {
                "Age": [27, 28, 29],
                "Points": [96, 112, 109],
                "Player": ["Connor McDavid"] * 3,
                "BaseName": ["Connor McDavid"] * 3,
            }
        )
        captured = {}

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.st, "caption"), patch.object(
            chart_module.components,
            "html",
        ), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age"),
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Points",
                team_mode=False,
                games_mode=False,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Skater",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                do_prime=False,
                do_era=False,
                selected_season="All",
            )

        fig = captured["fig"]
        player_trace = next(trace for trace in fig.data if trace.name == "Connor McDavid")
        glow_trace = next(trace for trace in fig.data if trace.name == "_age_marker_glow")

        self.assertEqual(player_trace.marker.size, chart_module.CLICKABLE_AGE_MARKER_SIZE)
        self.assertEqual(player_trace.marker.line.width, 1.35)
        self.assertEqual(player_trace.marker.line.color, chart_module.CLICKABLE_AGE_MARKER_OUTLINE)
        self.assertEqual(
            player_trace.hovertemplate,
            "<b>Click for details</b><br><br><b>%{customdata[1]}</b><br>Age %{x}<br>%{y:.0f} Points<extra></extra>",
        )
        self.assertEqual(list(player_trace.customdata[0]), ["Connor McDavid", "Connor McDavid"])
        self.assertEqual(player_trace.selected.marker.size, chart_module.CLICKABLE_AGE_MARKER_SIZE)
        self.assertEqual(player_trace.selected.marker.opacity, 1.0)
        self.assertEqual(player_trace.unselected.marker.opacity, 1.0)
        self.assertEqual(glow_trace.mode, "markers")
        self.assertEqual(glow_trace.marker.size, chart_module.CLICKABLE_AGE_MARKER_GLOW_SIZE)
        self.assertEqual(
            glow_trace.marker.color,
            chart_module._with_alpha(player_trace.line.color, chart_module.CLICKABLE_AGE_MARKER_GLOW_OPACITY),
        )

    def test_render_chart_single_season_click_opens_details_without_selection_highlight(self):
        """Open season details from the chart-click bridge without selection styling."""
        processed_df = pd.DataFrame(
            {
                "CumGP": [1, 2, 3, 4],
                "Age": [28, 28, 28, 28],
                "Points": [0, 1, 3, 1],
                "GameId": [1001, 1002, 1003, 1004],
                "GameDate": ["2024-10-09", "2024-10-12", "2024-10-15", "2024-10-18"],
                "GameType": ["Regular"] * 4,
                "Player": ["Connor McDavid"] * 4,
                "BaseName": ["Connor McDavid"] * 4,
            }
        )
        captured = {}
        fake_session_state = SimpleNamespace(
            do_predict=False,
            do_smooth=False,
            x_axis_mode="Games Played",
            teams={"TOR": "Toronto Maple Leafs"},
        )

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        def _bridge_result(**kwargs):
            bridge_data = json.loads(kwargs["data"])
            return SimpleNamespace(
                clicked=_build_chart_click_trigger(
                    bridge_data["chart_instance_id"],
                    nonce="bridge-1",
                    trace_name="Connor McDavid",
                    x=2,
                    y=1,
                    customdata=[
                        "Connor McDavid",
                        "Connor McDavid",
                        28,
                        1002,
                        "2024-10-12",
                        "Regular",
                    ],
                )
            )

        render_kwargs = dict(
            processed_dfs=[processed_df],
            metric="Points",
            team_mode=False,
            games_mode=True,
            do_cumul=False,
            do_base=False,
            do_smooth=False,
            stat_category="Skater",
            historical_baselines={},
            team_baselines={},
            raw_dfs_cache=[],
            ml_clones_dict={},
            season_type="Regular",
            sidebar_keys={},
            do_era=False,
            selected_season=2024,
        )

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.st, "caption"), patch.object(
            chart_module.components,
            "html",
        ), patch.object(
            chart_module,
            "show_season_details",
        ) as mock_show_season_details, patch.object(
            chart_module,
            "CHART_CLICK_BRIDGE",
            side_effect=_bridge_result,
        ), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            fake_session_state,
            create=True,
        ):
            chart_module.render_chart(**render_kwargs)

        player_trace = next(trace for trace in captured["fig"].data if trace.name == "Connor McDavid")
        self.assertEqual(player_trace.marker.size, chart_module.SEASON_MARKER_SIZE)
        self.assertEqual(player_trace.selected.marker.size, chart_module.SEASON_MARKER_SIZE)
        self.assertEqual(player_trace.selected.marker.opacity, 1.0)
        self.assertEqual(player_trace.unselected.marker.opacity, 1.0)
        self.assertFalse(any(trace.name == "_season_selected_glow" for trace in captured["fig"].data))
        mock_show_season_details.assert_called_once()
        self.assertEqual(mock_show_season_details.call_args.kwargs["game_id"], 1002)

    def test_render_chart_team_single_season_click_opens_team_details(self):
        """Open the team game snapshot dialog from a selected-season team point."""
        processed_df = pd.DataFrame(
            {
                "CumGP": [1, 2, 3],
                "GP": [1, 2, 3],
                "Points": [2, 4, 5],
                "Wins": [1, 2, 2],
                "Goals": [4, 7, 9],
                "GoalsAgainst": [3, 5, 8],
                "GameId": [2001, 2002, 2003],
                "GameDate": ["2023-10-10", "2023-10-12", "2023-10-15"],
                "GameType": ["Regular"] * 3,
                "OpponentAbbrev": ["MTL", "BOS", "BUF"],
                "OpponentName": ["Montreal Canadiens", "Boston Bruins", "Buffalo Sabres"],
                "HomeRoadFlag": ["H", "R", "H"],
                "ResultLabel": ["W", "W", "OTL"],
                "RecordLabel": ["1-0", "2-0", "2-0-1"],
                "Player": ["Toronto Maple Leafs"] * 3,
                "BaseName": ["TOR"] * 3,
            }
        )
        captured = {}
        fake_session_state = SimpleNamespace(
            do_predict=False,
            do_smooth=False,
            x_axis_mode="Games Played",
            teams={"TOR": "Toronto Maple Leafs"},
        )

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        def _bridge_result(**kwargs):
            bridge_data = json.loads(kwargs["data"])
            return SimpleNamespace(
                clicked=_build_chart_click_trigger(
                    bridge_data["chart_instance_id"],
                    nonce="team-bridge-1",
                    trace_name="Toronto Maple Leafs",
                    x=2,
                    y=4,
                    customdata=[
                        "TOR",
                        "Toronto Maple Leafs",
                        "2023-10-12",
                        "Regular",
                        "BOS",
                        "Boston Bruins",
                        "R",
                        "W",
                        7,
                        5,
                        2002,
                        "2-0",
                    ],
                )
            )

        render_kwargs = dict(
            processed_dfs=[processed_df],
            metric="Points",
            team_mode=True,
            games_mode=True,
            do_cumul=False,
            do_base=False,
            do_smooth=False,
            stat_category="Team",
            historical_baselines={},
            team_baselines={},
            raw_dfs_cache=[],
            ml_clones_dict={},
            season_type="Regular",
            sidebar_keys={},
            do_era=False,
            selected_season=2023,
        )

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module,
            "show_team_game_details",
        ) as mock_show_team_game_details, patch.object(
            chart_module,
            "CHART_CLICK_BRIDGE",
            side_effect=_bridge_result,
        ), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            fake_session_state,
            create=True,
        ):
            chart_module.render_chart(**render_kwargs)

        team_trace = next(trace for trace in captured["fig"].data if trace.name == "Toronto Maple Leafs")
        self.assertEqual(
            team_trace.hovertemplate,
            "<b>Click for details</b><br><br><b>%{customdata[1]}</b><br>%{customdata[2]} · %{customdata[3]}<br>Opponent: %{customdata[5]} (%{customdata[6]})<br>Result: %{customdata[7]} %{customdata[8]:.0f}-%{customdata[9]:.0f}<br>Game %{x}<br>%{y:.0f} Points<extra></extra>",
        )
        self.assertEqual(
            list(team_trace.customdata[0]),
            [
                "TOR",
                "Toronto Maple Leafs",
                "2023-10-10",
                "Regular",
                "MTL",
                "Montreal Canadiens",
                "H",
                "W",
                4,
                3,
                2001,
                "1-0",
            ],
        )
        mock_show_team_game_details.assert_called_once()
        self.assertEqual(mock_show_team_game_details.call_args.kwargs["game_id"], 2002)

    def test_render_chart_deduplicates_chart_bridge_nonce_on_rerun(self):
        """Do not reopen the same chart dialog when the bridge replays one nonce."""
        processed_df = pd.DataFrame(
            {
                "Age": [30, 31],
                "Points": [92, 88],
                "Player": ["Artemi Panarin", "Artemi Panarin"],
                "BaseName": ["Artemi Panarin", "Artemi Panarin"],
            }
        )
        fake_session_state = SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age")
        bridge_calls = {"count": 0}

        def _bridge_result(**kwargs):
            bridge_calls["count"] += 1
            bridge_data = json.loads(kwargs["data"])
            return SimpleNamespace(
                clicked=_build_chart_click_trigger(
                    bridge_data["chart_instance_id"],
                    nonce="nonce-1",
                    trace_name="Artemi Panarin",
                    x=30,
                    y=92,
                    customdata=["Artemi Panarin", "Artemi Panarin", 30],
                )
            )

        render_kwargs = dict(
            processed_dfs=[processed_df],
            metric="Points",
            team_mode=False,
            games_mode=False,
            do_cumul=False,
            do_base=False,
            do_smooth=False,
            stat_category="Skater",
            historical_baselines={},
            team_baselines={},
            raw_dfs_cache=[],
            ml_clones_dict={},
            season_type="Regular",
            sidebar_keys={},
            do_era=False,
        )

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module,
            "show_season_details",
        ) as mock_show_season_details, patch.object(
            chart_module,
            "CHART_CLICK_BRIDGE",
            side_effect=_bridge_result,
        ), patch.object(
            chart_module.st,
            "plotly_chart",
            return_value=None,
        ), patch.object(
            chart_module.st,
            "session_state",
            fake_session_state,
            create=True,
        ):
            chart_module.render_chart(**render_kwargs)
            chart_module.render_chart(**render_kwargs)

        mock_show_season_details.assert_called_once()

    def test_render_chart_suppressed_bridge_click_does_not_replay_later(self):
        """Remember suppressed bridge clicks so a later rerun does not reopen them."""
        processed_df = pd.DataFrame(
            {
                "Age": [30, 31],
                "Points": [92, 88],
                "Player": ["Artemi Panarin", "Artemi Panarin"],
                "BaseName": ["Artemi Panarin", "Artemi Panarin"],
            }
        )
        fake_session_state = SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age")

        def _bridge_result(**kwargs):
            bridge_data = json.loads(kwargs["data"])
            return SimpleNamespace(
                clicked=_build_chart_click_trigger(
                    bridge_data["chart_instance_id"],
                    nonce="suppressed-1",
                    trace_name="Artemi Panarin",
                    x=30,
                    y=92,
                    customdata=["Artemi Panarin", "Artemi Panarin", 30],
                )
            )

        render_kwargs = dict(
            processed_dfs=[processed_df],
            metric="Points",
            team_mode=False,
            games_mode=False,
            do_cumul=False,
            do_base=False,
            do_smooth=False,
            stat_category="Skater",
            historical_baselines={},
            team_baselines={},
            raw_dfs_cache=[],
            ml_clones_dict={},
            season_type="Regular",
            sidebar_keys={},
            do_era=False,
        )

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module,
            "show_season_details",
        ) as mock_show_season_details, patch.object(
            chart_module,
            "CHART_CLICK_BRIDGE",
            side_effect=_bridge_result,
        ), patch.object(
            chart_module.st,
            "plotly_chart",
            return_value=None,
        ), patch.object(
            chart_module.st,
            "session_state",
            fake_session_state,
            create=True,
        ):
            chart_module.render_chart(suppress_dialogs=True, **render_kwargs)
            chart_module.render_chart(suppress_dialogs=False, **render_kwargs)

        mock_show_season_details.assert_not_called()

    def test_render_chart_same_point_can_reopen_with_new_bridge_nonce(self):
        """Allow reopening the same point when the bridge emits a fresh nonce."""
        processed_df = pd.DataFrame(
            {
                "Age": [30, 31],
                "Points": [92, 88],
                "Player": ["Artemi Panarin", "Artemi Panarin"],
                "BaseName": ["Artemi Panarin", "Artemi Panarin"],
            }
        )
        fake_session_state = SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age")
        nonce_specs = ["nonce-a", "nonce-b"]

        def _bridge_result(**kwargs):
            bridge_data = json.loads(kwargs["data"])
            nonce = nonce_specs.pop(0)
            return SimpleNamespace(
                clicked=_build_chart_click_trigger(
                    bridge_data["chart_instance_id"],
                    nonce=nonce,
                    trace_name="Artemi Panarin",
                    x=30,
                    y=92,
                    customdata=["Artemi Panarin", "Artemi Panarin", 30],
                )
            )

        render_kwargs = dict(
            processed_dfs=[processed_df],
            metric="Points",
            team_mode=False,
            games_mode=False,
            do_cumul=False,
            do_base=False,
            do_smooth=False,
            stat_category="Skater",
            historical_baselines={},
            team_baselines={},
            raw_dfs_cache=[],
            ml_clones_dict={},
            season_type="Regular",
            do_era=False,
        )

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module,
            "show_season_details",
        ) as mock_show_season_details, patch.object(
            chart_module,
            "CHART_CLICK_BRIDGE",
            side_effect=_bridge_result,
        ), patch.object(
            chart_module.st,
            "plotly_chart",
            return_value=None,
        ), patch.object(
            chart_module.st,
            "session_state",
            fake_session_state,
            create=True,
        ):
            chart_module.render_chart(sidebar_keys={"search_term": "panarin"}, **render_kwargs)
            fake_session_state._dialog_opened_this_run = False
            chart_module.render_chart(sidebar_keys={"search_term": "panarin"}, **render_kwargs)

        self.assertEqual(mock_show_season_details.call_count, 2)

    def test_mount_chart_click_bridge_uses_stable_key_and_payload(self):
        """Mount the chart click bridge with one stable key and chart id payload."""
        fake_result = SimpleNamespace(clicked=None)

        with patch.object(chart_module, "CHART_CLICK_BRIDGE", return_value=fake_result) as mock_bridge:
            clicked = chart_module._mount_chart_click_bridge("chart_123")

        self.assertIsNone(clicked)
        mock_bridge.assert_called_once()
        self.assertEqual(mock_bridge.call_args.kwargs["key"], chart_module.CHART_CLICK_BRIDGE_MOUNT_KEY)
        self.assertIn("on_clicked_change", mock_bridge.call_args.kwargs)
        bridge_data = json.loads(mock_bridge.call_args.kwargs["data"])
        self.assertEqual(bridge_data["chart_instance_id"], "chart_123")
        self.assertEqual(bridge_data["anchor_id"], chart_module.CHART_CLICK_BRIDGE_ANCHOR_ID)

    def test_parse_chart_click_trigger_rejects_malformed_payloads(self):
        """Ignore malformed chart bridge payloads before dialog dispatch."""
        self.assertIsNone(chart_module._parse_chart_click_trigger(None, "chart_123"))
        self.assertIsNone(chart_module._parse_chart_click_trigger("not-json", "chart_123"))
        self.assertIsNone(
            chart_module._parse_chart_click_trigger(
                json.dumps({"nonce": "abc", "chart_instance_id": "other"}),
                "chart_123",
            )
        )
        self.assertIsNone(
            chart_module._parse_chart_click_trigger(
                json.dumps({"chart_instance_id": "chart_123"}),
                "chart_123",
            )
        )

    def test_show_chart_dialog_from_trigger_ignores_helper_trace_clicks(self):
        """Drop helper-trace bridge clicks so glow traces never open dialogs."""
        session_state = {}
        trigger_value = _build_chart_click_trigger(
            "chart_123",
            nonce="helper-1",
            trace_name="_age_marker_glow",
            x=30,
            y=92,
            customdata=["Artemi Panarin", "Artemi Panarin", 30],
        )

        with patch.object(chart_module, "show_season_details") as mock_show_season_details, patch.object(
            chart_module.st,
            "session_state",
            session_state,
            create=True,
        ):
            opened = chart_module._show_chart_dialog_from_trigger(
                trigger_value,
                "chart_123",
                suppress_dialogs=False,
                team_mode=False,
                games_mode=False,
                is_single_season_team_games=False,
                has_exact_game_custom_data=False,
                metric="Points",
                final_df=pd.DataFrame(),
                raw_dfs_cache=[],
                season_type="Regular",
                selected_season="All",
                do_cumul=False,
                ml_clones_dict={},
                historical_baselines={},
                stat_category="Skater",
                do_era=False,
            )

        self.assertFalse(opened)
        mock_show_season_details.assert_not_called()
        self.assertEqual(session_state[chart_module.LAST_HANDLED_CHART_CLICK_NONCE_SESSION_KEY], "helper-1")

    def test_render_chart_styles_reference_baseline_trace(self):
        """Keep the renamed baseline trace styled as a muted dashed reference line.

        Args:
            None.

        Returns:
            None.
        """
        processed_df = pd.DataFrame(
            {
                "Age": [30, 31, 30, 31],
                "Points": [92, 88, 44, 43],
                "Player": [
                    "Artemi Panarin",
                    "Artemi Panarin",
                    "Reference baseline",
                    "Reference baseline",
                ],
                "BaseName": ["Artemi Panarin", "Artemi Panarin", "Baseline", "Baseline"],
            }
        )
        captured = {}

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age"),
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Points",
                team_mode=False,
                games_mode=False,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Skater",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                do_era=False,
            )

        baseline_trace = next(trace for trace in captured["fig"].data if trace.name == "Reference baseline")

        self.assertEqual(baseline_trace.line.dash, "14px,10px")
        self.assertEqual(
            baseline_trace.line.color,
            chart_module._with_alpha(chart_module.BASELINE_LINE_COLOR, chart_module.BASELINE_LINE_OPACITY),
        )
        self.assertEqual(baseline_trace.line.width, 4)
        self.assertEqual(baseline_trace.marker.size, 8)
        self.assertEqual(baseline_trace.marker.color, "rgba(220, 220, 220, 0.92)")
        self.assertEqual(baseline_trace.marker.symbol, "circle")
        self.assertFalse(baseline_trace.showlegend)
        self.assertNotIn("Click for details", baseline_trace.hovertemplate)
        self.assertEqual(baseline_trace.selected.marker.opacity, 1.0)
        self.assertEqual(baseline_trace.unselected.marker.opacity, 1.0)

    def test_render_chart_injects_player_trace_toggle_js(self):
        """Wire the custom comparison-card toggle buttons to Plotly trace groups."""
        processed_df = pd.DataFrame(
            {
                "Age": [27, 28, 29],
                "Points": [95, 102, 99],
                "Player": ["Nathan MacKinnon"] * 3,
                "BaseName": ["Nathan MacKinnon"] * 3,
            }
        )
        captured = {}

        def _capture_plot(fig, **_kwargs):
            captured["fig"] = fig
            return None

        def _capture_html(script, **_kwargs):
            captured["script"] = script
            return None

        with patch.object(chart_module.st, "markdown"), patch.object(
            chart_module.components,
            "html",
            side_effect=_capture_html,
        ), patch.object(
            chart_module.st,
            "plotly_chart",
            side_effect=_capture_plot,
        ), patch.object(
            chart_module.st,
            "session_state",
            SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age"),
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Points",
                team_mode=False,
                games_mode=False,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Skater",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                do_era=False,
            )

        self.assertIn("var CHART_INSTANCE_ID = ", captured["script"])
        self.assertIn("var ENABLE_PLAYER_TRACE_TOGGLES = true;", captured["script"])
        self.assertIn("parent.__nhlAgeChartTraceToggleState", captured["script"])
        self.assertIn("function bindPlayerTraceToggleButtons(parent, Plotly)", captured["script"])
        self.assertIn("function patchHoverLabelRects(targetPlot)", captured["script"])
        self.assertIn('[data-nhl-trace-toggle="1"]', captured["script"])
        self.assertNotIn("function bindPlayerRemoveButtons(parent)", captured["script"])
        self.assertNotIn("findHiddenPlayerRemoveButton", captured["script"])
        self.assertNotIn('[data-nhl-player-remove="1"]', captured["script"])
        self.assertNotIn("ENABLE_CUSTOM_HOVER_POPOVER", captured["script"])
        self.assertNotIn("function bindCustomHoverHandlers(parent, Plotly, plot)", captured["script"])

    def test_render_chart_ignores_reference_baseline_clicks(self):
        """Ignore baseline point clicks so they do not open the details dialog.

        Args:
            None.

        Returns:
            None.
        """
        processed_df = pd.DataFrame(
            {
                "Age": [30, 31, 30, 31],
                "Points": [92, 88, 44, 43],
                "Player": [
                    "Artemi Panarin",
                    "Artemi Panarin",
                    "Reference baseline",
                    "Reference baseline",
                ],
                "BaseName": ["Artemi Panarin", "Artemi Panarin", "Baseline", "Baseline"],
            }
        )

        def _bridge_result(**kwargs):
            bridge_data = json.loads(kwargs["data"])
            return SimpleNamespace(
                clicked=_build_chart_click_trigger(
                    bridge_data["chart_instance_id"],
                    nonce="baseline-1",
                    trace_name="Reference baseline",
                    x=30,
                    y=44,
                    customdata=["Baseline", "Reference baseline", 30],
                )
            )

        with patch.object(chart_module.st, "markdown"), patch.object(chart_module.components, "html"), patch.object(
            chart_module,
            "show_season_details",
        ) as mock_show_season_details, patch.object(
            chart_module,
            "CHART_CLICK_BRIDGE",
            side_effect=_bridge_result,
        ), patch.object(
            chart_module.st,
            "plotly_chart",
            return_value=None,
        ), patch.object(
            chart_module.st,
            "session_state",
            SimpleNamespace(do_predict=False, do_smooth=False, x_axis_mode="Age"),
            create=True,
        ):
            chart_module.render_chart(
                processed_dfs=[processed_df],
                metric="Points",
                team_mode=False,
                games_mode=False,
                do_cumul=False,
                do_base=False,
                do_smooth=False,
                stat_category="Skater",
                historical_baselines={},
                team_baselines={},
                raw_dfs_cache=[],
                ml_clones_dict={},
                season_type="Regular",
                sidebar_keys={},
                do_era=False,
            )

        mock_show_season_details.assert_not_called()

    def test_chart_source_uses_reference_baseline_label(self):
        """Keep the shorter player baseline legend label in place.

        Args:
            None.

        Returns:
            None.
        """
        chart_text = (Path(__file__).resolve().parents[1] / "nhl" / "chart.py").read_text(encoding="utf-8")

        self.assertIn("base_label = 'Reference baseline'", chart_text)
        self.assertNotIn("Skater 75th Percentile Baseline", chart_text)

    def test_chart_source_uses_toolbar_copy_button_and_internal_modebar(self):
        """Keep share-link wiring in the toolbar and Plotly controls in-chart.

        Args:
            None.

        Returns:
            None.
        """
        chart_text = (Path(__file__).resolve().parents[1] / "nhl" / "chart.py").read_text(encoding="utf-8")

        self.assertIn("_build_chart_toolbar_markup(chart_header, share_button_id, toolbar_id)", chart_text)
        self.assertIn("var SHARE_BUTTON_ID = ", chart_text)
        self.assertIn("var TOOLBAR_ID = ", chart_text)
        self.assertIn("function resolveLiveShareButton(parent)", chart_text)
        self.assertIn("function bindShareButton(parent, attemptsLeft)", chart_text)
        self.assertIn("var CHART_INSTANCE_ID = ", chart_text)
        self.assertIn("var ENABLE_PLAYER_TRACE_TOGGLES = ", chart_text)
        self.assertIn("parent.__nhlAgeChartTraceToggleState", chart_text)
        self.assertIn("function bindPlayerTraceToggleButtons(parent, Plotly)", chart_text)
        self.assertIn('CHART_CLICK_BRIDGE_COMPONENT_NAME = "comparison_chart_click_bridge"', chart_text)
        self.assertIn("CHART_CLICK_BRIDGE = st.components.v2.component(", chart_text)
        self.assertIn("function getCurrentTargetPlot(parent)", chart_text)
        self.assertIn("plot.on('plotly_click', handler);", chart_text)
        self.assertIn("parent.document.getElementById(SHARE_BUTTON_ID)", chart_text)
        self.assertIn("toolbar.querySelector('.nhl-chart-share-btn')", chart_text)
        self.assertIn("parent.document.querySelectorAll('.nhl-chart-share-btn')", chart_text)
        self.assertIn("btn.classList.add('is-copied')", chart_text)
        self.assertIn("annotations = chart_axis_cues", chart_text)
        self.assertIn('template    = "plotly_dark"', chart_text)
        self.assertIn('modebar     = dict(bgcolor="rgba(0,0,0,0)")', chart_text)
        self.assertIn('paper_bgcolor = "rgba(0,0,0,0)"', chart_text)
        self.assertIn('plot_bgcolor  = "rgba(0,0,0,0)"', chart_text)
        self.assertIn("showlegend  = False", chart_text)
        self.assertIn('hovermode   = "closest"', chart_text)
        self.assertIn("hoverdistance = CHART_HOVER_DISTANCE", chart_text)
        self.assertNotIn("_get_chart_theme_colors", chart_text)
        self.assertIn("function applySettings(plot, Plotly)", chart_text)
        self.assertIn("Plotly.relayout(plot, updates)", chart_text)
        self.assertIn('return "baseline" in trace_name.casefold()', chart_text)
        self.assertIn("_apply_special_trace_styling(fig, player_colors)", chart_text)
        self.assertIn('groupclick="togglegroup"', chart_text)
        self.assertIn("trace.showlegend = False", chart_text)
        self.assertIn("legendgroup=proj['legendgroup']", chart_text)
        self.assertIn('BASELINE_LINE_DASH = "14px,10px"', chart_text)
        self.assertIn('BASELINE_LINE_COLOR = "rgba(190, 190, 190, 0.72)"', chart_text)
        self.assertIn('BASELINE_MARKER_COLOR = "rgba(220, 220, 220, 0.92)"', chart_text)
        self.assertIn('"displayModeBar": True', chart_text)
        self.assertIn('"toImageButtonOptions": {', chart_text)
        self.assertIn('"filename": _slugify_chart_export_name(chart_header)', chart_text)
        self.assertIn("height      = 430", chart_text)
        self.assertIn("CHART_HOVER_DISTANCE = 32", chart_text)
        self.assertIn('X_AXIS_TICK_COLOR = "rgba(255, 255, 255, 0.80)"', chart_text)
        self.assertIn('Y_AXIS_TICK_COLOR = "rgba(255, 255, 255, 0.80)"', chart_text)
        self.assertIn('X_AXIS_CUE_COLOR = "rgba(255, 255, 255, 0.25)"', chart_text)
        self.assertIn('Y_AXIS_CUE_COLOR = "rgba(255, 255, 255, 0.25)"', chart_text)
        self.assertIn("tickfont          = dict(size=16, family='Arial Black', color=X_AXIS_TICK_COLOR)", chart_text)
        self.assertIn("tickfont           = dict(size=16, family='Arial Black', color=Y_AXIS_TICK_COLOR)", chart_text)
        self.assertIn("function calcResponsiveAxisTickFontSize(width)", chart_text)
        self.assertIn("function syncToolbarTitleOffset(plot, parent)", chart_text)
        self.assertIn("var axisTickFontSize = calcResponsiveAxisTickFontSize(width);", chart_text)
        self.assertIn("'xaxis.tickfont.size': axisTickFontSize,", chart_text)
        self.assertIn("'yaxis.tickfont.size': yAxisTickFontSize,", chart_text)
        self.assertIn("'xaxis.tickfont.size': calcResponsiveAxisTickFontSize(width),", chart_text)
        self.assertIn("chart_click_trigger_value = _mount_chart_click_bridge(chart_key)", chart_text)
        self.assertIn("_show_chart_dialog_from_trigger(", chart_text)
        self.assertNotIn('clickmode   = \'event+select\'', chart_text)

    def test_chart_share_button_rebinds_with_standard_url_encoding(self):
        """Keep copy-link JS rerun-safe and standards-compliant."""
        chart_text = (Path(__file__).resolve().parents[1] / "nhl" / "chart.py").read_text(encoding="utf-8")

        self.assertIn("btn.onclick = function() {", chart_text)
        self.assertIn("var SearchParamsCtor = parent.URLSearchParams || URLSearchParams;", chart_text)
        self.assertIn("var searchParams = new SearchParamsCtor();", chart_text)
        self.assertIn("searchParams.set(String(key), clean);", chart_text)
        self.assertIn("var remainingAttempts = typeof attemptsLeft === 'number' ? attemptsLeft : 12;", chart_text)
        self.assertIn("bindShareButton(parent, remainingAttempts - 1);", chart_text)
        self.assertIn("}, 150);", chart_text)
        self.assertIn("bindShareButton(parent, 12);", chart_text)
        self.assertNotIn("btn.dataset.bound === '1'", chart_text)
        self.assertNotIn("btn.dataset.bound = '1';", chart_text)
        self.assertNotIn(".replace(/%3B/gi, ';')", chart_text)
        self.assertNotIn(".replace(/%2C/gi, ',')", chart_text)
        self.assertIn("'yaxis.tickfont.size': calcResponsiveYAxisTickFontSize(width),", chart_text)
        self.assertIn('var IS_SINGLE_SEASON_MODE = ', chart_text)
        self.assertIn('games_hover_label = "Game" if str(selected_season) != "All" else "Career Game"', chart_text)
        self.assertIn("parent.document.getElementById(TOOLBAR_ID)", chart_text)
        self.assertIn("title.style.paddingLeft = Math.max(0, Math.round(gutter)) + 'px';", chart_text)
        self.assertIn("if do_prime and not team_mode:", chart_text)
        self.assertIn("fig.add_vrect(", chart_text)
        self.assertNotIn("HEADER_ANCHOR_TEXT", chart_text)
        self.assertIn("clickmode   = 'event'", chart_text)
        self.assertNotIn('on_select           = "rerun"', chart_text)
        self.assertNotIn("selection_mode      = \"points\"", chart_text)

    def test_chart_source_repositions_y_axis_cue_for_mobile_widths(self):
        """Keep the y-axis cue responsive through the existing Plotly relayout JS.

        Args:
            None.

        Returns:
            None.
        """
        chart_text = (Path(__file__).resolve().parents[1] / "nhl" / "chart.py").read_text(encoding="utf-8")

        self.assertIn("function calcResponsiveYAxisCueX(width)", chart_text)
        self.assertIn("function calcResponsiveYAxisCueY(width)", chart_text)
        self.assertIn("if (width <= 480) return 0.094;", chart_text)
        self.assertIn("if (width <= 768) return 0.076;", chart_text)
        self.assertIn("if (width <= 480) return 1.000;", chart_text)
        self.assertIn("if (width <= 768) return 1.006;", chart_text)
        self.assertIn("updates['annotations[0].x'] = calcResponsiveYAxisCueX(width);", chart_text)
        self.assertIn("updates['annotations[0].y'] = calcResponsiveYAxisCueY(width);", chart_text)
        self.assertIn("'annotations[0].x': calcResponsiveYAxisCueX(width),", chart_text)
        self.assertIn("'annotations[0].y': calcResponsiveYAxisCueY(width),", chart_text)
        self.assertNotIn(".annotation-text, .annotation text", chart_text)


if __name__ == "__main__":
    unittest.main()
