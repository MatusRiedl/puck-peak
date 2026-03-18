import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import nhl.comparison as comparison_module
from nhl.comparison import (
    _build_card_stat_row,
    _build_live_game_card_html,
    _build_live_game_card_link_html,
    _prime_chart_season_picker,
    _sync_chart_season_picker,
    get_panel_tab_ids,
)


def _joined_markdown_output(mock_markdown) -> str:
    """Return concatenated markdown payloads in original render order."""
    return "\n".join(str(call.args[0]) for call in mock_markdown.call_args_list if call.args)


def _mock_columns(spec, **kwargs):
    """Return nullcontext columns for Streamlit column layout tests."""
    del kwargs
    count = int(spec) if isinstance(spec, int) else len(spec)
    return [nullcontext() for _ in range(count)]


class ComparisonTests(unittest.TestCase):
    """Cover the current comparison-panel surface."""

    def test_panel_tabs_include_overview_and_current_standings(self):
        """Keep the detail tabs aligned with the current runtime UI."""
        self.assertEqual(get_panel_tab_ids(), {"overview", "current-standings"})

    def test_live_game_card_html_includes_short_names_logos_and_probability_strip(self):
        """Render one unified predictions card with short team names and odds."""
        html = _build_live_game_card_html(
            {
                "away_abbr": "DET",
                "away_name": "Detroit Red Wings",
                "home_abbr": "TOR",
                "home_name": "Toronto Maple Leafs",
                "start_label_cest": "Tue 10 Mar, 01:00 CET",
                "venue": "Scotiabank Arena",
                "game_type": 2,
                "pregame_win_prob": {
                    "away_pct": 34,
                    "home_pct": 66,
                    "model_label": "Base model: TOR edge from season points %.",
                    "goalie_label": "Goalie proxy: TOR +2.0 pts from save% edge.",
                },
            }
        )

        self.assertIn("DET_light.svg", html)
        self.assertIn("TOR_light.svg", html)
        self.assertIn("Red Wings", html)
        self.assertIn("Maple Leafs", html)
        self.assertNotIn("Detroit Red Wings", html)
        self.assertNotIn("Toronto Maple Leafs", html)
        self.assertIn("Scotiabank Arena", html)
        self.assertIn("34%", html)
        self.assertIn("66%", html)
        self.assertIn("live-game-card--home-lead", html)
        self.assertIn("live-games-probability__divider", html)

    def test_live_game_card_html_falls_back_to_muted_copy_without_probability(self):
        """Show a muted placeholder when pregame odds are unavailable."""
        html = _build_live_game_card_html(
            {
                "away_abbr": "BUF",
                "away_name": "Buffalo Sabres",
                "home_abbr": "MTL",
                "home_name": "Montreal Canadiens",
                "start_label_cest": "Time TBD",
                "venue": "",
                "pregame_win_prob": None,
            }
        )

        self.assertIn("Estimate unavailable.", html)
        self.assertIn("live-game-card--no-prob", html)

    def test_live_game_card_link_html_preserves_shared_params_and_adds_matchup_history_query(self):
        """Wrap each prediction card in a full-card self-link for matchup history."""
        html = _build_live_game_card_link_html(
            {
                "away_abbr": "DET",
                "away_name": "Detroit Red Wings",
                "home_abbr": "TOR",
                "home_name": "Toronto Maple Leafs",
                "start_label_cest": "Tue 10 Mar, 01:00 CET",
                "venue": "Scotiabank Arena",
                "pregame_win_prob": None,
            },
            share_params={"cat": "T", "tm": "DET;TOR"},
        )

        self.assertIn("live-game-card-link", html)
        self.assertIn("data-nhl-matchup-history='1'", html)
        self.assertIn("data-matchup-history='DET,TOR'", html)
        self.assertIn("cat=T", html)
        self.assertIn("tm=DET%3BTOR", html)
        self.assertIn("mh=DET%2CTOR", html)

    def test_predictions_panel_renders_heading_anchor_and_shared_renderer(self):
        """Keep the predictions UI on the dedicated right-rail panel shell."""
        with patch.object(comparison_module.st, "markdown") as mock_markdown, patch.object(
            comparison_module.st,
            "tabs",
            return_value=[nullcontext()],
        ) as mock_tabs, patch.object(
            comparison_module,
            "_mount_matchup_history_click_bridge",
            return_value=None,
        ), patch.object(
            comparison_module,
            "_render_live_games_tab",
        ) as mock_render_live_games, patch.object(
            comparison_module,
            "_show_matchup_history_from_trigger",
            return_value=False,
        ), patch.object(
            comparison_module,
            "_consume_matchup_history_request",
            return_value=None,
        ), patch.object(
            comparison_module,
            "_show_pending_matchup_history_dialog",
            return_value=None,
        ):
            comparison_module.render_predictions_panel()

        markup = _joined_markdown_output(mock_markdown)
        self.assertIn("comparison-predictions-panel", markup)
        self.assertIn("comparison-panel-heading--predictions", markup)
        self.assertIn("Next matches prediction", markup)
        mock_tabs.assert_not_called()
        mock_render_live_games.assert_called_once_with(share_params=None)

    def test_predictions_panel_consumes_matchup_history_query_and_opens_dialog_once(self):
        """Use the old `mh` URL path only as fallback when JS does not trigger."""
        session_state = {}
        query_params = {"cat": "T", "mh": "EDM,DAL"}

        with patch.object(comparison_module.st, "markdown"), patch.object(
            comparison_module.st,
            "tabs",
            return_value=[nullcontext()],
        ), patch.object(
            comparison_module,
            "_mount_matchup_history_click_bridge",
            return_value=None,
        ), patch.object(
            comparison_module,
            "_render_live_games_tab",
        ) as mock_render_live_games, patch.object(
            comparison_module,
            "show_matchup_history",
        ) as mock_show_matchup_history, patch.object(
            comparison_module.st,
            "session_state",
            session_state,
            create=True,
        ), patch.object(
            comparison_module.st,
            "query_params",
            query_params,
            create=True,
        ):
            comparison_module.render_predictions_panel(share_params={"cat": "T"})

        self.assertNotIn("mh", query_params)
        self.assertNotIn("_pending_matchup_history", session_state)
        mock_render_live_games.assert_called_once_with(share_params={"cat": "T"})
        mock_show_matchup_history.assert_called_once_with(away_abbr="EDM", home_abbr="DAL")

    def test_predictions_panel_uses_pre_mounted_matchup_trigger_when_provided(self):
        """Avoid remounting the bridge when app.py already mounted it before the chart."""
        with patch.object(comparison_module.st, "markdown"), patch.object(
            comparison_module,
            "_mount_matchup_history_click_bridge",
        ) as mock_mount_bridge, patch.object(
            comparison_module,
            "_render_live_games_tab",
        ) as mock_render_live_games, patch.object(
            comparison_module,
            "_show_matchup_history_from_trigger",
            return_value=False,
        ) as mock_show_from_trigger, patch.object(
            comparison_module,
            "_consume_matchup_history_request",
            return_value=None,
        ), patch.object(
            comparison_module,
            "_show_pending_matchup_history_dialog",
            return_value=None,
        ):
            comparison_module.render_predictions_panel(
                share_params={"cat": "T"},
                matchup_history_trigger_value="EDM,DAL|nonce-1",
                matchup_history_bridge_mounted=True,
            )

        mock_mount_bridge.assert_not_called()
        mock_render_live_games.assert_called_once_with(share_params={"cat": "T"})
        mock_show_from_trigger.assert_called_once_with("EDM,DAL|nonce-1")

    def test_predictions_panel_does_not_remount_bridge_when_pre_mount_returns_none(self):
        """Treat a pre-mounted bridge with no click yet as already mounted."""
        with patch.object(comparison_module.st, "markdown"), patch.object(
            comparison_module,
            "_mount_matchup_history_click_bridge",
        ) as mock_mount_bridge, patch.object(
            comparison_module,
            "_render_live_games_tab",
        ) as mock_render_live_games, patch.object(
            comparison_module,
            "_show_matchup_history_from_trigger",
            return_value=False,
        ) as mock_show_from_trigger, patch.object(
            comparison_module,
            "_consume_matchup_history_request",
            return_value=None,
        ), patch.object(
            comparison_module,
            "_show_pending_matchup_history_dialog",
            return_value=None,
        ):
            comparison_module.render_predictions_panel(
                share_params={"cat": "T"},
                matchup_history_trigger_value=None,
                matchup_history_bridge_mounted=True,
            )

        mock_mount_bridge.assert_not_called()
        mock_render_live_games.assert_called_once_with(share_params={"cat": "T"})
        mock_show_from_trigger.assert_called_once_with(None)

    def test_predictions_panel_mounts_js_click_bridge(self):
        """Mount the JS component so card clicks can trigger in-app modal opens."""
        fake_result = SimpleNamespace(clicked=None)

        with patch.object(comparison_module, "_MATCHUP_HISTORY_CLICK_BRIDGE", return_value=fake_result) as mock_bridge:
            clicked = comparison_module._mount_matchup_history_click_bridge()

        self.assertIsNone(clicked)
        mock_bridge.assert_called_once()
        self.assertEqual(mock_bridge.call_args.kwargs["key"], "comparison_matchup_history_click_bridge")
        self.assertIn("on_clicked_change", mock_bridge.call_args.kwargs)

    def test_detail_tabs_mount_identity_card_click_bridge(self):
        """Mount the dedicated detail-card bridge so overview cards can open dialogs."""
        fake_result = SimpleNamespace(clicked=None)

        with patch.object(comparison_module, "_IDENTITY_CARD_CLICK_BRIDGE", return_value=fake_result) as mock_bridge:
            clicked = comparison_module._mount_identity_card_click_bridge()

        self.assertIsNone(clicked)
        mock_bridge.assert_called_once()
        self.assertEqual(mock_bridge.call_args.kwargs["key"], "comparison_identity_card_click_bridge")
        self.assertIn("on_clicked_change", mock_bridge.call_args.kwargs)

    def test_show_matchup_history_from_trigger_opens_once_per_nonce(self):
        """Handle valid JS payloads and ignore duplicates from the same rerun state."""
        session_state = {}
        with patch.object(
            comparison_module,
            "show_matchup_history",
        ) as mock_show_matchup_history, patch.object(
            comparison_module.st,
            "session_state",
            session_state,
            create=True,
        ):
            first = comparison_module._show_matchup_history_from_trigger("EDM,DAL|12345")
            duplicate = comparison_module._show_matchup_history_from_trigger("EDM,DAL|12345")
            session_state["_dialog_opened_this_run"] = False
            second = comparison_module._show_matchup_history_from_trigger("EDM,DAL|67890")

        self.assertTrue(first)
        self.assertFalse(duplicate)
        self.assertTrue(second)
        self.assertEqual(mock_show_matchup_history.call_count, 2)
        self.assertEqual(
            mock_show_matchup_history.call_args_list[0].kwargs,
            {"away_abbr": "EDM", "home_abbr": "DAL"},
        )
        self.assertEqual(session_state["_last_matchup_history_trigger_nonce"], "67890")

    def test_show_pending_matchup_history_dialog_keeps_request_if_slot_is_busy(self):
        """Preserve pending fallback requests until the dialog slot is actually free."""
        session_state = {
            "_pending_matchup_history": ("EDM", "DAL"),
            "_dialog_opened_this_run": True,
        }

        with patch.object(
            comparison_module,
            "show_matchup_history",
        ) as mock_show_matchup_history, patch.object(
            comparison_module.st,
            "session_state",
            session_state,
            create=True,
        ):
            comparison_module._show_pending_matchup_history_dialog()

        mock_show_matchup_history.assert_not_called()
        self.assertEqual(session_state["_pending_matchup_history"], ("EDM", "DAL"))

    def test_has_pending_matchup_history_dialog_request_detects_query_and_session_state(self):
        """Report pending prediction-modal work before the chart renders."""
        with patch.object(
            comparison_module.st,
            "session_state",
            {"_pending_matchup_history": ("EDM", "DAL")},
            create=True,
        ), patch.object(
            comparison_module.st,
            "query_params",
            {},
            create=True,
        ):
            self.assertTrue(comparison_module.has_pending_matchup_history_dialog_request())

        with patch.object(
            comparison_module.st,
            "session_state",
            {},
            create=True,
        ), patch.object(
            comparison_module.st,
            "query_params",
            {"mh": "VGK,EDM"},
            create=True,
        ):
            self.assertTrue(comparison_module.has_pending_matchup_history_dialog_request())

        with patch.object(
            comparison_module.st,
            "session_state",
            {},
            create=True,
        ), patch.object(
            comparison_module.st,
            "query_params",
            {},
            create=True,
        ):
            self.assertFalse(comparison_module.has_pending_matchup_history_dialog_request())

    def test_show_matchup_history_from_trigger_rejects_malformed_payloads(self):
        """Ignore malformed or incomplete JS trigger values."""
        session_state = {}
        with patch.object(
            comparison_module,
            "show_matchup_history",
        ) as mock_show_matchup_history, patch.object(
            comparison_module.st,
            "session_state",
            session_state,
            create=True,
        ):
            self.assertFalse(comparison_module._show_matchup_history_from_trigger(None))
            self.assertFalse(comparison_module._show_matchup_history_from_trigger("EDM,DAL"))
            self.assertFalse(comparison_module._show_matchup_history_from_trigger("EDM|12345"))
            self.assertFalse(comparison_module._show_matchup_history_from_trigger("EDM,EDM|12345"))
            self.assertFalse(comparison_module._show_matchup_history_from_trigger("XXX,DAL|12345"))

        mock_show_matchup_history.assert_not_called()

    def test_show_identity_card_from_trigger_opens_once_per_nonce(self):
        """Dispatch valid player and team payloads while ignoring duplicates."""
        session_state = {}
        with patch.object(
            comparison_module,
            "show_player_identity_details",
        ) as mock_show_player, patch.object(
            comparison_module,
            "show_team_identity_details",
        ) as mock_show_team, patch.object(
            comparison_module.st,
            "session_state",
            session_state,
            create=True,
        ):
            first = comparison_module._show_identity_card_from_trigger("player:8478402|12345")
            duplicate = comparison_module._show_identity_card_from_trigger("player:8478402|12345")
            session_state["_dialog_opened_this_run"] = False
            second = comparison_module._show_identity_card_from_trigger("team:EDM|67890")

        self.assertTrue(first)
        self.assertFalse(duplicate)
        self.assertTrue(second)
        mock_show_player.assert_called_once_with(8478402)
        mock_show_team.assert_called_once_with("EDM")
        self.assertEqual(session_state["_last_identity_card_trigger_nonce"], "67890")

    def test_show_identity_card_from_trigger_rejects_malformed_payloads(self):
        """Ignore malformed entity triggers before trying to open a dialog."""
        session_state = {}
        with patch.object(
            comparison_module,
            "show_player_identity_details",
        ) as mock_show_player, patch.object(
            comparison_module,
            "show_team_identity_details",
        ) as mock_show_team, patch.object(
            comparison_module.st,
            "session_state",
            session_state,
            create=True,
        ):
            self.assertFalse(comparison_module._show_identity_card_from_trigger(None))
            self.assertFalse(comparison_module._show_identity_card_from_trigger("player:8478402"))
            self.assertFalse(comparison_module._show_identity_card_from_trigger("player:abc|12345"))
            self.assertFalse(comparison_module._show_identity_card_from_trigger("team:XXX|12345"))
            self.assertFalse(comparison_module._show_identity_card_from_trigger("coach:EDM|12345"))

        mock_show_player.assert_not_called()
        mock_show_team.assert_not_called()

    def test_comparison_source_registers_v2_matchup_history_component(self):
        """Keep the JS click bridge on the official Streamlit v2 component API."""
        comparison_text = (Path(__file__).resolve().parents[1] / "nhl" / "comparison.py").read_text(encoding="utf-8")

        self.assertIn('st.components.v2.component(', comparison_text)
        self.assertIn('"comparison_matchup_history_click_bridge"', comparison_text)

    def test_comparison_source_registers_v2_identity_card_component(self):
        """Keep the detail-card bridge on the official Streamlit v2 component API."""
        comparison_text = (Path(__file__).resolve().parents[1] / "nhl" / "comparison.py").read_text(encoding="utf-8")

        self.assertIn('"comparison_identity_card_click_bridge"', comparison_text)
        self.assertIn("target.closest('[data-nhl-trace-toggle=\"1\"]')", comparison_text)

    def test_current_standings_tab_renders_board_meta_and_favorite_text(self):
        """Render the Current Standings board using the shared comparison-area styles."""
        board = {
            "generated_at_label": "Current as of Mar 12, 2026 20:28 UTC",
            "favorite_team": {"team_name": "Colorado Avalanche"},
            "divisions": [
                {
                    "division_name": "Central",
                    "conference_name": "Western",
                    "teams": [
                        {
                            "team_abbr": "COL",
                            "team_name": "Colorado Avalanche",
                            "team_common_name": "Avalanche",
                            "team_logo": "https://assets.nhle.com/logos/nhl/svg/COL_light.svg",
                            "games_played": 63,
                            "wins": 43,
                            "losses": 11,
                            "ot_losses": 9,
                            "points": 95,
                            "is_favorite": True,
                        },
                        {
                            "team_abbr": "DAL",
                            "team_name": "Dallas Stars",
                            "team_common_name": "Stars",
                            "team_logo": "https://assets.nhle.com/logos/nhl/svg/DAL_light.svg",
                            "games_played": 64,
                            "wins": 40,
                            "losses": 14,
                            "ot_losses": 10,
                            "points": 90,
                            "is_favorite": False,
                        },
                    ],
                }
            ],
        }

        with patch.object(
            comparison_module,
            "get_stanley_cup_board",
            return_value=board,
        ), patch.object(
            comparison_module.st,
            "markdown",
        ) as mock_markdown:
            comparison_module._render_current_standings_shared()

        markup = _joined_markdown_output(mock_markdown)
        self.assertIn("Cup pick: <strong>Colorado Avalanche</strong>", markup)
        self.assertIn("Current as of Mar 12, 2026 20:28 UTC", markup)
        self.assertIn("Central Division", markup)
        self.assertIn("Western Conference", markup)
        self.assertIn("COL_light.svg", markup)

    def test_current_standings_board_rows_are_clickable_team_identity_targets(self):
        """Expose the same team-detail click payloads on standings rows."""
        board = {
            "generated_at_label": "Current as of Mar 13, 2026 10:28 UTC",
            "favorite_team": {"team_name": "Colorado Avalanche", "rank": 1, "summary_text": "Colorado leads."},
            "divisions": [
                {
                    "division_name": "Central",
                    "conference_name": "Western",
                    "teams": [
                        {
                            "team_abbr": "COL",
                            "team_name": "Colorado Avalanche",
                            "team_common_name": "Avalanche",
                            "team_logo": "https://assets.nhle.com/logos/nhl/svg/COL_light.svg",
                            "games_played": 63,
                            "wins": 43,
                            "losses": 11,
                            "ot_losses": 9,
                            "points": 95,
                            "is_favorite": True,
                        }
                    ],
                }
            ],
        }

        markup = comparison_module._build_current_standings_board_markup(board)

        self.assertIn("comparison-card-shell--clickable stanley-cup-row-shell", markup)
        self.assertIn("data-nhl-identity-card='1'", markup)
        self.assertIn("data-identity-card='team:COL'", markup)

    def test_prime_chart_season_picker_seeds_widget_state_from_canonical_value(self):
        """Keep the selectbox widget pinned to the canonical session value."""
        session_state = {"chart_season": 2024}

        with patch.object(comparison_module.st, "session_state", session_state, create=True):
            _prime_chart_season_picker(["All", 2025, 2024])

        self.assertEqual(session_state["chart_season"], 2024)
        self.assertEqual(session_state["_chart_season_picker"], 2024)

    def test_sync_chart_season_picker_updates_canonical_state(self):
        """Copy the widget season into canonical session state."""
        session_state = {"chart_season": "All", "_chart_season_picker": 2023}

        with patch.object(comparison_module.st, "session_state", session_state, create=True):
            _sync_chart_season_picker()

        self.assertEqual(session_state["chart_season"], 2023)
        self.assertEqual(session_state["_chart_season_picker"], 2023)

    def test_card_stat_row_uses_one_cell_per_label_value_pair(self):
        """Build stat markup with a stable inline cell per stat."""
        html = _build_card_stat_row([("G", 652), ("A", 1094), ("Pts", 1746), ("GP", 1408)])

        self.assertIn("comparison-card-stats", html)
        self.assertEqual(html.count("comparison-card-stats__item"), 4)
        self.assertIn("comparison-card-stats__label'>G:&nbsp;", html)
        self.assertIn("comparison-card-stats__value'>1746", html)

    def test_overview_player_card_uses_chart_color_and_muted_context_rows(self):
        """Color the player name from the chart while keeping context copy muted."""
        processed_df = pd.DataFrame(
            {
                "Age": [18, 19],
                "GP": [79, 80],
                "Goals": [51, 55],
                "Assists": [86, 87],
                "Points": [137, 142],
                "Player": ["Wayne Gretzky", "Wayne Gretzky"],
                "BaseName": ["Wayne Gretzky", "Wayne Gretzky"],
            }
        )

        with patch.object(
            comparison_module,
            "get_player_headshot",
            return_value=None,
        ), patch.object(
            comparison_module,
            "get_player_current_team",
            return_value=None,
        ), patch.object(
            comparison_module,
            "get_player_roster_info",
            return_value=None,
        ), patch.object(
            comparison_module,
            "get_player_career_rank",
            return_value=1,
        ), patch.object(
            comparison_module.st,
            "container",
            return_value=nullcontext(),
        ), patch.object(
            comparison_module.st,
            "columns",
            return_value=[nullcontext(), nullcontext()],
        ), patch.object(
            comparison_module.st,
            "button",
            return_value=False,
        ), patch.object(
            comparison_module.st,
            "markdown",
        ) as mock_markdown, patch.object(
            comparison_module.st,
            "session_state",
            SimpleNamespace(player_chart_colors={"Wayne Gretzky": "#7b61ff"}),
            create=True,
        ):
            comparison_module._render_overview_players(
                processed_dfs=[processed_df],
                players={"99": "Wayne Gretzky"},
                peak_info={},
                metric="Points",
                stat_category="Skater",
                season_type="Regular",
            )

        card_markup = _joined_markdown_output(mock_markdown)
        self.assertIn("<strong style='color:#7b61ff;'>Wayne Gretzky</strong>", card_markup)
        self.assertIn("comparison-card-context-row", card_markup)
        self.assertIn("#1 in all-time Points", card_markup)
        self.assertIn("comparison-trace-toggle-row", card_markup)
        self.assertIn("comparison-card-shell--clickable", card_markup)
        self.assertIn("data-identity-card='player:99'", card_markup)

    def test_team_overview_all_time_card_uses_franchise_copy(self):
        """Label the all-time team card against franchise wins after lineage aggregation."""
        with patch.object(
            comparison_module,
            "get_team_all_time_stats",
            return_value={
                "TOR": {
                    "total_wins": 3100,
                    "total_points": 6900,
                    "total_goals": 12000,
                    "total_gp": 6200,
                    "wins_rank": 2,
                    "best_year": 1948,
                    "best_wins": 32,
                    "best_gp": 60,
                }
            },
        ), patch.object(
            comparison_module.st,
            "columns",
            return_value=[nullcontext(), nullcontext()],
        ), patch.object(
            comparison_module.st,
            "button",
            return_value=False,
        ), patch.object(
            comparison_module.st,
            "markdown",
        ) as mock_markdown, patch.object(
            comparison_module.st,
            "session_state",
            SimpleNamespace(player_chart_colors={"Toronto Maple Leafs": "#0055ff"}),
            create=True,
        ), patch.object(
            comparison_module.st,
            "container",
            return_value=nullcontext(),
        ):
            comparison_module._render_overview_teams(
                active_teams={"TOR": "Toronto Maple Leafs"},
                processed_dfs=[],
                metric="Wins",
                selected_season="All",
            )

        card_markup = _joined_markdown_output(mock_markdown)
        self.assertIn("<strong style='color:#0055ff;'>Toronto Maple Leafs</strong>", card_markup)
        self.assertIn("#2 in franchise Wins", card_markup)
        self.assertIn("Best season: 1947-48 -- 32 W in 60 GP", card_markup)
        self.assertIn("comparison-card-shell--clickable", card_markup)
        self.assertIn("data-identity-card='team:TOR'", card_markup)

    def test_comparison_source_no_longer_contains_legacy_live_tab_wrappers(self):
        """Keep removed compatibility wrappers from drifting back into the module."""
        comparison_text = (Path(__file__).resolve().parents[1] / "nhl" / "comparison.py").read_text(encoding="utf-8")

        self.assertNotIn("def render_comparison_area(", comparison_text)
        self.assertNotIn("def render_comparison_panel(", comparison_text)
        self.assertNotIn("def render_team_comparison_panel(", comparison_text)
        self.assertNotIn("def _render_live_games_players(", comparison_text)
        self.assertNotIn("def _render_live_games_teams(", comparison_text)


if __name__ == "__main__":
    unittest.main()
