import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from nhl import dialog


class DialogGuideTests(unittest.TestCase):
    """Lock the help modal structure and the new explainer copy."""

    def test_chart_dialogs_rerun_on_dismiss(self):
        """Keep chart-driven dialogs configured to rerun when dismissed."""
        dialog_path = Path(__file__).resolve().parents[1] / "nhl" / "dialog.py"
        module = ast.parse(dialog_path.read_text(encoding="utf-8"))
        decorator_targets = {
            "show_season_details": "Season Snapshot",
            "show_team_game_details": "Team Game Snapshot",
        }

        for function_name, dialog_title in decorator_targets.items():
            function_node = next(
                node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            dialog_decorator = next(
                decorator
                for decorator in function_node.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "dialog"
            )

            self.assertEqual(ast.literal_eval(dialog_decorator.args[0]), dialog_title)
            self.assertEqual(
                {
                    keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in dialog_decorator.keywords
                    if keyword.arg is not None
                }.get("on_dismiss"),
                "rerun",
            )

    def test_app_guide_uses_five_top_tabs(self):
        """Keep the help modal split into the intended explainer tabs.

        Args:
            None.

        Returns:
            None.
        """
        dialog_text = (Path(__file__).resolve().parents[1] / "nhl" / "dialog.py").read_text(encoding="utf-8")

        self.assertIn(
            'st.tabs(["Projection", "Baseline", "Era adjust skaters", "Era adjust goalies", "Smoothing"])',
            dialog_text,
        )
        self.assertIn('_render_projection_guide_tab()', dialog_text)
        self.assertIn('_render_baseline_guide_tab()', dialog_text)
        self.assertIn('_render_era_adjust_skaters_guide_tab()', dialog_text)
        self.assertIn('_render_era_adjust_goalies_guide_tab()', dialog_text)
        self.assertIn('_render_smoothing_guide_tab()', dialog_text)

    def test_app_guide_keeps_baseline_and_goalie_era_explainers(self):
        """Keep the less-technical baseline and goalie-era guidance in place.

        Args:
            None.

        Returns:
            None.
        """
        dialog_text = (Path(__file__).resolve().parents[1] / "nhl" / "dialog.py").read_text(encoding="utf-8")

        self.assertIn('The dashed baseline line is your historical benchmark.', dialog_text)
        self.assertIn('Goalie era adjust uses separate logic', dialog_text)
        self.assertIn('modern 2018+ environment', dialog_text)
        self.assertIn('With **Era off**, the chart shows raw league scoring.', dialog_text)
        self.assertIn('3-season rolling average', dialog_text)
        self.assertNotIn('era-adjusted value = raw value × multiplier', dialog_text)
        self.assertNotIn('100 raw points becomes 80 era-adjusted points', dialog_text)
        self.assertNotIn('100 raw points becomes 115 era-adjusted points', dialog_text)
        self.assertNotIn('For skaters, era adjust changes Goals, Assists, and Points independently.', dialog_text)
        self.assertNotIn('Short version: smoothing makes the chart calmer, not smarter.', dialog_text)


class MatchupCardTests(unittest.TestCase):
    """Cover exact-game matchup card rendering."""

    def test_matchup_card_helper_is_defined_once(self):
        """Keep one live top-level matchup card helper in the source file."""
        dialog_path = Path(__file__).resolve().parents[1] / "nhl" / "dialog.py"
        module = ast.parse(dialog_path.read_text(encoding="utf-8"))
        definition_count = sum(
            1 for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_build_matchup_card_html"
        )

        self.assertEqual(definition_count, 1)

    def test_matchup_card_labels_away_and_home_explicitly(self):
        """Make the matchup card say who is away and who is home."""
        html = dialog._build_matchup_card_html(
            {
                'away_abbr': 'MIN',
                'away_name': 'Minnesota Wild',
                'away_score': 5,
                'home_abbr': 'CBJ',
                'home_name': 'Columbus Blue Jackets',
                'home_score': 2,
                'venue': 'Nationwide Arena',
                'start_label_cest': 'Fri 19 Dec, 01:00 CET',
                'status_label': 'Final',
            }
        )

        self.assertIn('Away', html)
        self.assertIn('Home', html)
        self.assertIn('MIN', html)
        self.assertIn('CBJ', html)

    def test_matchup_history_card_uses_abbreviations_and_compact_details(self):
        """Keep history cards compact with abbreviations, year suffixes, and OT/SO context."""
        html = dialog._build_matchup_card_html(
            {
                'away_abbr': 'TOR',
                'away_name': 'Toronto Maple Leafs',
                'away_score': 5,
                'home_abbr': 'VGK',
                'home_name': 'Vegas Golden Knights',
                'home_score': 6,
                'game_date': '2026-01-16',
                'venue': 'T-Mobile Arena',
                'start_label_cest': 'Fri 16 Jan, 03:30 CET',
                'status_label': 'Final/OT',
            },
            compact_layout=True,
        )

        self.assertIn('TOR', html)
        self.assertIn('VGK', html)
        self.assertNotIn('Leafs', html)
        self.assertNotIn('Knights', html)
        self.assertNotIn('Toronto Maple Leafs', html)
        self.assertNotIn('Vegas Golden Knights', html)
        self.assertNotIn('Final', html)
        self.assertIn('OT', html)
        self.assertIn("Fri 16 Jan &#x27;26, 03:30 CET", html)
        self.assertIn('flex-wrap:nowrap', html)

    def test_matchup_history_summary_counts_wins_for_each_team(self):
        """Summarize the displayed history so users do not need to count wins manually."""
        summary = dialog._build_matchup_history_summary(
            "LAK",
            "NYI",
            [
                {
                    "away_abbr": "LAK",
                    "away_score": 4,
                    "home_abbr": "NYI",
                    "home_score": 2,
                },
                {
                    "away_abbr": "NYI",
                    "away_score": 5,
                    "home_abbr": "LAK",
                    "home_score": 1,
                },
                {
                    "away_abbr": "LAK",
                    "away_score": 3,
                    "home_abbr": "NYI",
                    "home_score": 3,
                },
            ],
        )

        self.assertEqual(summary, "LAK won 1, NYI won 1, with 1 tie in the last 3 meetings shown.")

    def test_matchup_history_dialog_renders_title_and_all_cards(self):
        """Render the matchup-history modal body with one card per prior meeting."""
        history_games = [
            {
                "away_abbr": "DAL",
                "away_name": "Dallas Stars",
                "away_score": 4,
                "home_abbr": "EDM",
                "home_name": "Edmonton Oilers",
                "home_score": 1,
                "game_date": "2025-11-15",
                "venue": "Rogers Place",
                "start_label_cest": "2025-11-15",
                "status_label": "Final",
            },
            {
                "away_abbr": "EDM",
                "away_name": "Edmonton Oilers",
                "away_score": 3,
                "home_abbr": "DAL",
                "home_name": "Dallas Stars",
                "home_score": 2,
                "game_date": "2025-11-20",
                "venue": "American Airlines Center",
                "start_label_cest": "2025-11-20",
                "status_label": "Final",
            },
        ]

        with patch.object(dialog, "get_matchup_history", return_value=history_games), patch.object(
            dialog.st,
            "markdown",
        ) as mock_markdown, patch.object(
            dialog.st,
            "info",
        ) as mock_info:
            dialog.show_matchup_history.__wrapped__("EDM", "DAL", 10)

        markdown_calls = [str(call.args[0]) for call in mock_markdown.call_args_list if call.args]
        self.assertIn("### EDM at DAL - Last 10 meetings", markdown_calls[0])
        self.assertIn("EDM won 1, DAL won 1 in the last 2 meetings shown.", markdown_calls[1])
        self.assertIn("American Airlines Center", markdown_calls[2])
        self.assertIn("Rogers Place", markdown_calls[3])
        mock_info.assert_not_called()

    def test_matchup_history_dialog_shows_empty_state_when_no_games_exist(self):
        """Show a friendly fallback when no completed meetings are available."""
        with patch.object(dialog, "get_matchup_history", return_value=[]), patch.object(
            dialog.st,
            "markdown",
        ), patch.object(
            dialog.st,
            "info",
        ) as mock_info:
            dialog.show_matchup_history.__wrapped__("EDM", "DAL", 10)

        mock_info.assert_called_once_with("No completed matchup history available right now.")


class IdentityDialogTests(unittest.TestCase):
    """Cover the overview-card identity modals."""

    def test_player_identity_dialog_renders_unique_identity_rows(self):
        """Show player-specific identity details without falling back to card stats."""
        summary = {
            "name": "Connor McDavid",
            "birth_date": "Jan 13, 1997",
            "age": 29,
            "birthplace": "Richmond Hill, Ontario, CAN",
            "shot_label": "Shoots",
            "shot_value": "L",
            "height": "6'1\" / 185 cm",
            "weight": "194 lb / 88 kg",
            "draft": "2015 | EDM | Round 1, pick 1 | 1 overall",
            "first_nhl_season_label": "2015-16",
            "debut_team": "Edmonton Oilers",
            "honors": ["NHL Top 100"],
            "trophies": [
                {"trophy": "Hart Memorial Trophy", "count": 3, "latest_label": "2022-23"},
                {"trophy": "Art Ross Trophy", "count": 5, "latest_label": "2022-23"},
            ],
        }

        with patch.object(dialog, "get_player_identity_summary", return_value=summary), patch.object(
            dialog.st,
            "markdown",
        ) as mock_markdown, patch.object(
            dialog.st,
            "columns",
            return_value=[nullcontext(), nullcontext()],
        ), patch.object(
            dialog.st,
            "session_state",
            {"player_chart_colors": {"Connor McDavid": "#00B4FF"}},
            create=True,
        ), patch.object(
            dialog.st,
            "info",
        ) as mock_info:
            dialog.show_player_identity_details.__wrapped__(8478402)

        markdown_calls = [str(call.args[0]) for call in mock_markdown.call_args_list if call.args]
        self.assertIn("Connor McDavid", markdown_calls[0])
        self.assertIn("#00B4FF", markdown_calls[0])
        self.assertTrue(any("Born" in line and "#00B4FF" in line and "Jan 13, 1997 | Age 29" in line for line in markdown_calls))
        self.assertTrue(any("Draft" in line and "2015 | EDM | Round 1, pick 1 | 1 overall" in line for line in markdown_calls))
        self.assertTrue(any("Debut team" in line and "Edmonton Oilers" in line for line in markdown_calls))
        self.assertTrue(any("Honors" in line and "#00B4FF" in line and "NHL Top 100" in line for line in markdown_calls))
        self.assertTrue(any("Trophies" in line and "#00B4FF" in line and "Hart Memorial Trophy: x3 (latest 2022-23)" in line for line in markdown_calls))
        mock_info.assert_not_called()

    def test_team_identity_dialog_renders_cleanly_without_live_alignment(self):
        """Keep the team modal usable even when standings-based conference data is missing."""
        summary = {
            "team_name": "Utah Hockey Club",
            "joined_nhl_label": "1979-80",
            "current_identity_since_label": "2024-25",
            "conference_name": "",
            "division_name": "",
            "total_nhl_seasons": 46,
            "stanley_cup_count": 1,
            "stanley_cup_labels": ["1978-79"],
            "lineage_label": (
                "Winnipeg Jets (1979-80 to 1995-96) -> Phoenix Coyotes (1996-97 to 2013-14) "
                "-> Arizona Coyotes (2014-15 to 2023-24) -> Utah Hockey Club (2024-25 to present)"
            ),
        }

        with patch.object(dialog, "get_team_identity_summary", return_value=summary), patch.object(
            dialog.st,
            "markdown",
        ) as mock_markdown, patch.object(
            dialog.st,
            "columns",
            return_value=[nullcontext(), nullcontext()],
        ), patch.object(
            dialog.st,
            "info",
        ) as mock_info:
            dialog.show_team_identity_details.__wrapped__("UTA")

        markdown_calls = [str(call.args[0]) for call in mock_markdown.call_args_list if call.args]
        self.assertIn("### Utah Hockey Club", markdown_calls[0])
        self.assertTrue(any("Joined NHL" in line and "1979-80" in line for line in markdown_calls))
        self.assertTrue(any("Current identity since" in line and "2024-25" in line for line in markdown_calls))
        self.assertTrue(any("Total NHL seasons" in line and "46" in line for line in markdown_calls))
        self.assertTrue(any("Franchise lineage" in line and "Utah Hockey Club" in line for line in markdown_calls))
        self.assertTrue(any("Stanley Cups" in line and "1" in line for line in markdown_calls))
        self.assertTrue(any("Stanley Cup wins" in line and "1978-79" in line for line in markdown_calls))
        mock_info.assert_not_called()



class SeasonSnapshotDialogTests(unittest.TestCase):
    """Cover the Season Snapshot player modal additions."""

    def test_percentile_label_does_not_round_sub_100_value_up_to_100(self):
        """Keep near-perfect percentiles honest when the rank is not first."""
        self.assertEqual(dialog._format_percentile_label(99.96466431095406), "99.96")

    def test_season_snapshot_renders_age_rarity_under_career_subtotals(self):
        """Show the age-rarity callout and keep League visible in the season table."""
        raw_df = pd.DataFrame(
            {
                "BaseName": ["Connor McDavid", "Connor McDavid"],
                "League": ["NHL", "NHL"],
                "Age": [25, 25],
                "SeasonYear": [2022, 2022],
                "GameType": ["Regular", "Regular"],
                "PlayerID": [97, 97],
                "PositionCode": ["C", "C"],
                "GP": [40.0, 42.0],
                "Points": [70.0, 83.0],
                "Goals": [30.0, 34.0],
                "Assists": [40.0, 49.0],
                "PIM": [8.0, 10.0],
                "+/-": [10.0, 12.0],
                "Shots": [120.0, 130.0],
                "TotalTOIMins": [780.0, 820.0],
                "Wins": [0.0, 0.0],
                "Shutouts": [0.0, 0.0],
                "Saves": [0.0, 0.0],
                "WeightedSV": [0.0, 0.0],
                "WeightedGAA": [0.0, 0.0],
            }
        )
        rarity_summary = {
            "season_label": "2022-23",
            "metric": "Points",
            "value": 153.0,
            "age": 25,
            "percentile": 99.7,
            "rank": 12,
            "sample_size": 4681,
            "role_label": "forwards",
            "role_percentile": 99.5,
            "role_rank": 8,
            "role_sample_size": 3712,
            "top_seasons": [
                {"display_rank": 1, "player_name": "Wayne Gretzky", "season_label": "1986-87", "value": 183.0},
                {"display_rank": 2, "player_name": "Connor McDavid", "season_label": "2022-23", "value": 153.0},
            ],
            "is_era_adjusted": False,
            "unavailable_reason": "",
        }

        with patch.object(dialog, "get_age_rarity_summary", return_value=rarity_summary), patch.object(
            dialog.st,
            "markdown",
        ) as mock_markdown, patch.object(
            dialog.st,
            "info",
        ), patch.object(
            dialog.st,
            "warning",
        ), patch.object(
            dialog.st,
            "dataframe",
        ) as mock_dataframe:
            dialog.show_season_details.__wrapped__(
                player_name="Connor McDavid",
                age=25,
                raw_dfs_list=[raw_df],
                metric="Points",
                val=153.0,
                is_cumul=False,
                full_df=pd.DataFrame(),
                s_type="Regular",
                ml_clones_dict={},
                historical_baselines={},
                stat_category="Skater",
                do_era=False,
            )

        markdown_calls = [str(call.args[0]) for call in mock_markdown.call_args_list if call.args]
        self.assertTrue(any("Age rarity" in line and "99.7th-percentile" in line for line in markdown_calls))
        self.assertTrue(any("Wayne Gretzky" in line and "1986-87" in line for line in markdown_calls))
        displayed_df = mock_dataframe.call_args.args[0]
        self.assertIn("League", displayed_df.columns)
        self.assertEqual(int(displayed_df.iloc[0]["Points"]), 153)

    def test_single_game_snapshot_returns_before_age_rarity(self):
        """Do not compute age rarity for exact-game snapshots."""
        raw_df = pd.DataFrame(
            {
                "BaseName": ["Connor McDavid"],
                "League": ["NHL"],
                "Age": [25],
                "SeasonYear": [2024],
                "GameType": ["Regular"],
                "GP": [1.0],
                "Points": [3.0],
                "Goals": [1.0],
                "Assists": [2.0],
                "PIM": [0.0],
                "+/-": [2.0],
                "Shots": [5.0],
                "TotalTOIMins": [22.0],
                "Wins": [0.0],
                "Shutouts": [0.0],
                "Saves": [0.0],
                "WeightedSV": [0.0],
                "WeightedGAA": [0.0],
                "GameDate": ["2024-10-10"],
                "GameId": [2024020001],
                "TeamAbbrev": ["EDM"],
                "OpponentAbbrev": ["WPG"],
                "HomeRoadFlag": ["H"],
                "TeamName": ["Edmonton Oilers"],
                "OpponentName": ["Winnipeg Jets"],
            }
        )

        with patch.object(dialog, "get_age_rarity_summary") as mock_rarity, patch.object(
            dialog,
            "get_game_details",
            return_value={},
        ), patch.object(
            dialog.st,
            "markdown",
        ), patch.object(
            dialog.st,
            "info",
        ), patch.object(
            dialog.st,
            "warning",
        ), patch.object(
            dialog.st,
            "dataframe",
        ):
            dialog.show_season_details.__wrapped__(
                player_name="Connor McDavid",
                age=25,
                raw_dfs_list=[raw_df],
                metric="Points",
                val=3.0,
                is_cumul=False,
                full_df=pd.DataFrame(),
                s_type="Regular",
                ml_clones_dict={},
                historical_baselines={},
                stat_category="Skater",
                do_era=False,
                game_id=2024020001,
                game_date="2024-10-10",
                clicked_game_type="Regular",
                game_number=1,
            )

        mock_rarity.assert_not_called()


if __name__ == "__main__":
    unittest.main()
