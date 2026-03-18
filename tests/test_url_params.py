import unittest

from nhl.url_params import (
    _parse_player_params,
    _parse_team_params,
    _resolve_shared_player_names,
    _resolve_shared_team_names,
    apply_params_to_state,
    encode_state_to_params,
)


class UrlParamsTests(unittest.TestCase):
    """Cover URL state encoding and decoding invariants."""

    def test_encode_state_to_params_omits_defaults_and_uses_short_entries(self):
        """Encode session state into compact URL params.

        Args:
            None.

        Returns:
            None.
        """
        session_state = {
            "stat_category": "Goalie",
            "skater_metric": "Goals",
            "goalie_metric": "Save %",
            "team_metric": "Wins",
            "season_type": "Both",
            "x_axis_mode": "Games Played",
            "league_filter": ["NHL", "AHL"],
            "do_smooth": True,
            "do_predict": False,
            "do_era": True,
            "do_cumul_toggle": False,
            "do_base": True,
            "do_prime": False,
            "panel_tab_skater": "overview",
            "panel_tab_goalie": "current-standings",
            "panel_tab_team": None,
            "players": {"97": "Connor McDavid", "30": "Henrik Lundqvist"},
            "teams": {"EDM": "Edmonton Oilers"},
        }

        params = encode_state_to_params(session_state)

        self.assertEqual(params["cat"], "G")
        self.assertEqual(params["sk_m"], "Goals")
        self.assertEqual(params["xm"], "GP")
        self.assertEqual(params["sp"], "Both")
        self.assertEqual(params["lg"], "NHL,AHL")
        self.assertEqual(params["sm"], "1")
        self.assertEqual(params["pr"], "0")
        self.assertEqual(params["era"], "1")
        self.assertEqual(params["pf"], "0")
        self.assertEqual(params["pt_g"], "current-standings")
        self.assertEqual(params["pl"], "97;30")
        self.assertEqual(params["tm"], "EDM")
        self.assertNotIn("go_m", params)
        self.assertNotIn("pt_s", params)
        self.assertNotIn("pt_t", params)
        self.assertNotIn("sk", params)
        self.assertNotIn("go", params)

    def test_encode_state_to_params_drops_all_default_values(self):
        """Skip writing query params when session state still matches defaults.

        Args:
            None.

        Returns:
            None.
        """
        session_state = {
            "stat_category": "Skater",
            "skater_metric": "Points",
            "goalie_metric": "Save %",
            "team_metric": "Points",
            "season_type": "Regular",
            "x_axis_mode": "Age",
            "league_filter": ["NHL"],
            "do_smooth": False,
            "do_predict": True,
            "do_era": False,
            "do_cumul_toggle": False,
            "do_base": True,
            "do_prime": True,
            "panel_tab_skater": "overview",
            "panel_tab_goalie": "overview",
            "panel_tab_team": "overview",
            "players": {},
            "teams": {},
        }

        params = encode_state_to_params(session_state)

        self.assertEqual(params, {})

    def test_apply_params_to_state_supports_legacy_players_and_short_entries(self):
        """Apply URL params into session state with backward compatibility.

        Args:
            None.

        Returns:
            None.
        """
        session_state = {
            "season_type": "Regular",
            "goalie_metric": "Wins",
            "team_metric": "Points",
        }
        params = {
            "cat": "T",
            "tm_m": "Wins",
            "sk_m": "Bad Metric",
            "xm": "SY",
            "lg": "NHL,KHL,NHL",
            "sm": "1",
            "pr": "0",
            "era": "1",
            "cu": "0",
            "bl": "1",
            "pf": "0",
            "pt_s": "trophies",
            "pt_g": "bad tab!",
            "pt_t": "stanley_cup",
            "sk": "97|Connor McDavid",
            "go": "30|Henrik Lundqvist",
            "pl": "88",
            "tm": "EDM",
        }

        apply_params_to_state(params, session_state)

        self.assertEqual(session_state["stat_category"], "Team")
        self.assertEqual(session_state["team_metric"], "Wins")
        self.assertEqual(session_state["goalie_metric"], "Wins")
        self.assertEqual(session_state["season_type"], "Regular")
        self.assertEqual(session_state["x_axis_mode"], "Season Year")
        self.assertEqual(session_state["league_filter"], ["NHL", "KHL"])
        self.assertTrue(session_state["do_smooth"])
        self.assertFalse(session_state["do_predict"])
        self.assertTrue(session_state["do_era"])
        self.assertFalse(session_state["do_cumul_toggle"])
        self.assertTrue(session_state["do_base"])
        self.assertFalse(session_state["do_prime"])
        self.assertEqual(session_state["panel_tab_skater"], "overview")
        self.assertEqual(session_state["panel_tab_goalie"], "overview")
        self.assertEqual(session_state["panel_tab_team"], "current-standings")
        self.assertEqual(
            session_state["players"],
            {"97": "Connor McDavid", "30": "Henrik Lundqvist", "88": "88"},
        )
        self.assertEqual(session_state["teams"], {"EDM": "EDM"})

    def test_encode_state_to_params_canonicalizes_legacy_panel_tab_aliases(self):
        """Rewrite legacy saved tab IDs to the current canonical query-string value."""
        session_state = {
            "stat_category": "Team",
            "skater_metric": "Points",
            "goalie_metric": "Save %",
            "team_metric": "Points",
            "season_type": "Regular",
            "x_axis_mode": "Season Year",
            "league_filter": ["NHL"],
            "do_smooth": False,
            "do_predict": True,
            "do_era": False,
            "do_cumul_toggle": False,
            "do_base": True,
            "do_prime": True,
            "panel_tab_skater": "overview",
            "panel_tab_goalie": "overview",
            "panel_tab_team": "stanley_cup",
            "players": {},
            "teams": {"COL": "Colorado Avalanche"},
        }

        params = encode_state_to_params(session_state)

        self.assertEqual(params["pt_t"], "current-standings")

    def test_encode_state_to_params_omits_x_axis_when_team_chart_season_forces_games(self):
        """Do not redundantly encode Games Played when team season mode already implies it."""
        session_state = {
            "stat_category": "Team",
            "skater_metric": "Points",
            "goalie_metric": "Save %",
            "team_metric": "Points",
            "season_type": "Regular",
            "chart_season": 2023,
            "x_axis_mode": "Games Played",
            "league_filter": ["NHL"],
            "do_smooth": False,
            "do_predict": True,
            "do_era": False,
            "do_cumul_toggle": False,
            "do_base": True,
            "do_prime": True,
            "panel_tab_skater": "overview",
            "panel_tab_goalie": "overview",
            "panel_tab_team": "overview",
            "players": {},
            "teams": {"TOR": "Toronto Maple Leafs"},
        }

        params = encode_state_to_params(session_state)

        self.assertEqual(params["cat"], "T")
        self.assertEqual(params["cs"], "2023")
        self.assertNotIn("xm", params)

    def test_parse_player_params_sanitizes_names_and_ignores_invalid_ids(self):
        """Keep only valid numeric IDs and sanitize legacy URL display names."""
        players = _parse_player_params("97|Connor <script>alert(1)</script>;abc|Bad;088;0|Zero")

        self.assertEqual(
            players,
            {
                "97": "Connor alert(1)",
                "88": "88",
            },
        )

    def test_parse_team_params_sanitizes_names_and_ignores_invalid_keys(self):
        """Keep only valid abbreviation-shaped keys and sanitize legacy team names."""
        teams = _parse_team_params("EDM|<b>Oilers</b>;sea;NY<script>|Nope")

        self.assertEqual(
            teams,
            {
                "EDM": "Oilers",
                "SEA": "SEA",
            },
        )

    def test_resolve_shared_player_names_prefers_canonical_lookup(self):
        """Use repo-known player names instead of trusting URL-provided display text."""
        resolved = _resolve_shared_player_names(
            {
                "97": "Connor <script>alert(1)</script>",
                "12345": "<b>Prospect</b>",
            },
            {"97": "Connor McDavid"},
        )

        self.assertEqual(
            resolved,
            {
                "97": "Connor McDavid",
                "12345": "Prospect",
            },
        )

    def test_resolve_shared_team_names_prefers_canonical_lookup(self):
        """Use repo-known franchise names instead of trusting URL-provided display text."""
        resolved = _resolve_shared_team_names(
            {
                "EDM": "Not the Oilers",
                "QMJ": "<b>Quebec Nordiques</b>",
            }
        )

        self.assertEqual(
            resolved,
            {
                "EDM": "Edmonton Oilers",
                "QMJ": "Quebec Nordiques",
            },
        )


if __name__ == "__main__":
    unittest.main()
