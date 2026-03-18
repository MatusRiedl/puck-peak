import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from nhl import schedule
from nhl.win_prob import WIN_PROB_FEATURE_ORDER


class ScheduleTests(unittest.TestCase):
    """Cover upcoming-game and featured-player schedule helpers."""

    def setUp(self):
        """Clear cached schedule wrappers so mocks stay honest.

        Args:
            None.

        Returns:
            None.
        """
        for cached_func in (
            schedule.get_live_or_recent_game,
            schedule.get_featured_players,
            schedule.get_upcoming_games,
            schedule.get_game_details,
            schedule.get_matchup_history,
            schedule.get_game_win_probabilities,
            getattr(schedule, "get_team_goalie_proxy_save_percentage", None),
            getattr(schedule, "_get_cached_club_stats", None),
        ):
            if hasattr(cached_func, "clear"):
                cached_func.clear()

    def test_format_game_time_cest_handles_winter_and_summer_offsets(self):
        """Format UTC start times into Central European local time.

        Args:
            None.

        Returns:
            None.
        """
        self.assertEqual(
            schedule._format_game_time_cest("2026-03-07T17:30:00Z"),
            "Sat 07 Mar, 18:30 CET",
        )
        self.assertEqual(
            schedule._format_game_time_cest("2026-04-07T17:30:00Z"),
            "Tue 07 Apr, 19:30 CEST",
        )

    def test_extract_upcoming_games_filters_invalid_rows_and_sorts_by_start(self):
        """Keep only valid future regular-season or playoff games.

        Args:
            None.

        Returns:
            None.
        """
        now_utc = datetime(2026, 3, 7, 12, 0, tzinfo=timezone.utc)
        games = [
            {
                "id": 2,
                "gameType": 2,
                "gameState": "FUT",
                "startTimeUTC": "2026-03-07T20:00:00Z",
                "awayTeam": {"abbrev": "MTL", "name": {"default": "Canadiens"}},
                "homeTeam": {"abbrev": "NYR", "name": {"default": "Rangers"}},
                "venue": {"default": "Madison Square Garden"},
            },
            {
                "id": 1,
                "gameType": 2,
                "gameState": "FUT",
                "startTimeUTC": "2026-03-07T17:30:00Z",
                "awayTeam": {"abbrev": "WSH", "name": {"default": "Capitals"}},
                "homeTeam": {"abbrev": "BOS", "name": {"default": "Bruins"}},
                "venue": {"default": "TD Garden"},
            },
            {
                "id": 3,
                "gameType": 2,
                "gameState": "FINAL",
                "startTimeUTC": "2026-03-07T01:00:00Z",
                "awayTeam": {"abbrev": "PIT", "name": {"default": "Penguins"}},
                "homeTeam": {"abbrev": "PHI", "name": {"default": "Flyers"}},
                "venue": {"default": "Wells Fargo Center"},
            },
            {
                "id": 4,
                "gameType": 1,
                "gameState": "FUT",
                "startTimeUTC": "2026-03-07T18:00:00Z",
                "awayTeam": {"abbrev": "OTT", "name": {"default": "Senators"}},
                "homeTeam": {"abbrev": "BUF", "name": {"default": "Sabres"}},
                "venue": {"default": "KeyBank Center"},
            },
        ]

        upcoming = schedule._extract_upcoming_games(games, now_utc)

        self.assertEqual([game["game_id"] for game in upcoming], [1, 2])
        self.assertEqual(upcoming[0]["game_type"], 2)
        self.assertEqual(upcoming[0]["matchup"], "Washington Capitals at Boston Bruins")
        self.assertEqual(upcoming[0]["venue"], "TD Garden")
        self.assertEqual(upcoming[0]["start_label_cest"], "Sat 07 Mar, 18:30 CET")

    def test_extract_game_details_from_payload_keeps_score_and_final_label(self):
        """Normalize one finished game into the exact-match dialog shape."""
        payload = {
            "games": [
                {
                    "id": 55,
                    "gameDate": "2026-03-07",
                    "gameType": 2,
                    "gameState": "FINAL",
                    "startTimeUTC": "2026-03-07T17:30:00Z",
                    "awayTeam": {"abbrev": "EDM", "name": {"default": "Oilers"}, "score": 4},
                    "homeTeam": {"abbrev": "CGY", "name": {"default": "Flames"}, "score": 2},
                    "venue": {"default": "Scotiabank Saddledome"},
                    "periodDescriptor": {"periodType": "REG"},
                }
            ]
        }

        details = schedule._extract_game_details_from_payload(payload, 55)

        self.assertEqual(details["away_abbr"], "EDM")
        self.assertEqual(details["home_abbr"], "CGY")
        self.assertEqual(details["away_score"], 4)
        self.assertEqual(details["home_score"], 2)
        self.assertEqual(details["status_label"], "Final")
        self.assertEqual(details["venue"], "Scotiabank Saddledome")
        self.assertEqual(details["start_label_cest"], "Sat 07 Mar, 18:30 CET")

    @patch("nhl.schedule.get_game_details")
    @patch("nhl.schedule.get_team_season_game_log")
    @patch("nhl.schedule.get_team_available_nhl_seasons")
    def test_get_matchup_history_collects_multi_season_games_newest_first_and_caps_limit(
        self,
        mock_available_seasons,
        mock_game_log,
        mock_game_details,
    ):
        """Assemble the latest 10 completed meetings across seasons and game types."""
        mock_available_seasons.return_value = [2025, 2024]
        mock_game_details.return_value = {}
        mock_game_log.side_effect = [
            schedule.pd.DataFrame(
                [
                    {
                        "GameDate": "2025-11-20",
                        "GameId": 5011,
                        "GameType": "Regular",
                        "gameTypeId": 2,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "DAL",
                        "OpponentName": "Dallas Stars",
                        "HomeRoadFlag": "H",
                        "Goals": 5,
                        "GoalsAgainst": 3,
                    },
                    {
                        "GameDate": "2025-11-15",
                        "GameId": 5010,
                        "GameType": "Playoffs",
                        "gameTypeId": 3,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "DAL",
                        "OpponentName": "Dallas Stars",
                        "HomeRoadFlag": "R",
                        "Goals": 2,
                        "GoalsAgainst": 4,
                    },
                    {
                        "GameDate": "2025-11-01",
                        "GameId": 5009,
                        "GameType": "Regular",
                        "gameTypeId": 2,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "DAL",
                        "OpponentName": "Dallas Stars",
                        "HomeRoadFlag": "H",
                        "Goals": 4,
                        "GoalsAgainst": 1,
                    },
                    {
                        "GameDate": "2025-10-25",
                        "GameId": 5008,
                        "GameType": "Regular",
                        "gameTypeId": 2,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "DAL",
                        "OpponentName": "Dallas Stars",
                        "HomeRoadFlag": "R",
                        "Goals": 3,
                        "GoalsAgainst": 2,
                    },
                    {
                        "GameDate": "2025-10-20",
                        "GameId": 5007,
                        "GameType": "Regular",
                        "gameTypeId": 2,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "DAL",
                        "OpponentName": "Dallas Stars",
                        "HomeRoadFlag": "H",
                        "Goals": 6,
                        "GoalsAgainst": 5,
                    },
                    {
                        "GameDate": "2025-10-18",
                        "GameId": 5006,
                        "GameType": "Regular",
                        "gameTypeId": 2,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "DAL",
                        "OpponentName": "Dallas Stars",
                        "HomeRoadFlag": "R",
                        "Goals": 1,
                        "GoalsAgainst": 2,
                    },
                    {
                        "GameDate": "2025-10-12",
                        "GameId": 5005,
                        "GameType": "Regular",
                        "gameTypeId": 2,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "DAL",
                        "OpponentName": "Dallas Stars",
                        "HomeRoadFlag": "H",
                        "Goals": 4,
                        "GoalsAgainst": 0,
                    },
                    {
                        "GameDate": "2025-10-10",
                        "GameId": 5004,
                        "GameType": "Regular",
                        "gameTypeId": 2,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "CGY",
                        "OpponentName": "Calgary Flames",
                        "HomeRoadFlag": "H",
                        "Goals": 4,
                        "GoalsAgainst": 1,
                    },
                ]
            ),
            schedule.pd.DataFrame(
                [
                    {
                        "GameDate": "2024-12-22",
                        "GameId": 4004,
                        "GameType": "Regular",
                        "gameTypeId": 2,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "DAL",
                        "OpponentName": "Dallas Stars",
                        "HomeRoadFlag": "R",
                        "Goals": 3,
                        "GoalsAgainst": 1,
                    },
                    {
                        "GameDate": "2024-12-01",
                        "GameId": 4003,
                        "GameType": "Regular",
                        "gameTypeId": 2,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "DAL",
                        "OpponentName": "Dallas Stars",
                        "HomeRoadFlag": "H",
                        "Goals": 5,
                        "GoalsAgainst": 4,
                    },
                    {
                        "GameDate": "2024-11-15",
                        "GameId": 4002,
                        "GameType": "Regular",
                        "gameTypeId": 2,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "DAL",
                        "OpponentName": "Dallas Stars",
                        "HomeRoadFlag": "R",
                        "Goals": 2,
                        "GoalsAgainst": 1,
                    },
                    {
                        "GameDate": "2024-10-12",
                        "GameId": 4001,
                        "GameType": "Regular",
                        "gameTypeId": 2,
                        "TeamAbbrev": "EDM",
                        "TeamName": "Edmonton Oilers",
                        "OpponentAbbrev": "DAL",
                        "OpponentName": "Dallas Stars",
                        "HomeRoadFlag": "H",
                        "Goals": 1,
                        "GoalsAgainst": 0,
                    },
                ]
            ),
        ]

        history = schedule.get_matchup_history("EDM", "DAL", limit=10)

        self.assertEqual(len(history), 10)
        self.assertEqual(history[0]["game_date"], "2025-11-20")
        self.assertEqual(history[-1]["game_date"], "2024-11-15")
        self.assertEqual(history[0]["home_abbr"], "EDM")
        self.assertEqual(history[0]["home_score"], 5)
        self.assertEqual(history[1]["away_abbr"], "EDM")
        self.assertEqual(history[1]["away_score"], 2)
        self.assertTrue(any(game["game_type"] == 3 for game in history))

    @patch("nhl.schedule.get_game_details")
    @patch("nhl.schedule.get_team_season_game_log")
    @patch("nhl.schedule.get_team_available_nhl_seasons")
    def test_get_matchup_history_matches_franchise_aliases_and_prefers_score_details(
        self,
        mock_available_seasons,
        mock_game_log,
        mock_game_details,
    ):
        """Match lineage aliases to current teams while keeping enriched game details."""
        mock_available_seasons.return_value = [2024]
        mock_game_log.return_value = schedule.pd.DataFrame(
            [
                {
                    "GameDate": "2024-02-15",
                    "GameId": 77,
                    "GameType": "Playoffs",
                    "gameTypeId": 3,
                    "TeamAbbrev": "WPG",
                    "TeamName": "Winnipeg Jets",
                    "OpponentAbbrev": "PHX",
                    "OpponentName": "Phoenix Coyotes",
                    "HomeRoadFlag": "H",
                    "Goals": 3,
                    "GoalsAgainst": 2,
                }
            ]
        )
        mock_game_details.return_value = {
            "game_id": 77,
            "game_date": "2024-02-15",
            "game_type": 3,
            "away_abbr": "PHX",
            "away_name": "Phoenix Coyotes",
            "away_score": 2,
            "home_abbr": "WPG",
            "home_name": "Winnipeg Jets",
            "home_score": 3,
            "venue": "Canada Life Centre",
            "start_label_cest": "Thu 15 Feb, 02:00 CET",
            "status_label": "Final/OT",
        }

        history = schedule.get_matchup_history("WPG", "UTA", limit=10)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["away_abbr"], "PHX")
        self.assertEqual(history[0]["home_abbr"], "WPG")
        self.assertEqual(history[0]["venue"], "Canada Life Centre")
        self.assertEqual(history[0]["status_label"], "Final/OT")

    def test_select_best_goalie_prefers_the_established_starter(self):
        """Prefer the current starter profile over a smaller-sample backup.

        Args:
            None.

        Returns:
            None.
        """
        goalies = [
            {"playerId": 1, "name": "Workhorse", "gamesPlayed": 40, "wins": 24, "savePercentage": 0.905},
            {"playerId": 2, "name": "Wall", "gamesPlayed": 22, "wins": 14, "savePercentage": 0.918},
        ]

        self.assertEqual(schedule._select_best_goalie(goalies)["name"], "Workhorse")

    @patch("nhl.schedule._fetch_club_stats")
    def test_get_featured_players_returns_points_leaders_and_current_starters(self, mock_fetch):
        """Select the current points leaders and starter goalies for both teams.

        Args:
            mock_fetch: Mocked team stat fetch helper.

        Returns:
            None.
        """
        mock_fetch.side_effect = [
            {
                "skaters": [
                    {"playerId": 10, "name": "Home Star", "points": 88},
                    {"playerId": 11, "name": "Home Support", "points": 63},
                ],
                "goalies": [
                    {"playerId": 20, "name": "Home Starter", "gamesPlayed": 40, "wins": 24, "savePercentage": 0.904},
                    {"playerId": 21, "name": "Home Ace", "gamesPlayed": 24, "wins": 15, "savePercentage": 0.919},
                ],
            },
            {
                "skaters": [
                    {"playerId": 30, "name": "Away Star", "points": 91},
                ],
                "goalies": [
                    {"playerId": 40, "name": "Away Ace", "gamesPlayed": 33, "wins": 20, "savePercentage": 0.916},
                ],
            },
        ]

        featured = schedule.get_featured_players("TOR", "MTL")

        self.assertEqual(featured["teams"]["TOR"], schedule.ACTIVE_TEAMS["TOR"])
        self.assertEqual(featured["teams"]["MTL"], schedule.ACTIVE_TEAMS["MTL"])
        self.assertEqual(
            featured["players"],
            {10: "Home Star", 20: "Home Starter", 30: "Away Star", 40: "Away Ace"},
        )

    @patch("nhl.schedule.load_win_prob_weights")
    @patch("nhl.schedule._get_cached_club_stats")
    @patch("nhl.schedule.get_team_season_game_log")
    def test_get_game_win_probabilities_blends_team_form_home_ice_and_goalie_proxy(
        self,
        mock_game_log,
        mock_club_stats,
        mock_load_weights,
    ):
        """Lean toward the stronger home team without pretending certainty."""
        mock_load_weights.return_value = {
            "model_version": 1,
            "feature_order": WIN_PROB_FEATURE_ORDER,
            "coefficients": [1.2, 0.8, 0.6, 0.5, 0.05],
            "intercept": 0.1,
            "scaler_mean": [0.0] * len(WIN_PROB_FEATURE_ORDER),
            "scaler_scale": [1.0] * len(WIN_PROB_FEATURE_ORDER),
            "selected_c": 1.0,
            "min_games": 5,
        }
        mock_game_log.side_effect = [
            schedule.pd.DataFrame(
                {
                    "GameType": ["Regular"] * 10,
                    "GameDate": [f"2026-01-{day:02d}" for day in range(1, 11)],
                    "GameId": list(range(10)),
                    "GP": [1] * 10,
                    "Points": [2, 2, 2, 0, 2, 0, 2, 2, 0, 0],
                    "Goals": [4, 3, 5, 1, 4, 2, 3, 4, 2, 1],
                    "GoalsAgainst": [2, 2, 3, 4, 2, 4, 2, 1, 4, 3],
                    "PP%": [24, 21, 26, 18, 25, 20, 23, 27, 17, 19],
                }
            ),
            schedule.pd.DataFrame(
                {
                    "GameType": ["Regular"] * 10,
                    "GameDate": [f"2026-01-{day:02d}" for day in range(1, 11)],
                    "GameId": list(range(20, 30)),
                    "GP": [1] * 10,
                    "Points": [2, 2, 0, 2, 2, 2, 2, 0, 2, 2],
                    "Goals": [4, 5, 2, 3, 4, 4, 5, 2, 4, 3],
                    "GoalsAgainst": [2, 1, 4, 2, 2, 1, 2, 3, 1, 2],
                    "PP%": [23, 25, 19, 24, 22, 24, 26, 18, 25, 23],
                }
            ),
        ]
        mock_club_stats.side_effect = [
            {"goalies": [{"playerId": 1, "name": "Away Goalie", "gamesPlayed": 38, "wins": 21, "savePercentage": 0.907}]},
            {"goalies": [{"playerId": 2, "name": "Home Goalie", "gamesPlayed": 42, "wins": 28, "savePercentage": 0.918}]},
        ]

        probability = schedule.get_game_win_probabilities("EDM", "DAL")

        self.assertEqual(probability["away_pct"] + probability["home_pct"], 100)
        self.assertGreater(probability["home_pct"], probability["away_pct"])
        self.assertIn("Base model:", probability["model_label"])
        self.assertIn("Goalie proxy", probability["goalie_label"])

    def test_coerce_save_percentage_handles_percent_scale_payloads(self):
        """Normalize goalie save percentage whether the API sends 0.915 or 91.5."""
        self.assertAlmostEqual(schedule._coerce_save_percentage(0.915), 0.915)
        self.assertAlmostEqual(schedule._coerce_save_percentage(91.5), 0.915)

    @patch("nhl.schedule._aggregate_team_save_percentage")
    @patch("nhl.schedule._select_best_goalie")
    def test_build_goalie_proxy_save_percentage_shrinks_toward_team_average(
        self,
        mock_select_best_goalie,
        mock_team_save_pct,
    ):
        """Keep the goalie proxy as the starter save% blended toward team context."""
        mock_select_best_goalie.return_value = {
            "playerId": 1,
            "name": "Starter",
            "gamesPlayed": 10,
            "savePercentage": 0.920,
        }
        mock_team_save_pct.return_value = 0.910

        goalie_proxy = schedule._build_goalie_proxy_save_percentage(
            [{"playerId": 1, "name": "Starter", "gamesPlayed": 10, "savePercentage": 0.920}]
        )

        self.assertAlmostEqual(goalie_proxy, 0.914)
        mock_select_best_goalie.assert_called_once()
        mock_team_save_pct.assert_called_once()


    def test_find_game_from_data_picks_live_over_final(self):
        """Live games have higher priority than finished games."""
        data = {
            "gamesByDate": [{
                "games": [
                    {
                        "gameType": 2,
                        "gameState": "FINAL",
                        "startTimeUTC": "2026-03-18T20:00:00Z",
                        "homeTeam": {"abbrev": "BOS"},
                        "awayTeam": {"abbrev": "NYR"},
                    },
                    {
                        "gameType": 2,
                        "gameState": "LIVE",
                        "startTimeUTC": "2026-03-18T23:00:00Z",
                        "homeTeam": {"abbrev": "TOR"},
                        "awayTeam": {"abbrev": "MTL"},
                    },
                ]
            }]
        }

        result = schedule._find_game_from_data(data)

        self.assertEqual(result, ("TOR", "MTL"))

    def test_find_game_from_data_returns_none_on_empty(self):
        """No games in payload returns None."""
        self.assertIsNone(schedule._find_game_from_data({"gamesByDate": []}))
        self.assertIsNone(schedule._find_game_from_data({}))

    @patch("nhl.schedule.get_client")
    def test_get_live_or_recent_game_uses_scoreboard(self, mock_get_client):
        """Scoreboard path returns the most recent finished game."""
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "gamesByDate": [{
                "games": [{
                    "gameType": 2,
                    "gameState": "FINAL",
                    "startTimeUTC": "2026-03-18T00:00:00Z",
                    "homeTeam": {"abbrev": "TOR"},
                    "awayTeam": {"abbrev": "MTL"},
                }]
            }]
        }
        mock_get_client.return_value = mock_client

        result = schedule.get_live_or_recent_game()

        self.assertEqual(result, ("TOR", "MTL"))
        mock_client.get.assert_called_once()

    @patch("nhl.schedule.get_client")
    def test_get_live_or_recent_game_returns_none_on_total_failure(self, mock_get_client):
        """All NHLClient.get calls return None — function returns None."""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client

        result = schedule.get_live_or_recent_game()

        self.assertIsNone(result)


class ScheduleMigrationTests(unittest.TestCase):
    """Verify Phase 2b schedule functions route HTTP through NHLClient."""

    def setUp(self):
        schedule.get_game_details.clear()
        schedule.get_upcoming_games.clear()
        schedule._get_cached_club_stats.clear()

    def tearDown(self):
        schedule.get_game_details.clear()
        schedule.get_upcoming_games.clear()
        schedule._get_cached_club_stats.clear()

    @patch("nhl.schedule.get_client")
    def test_fetch_club_stats_routes_through_nhl_client(self, mock_get_client):
        """Verify cache key club_stats:{abbr}."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"skaters": [], "goalies": []}
        mock_get_client.return_value = mock_client

        result = schedule._fetch_club_stats("TOR")

        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args.kwargs
        self.assertEqual(call_kwargs["cache_key"], "club_stats:TOR")
        self.assertEqual(call_kwargs["ttl"], 3600)
        self.assertEqual(result, {"skaters": [], "goalies": []})

    @patch("nhl.schedule.get_client")
    def test_get_game_details_routes_through_nhl_client(self, mock_get_client):
        """Verify score:{date} cache key."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"games": []}
        mock_get_client.return_value = mock_client

        schedule.get_game_details("2025-01-15", 2025020001)

        call_kwargs = mock_client.get.call_args.kwargs
        self.assertEqual(call_kwargs["cache_key"], "score:2025-01-15")

    @patch("nhl.schedule.get_game_win_probabilities", return_value=None)
    @patch("nhl.schedule.get_client")
    def test_get_upcoming_games_routes_through_nhl_client(self, mock_get_client, _):
        """Verify per-date score: keys are used."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"games": []}
        mock_get_client.return_value = mock_client

        schedule.get_upcoming_games(limit=1, days_ahead=1)

        self.assertTrue(mock_client.get.call_count >= 1)
        for call in mock_client.get.call_args_list:
            self.assertTrue(call.kwargs["cache_key"].startswith("score:"))

    @patch("nhl.schedule.get_client")
    def test_fetch_club_stats_returns_none_on_failure(self, mock_get_client):
        """NHLClient returns None — _fetch_club_stats returns None."""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client

        self.assertIsNone(schedule._fetch_club_stats("TOR"))


if __name__ == "__main__":
    unittest.main()
