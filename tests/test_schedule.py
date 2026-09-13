import math
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from nhl import schedule
from nhl.ledger import load_ledger
from nhl.win_prob import validate_model_artifact


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
            schedule.get_current_team_ratings,
            schedule.get_season_projection,
            getattr(schedule, "_get_schedule_back_to_back_flags", None),
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
        """Keep future preseason, regular-season and playoff games, sorted by start.

        Preseason (gameType 1) is included on purpose: for most of September it is
        the only NHL hockey on the calendar, and excluding it left the predictions
        panel completely empty. Finished games are still dropped.

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

        # id 3 is FINAL and dropped; ids 1, 4, 2 are future and ordered by start time,
        # with the preseason game (id 4, gameType 1) kept in its chronological place.
        self.assertEqual([game["game_id"] for game in upcoming], [1, 4, 2])
        self.assertEqual(upcoming[0]["game_type"], 2)
        self.assertEqual(upcoming[0]["matchup"], "Washington Capitals at Boston Bruins")
        self.assertEqual(upcoming[0]["venue"], "TD Garden")
        self.assertEqual(upcoming[0]["start_label_cest"], "Sat 07 Mar, 18:30 CET")
        self.assertEqual(upcoming[1]["game_type"], 1)

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

    @patch("nhl.schedule._get_schedule_back_to_back_flags")
    @patch("nhl.schedule.get_current_team_ratings")
    @patch("nhl.schedule.load_win_prob_weights")
    def test_get_game_win_probabilities_scores_the_stronger_home_team_with_fair_odds(
        self,
        mock_load_weights,
        mock_ratings,
        mock_flags,
    ):
        """Lean toward the stronger home team and report matching fair odds."""
        mock_load_weights.return_value = validate_model_artifact(
            {
                "model_version": 2,
                "feature_order": ["elo_diff", "away_back_to_back"],
                "coefficients": [0.5, 0.1],
                "intercept": 0.15,
                "scaler_mean": [0.0, 0.0],
                "scaler_scale": [50.0, 1.0],
            }
        )
        mock_ratings.return_value = {
            "season_year": 2026,
            "teams": {
                "DAL": {"elo": 1580.0, "goal_diff_shrunk": 0.3, "sat_share_shrunk": 0.53, "sog_share_shrunk": 0.52, "games_played": 30},
                "EDM": {"elo": 1520.0, "goal_diff_shrunk": 0.1, "sat_share_shrunk": 0.51, "sog_share_shrunk": 0.5, "games_played": 4},
            },
        }
        mock_flags.return_value = {2026020123: (False, True)}

        probability = schedule.get_game_win_probabilities("EDM", "DAL", 2026020123, 2)

        self.assertEqual(probability["away_pct"] + probability["home_pct"], 100)
        self.assertGreater(probability["home_pct"], probability["away_pct"])
        self.assertAlmostEqual(probability["fair_odds_home"], round(1.0 / probability["home_win_prob"], 2))
        self.assertTrue(probability["away_back_to_back"])
        self.assertTrue(probability["early_season"])
        self.assertIn("Model:", probability["model_label"])
        self.assertIsNone(probability["markets"])
        mock_ratings.assert_called_once_with(2026)

    @patch("nhl.schedule._get_schedule_back_to_back_flags", return_value={})
    @patch("nhl.schedule.get_current_team_ratings")
    @patch("nhl.schedule.load_win_prob_weights")
    def test_get_game_win_probabilities_prices_markets_consistent_with_the_moneyline(
        self,
        mock_load_weights,
        mock_ratings,
        _mock_flags,
    ):
        """With a goal model the card gets a 60-minute result and puck line that agree with the win probability."""
        mock_load_weights.return_value = validate_model_artifact(
            {
                "model_version": 2,
                "feature_order": ["elo_diff"],
                "coefficients": [0.5],
                "intercept": 0.15,
                "scaler_mean": [0.0],
                "scaler_scale": [50.0],
                "goal_model": {"rate_scale": 0.98, "tie_inflation": 0.48, "lead1_transfer": 0.31, "lead2_transfer": 0.46, "overtime_intercept": -0.04, "overtime_logit_coef": 0.39},
            }
        )
        mock_ratings.return_value = {
            "season_year": 2026,
            "scoring_environment": 3.05,
            "teams": {
                "TOR": {"elo": 1590.0, "goal_diff_shrunk": 0.3, "sat_share_shrunk": 0.52, "sog_share_shrunk": 0.52, "games_played": 20},
                "MTL": {"elo": 1500.0, "goal_diff_shrunk": -0.1, "sat_share_shrunk": 0.49, "sog_share_shrunk": 0.49, "games_played": 20},
            },
        }

        probability = schedule.get_game_win_probabilities("MTL", "TOR", 2026020200, 2)
        markets = probability["markets"]
        regulation = markets["regulation"]

        self.assertAlmostEqual(regulation["home"] + regulation["draw"] + regulation["away"], 1.0, places=9)
        self.assertGreater(regulation["home"], regulation["away"])
        self.assertLess(markets["puck_line"]["home_minus_1_5"], regulation["home"])
        self.assertNotIn("totals", markets)
        tie_break = 1 / (1 + math.exp(-(-0.04 + 0.39 * math.log(probability["home_win_prob"] / (1 - probability["home_win_prob"])))))
        self.assertAlmostEqual(regulation["home"] + regulation["draw"] * tie_break, probability["home_win_prob"], places=5)

    @patch("nhl.schedule.load_win_prob_weights")
    def test_get_game_win_probabilities_skips_preseason_games(self, mock_load_weights):
        """Exhibition lineups are mostly prospects, so they never get a prediction."""
        self.assertIsNone(schedule.get_game_win_probabilities("BOS", "PHI", 2026010001, 1))
        mock_load_weights.assert_not_called()

    @patch("nhl.schedule.get_league_schedule")
    @patch("nhl.schedule.get_current_nhl_standings")
    @patch("nhl.schedule.get_playoff_bracket")
    @patch("nhl.schedule.load_win_prob_weights")
    def test_get_season_projection_reports_a_decided_cup_instead_of_simulating(
        self,
        mock_load_weights,
        mock_bracket,
        mock_standings,
        mock_schedule,
    ):
        """After the Final the board names the champion and nothing is simulated."""
        schedule.get_season_projection.clear()
        mock_load_weights.return_value = {"model_version": 2}
        mock_bracket.return_value = {
            "series": [
                {
                    "seriesLetter": "O",
                    "playoffRound": 4,
                    "topSeedTeam": {"id": 12, "abbrev": "CAR"},
                    "bottomSeedTeam": {"id": 54, "abbrev": "VGK"},
                    "topSeedWins": 4,
                    "bottomSeedWins": 2,
                    "winningTeamId": 12,
                }
            ]
        }

        projection = schedule.get_season_projection()
        schedule.get_season_projection.clear()

        self.assertEqual(projection["state"], "champion")
        self.assertEqual(projection["champion"], "CAR")
        mock_standings.assert_not_called()
        mock_schedule.assert_not_called()

    @patch("nhl.schedule.get_league_game_table", return_value=schedule.pd.DataFrame())
    @patch("nhl.schedule.get_game_win_probabilities")
    @patch("nhl.schedule.load_win_prob_weights")
    @patch("nhl.schedule.get_client")
    def test_capture_prediction_ledger_logs_upcoming_games_and_market_prices(
        self,
        mock_get_client,
        mock_load_weights,
        mock_probabilities,
        _mock_games,
    ):
        """Regular-season games inside the window are logged with their market prices; others are not."""
        def _game(game_id, game_type, start):
            return {
                "id": game_id, "gameType": game_type, "gameState": "FUT", "startTimeUTC": start,
                "awayTeam": {"abbrev": "FLA", "name": {"default": "Panthers"}},
                "homeTeam": {"abbrev": "CAR", "name": {"default": "Hurricanes"}},
                "venue": {"default": "Lenovo Center"},
            }

        scoreboard = {"gamesByDate": [{"games": [
            _game(2026020001, 2, "2026-09-29T21:00:00Z"),
            _game(2026010099, 1, "2026-09-29T23:00:00Z"),
            _game(2026020060, 2, "2026-10-03T23:00:00Z"),
        ]}]}
        partner = {
            "bettingPartner": {"name": "FanDuel", "country": "CAN"},
            "games": [{"gameId": 2026020001, "gameType": 2, "startTimeUTC": "2026-09-29T21:00:00Z",
                       "homeTeam": {"abbrev": "CAR", "odds": [{"description": "MONEY_LINE_2_WAY", "value": -125.0, "qualifier": ""}]},
                       "awayTeam": {"abbrev": "FLA", "odds": [{"description": "MONEY_LINE_2_WAY", "value": 104.0, "qualifier": ""}]}}],
        }

        def _fake_get(url, params=None, cache_key=None, ttl=None, timeout=None):
            if "scoreboard" in url:
                return scoreboard
            return partner if url.endswith("/CA/now") else {}

        mock_client = MagicMock()
        mock_client.get.side_effect = _fake_get
        mock_get_client.return_value = mock_client
        mock_load_weights.return_value = {"generated_at_utc": "2026-09-13T00:00:00Z"}
        mock_probabilities.return_value = {
            "home_win_prob": 0.66,
            "early_season": True,
            "markets": {"regulation": {"home": 0.54, "draw": 0.21, "away": 0.25}, "puck_line": {"home_minus_1_5": 0.43, "away_minus_1_5": 0.17}},
        }

        directory = tempfile.mkdtemp()
        try:
            with patch.dict(os.environ, {"PUCKPEAK_DATA_DIR": directory}):
                summary = schedule.capture_prediction_ledger(datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc))
                predictions, market = load_ledger()
        finally:
            shutil.rmtree(directory, ignore_errors=True)

        self.assertEqual(summary, {"predictions": 1, "market_rows": 1, "graded": 0})
        mock_probabilities.assert_called_once_with("FLA", "CAR", 2026020001, 2)
        self.assertEqual(list(predictions["game_id"]), [2026020001])
        self.assertAlmostEqual(float(predictions.iloc[0]["regulation_draw"]), 0.21)
        self.assertEqual(predictions.iloc[0]["model_version"], "2026-09-13T00:00:00Z")
        self.assertEqual(list(market["partner"]), ["FanDuel (CAN)"])

    def test_coerce_save_percentage_handles_percent_scale_payloads(self):
        """Normalize goalie save percentage whether the API sends 0.915 or 91.5."""
        self.assertAlmostEqual(schedule._coerce_save_percentage(0.915), 0.915)
        self.assertAlmostEqual(schedule._coerce_save_percentage(91.5), 0.915)


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
        """Read the scoreboard first, then fall back to per-date score: keys."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"games": []}
        mock_get_client.return_value = mock_client

        schedule.get_upcoming_games(limit=1, days_ahead=1)

        cache_keys = [call.kwargs["cache_key"] for call in mock_client.get.call_args_list]
        self.assertTrue(cache_keys)
        # The multi-day scoreboard is tried first; it covers ~11 days in one request.
        self.assertEqual(cache_keys[0], "scoreboard")
        # It returned nothing here, so the per-date walk still runs behind it.
        self.assertTrue(all(key.startswith("score:") for key in cache_keys[1:]))

    @patch("nhl.schedule.get_game_win_probabilities", return_value=None)
    @patch("nhl.schedule.get_client")
    def test_get_upcoming_games_skips_date_walk_when_scoreboard_suffices(
        self, mock_get_client, _
    ):
        """One scoreboard request is enough — do not walk 60 individual dates."""
        future = (
            datetime.now(timezone.utc) + timedelta(days=20)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "gamesByDate": [
                {
                    "games": [
                        {
                            "id": 11,
                            "gameType": 2,
                            "gameState": "FUT",
                            "startTimeUTC": future,
                            "awayTeam": {"abbrev": "EDM", "name": {"default": "Oilers"}},
                            "homeTeam": {"abbrev": "DAL", "name": {"default": "Stars"}},
                            "venue": {"default": "American Airlines Center"},
                        }
                    ]
                }
            ]
        }
        mock_get_client.return_value = mock_client

        games = schedule.get_upcoming_games(limit=1, days_ahead=60)

        self.assertEqual([game["game_id"] for game in games], [11])
        cache_keys = [call.kwargs["cache_key"] for call in mock_client.get.call_args_list]
        self.assertEqual(cache_keys, ["scoreboard"])

    def test_find_game_from_data_falls_back_to_soonest_upcoming_game(self):
        """In the offseason, seed from the next scheduled game rather than nothing.

        Between the Cup final and opening night no payload contains a live or
        finished game, so without this fallback the landing board seeds empty.
        """
        payload = {
            "gamesByDate": [
                {
                    "games": [
                        {
                            "id": 20,
                            "gameType": 2,
                            "gameState": "FUT",
                            "startTimeUTC": "2026-10-01T23:00:00Z",
                            "awayTeam": {"abbrev": "COL"},
                            "homeTeam": {"abbrev": "VGK"},
                        },
                        {
                            "id": 21,
                            "gameType": 1,
                            "gameState": "FUT",
                            "startTimeUTC": "2026-09-24T23:00:00Z",
                            "awayTeam": {"abbrev": "BOS"},
                            "homeTeam": {"abbrev": "PHI"},
                        },
                    ]
                }
            ]
        }

        # Soonest first, and preseason counts.
        self.assertEqual(schedule._find_game_from_data(payload), ("PHI", "BOS"))

    def test_find_game_from_data_still_prefers_finished_over_upcoming(self):
        """A finished game outranks a scheduled one — the fallback is last resort."""
        payload = {
            "gamesByDate": [
                {
                    "games": [
                        {
                            "id": 30,
                            "gameType": 2,
                            "gameState": "FUT",
                            "startTimeUTC": "2026-10-01T23:00:00Z",
                            "awayTeam": {"abbrev": "COL"},
                            "homeTeam": {"abbrev": "VGK"},
                        },
                        {
                            "id": 31,
                            "gameType": 2,
                            "gameState": "FINAL",
                            "startTimeUTC": "2026-09-30T23:00:00Z",
                            "awayTeam": {"abbrev": "EDM"},
                            "homeTeam": {"abbrev": "CGY"},
                        },
                    ]
                }
            ]
        }

        self.assertEqual(schedule._find_game_from_data(payload), ("CGY", "EDM"))

    @patch("nhl.schedule.get_client")
    def test_fetch_club_stats_returns_none_on_failure(self, mock_get_client):
        """NHLClient returns None — _fetch_club_stats returns None."""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client

        self.assertIsNone(schedule._fetch_club_stats("TOR"))


if __name__ == "__main__":
    unittest.main()
