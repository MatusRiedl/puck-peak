import unittest
from unittest.mock import Mock, mock_open, patch, MagicMock

import pandas as pd

import nhl.data_loaders as data_loaders


class DataLoaderInvariantTests(unittest.TestCase):
    """Cover parquet sanitation and fallback data-loading rules."""

    def setUp(self):
        """Clear cached historical data before each test.

        Args:
            None.

        Returns:
            None.
        """
        data_loaders.load_historical_data.clear()

    def tearDown(self):
        """Clear cached historical data after each test.

        Args:
            None.

        Returns:
            None.
        """
        data_loaders.load_historical_data.clear()

    def test_normalize_historical_goalie_rates_only_changes_goalies(self):
        """Convert legacy goalie SavePct values into 0-1 scale.

        Args:
            None.

        Returns:
            None.
        """
        raw_df = pd.DataFrame(
            {
                "Position": ["G", "G", "G", "G", "S"],
                "SavePct": [91.3, 0.915, 105.0, None, 150.0],
            }
        )

        normalized = data_loaders._normalize_historical_goalie_rates(raw_df)

        self.assertAlmostEqual(float(normalized.loc[0, "SavePct"]), 0.913)
        self.assertAlmostEqual(float(normalized.loc[1, "SavePct"]), 0.915)
        self.assertAlmostEqual(float(normalized.loc[2, "SavePct"]), 1.0)
        self.assertAlmostEqual(float(normalized.loc[3, "SavePct"]), 0.0)
        self.assertAlmostEqual(float(normalized.loc[4, "SavePct"]), 150.0)

    def test_load_historical_data_infers_goalies_and_adds_derived_columns(self):
        """Infer goalie rows when parquet lacks explicit goalie position labels.

        Args:
            None.

        Returns:
            None.
        """
        parquet_df = pd.DataFrame(
            {
                "Points": [10.0, 0.0],
                "GP": [20.0, 10.0],
                "SavePct": [0.500, 91.3],
                "Saves": [0.0, 300.0],
                "Wins": [0.0, 20.0],
                "Shutouts": [0.0, 2.0],
            }
        )

        with patch("nhl.data_loaders.os.path.exists", return_value=True), patch(
            "nhl.data_loaders.pd.read_parquet", return_value=parquet_df
        ):
            loaded = data_loaders.load_historical_data()

        self.assertEqual(loaded.loc[0, "Position"], "S")
        self.assertEqual(loaded.loc[1, "Position"], "G")
        self.assertAlmostEqual(float(loaded.loc[0, "PPG"]), 0.5)
        self.assertAlmostEqual(float(loaded.loc[1, "SavePct"]), 0.913)
        self.assertAlmostEqual(float(loaded.loc[1, "Save %"]), 91.3)
        self.assertAlmostEqual(float(loaded.loc[0, "SH%"]), 0.0)
        self.assertAlmostEqual(float(loaded.loc[0, "TOI"]), 0.0)
        self.assertIn("Shots", loaded.columns)
        self.assertIn("TotalTOIMins", loaded.columns)

    def test_load_historical_data_builds_sh_pct_and_toi_when_new_parquet_columns_exist(self):
        """Derive SH% and TOI from additive historical parquet columns."""
        parquet_df = pd.DataFrame(
            {
                "Position": ["S"],
                "Points": [30.0],
                "Goals": [12.0],
                "GP": [20.0],
                "Shots": [60.0],
                "TotalTOIMins": [400.0],
                "SavePct": [0.0],
                "Saves": [0.0],
                "Wins": [0.0],
                "Shutouts": [0.0],
            }
        )

        with patch("nhl.data_loaders.os.path.exists", return_value=True), patch(
            "nhl.data_loaders.pd.read_parquet", return_value=parquet_df
        ):
            loaded = data_loaders.load_historical_data()

        self.assertAlmostEqual(float(loaded.loc[0, "SH%"]), 20.0)
        self.assertAlmostEqual(float(loaded.loc[0, "TOI"]), 20.0)


class WinProbArtifactLoaderTests(unittest.TestCase):
    """Cover loading the frozen win-probability artifact from disk."""

    def setUp(self):
        data_loaders.load_win_prob_weights.clear()

    def tearDown(self):
        data_loaders.load_win_prob_weights.clear()

    def test_load_win_prob_weights_validates_and_returns_json_payload(self):
        payload = """
        {
          "model_version": 1,
          "feature_order": ["point_pct_to_date", "goal_diff_per_game_to_date", "l10_point_pct", "l10_goal_diff_per_game", "power_play_pct_to_date"],
          "coefficients": [0.1, 0.2, 0.3, 0.4, 0.5],
          "intercept": 0.25,
          "scaler_mean": [0, 0, 0, 0, 0],
          "scaler_scale": [1, 1, 1, 1, 1],
          "selected_c": 1.0,
          "min_games": 5
        }
        """

        with patch("nhl.data_loaders.os.path.exists", return_value=True), patch(
            "builtins.open",
            mock_open(read_data=payload),
        ):
            loaded = data_loaders.load_win_prob_weights()

        self.assertEqual(loaded["feature_order"][0], "point_pct_to_date")
        self.assertEqual(len(loaded["coefficients"]), 5)
        self.assertEqual(int(loaded["min_games"]), 5)


class PlayerLandingLoaderTests(unittest.TestCase):
    """Cover the shared player landing payload helpers."""

    def setUp(self):
        """Clear shared landing caches before each test.

        Args:
            None.

        Returns:
            None.
        """
        data_loaders.get_player_landing.clear()
        data_loaders.get_player_raw_stats.clear()
        data_loaders.get_player_identity_summary.clear()

    def tearDown(self):
        """Clear shared landing caches after each test.

        Args:
            None.

        Returns:
            None.
        """
        data_loaders.get_player_landing.clear()
        data_loaders.get_player_raw_stats.clear()
        data_loaders.get_player_identity_summary.clear()

    def test_helpers_share_one_cached_landing_payload(self):
        """Use one landing fetch across the small player helpers.

        Args:
            None.

        Returns:
            None.
        """
        payload = {
            "birthDate": "1997-01-13",
            "position": "L",
            "currentTeamAbbrev": "EDM",
            "sweaterNumber": "97",
            "headshot": "https://img/headshot.png",
            "heroImage": "https://img/hero.png",
            "awards": [{"trophy": "Hart Memorial Trophy"}],
            "seasonTotals": [
                {
                    "leagueAbbrev": "NHL",
                    "gameTypeId": "2",
                    "season": "20232024",
                    "gamesPlayed": 82,
                    "points": 130,
                    "goals": 44,
                    "assists": 86,
                    "pim": 26,
                    "plusMinus": 35,
                    "shots": 300,
                    "avgToi": "21:15",
                    "wins": 0,
                    "shutouts": 0,
                    "saves": 0,
                    "savePctg": 0.0,
                    "goalsAgainstAvg": 0.0,
                }
            ],
        }
        mock_client = MagicMock()
        mock_client.get.return_value = payload

        with patch("nhl.data_loaders.get_client", return_value=mock_client):
            self.assertEqual(data_loaders.get_player_headshot(97), "https://img/headshot.png")
            self.assertEqual(data_loaders.get_player_current_team(97), "EDM")
            self.assertEqual(
                data_loaders.get_player_roster_info(97),
                {"position": "LW", "sweater_number": 97},
            )
            self.assertEqual(data_loaders.get_player_hero_image(97), "https://img/hero.png")
            self.assertEqual(data_loaders.get_player_awards(97), [{"trophy": "Hart Memorial Trophy"}])
            self.assertEqual(data_loaders.get_player_league_abbrevs(97), ["NHL"])

            raw_df, base_name, position = data_loaders.get_player_raw_stats(97, "Connor McDavid")

        self.assertEqual(mock_client.get.call_count, 1)
        self.assertEqual(base_name, "Connor McDavid")
        self.assertEqual(position, "L")
        self.assertEqual(len(raw_df), 1)
        self.assertEqual(int(raw_df.iloc[0]["Points"]), 130)
        self.assertEqual(int(raw_df.iloc[0]["SeasonYear"]), 2023)
        self.assertEqual(int(raw_df.iloc[0]["PlayerID"]), 97)
        self.assertEqual(str(raw_df.iloc[0]["PositionCode"]), "L")

    def test_landing_failure_falls_back_cleanly(self):
        """Keep helper fallbacks intact when the landing request blows up.

        Args:
            None.

        Returns:
            None.
        """
        mock_client = MagicMock()
        mock_client.get.return_value = None

        with patch("nhl.data_loaders.get_client", return_value=mock_client):
            self.assertEqual(data_loaders.get_player_landing(8), {})
            self.assertEqual(data_loaders.get_player_headshot(8), "")
            self.assertEqual(data_loaders.get_player_current_team(8), "")
            self.assertEqual(data_loaders.get_player_roster_info(8), {})
            self.assertEqual(data_loaders.get_player_hero_image(8), "")
            self.assertEqual(data_loaders.get_player_awards(8), [])
            self.assertEqual(data_loaders.get_player_league_abbrevs(8), [])

            raw_df, base_name, position = data_loaders.get_player_raw_stats(8, "Alex Ovechkin")

        self.assertTrue(raw_df.empty)
        self.assertEqual(base_name, "Alex Ovechkin")
        self.assertEqual(position, "S")

    def test_roster_info_requires_active_team_and_complete_fields(self):
        """Return empty roster metadata for inactive or incomplete player rows.

        Args:
            None.

        Returns:
            None.
        """
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "position": "C",
            "sweaterNumber": 87,
        }

        with patch("nhl.data_loaders.get_client", return_value=mock_client):
            self.assertEqual(data_loaders.get_player_roster_info(87), {})

    def test_hero_image_falls_back_to_headshot(self):
        """Use the headshot when the landing payload lacks a hero image.

        Args:
            None.

        Returns:
            None.
        """
        mock_client = MagicMock()
        mock_client.get.return_value = {"headshot": "https://img/fallback.png"}

        with patch("nhl.data_loaders.get_client", return_value=mock_client):
            self.assertEqual(data_loaders.get_player_hero_image(29), "https://img/fallback.png")

    def test_awards_helper_rejects_non_list_payloads(self):
        """Ignore malformed awards shapes from the landing payload.

        Args:
            None.

        Returns:
            None.
        """
        mock_client = MagicMock()
        mock_client.get.return_value = {"awards": {"trophy": "Hart Memorial Trophy"}}

        with patch("nhl.data_loaders.get_client", return_value=mock_client):
            self.assertEqual(data_loaders.get_player_awards(29), [])

    def test_player_identity_summary_normalizes_modal_fields(self):
        """Build the player modal summary from landing metadata plus derived NHL context."""
        payload = {
            "firstName": {"default": "Connor"},
            "lastName": {"default": "McDavid"},
            "birthDate": "1997-01-13",
            "birthCity": {"default": "Richmond Hill"},
            "birthStateProvince": {"default": "Ontario"},
            "birthCountry": "CAN",
            "position": "C",
            "shootsCatches": "L",
            "heightInInches": 73,
            "heightInCentimeters": 185,
            "weightInPounds": 194,
            "weightInKilograms": 88,
            "draftDetails": {
                "year": 2015,
                "teamAbbrev": "EDM",
                "round": 1,
                "pickInRound": 1,
                "overallPick": 1,
            },
            "inTop100AllTime": 1,
            "awards": [
                {
                    "trophy": {"default": "Hart Memorial Trophy"},
                    "seasons": [{"seasonId": 20162017}, {"seasonId": 20202021}, {"seasonId": 20222023}],
                },
                {
                    "trophy": {"default": "Ted Lindsay Award"},
                    "seasons": [{"seasonId": 20162017}, {"seasonId": 20172018}],
                },
            ],
            "seasonTotals": [
                {
                    "leagueAbbrev": "OHL",
                    "gameTypeId": "2",
                    "season": "20142015",
                    "teamName": {"default": "Erie Otters"},
                },
                {
                    "leagueAbbrev": "NHL",
                    "gameTypeId": "2",
                    "season": "20152016",
                    "sequence": 1,
                    "teamName": {"default": "Edmonton Oilers"},
                },
            ],
        }

        with patch.object(data_loaders, "get_player_landing", return_value=payload):
            summary = data_loaders.get_player_identity_summary(97)

        self.assertEqual(summary["name"], "Connor McDavid")
        self.assertEqual(summary["birthplace"], "Richmond Hill, Ontario, CAN")
        self.assertEqual(summary["shot_label"], "Shoots")
        self.assertEqual(summary["shot_value"], "L")
        self.assertEqual(summary["height"], "6'1\" / 185 cm")
        self.assertEqual(summary["weight"], "194 lb / 88 kg")
        self.assertEqual(summary["draft"], "2015 | EDM | Round 1, pick 1 | 1 overall")
        self.assertEqual(summary["first_nhl_season"], 2015)
        self.assertEqual(summary["first_nhl_season_label"], "2015-16")
        self.assertEqual(summary["debut_team"], "Edmonton Oilers")
        self.assertIn("NHL Top 100", summary["honors"])
        self.assertEqual(
            summary["trophies"],
            [
                {
                    "trophy": "Hart Memorial Trophy",
                    "count": 3,
                    "latest": 20222023,
                    "latest_label": "2022-23",
                },
                {
                    "trophy": "Ted Lindsay Award",
                    "count": 2,
                    "latest": 20172018,
                    "latest_label": "2017-18",
                },
            ],
        )
        self.assertIsInstance(summary["age"], int)

    def test_player_identity_summary_handles_undrafted_partial_payload(self):
        """Fall back cleanly when optional player identity fields are missing."""
        payload = {
            "firstName": {"default": "Frederik"},
            "lastName": {"default": "Andersen"},
            "birthDate": "1989-10-02",
            "position": "G",
            "shootsCatches": "L",
            "seasonTotals": [
                {
                    "leagueAbbrev": "NHL",
                    "gameTypeId": "3",
                    "season": "20132014",
                    "sequence": 2,
                    "teamName": {"default": "Anaheim Ducks"},
                }
            ],
        }

        with patch.object(data_loaders, "get_player_landing", return_value=payload):
            summary = data_loaders.get_player_identity_summary(31)

        self.assertEqual(summary["name"], "Frederik Andersen")
        self.assertEqual(summary["shot_label"], "Catches")
        self.assertEqual(summary["shot_value"], "L")
        self.assertEqual(summary["draft"], "Undrafted")
        self.assertEqual(summary["first_nhl_season_label"], "2013-14")
        self.assertEqual(summary["debut_team"], "Anaheim Ducks")
        self.assertEqual(summary["honors"], [])
        self.assertEqual(summary["trophies"], [])


class PlayerGameLogNormalizationTests(unittest.TestCase):
    """Cover exact-game metadata preservation in season game logs."""

    def test_normalize_player_game_log_rows_keeps_matchup_metadata(self):
        """Preserve game identifiers and matchup fields needed for click dialogs."""
        rows = [
            {
                "gameId": "2024020001",
                "gameDate": "2024-10-09",
                "teamAbbrev": "EDM",
                "opponentAbbrev": "WPG",
                "homeRoadFlag": "H",
                "commonName": {"default": "Oilers"},
                "opponentCommonName": {"default": "Jets"},
                "points": 2,
                "goals": 1,
                "assists": 1,
                "toi": "21:45",
            }
        ]

        normalized = data_loaders._normalize_player_game_log_rows(
            rows=rows,
            season_year=2024,
            birth_year=1997,
            game_type="Regular",
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["GameId"], 2024020001)
        self.assertEqual(normalized[0]["GameDate"], "2024-10-09")
        self.assertEqual(normalized[0]["TeamAbbrev"], "EDM")
        self.assertEqual(normalized[0]["OpponentAbbrev"], "WPG")
        self.assertEqual(normalized[0]["HomeRoadFlag"], "H")
        self.assertEqual(normalized[0]["TeamName"], "Oilers")
        self.assertEqual(normalized[0]["OpponentName"], "Jets")


class TeamSeasonLoaderTests(unittest.TestCase):
    """Cover team selected-season helpers and aggregation rules."""

    def setUp(self):
        """Clear cached team-season helpers before each test."""
        data_loaders.get_team_available_nhl_seasons.clear()
        data_loaders.get_team_season_game_log.clear()
        data_loaders.get_team_season_summary.clear()
        data_loaders.get_team_season_rank_map.clear()
        data_loaders.get_team_all_time_stats.clear()

    def tearDown(self):
        """Clear cached team-season helpers after each test."""
        data_loaders.get_team_available_nhl_seasons.clear()
        data_loaders.get_team_season_game_log.clear()
        data_loaders.get_team_season_summary.clear()
        data_loaders.get_team_season_rank_map.clear()
        data_loaders.get_team_all_time_stats.clear()

    def test_normalize_team_game_log_rows_keeps_matchup_metadata(self):
        """Preserve identifiers and team matchup fields needed for team clicks."""
        rows = [
            {
                "gameId": "2023020001",
                "gameDate": "2023-10-11",
                "goalsFor": 4,
                "goalsAgainst": 2,
                "wins": 1,
                "points": 2,
                "powerPlayPct": 0.25,
                "homeRoad": "H",
                "opponentTeamAbbrev": "MTL",
            }
        ]

        normalized = data_loaders._normalize_team_game_log_rows(
            rows=rows,
            season_year=2023,
            team_abbr="TOR",
            team_name="Toronto Maple Leafs",
            game_type="Regular",
            game_type_id=2,
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["GameId"], 2023020001)
        self.assertEqual(normalized[0]["GameDate"], "2023-10-11")
        self.assertEqual(normalized[0]["GameType"], "Regular")
        self.assertEqual(normalized[0]["TeamAbbrev"], "TOR")
        self.assertEqual(normalized[0]["OpponentAbbrev"], "MTL")
        self.assertEqual(normalized[0]["HomeRoadFlag"], "H")
        self.assertEqual(normalized[0]["ResultCode"], "W")
        self.assertAlmostEqual(float(normalized[0]["PP%"]), 25.0)

    def test_get_team_season_summary_combines_both_types_and_builds_ranks(self):
        """Combine regular and playoff summary rows before ranking one season metric."""
        all_team_df = pd.DataFrame(
            [
                {
                    "teamAbbrev": "TOR",
                    "teamFullName": "Toronto Maple Leafs",
                    "teamId": 10,
                    "SeasonYear": 2023,
                    "gameTypeId": 2,
                    "GP": 82,
                    "Wins": 46,
                    "Losses": 26,
                    "OTLosses": 10,
                    "Ties": 0,
                    "Points": 102,
                    "Goals": 298,
                    "Win%": 62.2,
                    "GF/G": 3.634,
                    "GA/G": 3.073,
                    "PPG": 9.812,
                    "PP%": 24.1,
                },
                {
                    "teamAbbrev": "TOR",
                    "teamFullName": "Toronto Maple Leafs",
                    "teamId": 10,
                    "SeasonYear": 2023,
                    "gameTypeId": 3,
                    "GP": 7,
                    "Wins": 3,
                    "Losses": 4,
                    "OTLosses": 0,
                    "Ties": 0,
                    "Points": 6,
                    "Goals": 24,
                    "Win%": 42.9,
                    "GF/G": 3.429,
                    "GA/G": 3.571,
                    "PPG": 9.257,
                    "PP%": 28.0,
                },
                {
                    "teamAbbrev": "EDM",
                    "teamFullName": "Edmonton Oilers",
                    "teamId": 22,
                    "SeasonYear": 2023,
                    "gameTypeId": 2,
                    "GP": 82,
                    "Wins": 49,
                    "Losses": 27,
                    "OTLosses": 6,
                    "Ties": 0,
                    "Points": 104,
                    "Goals": 294,
                    "Win%": 63.4,
                    "GF/G": 3.585,
                    "GA/G": 2.951,
                    "PPG": 9.683,
                    "PP%": 26.3,
                },
            ]
        )

        with patch.object(data_loaders, "load_all_team_seasons", return_value=all_team_df):
            summary = data_loaders.get_team_season_summary(2023, "Both")
            ranks = data_loaders.get_team_season_rank_map(2023, "Both", "Points")

        tor_row = summary[summary["teamAbbrev"] == "TOR"].iloc[0]
        self.assertEqual(int(tor_row["GP"]), 89)
        self.assertEqual(int(tor_row["Wins"]), 49)
        self.assertEqual(int(tor_row["Points"]), 108)
        self.assertEqual(int(tor_row["Goals"]), 322)
        self.assertEqual(ranks["TOR"], 1)
        self.assertEqual(ranks["EDM"], 2)

    def test_get_team_available_nhl_seasons_reads_team_history(self):
        """Build the picker seasons from team history instead of player seasons."""
        all_team_df = pd.DataFrame(
            [
                {"teamAbbrev": "TOR", "SeasonYear": 2024},
                {"teamAbbrev": "TOR", "SeasonYear": 2023},
                {"teamAbbrev": "TOR", "SeasonYear": 2023},
                {"teamAbbrev": "EDM", "SeasonYear": 2022},
            ]
        )

        with patch.object(data_loaders, "load_all_team_seasons", return_value=all_team_df):
            seasons = data_loaders.get_team_available_nhl_seasons("TOR")

        self.assertEqual(seasons, [2024, 2023])

    def test_get_team_available_nhl_seasons_merges_franchise_aliases(self):
        """Expose one franchise season list across historical team abbreviations."""
        all_team_df = pd.DataFrame(
            [
                {"teamAbbrev": "WIN", "SeasonYear": 1979},
                {"teamAbbrev": "PHX", "SeasonYear": 1997},
                {"teamAbbrev": "ARI", "SeasonYear": 2021},
                {"teamAbbrev": "UTA", "SeasonYear": 2024},
                {"teamAbbrev": "TOR", "SeasonYear": 2024},
            ]
        )

        with patch.object(data_loaders, "load_all_team_seasons", return_value=all_team_df):
            seasons = data_loaders.get_team_available_nhl_seasons("UTA")

        self.assertEqual(seasons, [2024, 2021, 1997, 1979])

    def test_get_team_season_summary_normalizes_historical_alias_to_active_franchise(self):
        """Return active franchise keys even when historical team abbreviations are loaded."""
        all_team_df = pd.DataFrame(
            [
                {
                    "teamAbbrev": "PHX",
                    "teamFullName": "Phoenix Coyotes",
                    "teamId": 28,
                    "SeasonYear": 2013,
                    "gameTypeId": 2,
                    "GP": 82,
                    "Wins": 37,
                    "Losses": 30,
                    "OTLosses": 15,
                    "Ties": 0,
                    "Points": 89,
                    "Goals": 216,
                    "Win%": 54.3,
                    "GF/G": 2.634,
                    "GA/G": 2.500,
                    "PPG": 7.112,
                    "PP%": 19.5,
                }
            ]
        )

        with patch.object(data_loaders, "load_all_team_seasons", return_value=all_team_df):
            summary = data_loaders.get_team_season_summary(2013, "Regular")

        self.assertEqual(summary.iloc[0]["teamAbbrev"], "UTA")
        self.assertEqual(summary.iloc[0]["teamFullName"], "Utah Hockey Club")

    def test_get_team_all_time_stats_aggregates_franchise_lineage(self):
        """Aggregate all-time franchise totals across relocated team abbreviations."""
        all_team_df = pd.DataFrame(
            [
                {
                    "teamAbbrev": "WIN",
                    "SeasonYear": 1979,
                    "gameTypeId": 2,
                    "GP": 80,
                    "Wins": 26,
                    "Points": 63,
                    "Goals": 230,
                },
                {
                    "teamAbbrev": "PHX",
                    "SeasonYear": 2000,
                    "gameTypeId": 2,
                    "GP": 82,
                    "Wins": 38,
                    "Points": 90,
                    "Goals": 214,
                },
                {
                    "teamAbbrev": "ARI",
                    "SeasonYear": 2021,
                    "gameTypeId": 2,
                    "GP": 56,
                    "Wins": 24,
                    "Points": 54,
                    "Goals": 153,
                },
                {
                    "teamAbbrev": "UTA",
                    "SeasonYear": 2024,
                    "gameTypeId": 2,
                    "GP": 82,
                    "Wins": 40,
                    "Points": 91,
                    "Goals": 230,
                },
                {
                    "teamAbbrev": "TOR",
                    "SeasonYear": 2024,
                    "gameTypeId": 2,
                    "GP": 82,
                    "Wins": 45,
                    "Points": 102,
                    "Goals": 280,
                },
            ]
        )

        with patch.object(data_loaders, "load_all_team_seasons", return_value=all_team_df):
            team_stats = data_loaders.get_team_all_time_stats()

        self.assertEqual(team_stats["UTA"]["total_wins"], 128)
        self.assertEqual(team_stats["UTA"]["total_gp"], 300)
        self.assertEqual(team_stats["UTA"]["total_points"], 298)
        self.assertEqual(team_stats["UTA"]["total_goals"], 827)
        self.assertEqual(team_stats["UTA"]["wins_rank"], 1)
        self.assertEqual(team_stats["UTA"]["best_year"], 2024)
        self.assertEqual(team_stats["UTA"]["best_wins"], 40)
        self.assertEqual(team_stats["UTA"]["best_gp"], 82)


class CurrentStandingsLoaderTests(unittest.TestCase):
    """Cover live standings normalization for the Stanley Cup board."""

    def setUp(self):
        """Clear cached live standings before each test."""
        data_loaders.get_current_nhl_standings.clear()

    def tearDown(self):
        """Clear cached live standings after each test."""
        data_loaders.get_current_nhl_standings.clear()

    @patch.object(data_loaders, "get_team_season_summary")
    @patch("nhl.data_loaders.get_client")
    def test_get_current_nhl_standings_normalizes_live_payload_and_merges_pp_pct(
        self,
        mock_get_client,
        mock_team_summary,
    ):
        """Normalize nested standings rows into the app's current-season team shape."""
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "standingsDateTimeUtc": "2026-03-12T20:28:00Z",
            "standings": [
                {
                    "teamAbbrev": {"default": "COL"},
                    "teamName": {"default": "Colorado Avalanche"},
                    "teamCommonName": {"default": "Avalanche"},
                    "teamLogo": "https://assets.nhle.com/logos/nhl/svg/COL_light.svg",
                    "conferenceName": "Western",
                    "divisionName": "Central",
                    "gamesPlayed": 63,
                    "wins": 43,
                    "losses": 11,
                    "otLosses": 9,
                    "points": 95,
                    "goalDifferential": 82,
                    "goalFor": 241,
                    "goalAgainst": 159,
                    "pointPctg": 0.753968,
                    "regulationWins": 38,
                    "regulationPlusOtWinPctg": 0.634921,
                    "streakCode": "L",
                    "streakCount": 1,
                    "divisionSequence": 1,
                    "conferenceSequence": 1,
                    "leagueSequence": 1,
                    "l10GamesPlayed": 10,
                    "l10Wins": 7,
                    "l10Losses": 3,
                    "l10OtLosses": 0,
                    "l10Points": 14,
                    "l10GoalDifferential": 8,
                },
                {
                    "teamAbbrev": {"default": "DAL"},
                    "teamName": {"default": "Dallas Stars"},
                    "teamCommonName": {"default": "Stars"},
                    "teamLogo": "https://assets.nhle.com/logos/nhl/svg/DAL_light.svg",
                    "conferenceName": "Western",
                    "divisionName": "Central",
                    "gamesPlayed": 64,
                    "wins": 40,
                    "losses": 14,
                    "otLosses": 10,
                    "points": 90,
                    "goalDifferential": 48,
                    "goalFor": 218,
                    "goalAgainst": 170,
                    "pointPctg": 0.703125,
                    "regulationWins": 35,
                    "regulationPlusOtWinPctg": 0.59375,
                    "streakCode": "W",
                    "streakCount": 2,
                    "divisionSequence": 2,
                    "conferenceSequence": 2,
                    "leagueSequence": 3,
                    "l10GamesPlayed": 10,
                    "l10Wins": 9,
                    "l10Losses": 1,
                    "l10OtLosses": 0,
                    "l10Points": 18,
                    "l10GoalDifferential": 18,
                },
            ],
        }
        mock_get_client.return_value = mock_client
        mock_team_summary.return_value = pd.DataFrame(
            {
                "teamAbbrev": ["COL", "DAL"],
                "PP%": [16.0, 29.7],
            }
        )

        standings = data_loaders.get_current_nhl_standings()

        self.assertEqual(list(standings["teamAbbrev"]), ["COL", "DAL"])
        col_row = standings[standings["teamAbbrev"] == "COL"].iloc[0]
        self.assertEqual(col_row["teamCommonName"], "Avalanche")
        self.assertEqual(col_row["recordLabel"], "43-11-9")
        self.assertEqual(col_row["l10RecordLabel"], "7-3-0")
        self.assertEqual(col_row["standingsDateTimeUtc"], "2026-03-12T20:28:00Z")
        self.assertAlmostEqual(float(col_row["goalDiffPerGame"]), 82 / 63, places=6)
        self.assertAlmostEqual(float(col_row["l10PointPctg"]), 14 / 20, places=6)
        self.assertAlmostEqual(float(col_row["PP%"]), 16.0)


class TeamIdentitySummaryTests(unittest.TestCase):
    """Cover the team identity summaries used by overview-card modals."""

    def setUp(self):
        data_loaders.get_team_identity_summary.clear()
        data_loaders.get_current_nhl_standings.clear()
        data_loaders.get_team_trophy_summary.clear()

    def tearDown(self):
        data_loaders.get_team_identity_summary.clear()
        data_loaders.get_current_nhl_standings.clear()
        data_loaders.get_team_trophy_summary.clear()

    def test_team_identity_summary_uses_founded_year_and_live_alignment(self):
        """Keep normal franchises aligned to joined year plus current conference context."""
        team_history = pd.DataFrame(
            [
                {
                    "teamAbbrev": "EDM",
                    "FranchiseAbbrev": "EDM",
                    "teamFullName": "Edmonton Oilers",
                    "SeasonYear": 1979,
                    "gameTypeId": 2,
                },
                {
                    "teamAbbrev": "EDM",
                    "FranchiseAbbrev": "EDM",
                    "teamFullName": "Edmonton Oilers",
                    "SeasonYear": 2025,
                    "gameTypeId": 2,
                },
            ]
        )
        standings_df = pd.DataFrame(
            [
                {
                    "teamAbbrev": "EDM",
                    "conferenceName": "Western",
                    "divisionName": "Pacific",
                }
            ]
        )

        with patch.object(data_loaders, "load_all_team_seasons", return_value=team_history), patch.object(
            data_loaders,
            "get_current_nhl_standings",
            return_value=standings_df,
        ), patch.object(
            data_loaders,
            "get_team_trophy_summary",
            return_value={"EDM": {"stanley_cups": 5, "cup_seasons": [19901991, 19891990], "cup_labels": ["1990-91", "1989-90"]}},
        ):
            summary = data_loaders.get_team_identity_summary("EDM")

        self.assertEqual(summary["team_name"], "Edmonton Oilers")
        self.assertEqual(summary["joined_nhl_year"], 1979)
        self.assertEqual(summary["joined_nhl_label"], "1979-80")
        self.assertEqual(summary["current_identity_since_year"], 1979)
        self.assertEqual(summary["conference_name"], "Western")
        self.assertEqual(summary["division_name"], "Pacific")
        self.assertEqual(summary["total_nhl_seasons"], 2)
        self.assertEqual(summary["stanley_cup_count"], 5)
        self.assertEqual(summary["stanley_cup_labels"], ["1990-91", "1989-90"])
        self.assertIn("Edmonton Oilers", summary["lineage_label"])

    def test_team_identity_summary_handles_lineage_franchises(self):
        """Build a lineage path for relocated franchises like Utah."""
        team_history = pd.DataFrame(
            [
                {
                    "teamAbbrev": "WIN",
                    "FranchiseAbbrev": "UTA",
                    "teamFullName": "Winnipeg Jets",
                    "SeasonYear": 1979,
                    "gameTypeId": 2,
                },
                {
                    "teamAbbrev": "PHX",
                    "FranchiseAbbrev": "UTA",
                    "teamFullName": "Phoenix Coyotes",
                    "SeasonYear": 1997,
                    "gameTypeId": 2,
                },
                {
                    "teamAbbrev": "ARI",
                    "FranchiseAbbrev": "UTA",
                    "teamFullName": "Arizona Coyotes",
                    "SeasonYear": 2021,
                    "gameTypeId": 2,
                },
                {
                    "teamAbbrev": "UTA",
                    "FranchiseAbbrev": "UTA",
                    "teamFullName": "Utah Hockey Club",
                    "SeasonYear": 2024,
                    "gameTypeId": 2,
                },
            ]
        )

        with patch.object(data_loaders, "load_all_team_seasons", return_value=team_history), patch.object(
            data_loaders,
            "get_current_nhl_standings",
            return_value=pd.DataFrame(),
        ), patch.object(
            data_loaders,
            "get_team_trophy_summary",
            return_value={"UTA": {"stanley_cups": 0, "cup_seasons": [], "cup_labels": []}},
        ):
            summary = data_loaders.get_team_identity_summary("UTA")

        self.assertEqual(summary["joined_nhl_year"], 1979)
        self.assertEqual(summary["current_identity_since_year"], 2024)
        self.assertEqual(summary["current_identity_since_label"], "2024-25")
        self.assertEqual(summary["total_nhl_seasons"], 4)
        self.assertEqual(
            [segment["abbr"] for segment in summary["lineage"]],
            ["WIN", "PHX", "ARI", "UTA"],
        )
        self.assertIn("Utah Hockey Club", summary["lineage_label"])
        self.assertEqual(summary["conference_name"], "")
        self.assertEqual(summary["division_name"], "")


class SeasonLeaderboardLoaderTests(unittest.TestCase):
    """Cover season-summary leaderboard aggregation and ranking."""

    def setUp(self):
        """Clear cached season leaderboard helpers before each test."""
        data_loaders.get_season_leaderboard.clear()
        data_loaders.get_player_season_rank_map.clear()

    def tearDown(self):
        """Clear cached season leaderboard helpers after each test."""
        data_loaders.get_season_leaderboard.clear()
        data_loaders.get_player_season_rank_map.clear()

    def test_get_season_leaderboard_merges_both_skater_game_types(self):
        """Combine regular and playoff skater rows and recompute rate stats."""
        regular_payload = {
            "data": [
                {
                    "playerId": 1,
                    "skaterFullName": "Player One",
                    "gamesPlayed": 2,
                    "points": 3,
                    "goals": 1,
                    "assists": 2,
                    "penaltyMinutes": 4,
                    "plusMinus": 2,
                    "shots": 4,
                    "timeOnIcePerGame": 600,
                },
                {
                    "playerId": 1,
                    "skaterFullName": "Player One",
                    "gamesPlayed": 1,
                    "points": 2,
                    "goals": 1,
                    "assists": 1,
                    "penaltyMinutes": 2,
                    "plusMinus": 1,
                    "shots": 2,
                    "timeOnIcePerGame": 300,
                },
                {
                    "playerId": 2,
                    "skaterFullName": "Player Two",
                    "gamesPlayed": 3,
                    "points": 4,
                    "goals": 2,
                    "assists": 2,
                    "penaltyMinutes": 0,
                    "plusMinus": 0,
                    "shots": 8,
                    "timeOnIcePerGame": 900,
                },
            ]
        }
        playoff_payload = {
            "data": [
                {
                    "playerId": 1,
                    "skaterFullName": "Player One",
                    "gamesPlayed": 1,
                    "points": 1,
                    "goals": 0,
                    "assists": 1,
                    "penaltyMinutes": 0,
                    "plusMinus": -1,
                    "shots": 1,
                    "timeOnIcePerGame": 600,
                }
            ]
        }

        with patch("nhl.data_loaders.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get.side_effect = [regular_payload, playoff_payload]
            mock_get_client.return_value = mock_client
            leaderboard = data_loaders.get_season_leaderboard("Skater", 2024, "Both")

        player_one = leaderboard.loc[leaderboard["playerId"] == 1].iloc[0]
        self.assertEqual(int(player_one["GP"]), 4)
        self.assertEqual(int(player_one["Points"]), 6)
        self.assertEqual(int(player_one["Goals"]), 2)
        self.assertEqual(int(player_one["Assists"]), 4)
        self.assertAlmostEqual(float(player_one["PPG"]), 1.5)
        self.assertAlmostEqual(float(player_one["SH%"]), 28.57142857, places=5)
        self.assertAlmostEqual(float(player_one["TOI"]), 8.75)

    def test_get_player_season_rank_map_sorts_gaa_low_to_high_with_ties(self):
        """Rank GAA ascending and give tied goalies the same place."""
        leaderboard = pd.DataFrame(
            {
                "playerId": [31, 30, 35],
                "GAA": [2.25, 1.9, 1.9],
            }
        )

        with patch("nhl.data_loaders.get_season_leaderboard", return_value=leaderboard):
            ranks = data_loaders.get_player_season_rank_map("Goalie", 2024, "Regular", "GAA")

        self.assertEqual(ranks, {30: 1, 35: 1, 31: 3})


class PlayerSearchLoaderTests(unittest.TestCase):
    """Cover search payload normalization and local fallback matching."""

    def setUp(self):
        """Clear the cached search endpoint wrapper before each test."""
        data_loaders.search_player.clear()

    def tearDown(self):
        """Clear the cached search endpoint wrapper after each test."""
        data_loaders.search_player.clear()

    def test_search_player_rejects_string_error_payloads(self):
        """Treat string API error payloads as an empty search result."""
        with patch("nhl.data_loaders.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get.return_value = "Something wrong happened"
            mock_get_client.return_value = mock_client
            self.assertEqual(data_loaders.search_player("McDavid"), [])

    def test_search_player_normalizes_list_payloads(self):
        """Keep only valid search rows and normalize names into one shape."""
        with patch("nhl.data_loaders.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get.return_value = [
                "bad row",
                {"playerId": "8478402", "name": "Connor McDavid", "teamAbbrev": "EDM"},
                {"playerId": "8471214", "firstName": "Alex", "lastName": "Ovechkin"},
                {"playerId": "oops", "name": "Broken Id"},
                {"playerId": 1234},
            ]
            mock_get_client.return_value = mock_client
            self.assertEqual(
                data_loaders.search_player("m"),
                [
                    {"playerId": 8478402, "name": "Connor McDavid", "teamAbbrev": "EDM"},
                    {"playerId": 8471214, "name": "Alex Ovechkin", "teamAbbrev": ""},
                ],
            )

    def test_search_local_players_matches_multi_token_queries(self):
        """Keep local search useful when the live D3 endpoint is dead."""
        with patch(
            "nhl.data_loaders.get_id_to_name_map",
            return_value={8478402: "Connor McDavid", 8471675: "Sidney Crosby"},
        ), patch(
            "nhl.data_loaders.get_clone_details_map",
            return_value={8478402: {"team": "EDM"}, 8471675: {"team": "PIT"}},
        ), patch("nhl.data_loaders.search_player", return_value=[]):
            self.assertEqual(
                data_loaders.search_local_players("Connor Mc", "Skater"),
                {"[EDM] Connor McDavid": 8478402},
            )


class NetworkReliabilityTests(unittest.TestCase):
    """Cover guarded JSON request fallbacks in audited loader paths."""

    def setUp(self):
        data_loaders.get_team_roster.clear()
        data_loaders.get_player_landing.clear()

    def tearDown(self):
        data_loaders.get_team_roster.clear()
        data_loaders.get_player_landing.clear()

    def test_request_json_dict_returns_empty_on_http_error_status(self):
        """NHLClient returns None on failure — function returns {}."""
        with patch("nhl.data_loaders.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get.return_value = None
            mock_get_client.return_value = mock_client
            self.assertEqual(data_loaders.get_player_landing(97), {})

    def test_get_team_roster_returns_empty_when_json_decode_fails(self):
        """NHLClient returns None on failure — roster falls back to {}."""
        with patch("nhl.data_loaders.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get.return_value = None
            mock_get_client.return_value = mock_client
            self.assertEqual(data_loaders.get_team_roster("EDM"), {})


class PlayerLandingMigrationTests(unittest.TestCase):
    """Verify get_player_landing routes through NHLClient."""

    def setUp(self):
        """Clear st.cache_data for get_player_landing between tests."""
        data_loaders.get_player_landing.clear()

    def tearDown(self):
        data_loaders.get_player_landing.clear()

    @patch("nhl.data_loaders.get_client")
    def test_get_player_landing_returns_dict_on_success(self, mock_get_client):
        """NHLClient.get returns a valid dict — passed through unchanged."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"playerId": 8471675, "position": "C"}
        mock_get_client.return_value = mock_client

        result = data_loaders.get_player_landing(8471675)

        self.assertEqual(result, {"playerId": 8471675, "position": "C"})
        mock_client.get.assert_called_once_with(
            url="https://api-web.nhle.com/v1/player/8471675/landing",
            cache_key="player_landing:8471675",
            ttl=7200,
            timeout=5,
        )

    @patch("nhl.data_loaders.get_client")
    def test_get_player_landing_returns_empty_dict_on_none(self, mock_get_client):
        """NHLClient.get returns None on failure — function returns {}."""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client

        result = data_loaders.get_player_landing(8471675)

        self.assertEqual(result, {})

    @patch("nhl.data_loaders.get_client")
    def test_get_player_landing_returns_empty_dict_on_non_dict(self, mock_get_client):
        """NHLClient.get returns a non-dict — function returns {}."""
        mock_client = MagicMock()
        mock_client.get.return_value = [1, 2, 3]
        mock_get_client.return_value = mock_client

        result = data_loaders.get_player_landing(8471675)

        self.assertEqual(result, {})


class BroadMigrationTests(unittest.TestCase):
    """Verify Phase 2b functions route HTTP through NHLClient."""

    def setUp(self):
        data_loaders.load_all_team_seasons.clear()
        data_loaders.get_top_50.clear()
        data_loaders.get_team_roster.clear()
        data_loaders.get_current_nhl_standings.clear()
        data_loaders.search_player.clear()
        data_loaders.get_player_season_game_log.clear()
        data_loaders.fetch_all_time_records.clear()

    def tearDown(self):
        data_loaders.load_all_team_seasons.clear()
        data_loaders.get_top_50.clear()
        data_loaders.get_team_roster.clear()
        data_loaders.get_current_nhl_standings.clear()
        data_loaders.search_player.clear()
        data_loaders.get_player_season_game_log.clear()
        data_loaders.fetch_all_time_records.clear()

    @patch("nhl.data_loaders.get_client")
    def test_load_all_team_seasons_routes_through_nhl_client(self, mock_get_client):
        """Three client.get calls: team_list, team_summary:2, team_summary:3."""
        mock_client = MagicMock()
        mock_client.get.side_effect = [
            {"data": [{"id": 1, "triCode": "TST"}]},
            {"data": [
                {"teamId": 1, "seasonId": 20242025, "gamesPlayed": 82, "wins": 40,
                 "losses": 30, "otLosses": 12, "points": 92, "ties": 0,
                 "goalsFor": 210, "goalsAgainst": 190, "goalsForPerGame": 2.56,
                 "goalsAgainstPerGame": 2.31, "pointPct": 0.561, "powerPlayPct": 0.20},
            ]},
            {"data": []},
        ]
        mock_get_client.return_value = mock_client

        result = data_loaders.load_all_team_seasons()

        self.assertEqual(mock_client.get.call_count, 3)
        keys = [call.kwargs.get("cache_key") for call in mock_client.get.call_args_list]
        self.assertEqual(keys, ["team_list", "team_summary:2", "team_summary:3"])
        self.assertFalse(result.empty)

    @patch("nhl.data_loaders.get_cache")
    @patch("nhl.data_loaders.get_client")
    def test_fetch_all_time_records_checks_disk_cache(self, mock_get_client, mock_get_cache):
        """Return cached result from disk without paginating."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = [{"playerId": 1, "points": 100}]
        mock_get_cache.return_value = mock_cache

        result = data_loaders.fetch_all_time_records("Skater", "Regular")

        mock_cache.get.assert_called_once_with("records:Skater:Regular")
        self.assertEqual(result, [{"playerId": 1, "points": 100}])
        mock_get_client.return_value.get.assert_not_called()

    @patch("nhl.data_loaders.get_client")
    def test_get_top_50_routes_through_nhl_client(self, mock_get_client):
        """Verify cache key top50_skater:Points and TTL T1."""
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "data": [
                {"playerId": 99, "firstName": "Wayne", "lastName": "Gretzky", "points": 2857},
            ]
        }
        mock_get_client.return_value = mock_client

        result = data_loaders.get_top_50("Points")

        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args.kwargs
        self.assertEqual(call_kwargs["cache_key"], "top50_skater:Points")
        self.assertEqual(call_kwargs["ttl"], 86400)
        self.assertIn("Wayne Gretzky", list(result.keys())[0])

    @patch("nhl.data_loaders.get_client")
    def test_get_team_roster_returns_empty_on_none(self, mock_get_client):
        """NHLClient returns None — roster returns {}."""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client

        self.assertEqual(data_loaders.get_team_roster("EDM"), {})

    @patch("nhl.data_loaders.get_client")
    def test_get_current_nhl_standings_routes_through_nhl_client(self, mock_get_client):
        """Verify cache key 'standings'."""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_get_client.return_value = mock_client

        result = data_loaders.get_current_nhl_standings()

        call_kwargs = mock_client.get.call_args.kwargs
        self.assertEqual(call_kwargs["cache_key"], "standings")
        self.assertTrue(result.empty)

    @patch("nhl.data_loaders.get_client")
    def test_search_player_routes_through_nhl_client(self, mock_get_client):
        """Verify normalized cache key for search."""
        mock_client = MagicMock()
        mock_client.get.return_value = []
        mock_get_client.return_value = mock_client

        data_loaders.search_player("  McDavid  ")

        call_kwargs = mock_client.get.call_args.kwargs
        self.assertEqual(call_kwargs["cache_key"], "search:mcdavid")

    @patch("nhl.data_loaders.get_client")
    def test_get_player_season_game_log_uses_effective_ttl(self, mock_get_client):
        """Past season gets T1_TTL (86400), current gets T2."""
        mock_client = MagicMock()
        mock_client.get.side_effect = [
            {"birthDate": "1997-01-13", "position": "C", "seasonTotals": []},
            {"gameLog": []},
            {"gameLog": []},
        ]
        mock_get_client.return_value = mock_client

        data_loaders.get_player_season_game_log(8478402, "McDavid", 2020)

        gamelog_calls = [c for c in mock_client.get.call_args_list
                         if c.kwargs.get("cache_key", "").startswith("player_gamelog:")]
        self.assertTrue(len(gamelog_calls) >= 1)
        self.assertEqual(gamelog_calls[0].kwargs["ttl"], 86400)


if __name__ == "__main__":
    unittest.main()
