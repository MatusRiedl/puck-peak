import unittest

import pandas as pd

from nhl import team_ratings

TEAM_MAP = {1: "TOR", 2: "MTL", 3: "ARI", 4: "BOS"}


def _row(game_id, date, team_id, opponent, home_road, goals_for, goals_against, wins,
         regulation_win=0, shootout_win=0, ot_loss=0, shots=30):
    """Build one stats-API ``team/summary?isGame=true`` row."""
    return {
        "gameId": game_id,
        "gameDate": date,
        "teamId": team_id,
        "opponentTeamAbbrev": opponent,
        "homeRoad": home_road,
        "goalsFor": goals_for,
        "goalsAgainst": goals_against,
        "wins": wins,
        "winsInRegulation": regulation_win,
        "winsInShootout": shootout_win,
        "otLosses": ot_loss,
        "shotsForPerGame": shots,
    }


def _game(game_id, date, home_id, away_id, home_goals, away_goals, home_won, result="REG"):
    """Build both team rows for one game. ``result`` is REG, OT or SO."""
    home_abbr, away_abbr = TEAM_MAP[home_id], TEAM_MAP[away_id]
    regulation = 1 if result == "REG" else 0
    shootout = 1 if result == "SO" else 0
    loser_otl = 0 if result == "REG" else 1
    return [
        _row(game_id, date, home_id, away_abbr, "H", home_goals, away_goals, int(home_won),
             regulation_win=regulation if home_won else 0, shootout_win=shootout if home_won else 0,
             ot_loss=0 if home_won else loser_otl),
        _row(game_id, date, away_id, home_abbr, "R", away_goals, home_goals, int(not home_won),
             regulation_win=regulation if not home_won else 0, shootout_win=shootout if not home_won else 0,
             ot_loss=loser_otl if home_won else 0),
    ]


def _table(*games, shooting=None):
    """Flatten game rows into a league game table."""
    rows = [row for game in games for row in game]
    return team_ratings.build_league_game_table(rows, shooting, TEAM_MAP)


class LeagueGameTableTests(unittest.TestCase):
    """Cover pairing per-team rows into one labelled row per game."""

    def test_shootout_winner_comes_from_wins_not_goals(self):
        """A home shootout win must be a home win even though goalsFor is level.

        The stats API leaves the shootout goal out of goalsFor. The old trainer compared
        goals, so every home shootout win was recorded as an away win.
        """
        table = _table(
            _game(2024020001, "2024-10-10", 1, 2, 3, 3, home_won=True, result="SO"),
            _game(2024020002, "2024-10-11", 2, 1, 2, 2, home_won=False, result="SO"),
        )

        first, second = table.iloc[0], table.iloc[1]
        self.assertEqual(int(first["HomeWin"]), 1)
        self.assertEqual(first["ResultType"], "SO")
        self.assertEqual(int(second["HomeWin"]), 0)
        self.assertEqual(second["ResultType"], "SO")

    def test_result_type_separates_regulation_and_overtime(self):
        """Regulation and overtime results are labelled from the winner's row."""
        table = _table(
            _game(2024020003, "2024-10-12", 1, 2, 4, 1, home_won=True, result="REG"),
            _game(2024020004, "2024-10-13", 2, 1, 2, 3, home_won=False, result="OT"),
        )

        self.assertEqual(list(table["ResultType"]), ["REG", "OT"])
        self.assertEqual(list(table["HomeWin"]), [1, 0])
        self.assertEqual(list(table["HomeTeam"]), ["TOR", "MTL"])

    def test_relocated_franchise_is_keyed_by_active_abbreviation(self):
        """Arizona rows carry over to Utah so ratings survive the relocation."""
        table = _table(_game(2023020001, "2023-10-12", 3, 4, 2, 1, home_won=True))

        self.assertEqual(table.iloc[0]["HomeTeam"], "UTA")
        self.assertEqual(int(table.iloc[0]["SeasonYear"]), 2023)
        self.assertEqual(int(table.iloc[0]["GameTypeId"]), 2)

    def test_shot_attempts_merge_by_game_and_team(self):
        """5v5 shot attempts attach to the home side of each game."""
        shooting = [
            {"gameId": 2024020001, "teamId": 1, "satFor": 55, "satAgainst": 45},
            {"gameId": 2024020001, "teamId": 2, "satFor": 45, "satAgainst": 55},
        ]
        table = _table(_game(2024020001, "2024-10-10", 1, 2, 3, 2, home_won=True), shooting=shooting)

        self.assertEqual(float(table.iloc[0]["HomeSatFor"]), 55.0)
        self.assertEqual(float(table.iloc[0]["HomeSatAgainst"]), 45.0)

    def test_rows_without_exactly_one_winner_are_dropped(self):
        """A half-published game (no winner yet) never becomes a training label."""
        rows = _game(2024020009, "2024-10-10", 1, 2, 1, 1, home_won=True)
        rows[0]["wins"] = 0
        table = team_ratings.build_league_game_table(rows, None, TEAM_MAP)

        self.assertTrue(table.empty)


class EloTests(unittest.TestCase):
    """Cover the online Elo pass."""

    def test_pregame_difference_excludes_the_game_result(self):
        """The first meeting starts level; the rematch reflects the first result."""
        table = _table(
            _game(2024020001, "2024-10-10", 1, 2, 5, 1, home_won=True),
            _game(2024020002, "2024-10-12", 2, 1, 2, 3, home_won=False),
        )
        run = team_ratings.run_elo(table)

        self.assertEqual(float(run.pregame_diff.iloc[0]), 0.0)
        # MTL hosts game two after losing game one, so home minus away is negative.
        self.assertLess(float(run.pregame_diff.iloc[1]), 0.0)
        self.assertGreater(run.ratings["TOR"], team_ratings.LEAGUE_MEAN_ELO)

    def test_ratings_regress_toward_the_mean_between_seasons(self):
        """A season boundary keeps only the carryover share of each team's edge."""
        params = {"elo_carryover": 0.5}
        table = _table(
            _game(2024020001, "2024-10-10", 1, 2, 5, 1, home_won=True),
            _game(2025020001, "2025-10-10", 1, 2, 3, 2, home_won=True),
        )
        first_season = team_ratings.run_elo(table.iloc[:1], params)
        both = team_ratings.run_elo(table, params)

        edge_after_first = first_season.ratings["TOR"] - first_season.ratings["MTL"]
        self.assertAlmostEqual(float(both.pregame_diff.iloc[1]), edge_after_first * 0.5, places=6)


class TeamFormTests(unittest.TestCase):
    """Cover the shrunk season-to-date form features."""

    def test_opening_game_uses_the_regressed_prior_and_later_games_blend_results(self):
        """Game one of a season shows only the prior; game two includes game one."""
        params = {"form_prior_weight": 10.0, "goal_diff_prior_keep": 0.5}
        table = _table(
            _game(2024020001, "2024-10-10", 1, 2, 4, 2, home_won=True),
            _game(2024020002, "2024-10-12", 2, 1, 1, 3, home_won=False),
            _game(2025020001, "2025-10-10", 1, 2, 5, 1, home_won=True),
            _game(2025020002, "2025-10-12", 1, 2, 2, 1, home_won=True),
        )
        form = team_ratings.compute_team_form(table, params)
        toronto_2025 = form[(form["Team"] == "TOR") & (form["SeasonYear"] == 2025)].reset_index(drop=True)

        # 2024 average goal diff for TOR is +2; the prior keeps half of it.
        self.assertEqual(int(toronto_2025.loc[0, "GamesBefore"]), 0)
        self.assertAlmostEqual(float(toronto_2025.loc[0, "goal_diff_shrunk"]), 1.0)
        # Game two blends the prior (worth 10 games) with game one's +4.
        self.assertAlmostEqual(float(toronto_2025.loc[1, "goal_diff_shrunk"]), (10 * 1.0 + 4.0) / 11)

    def test_team_without_previous_season_starts_at_league_mean(self):
        """A team with no prior season on file gets a neutral prior."""
        table = _table(_game(2024020001, "2024-10-10", 1, 2, 4, 2, home_won=True))
        form = team_ratings.compute_team_form(table)

        self.assertTrue((form["goal_diff_shrunk"] == 0.0).all())
        self.assertTrue((form["sog_share_shrunk"] == 0.5).all())


class ScheduleFlagTests(unittest.TestCase):
    """Cover back-to-back detection."""

    def test_back_to_back_needs_consecutive_calendar_days(self):
        """Flag the second night only when the previous game was yesterday."""
        schedule = pd.DataFrame(
            {
                "GameId": [1, 2, 3],
                "GameDate": ["2025-01-01", "2025-01-02", "2025-01-04"],
                "HomeTeam": ["TOR", "BOS", "TOR"],
                "AwayTeam": ["MTL", "TOR", "BOS"],
            }
        )
        flags = team_ratings.back_to_back_flags(schedule).set_index("GameId")

        self.assertEqual(int(flags.loc[2, "AwayBackToBack"]), 1)
        self.assertEqual(int(flags.loc[2, "HomeBackToBack"]), 0)
        self.assertEqual(int(flags.loc[3, "HomeBackToBack"]), 0)
        self.assertEqual(int(flags.loc[1, "HomeBackToBack"]), 0)


class SnapshotAndFeatureTests(unittest.TestCase):
    """Cover the runtime snapshot and the leak-safe training table."""

    def test_snapshot_before_opening_night_uses_prior_and_regressed_rating(self):
        """Opening-night estimates come from last season, regressed, with zero games."""
        table = _table(
            _game(2025020001, "2025-10-10", 1, 2, 5, 1, home_won=True),
            _game(2025020002, "2025-10-12", 2, 1, 1, 4, home_won=False),
        )
        snapshot = team_ratings.current_team_snapshot(table, 2026, {"elo_carryover": 0.5})
        last_season = team_ratings.run_elo(table, {"elo_carryover": 0.5})

        self.assertEqual(snapshot["TOR"]["games_played"], 0)
        expected_elo = team_ratings.LEAGUE_MEAN_ELO + (last_season.ratings["TOR"] - team_ratings.LEAGUE_MEAN_ELO) * 0.5
        self.assertAlmostEqual(snapshot["TOR"]["elo"], expected_elo, places=6)
        self.assertGreater(snapshot["TOR"]["goal_diff_shrunk"], 0.0)
        self.assertLess(snapshot["MTL"]["goal_diff_shrunk"], 0.0)

    def test_matchup_features_difference_home_minus_away(self):
        """Feature values are home minus away plus both schedule flags."""
        home = {"elo": 1550.0, "goal_diff_shrunk": 0.4, "sat_share_shrunk": 0.53, "sog_share_shrunk": 0.52}
        away = {"elo": 1500.0, "goal_diff_shrunk": -0.1, "sat_share_shrunk": 0.49, "sog_share_shrunk": 0.5}
        features = team_ratings.matchup_feature_values(home, away, home_back_to_back=False, away_back_to_back=True)

        self.assertEqual(set(features), set(team_ratings.MODEL_FEATURES))
        self.assertAlmostEqual(features["elo_diff"], 50.0)
        self.assertAlmostEqual(features["goal_diff_shrunk_diff"], 0.5)
        self.assertEqual(features["away_back_to_back"], 1.0)

    def test_training_features_never_see_their_own_result(self):
        """Flipping one game's result leaves that game's features unchanged."""
        games = [
            _game(2024020001, "2024-10-10", 1, 2, 4, 2, home_won=True),
            _game(2024020002, "2024-10-12", 2, 1, 3, 1, home_won=True),
            _game(2024020003, "2024-10-14", 1, 2, 2, 5, home_won=False),
        ]
        flipped = [list(game) for game in games]
        flipped[1] = _game(2024020002, "2024-10-12", 2, 1, 0, 6, home_won=False)

        original = team_ratings.build_model_features(_table(*games)).set_index("GameId")
        changed = team_ratings.build_model_features(_table(*flipped)).set_index("GameId")
        features = list(team_ratings.MODEL_FEATURES)

        pd.testing.assert_series_equal(original.loc[2024020002, features], changed.loc[2024020002, features])
        self.assertNotAlmostEqual(
            float(original.loc[2024020003, "elo_diff"]),
            float(changed.loc[2024020003, "elo_diff"]),
        )


if __name__ == "__main__":
    unittest.main()
