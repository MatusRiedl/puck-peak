import math
import unittest

import numpy as np
import pandas as pd

from nhl import xg

HOME_ID, AWAY_ID = 10, 8


def _play(sort_order, kind, clock, owner=None, x=None, y=None, zone=None, situation="1551",
          period=1, period_type="REG", goalie=None, shot_type="wrist", side="left"):
    """Build one play-by-play event."""
    details = {}
    if owner is not None:
        details["eventOwnerTeamId"] = owner
    if x is not None:
        details.update({"xCoord": x, "yCoord": y})
    if zone is not None:
        details["zoneCode"] = zone
    if goalie is not None:
        details["goalieInNetId"] = goalie
    if kind in xg.ATTEMPT_TYPES:
        details["shotType"] = shot_type
    play = {
        "sortOrder": sort_order,
        "typeDescKey": kind,
        "timeInPeriod": clock,
        "situationCode": situation,
        "periodDescriptor": {"number": period, "periodType": period_type},
        "details": details,
    }
    if side is not None:
        play["homeTeamDefendingSide"] = side
    return play


def _payload(plays):
    """Wrap plays in a play-by-play payload between TOR (home) and MTL (away)."""
    return {
        "id": 2025020001,
        "season": 20252026,
        "gameType": 2,
        "gameDate": "2025-10-08",
        "homeTeam": {"id": HOME_ID, "abbrev": "TOR"},
        "awayTeam": {"id": AWAY_ID, "abbrev": "MTL"},
        "rosterSpots": [
            {"teamId": HOME_ID, "playerId": 100, "firstName": {"default": "Home"}, "lastName": {"default": "Starter"}, "positionCode": "G"},
            {"teamId": AWAY_ID, "playerId": 200, "firstName": {"default": "Away"}, "lastName": {"default": "Starter"}, "positionCode": "G"},
            {"teamId": AWAY_ID, "playerId": 201, "firstName": {"default": "Away"}, "lastName": {"default": "Backup"}, "positionCode": "G"},
        ],
        "plays": plays,
    }


class ParsePlayByPlayTests(unittest.TestCase):
    """Cover shot extraction from play-by-play payloads."""

    def setUp(self):
        self.parsed = xg.parse_play_by_play(
            _payload(
                [
                    _play(1, "faceoff", "00:00", owner=HOME_ID, x=0, y=0, zone="N"),
                    _play(2, "shot-on-goal", "00:03", owner=HOME_ID, x=80, y=0, zone="O", goalie=200),
                    _play(3, "blocked-shot", "00:10", owner=AWAY_ID, x=70, y=5, zone="D"),
                    _play(4, "missed-shot", "00:12", owner=HOME_ID, x=85, y=-3, zone="O", goalie=200),
                    _play(5, "goal", "01:00", owner=AWAY_ID, x=-80, y=0, zone="O", situation="1541", goalie=100, shot_type="snap"),
                    _play(6, "shot-on-goal", "02:00", owner=HOME_ID, x=60, y=0, zone="O", situation="0651"),
                    _play(7, "shot-on-goal", "05:00", owner=HOME_ID, x=80, y=0, zone="O", situation="0101", goalie=200),
                    _play(8, "goal", "00:00", owner=AWAY_ID, x=-80, y=0, zone="O", period=5, period_type="SO", goalie=100),
                ]
            )
        )
        self.shots = self.parsed["shots"]

    def test_only_unblocked_non_shootout_non_penalty_shots_are_kept(self):
        """Blocked, shootout and penalty-shot attempts are excluded."""
        self.assertEqual(len(self.shots), 4)
        self.assertEqual(list(self.shots["ShootingTeam"]), ["TOR", "TOR", "MTL", "TOR"])

    def test_coordinates_are_normalized_toward_the_attacked_net(self):
        """Both teams' shots from the same spot in front of their target are 9 feet out."""
        self.assertAlmostEqual(float(self.shots.iloc[0]["Distance"]), 9.0)
        self.assertAlmostEqual(float(self.shots.iloc[2]["Distance"]), 9.0)
        self.assertAlmostEqual(float(self.shots.iloc[0]["Angle"]), 0.0)

    def test_rush_rebound_strength_and_empty_net_flags(self):
        """A shot right after a neutral-zone faceoff is a rush; a shot 2 s after our blocked attempt is a rebound."""
        first, rebound, power_play_goal, empty_net = (self.shots.iloc[i] for i in range(4))
        self.assertTrue(bool(first["Rush"]))
        self.assertFalse(bool(first["Rebound"]))
        self.assertTrue(bool(rebound["Rebound"]))
        self.assertTrue(bool(power_play_goal["IsGoal"]))
        self.assertEqual(int(power_play_goal["ShooterSkaters"]), 5)
        self.assertEqual(int(power_play_goal["DefenderSkaters"]), 4)
        self.assertFalse(bool(power_play_goal["FiveOnFive"]))
        self.assertTrue(bool(empty_net["EmptyNet"]))

    def test_starters_are_the_first_goalies_to_face_a_shot(self):
        """The first goalie in net for a shot against each team is its starter."""
        meta = self.parsed["meta"]
        self.assertEqual(meta["home_starter"], 100)
        self.assertEqual(meta["away_starter"], 200)
        self.assertEqual(meta["goalie_names"][201], "Away Backup")

    def test_attack_direction_is_inferred_when_the_defending_side_is_missing(self):
        """2017-18 payloads have no defending side; offensive-zone shots reveal the direction."""
        parsed = xg.parse_play_by_play(
            _payload(
                [
                    _play(1, "shot-on-goal", "00:30", owner=HOME_ID, x=-75, y=4, zone="O", goalie=200, side=None),
                    _play(2, "shot-on-goal", "00:50", owner=HOME_ID, x=-70, y=0, zone="O", goalie=200, side=None),
                ]
            )
        )
        self.assertAlmostEqual(float(parsed["shots"].iloc[1]["Distance"]), 19.0)


class DesignAndScoringTests(unittest.TestCase):
    """Cover the xG design matrix and runtime scoring."""

    def test_design_matrix_columns_and_shot_type_indicators(self):
        """Wrist shots are the baseline; unknown types land in type_other."""
        shots = pd.DataFrame(
            {
                "Distance": [10.0, 30.0, 45.0],
                "Angle": [0.1, 0.5, 1.0],
                "ShotType": ["wrist", "tip-in", "cradle"],
                "X": [79.0, 60.0, 92.0],
                "Rebound": [True, False, False],
                "Rush": [False, True, False],
                "ShooterSkaters": [6, 5, 4],
                "DefenderSkaters": [5, 4, 5],
                "EmptyNet": [False, False, True],
                "SecondsSincePrevious": [2.0, 10.0, 99.0],
            }
        )
        matrix = pd.DataFrame(xg.xg_design_matrix(shots), columns=xg.XG_FEATURES)

        self.assertEqual(list(matrix.columns), list(xg.XG_FEATURES))
        self.assertEqual(float(matrix.loc[0, ["type_snap", "type_tip_in", "type_other"]].sum()), 0.0)
        self.assertEqual(float(matrix.loc[1, "type_tip_in"]), 1.0)
        self.assertEqual(float(matrix.loc[2, "type_other"]), 1.0)
        self.assertEqual(list(matrix["power_play"]), [0.0, 1.0, 0.0])
        self.assertEqual(list(matrix["behind_net"]), [0.0, 0.0, 1.0])

    def test_scoring_applies_the_exported_logistic_model(self):
        """With zero coefficients every shot gets the intercept's probability."""
        payload = {
            "feature_order": list(xg.XG_FEATURES),
            "coefficients": [0.0] * len(xg.XG_FEATURES),
            "intercept": math.log(0.1 / 0.9),
            "scaler_mean": [0.0] * len(xg.XG_FEATURES),
            "scaler_scale": [1.0] * len(xg.XG_FEATURES),
        }
        shots = xg.parse_play_by_play(_payload([_play(1, "shot-on-goal", "00:30", owner=HOME_ID, x=80, y=0, zone="O", goalie=200)]))["shots"]

        probabilities = xg.score_expected_goals(shots, xg.validate_xg_model(payload))

        self.assertAlmostEqual(float(probabilities[0]), 0.1)
        self.assertIsNone(xg.validate_xg_model({"feature_order": ["distance"]}))


class SummaryTests(unittest.TestCase):
    """Cover per-game team xG and goalie workloads."""

    def test_empty_net_shots_count_nowhere_but_five_on_five_splits_by_side(self):
        """Team xG and goalie workloads exclude empty-net attempts."""
        parsed = xg.parse_play_by_play(
            _payload(
                [
                    _play(1, "shot-on-goal", "00:30", owner=HOME_ID, x=80, y=0, zone="O", goalie=200),
                    _play(2, "goal", "01:30", owner=AWAY_ID, x=-80, y=0, zone="O", goalie=100),
                    _play(3, "shot-on-goal", "03:00", owner=AWAY_ID, x=-60, y=10, zone="O", situation="1541", goalie=100),
                    _play(4, "shot-on-goal", "19:00", owner=HOME_ID, x=40, y=0, zone="O", situation="0651"),
                ]
            )
        )
        expected_goals = np.array([0.2, 0.3, 0.1, 0.05])

        games, goalies = xg.summarize_games(parsed["shots"], expected_goals, [parsed["meta"]])

        game = games.iloc[0]
        self.assertAlmostEqual(float(game["HomeXgf5v5"]), 0.2)
        self.assertAlmostEqual(float(game["AwayXgf5v5"]), 0.3)
        self.assertAlmostEqual(float(game["HomeXgfAll"]), 0.2)
        self.assertAlmostEqual(float(game["AwayXgfAll"]), 0.4)
        home_goalie = goalies[goalies["GoalieId"] == 100].iloc[0]
        self.assertEqual(int(home_goalie["Shots"]), 2)
        self.assertAlmostEqual(float(home_goalie["XgAgainst"]), 0.4)
        self.assertEqual(int(home_goalie["GoalsAgainst"]), 1)
        self.assertTrue(bool(home_goalie["Started"]))
        self.assertEqual(home_goalie["GoalieName"], "Home Starter")

    def test_research_features_are_pregame_shrunk_and_skip_games_without_play_by_play(self):
        """Game 2 sees only game 1; game 3 has no play-by-play and does not count; playoffs use season end."""
        games = pd.DataFrame(
            {
                "SeasonYear": [2025, 2025, 2025, 2025],
                "GameTypeId": [2, 2, 2, 3],
                "GameId": [1, 2, 3, 4],
                "GameDate": ["2025-10-08", "2025-10-10", "2025-10-12", "2026-04-20"],
                "HomeTeam": ["TOR", "TOR", "TOR", "TOR"],
                "AwayTeam": ["MTL", "MTL", "MTL", "MTL"],
            }
        )
        xg_games = pd.DataFrame(
            {"GameId": [1, 2], "HomeXgf5v5": [3.0, 1.0], "AwayXgf5v5": [1.0, 1.0], "HomeXgfAll": [3.5, 1.0], "AwayXgfAll": [1.5, 1.0]}
        )
        goalies = pd.DataFrame({"GameId": [2], "HomeGoalieGsax": [0.3], "AwayGoalieGsax": [-0.1]})

        features = xg.research_game_features(games, xg_games, goalies, prior_weight=4.0).set_index("GameId")

        self.assertAlmostEqual(float(features.loc[1, "xg_share_shrunk_diff"]), 0.0)
        toronto_before_game_two = (4 * 0.5 + 0.75) / 5
        self.assertAlmostEqual(float(features.loc[2, "xg_share_shrunk_diff"]), toronto_before_game_two - (1 - toronto_before_game_two))
        toronto_end = (4 * 0.5 + 0.75 + 0.5) / 6
        self.assertAlmostEqual(float(features.loc[4, "xg_share_shrunk_diff"]), toronto_end - (1 - toronto_end))
        self.assertAlmostEqual(float(features.loc[2, "goalie_gsax_diff"]), 0.4)
        self.assertEqual(float(features.loc[3, "goalie_gsax_diff"]), 0.0)


class GoalieHistoryTests(unittest.TestCase):
    """Cover point-in-time goalie ratings and projected starters."""

    def _history(self, rows, **params):
        frame = pd.DataFrame(
            rows,
            columns=["GameId", "SeasonYear", "GameTypeId", "GameDate", "Team", "GoalieId", "GoalieName", "Shots", "XgAgainst", "GoalsAgainst", "Started"],
        )
        return xg.GoalieHistory(frame, params)

    def test_rating_is_shrunk_and_ignores_same_day_games(self):
        """Saving 3 goals above expected on 30 shots with a 270-shot prior is +0.3 per 30 shots."""
        history = self._history(
            [(1, 2025, 2, "2025-10-08", "TOR", 100, "A", 30, 4.0, 1, True)],
            goalie_prior_shots=270.0,
        )
        self.assertEqual(history.rating(100, "2025-10-08", 2025), 0.0)
        self.assertAlmostEqual(history.rating(100, "2025-10-09", 2025), 30.0 * 3.0 / 300.0)
        self.assertEqual(history.rating(999, "2025-10-09", 2025), 0.0)

    def test_older_seasons_decay(self):
        """A season boundary keeps only the decay share of past workload."""
        history = self._history(
            [(1, 2024, 2, "2024-10-08", "TOR", 100, "A", 100, 12.0, 2, True)],
            goalie_prior_shots=100.0,
            goalie_season_decay=0.5,
        )
        self.assertAlmostEqual(history.rating(100, "2025-10-01", 2025), 30.0 * 5.0 / (50.0 + 100.0))

    def test_projected_starter_follows_recent_starts_and_rests_on_back_to_backs(self):
        """The busiest recent goalie starts, except on the second night after starting last night."""
        rows = []
        for day, goalie in enumerate([100, 100, 101, 100, 100], start=1):
            rows.append((day, 2025, 2, f"2025-10-{day:02d}", "TOR", goalie, "", 30, 3.0, 3, True))
        history = self._history(rows)

        self.assertEqual(history.projected_starter("TOR", "2025-10-07"), 100)
        self.assertEqual(history.projected_starter("TOR", "2025-10-06", back_to_back=True), 101)
        self.assertIsNone(history.projected_starter("MTL", "2025-10-06"))
        self.assertEqual(history.projected_starter("TOR", "2025-10-01"), None)

    def test_back_to_back_projection_a_day_ahead_rests_the_first_choice(self):
        """If last night's game is not on record yet, assume the first choice plays it."""
        starts = [("2025-10-01", 1, 100), ("2025-10-03", 2, 100), ("2025-10-05", 3, 101)]

        self.assertEqual(xg.choose_starter(starts, "2025-10-08", back_to_back=True), 101)
        self.assertEqual(xg.choose_starter(starts, "2025-10-06", back_to_back=True), 100)
        self.assertEqual(xg.choose_starter(starts, "2025-10-08", back_to_back=False), 100)


if __name__ == "__main__":
    unittest.main()
