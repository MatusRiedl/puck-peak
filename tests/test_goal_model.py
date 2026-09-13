import math
import unittest

import numpy as np
import pandas as pd

from nhl import goal_model

PLAIN = goal_model.validate_goal_model(
    {"rate_scale": 1.0, "tie_inflation": 0.0, "lead1_transfer": 0.0, "lead2_transfer": 0.0, "overtime_intercept": 0.0, "overtime_logit_coef": 0.5}
)
FITTED = goal_model.validate_goal_model(dict(goal_model.DEFAULT_GOAL_MODEL))


class ScoreGridTests(unittest.TestCase):
    """Cover the corrected Poisson score grid."""

    def test_plain_grid_is_independent_poisson(self):
        """Without corrections the grid is the product of two Poisson distributions."""
        grid = goal_model.score_grid(3.0, 2.5, PLAIN)[0]
        expected = (math.exp(-3.0) * 3.0 ** 2 / 2) * (math.exp(-2.5) * 2.5 ** 1)

        self.assertAlmostEqual(float(grid.sum()), 1.0)
        self.assertAlmostEqual(float(grid[2, 1]), expected / float(goal_model.score_grid(3.0, 2.5, PLAIN)[0].sum()), places=6)

    def test_tie_inflation_adds_draws(self):
        """Inflating the diagonal raises the regulation draw probability."""
        inflated = dict(PLAIN, tie_inflation=0.4)
        plain_draw = goal_model.market_probabilities(goal_model.score_grid(3.0, 3.0, PLAIN))["draw"][0]
        inflated_draw = goal_model.market_probabilities(goal_model.score_grid(3.0, 3.0, inflated))["draw"][0]

        self.assertGreater(inflated_draw, plain_draw + 0.03)

    def test_empty_net_transfers_widen_margins_but_keep_winners(self):
        """Moving one-goal leads to two-goal leads leaves who wins untouched."""
        transfers = dict(PLAIN, lead1_transfer=0.3, lead2_transfer=0.4)
        plain = goal_model.market_probabilities(goal_model.score_grid(3.2, 2.8, PLAIN))
        shifted = goal_model.market_probabilities(goal_model.score_grid(3.2, 2.8, transfers))

        self.assertAlmostEqual(float(plain["home_regulation"][0]), float(shifted["home_regulation"][0]), places=4)
        self.assertAlmostEqual(float(plain["draw"][0]), float(shifted["draw"][0]), places=6)
        self.assertGreater(float(shifted["home_minus_1_5"][0]), float(plain["home_minus_1_5"][0]))
        self.assertGreater(float(shifted["expected_total"][0]), float(plain["expected_total"][0]))


class PricingTests(unittest.TestCase):
    """Cover the moneyline inversion and market consistency."""

    def test_solved_split_reproduces_the_win_probability(self):
        """Regulation home win plus the tied share won in OT equals the win model's number."""
        targets = np.array([0.3, 0.5, 0.64, 0.8])
        priced = goal_model.price_games(targets, 3.05, FITTED)
        tie_break = goal_model.overtime_home_probability(targets, FITTED)

        implied = priced["home_regulation"] + priced["draw"] * tie_break
        np.testing.assert_allclose(implied, targets, atol=1e-6)

    def test_markets_are_mutually_consistent(self):
        """1X2 sums to one; a -1.5 win is rarer than any regulation win; favourites cover more often."""
        priced = goal_model.price_games(np.array([0.62]), 3.05, FITTED)

        self.assertAlmostEqual(float(priced["home_regulation"][0] + priced["draw"][0] + priced["away_regulation"][0]), 1.0)
        self.assertLess(float(priced["home_minus_1_5"][0]), float(priced["home_regulation"][0]))
        self.assertGreater(float(priced["home_minus_1_5"][0]), float(priced["away_minus_1_5"][0]))
        self.assertGreater(float(priced["home_share"][0]), 0.5)

    def test_game_markets_needs_a_model_and_falls_back_on_scoring_level(self):
        """No goal model, no markets; a missing scoring level uses the model default."""
        self.assertIsNone(goal_model.game_markets(0.55, 3.0, None))
        markets = goal_model.game_markets(0.55, None, FITTED)

        self.assertAlmostEqual(sum(markets["regulation"].values()), 1.0, places=9)
        self.assertLess(markets["puck_line"]["home_minus_1_5"], markets["regulation"]["home"])

    def test_overtime_probability_follows_the_logit(self):
        """An even game's tie-break chance is the sigmoid of the intercept."""
        model = dict(PLAIN, overtime_intercept=0.2, overtime_logit_coef=0.5)
        self.assertAlmostEqual(float(goal_model.overtime_home_probability(0.5, model)[0]), 1 / (1 + math.exp(-0.2)))

    def test_invalid_blocks_are_rejected(self):
        """Out-of-range shape parameters disable the markets rather than mispricing."""
        self.assertIsNone(goal_model.validate_goal_model({"lead1_transfer": 1.5}))
        self.assertIsNone(goal_model.validate_goal_model({"rate_scale": "x"}))
        self.assertIsNone(goal_model.validate_goal_model(None))


class ScoringDataTests(unittest.TestCase):
    """Cover regulation goals and the league scoring level."""

    def _games(self):
        return pd.DataFrame(
            {
                "SeasonYear": [2024, 2024, 2025, 2025, 2025, 2025],
                "GameTypeId": [2, 2, 2, 2, 2, 3],
                "GameId": [1, 2, 3, 4, 5, 6],
                "GameDate": ["2024-10-01", "2024-10-02", "2025-10-01", "2025-10-01", "2025-10-03", "2026-04-20"],
                "HomeGoals": [4.0, 3.0, 5.0, 2.0, 3.0, 2.0],
                "AwayGoals": [2.0, 2.0, 5.0, 3.0, 2.0, 1.0],
                "ResultType": ["REG", "OT", "SO", "OT", "REG", "REG"],
                "HomeWin": [1, 1, 1, 0, 1, 1],
            }
        )

    def test_regulation_goals_remove_the_overtime_winner(self):
        """An OT goal comes off the winner; shootouts never counted in goals."""
        home, away = goal_model.regulation_goals(self._games())

        self.assertEqual(list(home), [4.0, 2.0, 5.0, 2.0, 3.0, 2.0])
        self.assertEqual(list(away), [2.0, 2.0, 5.0, 2.0, 2.0, 1.0])

    def test_environment_blends_last_season_with_games_before_each_date(self):
        """Opening night uses last season; later dates add games already played; playoffs use season end."""
        games = self._games()
        environment = goal_model.scoring_environment(games, prior_team_games=4.0)
        last_season = (6.0 + 4.0) / 4.0

        self.assertAlmostEqual(float(environment.iloc[2]), last_season)
        self.assertAlmostEqual(float(environment.iloc[3]), last_season)
        self.assertAlmostEqual(float(environment.iloc[4]), (4.0 * last_season + 10.0 + 4.0) / (4.0 + 4.0))
        self.assertAlmostEqual(float(environment.iloc[5]), (4.0 * last_season + 10.0 + 4.0 + 5.0) / (4.0 + 6.0))

    def test_current_environment_before_opening_night_is_last_season(self):
        """With no games yet this season the level is last season's."""
        games = self._games()
        self.assertAlmostEqual(goal_model.current_scoring_environment(games[games["SeasonYear"] == 2024], 2025, 4.0), 2.5)
        self.assertIsNone(goal_model.current_scoring_environment(pd.DataFrame(), 2025))


if __name__ == "__main__":
    unittest.main()
