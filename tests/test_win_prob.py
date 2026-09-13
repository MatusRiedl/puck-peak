import math
import unittest

import numpy as np
import pandas as pd

import nhl.win_prob as win_prob


def _artifact(**overrides):
    """Return a small valid version-2 artifact."""
    payload = {
        "model_version": 2,
        "feature_order": ["elo_diff", "sog_share_shrunk_diff", "away_back_to_back"],
        "coefficients": [0.4, 0.2, 0.1],
        "intercept": 0.15,
        "scaler_mean": [2.0, 0.01, 0.2],
        "scaler_scale": [50.0, 0.04, 0.4],
        "selected_c": 0.3,
    }
    payload.update(overrides)
    return payload


class ArtifactValidationTests(unittest.TestCase):
    """Cover the artifact contract between the trainer and the runtime."""

    def test_version_one_artifact_is_rejected(self):
        """The old standings-form model must not be scored with the new features."""
        with self.assertRaises(ValueError):
            win_prob.validate_model_artifact(
                {
                    "model_version": 1,
                    "feature_order": ["point_pct_to_date"],
                    "coefficients": [1.0],
                    "scaler_mean": [0.0],
                    "scaler_scale": [1.0],
                }
            )

    def test_unknown_or_mismatched_features_are_rejected(self):
        """A feature the runtime cannot build, or a shape mismatch, fails loudly."""
        with self.assertRaises(ValueError):
            win_prob.validate_model_artifact(_artifact(feature_order=["elo_diff", "power_play_pct", "away_back_to_back"]))
        with self.assertRaises(ValueError):
            win_prob.validate_model_artifact(_artifact(coefficients=[0.4, 0.2]))

    def test_optional_blocks_are_filled_from_defaults(self):
        """Missing rating, overtime and simulation blocks fall back to defaults."""
        validated = win_prob.validate_model_artifact(_artifact())

        self.assertEqual(validated["rating_params"]["elo_k"], win_prob.DEFAULT_RATING_PARAMS["elo_k"])
        self.assertEqual(validated["overtime_model"], win_prob.DEFAULT_OVERTIME_MODEL)
        self.assertEqual(validated["simulation"], win_prob.DEFAULT_SIMULATION_PARAMS)
        self.assertIsNone(validated["goal_model"])

    def test_goal_model_block_is_validated_and_kept(self):
        """A valid goal model survives validation; a broken one switches the markets off."""
        kept = win_prob.validate_model_artifact(_artifact(goal_model={"tie_inflation": 0.48, "lead1_transfer": 0.31}))
        broken = win_prob.validate_model_artifact(_artifact(goal_model={"lead1_transfer": 3.0}))

        self.assertAlmostEqual(kept["goal_model"]["tie_inflation"], 0.48)
        self.assertIsNone(broken["goal_model"])


class ScoringTests(unittest.TestCase):
    """Cover scalar, vectorized and decomposed scoring."""

    def test_scalar_scoring_matches_exported_weights(self):
        """Runtime scoring honours exported means, scales, weights and intercept exactly."""
        features = {"elo_diff": 52.0, "sog_share_shrunk_diff": 0.05, "away_back_to_back": 1.0}
        scored = win_prob.score_home_win_probability(features, _artifact())

        logit = 0.15 + 0.4 * (52.0 - 2.0) / 50.0 + 0.2 * (0.05 - 0.01) / 0.04 + 0.1 * (1.0 - 0.2) / 0.4
        self.assertAlmostEqual(scored["logit"], logit)
        self.assertAlmostEqual(scored["home_win_prob"], 1.0 / (1.0 + math.exp(-logit)))
        self.assertAlmostEqual(scored["contributions"]["away_back_to_back"], 0.1 * 2.0)

    def test_vectorized_logits_match_scalar_scoring(self):
        """The simulator's batch path and the card's scalar path agree."""
        artifact = win_prob.validate_model_artifact(_artifact())
        frame = pd.DataFrame(
            {
                "elo_diff": [52.0, -30.0],
                "sog_share_shrunk_diff": [0.05, -0.02],
                "away_back_to_back": [1.0, 0.0],
                "home_back_to_back": [0.0, 1.0],
            }
        )
        batch = win_prob.score_home_win_logits(frame, artifact)
        for position in range(len(frame)):
            scalar = win_prob.score_home_win_probability(frame.iloc[position].to_dict(), artifact)["logit"]
            self.assertAlmostEqual(float(batch[position]), scalar)

    def test_decomposition_reproduces_the_logit_from_team_strengths(self):
        """constant + strength[home] - strength[away] + flags equals the full logit."""
        artifact = win_prob.validate_model_artifact(_artifact())
        home = {"elo": 1560.0, "sog_share_shrunk": 0.53}
        away = {"elo": 1508.0, "sog_share_shrunk": 0.48}
        constant, attribute_weights, flag_weights = win_prob.decompose_linear_model(artifact)

        def strength(team):
            return sum(weight * team[attribute] for attribute, weight in attribute_weights.items())

        decomposed = constant + strength(home) - strength(away) + flag_weights["away_back_to_back"] * 1.0
        features = {"elo_diff": 52.0, "sog_share_shrunk_diff": 0.05, "away_back_to_back": 1.0}
        self.assertAlmostEqual(decomposed, win_prob.score_home_win_probability(features, artifact)["logit"])

    def test_overtime_is_likelier_in_close_games(self):
        """A negative slope on |logit| means mismatches reach overtime less often."""
        artifact = win_prob.validate_model_artifact(_artifact(overtime_model={"intercept": -1.1, "abs_logit_coef": -0.3, "shootout_share": 0.33}))
        probabilities = win_prob.overtime_probability(np.array([0.0, 1.5, -1.5]), artifact)

        self.assertAlmostEqual(float(probabilities[0]), 1.0 / (1.0 + math.exp(1.1)))
        self.assertGreater(float(probabilities[0]), float(probabilities[1]))
        self.assertAlmostEqual(float(probabilities[1]), float(probabilities[2]))

    def test_top_feature_driver_picks_the_largest_absolute_contribution(self):
        """The card label names the strongest single driver."""
        name, value = win_prob.get_top_feature_driver({"contributions": {"elo_diff": 0.2, "away_back_to_back": -0.35}})

        self.assertEqual(name, "away_back_to_back")
        self.assertAlmostEqual(value, -0.35)


if __name__ == "__main__":
    unittest.main()
