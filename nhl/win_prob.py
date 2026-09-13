"""Pregame win-probability model contract: feature names, artifact validation, scoring.

Features are built in ``nhl.team_ratings``. This module only knows how to validate the
exported ``win_prob_weights.json`` artifact and turn feature values into probabilities,
so runtime scoring stays numpy-only (scikit-learn lives in ``train_win_prob.py``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nhl.goal_model import validate_goal_model
from nhl.team_ratings import (
    DEFAULT_RATING_PARAMS,
    DIFF_FEATURE_ATTRIBUTES,
    FLAG_FEATURE_SIDES,
    MODEL_FEATURES,
    resolve_rating_params,
)

WIN_PROB_FEATURE_ORDER = list(MODEL_FEATURES)
"""Every feature the runtime can build. The artifact's ``feature_order`` is a subset."""

WIN_PROB_FEATURE_LABELS = {
    "elo_diff": "team rating",
    "goal_diff_shrunk_diff": "goal differential",
    "sat_share_shrunk_diff": "5v5 shot-attempt share",
    "sog_share_shrunk_diff": "shots-on-goal share",
    "home_back_to_back": "back-to-back",
    "away_back_to_back": "back-to-back",
}
WIN_PROB_MODEL_VERSION = 2

DEFAULT_OVERTIME_MODEL = {
    "intercept": -1.1,
    "abs_logit_coef": -0.5,
    "shootout_share": 0.42,
}
"""P(tied after 60) = sigmoid(intercept + abs_logit_coef * |home-win logit|)."""

DEFAULT_SIMULATION_PARAMS = {
    "strength_sd_preseason": 0.25,
    "strength_sd_late": 0.1,
}
"""Per-simulation team strength noise on the logit scale, interpolated by season progress."""


def sigmoid(value: float) -> float:
    """Return a numerically stable sigmoid."""
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _float_mapping(payload: object, defaults: dict[str, float]) -> dict[str, float]:
    """Merge a JSON sub-object over defaults, keeping only finite numeric values."""
    resolved = dict(defaults)
    if not isinstance(payload, dict):
        return resolved
    for key in defaults:
        try:
            value = float(payload.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            resolved[key] = value
    return resolved


def validate_model_artifact(payload: object) -> dict:
    """Validate and normalize the exported JSON model artifact.

    Args:
        payload: Parsed ``win_prob_weights.json`` content.

    Returns:
        A normalized artifact dict with every optional block filled from defaults.

    Raises:
        ValueError: The payload is not a version-2 artifact with a consistent shape.
    """
    if not isinstance(payload, dict):
        raise ValueError("Win-probability artifact must be a dict.")

    model_version = int(payload.get("model_version", 0) or 0)
    if model_version != WIN_PROB_MODEL_VERSION:
        raise ValueError(
            f"Win-probability artifact version {model_version} is not supported; "
            f"retrain with train_win_prob.py (expects version {WIN_PROB_MODEL_VERSION})."
        )

    feature_order = [str(name) for name in (payload.get("feature_order") or [])]
    if not feature_order:
        raise ValueError("Win-probability artifact has no features.")
    unknown = [name for name in feature_order if name not in WIN_PROB_FEATURE_ORDER]
    if unknown or len(set(feature_order)) != len(feature_order):
        raise ValueError("Win-probability artifact feature order does not match runtime features.")

    coefficients = [float(value) for value in payload.get("coefficients", [])]
    scaler_mean = [float(value) for value in payload.get("scaler_mean", [])]
    scaler_scale = [float(value) for value in payload.get("scaler_scale", [])]
    if not (len(coefficients) == len(scaler_mean) == len(scaler_scale) == len(feature_order)):
        raise ValueError("Win-probability artifact coefficient shape is invalid.")

    overtime_model = _float_mapping(payload.get("overtime_model"), DEFAULT_OVERTIME_MODEL)
    overtime_model["shootout_share"] = min(max(overtime_model["shootout_share"], 0.0), 1.0)
    simulation = _float_mapping(payload.get("simulation"), DEFAULT_SIMULATION_PARAMS)
    for key in simulation:
        simulation[key] = max(simulation[key], 0.0)

    return {
        "model_version": model_version,
        "feature_order": feature_order,
        "coefficients": coefficients,
        "intercept": float(payload.get("intercept", 0.0)),
        "scaler_mean": scaler_mean,
        "scaler_scale": [value if value != 0 else 1.0 for value in scaler_scale],
        "selected_c": float(payload.get("selected_c", 0.0) or 0.0),
        "rating_params": resolve_rating_params(payload.get("rating_params") or DEFAULT_RATING_PARAMS),
        "overtime_model": overtime_model,
        "simulation": simulation,
        "goal_model": validate_goal_model(payload.get("goal_model")),
        "training_seasons": [int(season) for season in payload.get("training_seasons", [])],
        "generated_at_utc": str(payload.get("generated_at_utc", "") or ""),
        "validation_metrics": payload.get("validation_metrics", {}),
    }


def score_home_win_probability(
    feature_values: dict[str, float],
    artifact: dict,
) -> dict[str, object]:
    """Score one home-win probability from raw feature values and exported weights.

    Args:
        feature_values: Feature name -> raw value; must cover the artifact's features.
        artifact: Model artifact (validated here).

    Returns:
        ``home_win_prob``, ``logit``, standardized values, per-feature logit
        contributions and the raw values that were scored.
    """
    validated_artifact = validate_model_artifact(artifact)
    raw_vector = np.array(
        [float(feature_values[feature_name]) for feature_name in validated_artifact["feature_order"]],
        dtype=float,
    )
    mean_vector = np.array(validated_artifact["scaler_mean"], dtype=float)
    scale_vector = np.array(validated_artifact["scaler_scale"], dtype=float)
    standardized_vector = (raw_vector - mean_vector) / scale_vector
    coefficient_vector = np.array(validated_artifact["coefficients"], dtype=float)
    logit = float(validated_artifact["intercept"] + np.dot(coefficient_vector, standardized_vector))
    probability = sigmoid(logit)
    contributions = {
        feature_name: float(value)
        for feature_name, value in zip(
            validated_artifact["feature_order"],
            coefficient_vector * standardized_vector,
        )
    }
    return {
        "home_win_prob": probability,
        "logit": logit,
        "standardized_values": {
            feature_name: float(value)
            for feature_name, value in zip(validated_artifact["feature_order"], standardized_vector)
        },
        "contributions": contributions,
        "feature_values": {feature_name: float(feature_values[feature_name]) for feature_name in validated_artifact["feature_order"]},
    }


def score_home_win_logits(features: pd.DataFrame, artifact: dict) -> np.ndarray:
    """Score many games at once.

    Args:
        features: Frame holding at least the artifact's feature columns.
        artifact: Validated model artifact.

    Returns:
        Home-win logits, one per row.
    """
    matrix = features[artifact["feature_order"]].to_numpy(dtype=float)
    standardized = (matrix - np.array(artifact["scaler_mean"], dtype=float)) / np.array(artifact["scaler_scale"], dtype=float)
    return artifact["intercept"] + standardized @ np.array(artifact["coefficients"], dtype=float)


def decompose_linear_model(artifact: dict) -> tuple[float, dict[str, float], dict[str, float]]:
    """Rewrite the logistic model as a constant plus per-team strength plus schedule flags.

    Every difference feature is ``attribute[home] - attribute[away]``, so the logit is
    ``constant + strength[home] - strength[away] + flag terms`` with
    ``strength[team] = sum(weight * attribute[team])``. The season simulator uses this
    to score any pairing without rebuilding feature rows.

    Args:
        artifact: Validated model artifact.

    Returns:
        ``(constant, attribute_weights, flag_weights)``.
    """
    constant = float(artifact["intercept"])
    attribute_weights: dict[str, float] = {}
    flag_weights: dict[str, float] = {}
    for name, coefficient, mean, scale in zip(
        artifact["feature_order"],
        artifact["coefficients"],
        artifact["scaler_mean"],
        artifact["scaler_scale"],
    ):
        weight = float(coefficient) / float(scale)
        constant -= weight * float(mean)
        if name in DIFF_FEATURE_ATTRIBUTES:
            attribute = DIFF_FEATURE_ATTRIBUTES[name]
            attribute_weights[attribute] = attribute_weights.get(attribute, 0.0) + weight
        elif name in FLAG_FEATURE_SIDES:
            flag_weights[name] = weight
    return constant, attribute_weights, flag_weights


def overtime_probability(home_win_logit: float | np.ndarray, artifact: dict) -> float | np.ndarray:
    """Return the chance a game is still tied after 60 minutes.

    Close matchups go to overtime more often, so the model keys off the absolute
    home-win logit.

    Args:
        home_win_logit: Scalar or array of home-win logits.
        artifact: Validated model artifact.

    Returns:
        Probability (same shape as the input).
    """
    overtime_model = artifact["overtime_model"]
    linear = overtime_model["intercept"] + overtime_model["abs_logit_coef"] * np.abs(home_win_logit)
    return 1.0 / (1.0 + np.exp(-linear))


def get_top_feature_driver(
    scored_probability: dict[str, object],
) -> tuple[str, float]:
    """Return the strongest single feature contribution from one scored matchup."""
    contributions = scored_probability.get("contributions", {})
    if not isinstance(contributions, dict) or not contributions:
        return ("", 0.0)

    feature_name = max(
        contributions,
        key=lambda current_feature: abs(float(contributions.get(current_feature, 0.0))),
    )
    return feature_name, float(contributions.get(feature_name, 0.0))
