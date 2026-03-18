"""nhl.baselines — Historical and team 75th-percentile baseline builders.

Historical baselines answer a simple question: what does a strong historical NHL
player or goalie usually look like at each age? The app keeps them permanently
cached because the parquet source is effectively static during runtime.

Imports from project:
    nhl.constants — TEAM_METRICS
    nhl.data_loaders — load_historical_data(), load_all_team_seasons()
"""

import pandas as pd
import streamlit as st

from nhl.constants import TEAM_METRICS
from nhl.data_loaders import load_all_team_seasons, load_historical_data


def _estimate_recent_decline_ratio(base: pd.DataFrame, col: str, anchor_age: int) -> float:
    """Estimate a gentle late-career decline ratio from recent baseline ages.

    Args:
        base: Baseline DataFrame indexed by age.
        col: Metric column to inspect.
        anchor_age: Latest trusted age to use as the tail anchor.

    Returns:
        Multiplicative decline ratio clipped to a realistic late-career range.
    """
    ratios = []
    for age in range(anchor_age - 2, anchor_age + 1):
        if age not in base.index or (age - 1) not in base.index:
            continue
        prev = base.loc[age - 1, col]
        curr = base.loc[age, col]
        if pd.isna(prev) or pd.isna(curr) or prev <= 0 or curr <= 0:
            continue
        ratios.append(float(curr) / float(prev))

    if not ratios:
        return 0.98

    return min(0.995, max(0.97, float(pd.Series(ratios).median())))


def _apply_survivorship_decay(
    base: pd.DataFrame,
    skip_cols: set[str] | None = None,
) -> pd.DataFrame:
    """Apply the classic post-prime survivorship guard.

    Args:
        base: Raw percentile baseline indexed by age.
        skip_cols: Optional metric columns to leave untouched by the multiplicative cap.

    Returns:
        Smoothed baseline DataFrame with post-31 rises capped by the legacy
        ``prev * 0.92`` rule.
    """
    if base.empty:
        return base

    skip_cols = skip_cols or set()
    base = base.sort_index().copy()
    base_smoothed = base.rolling(window=3, min_periods=1, center=True).mean()
    for col in base.columns:
        base[col] = base_smoothed[col]
        if col in skip_cols:
            continue

        for age in range(32, 42):
            if age not in base.index or (age - 1) not in base.index:
                continue

            prev = base.loc[age - 1, col]
            curr = base.loc[age, col]
            if pd.isna(prev) or pd.isna(curr) or prev <= 0:
                continue

            max_allowed = float(prev) * 0.92
            if curr > max_allowed:
                base.loc[age, col] = max_allowed

    return base


def _shape_skater_late_tail(base: pd.DataFrame, age_counts: pd.Series) -> pd.DataFrame:
    """Repair synthetic late skater tails using recent trusted ages plus sparse-data blend.

    Args:
        base: Skater baseline DataFrame indexed by age.
        age_counts: Series mapping age to qualifying skater-season count.

    Returns:
        Baseline DataFrame with ages 36-41 gently shaped when survivorship rules
        or sparse ages create an overly mechanical tail.
    """
    base = base.reindex(base.index.union(range(36, 42)))

    for col in base.columns:
        anchor_age = next(
            (
                age for age in (35, 34, 33)
                if age in base.index and not pd.isna(base.loc[age, col]) and base.loc[age, col] > 0
            ),
            None,
        )
        if anchor_age is None or col == '+/-':
            continue

        tail_ratio = _estimate_recent_decline_ratio(base, col, anchor_age)
        est_val = float(base.loc[anchor_age, col])

        for age in range(anchor_age + 1, 42):
            year_offset = age - anchor_age - 1
            decline_ratio = max(0.88, tail_ratio - 0.012 * year_offset)
            est_val *= decline_ratio

            curr = base.loc[age, col] if age in base.index else float('nan')
            prev = base.loc[age - 1, col] if (age - 1) in base.index else float('nan')
            count = int(age_counts.get(age, 0))
            data_weight = min(1.0, count / 600.0)

            synthetic_chain = (
                pd.notna(curr)
                and pd.notna(prev)
                and prev > 0
                and abs((float(curr) / float(prev)) - 0.92) <= 0.01
            )
            if synthetic_chain:
                data_weight = min(data_weight, 0.25)

            shaped = est_val if pd.isna(curr) else float(curr) * data_weight + est_val * (1 - data_weight)
            if pd.notna(prev) and prev > 0:
                shaped = min(shaped, float(prev) * 0.995)
                shaped = max(shaped, float(prev) * 0.85)
            base.loc[age, col] = shaped

    return base


def _shape_goalie_save_tail(goalie_base: pd.DataFrame, age_counts: pd.Series) -> pd.DataFrame:
    """Blend sparse late-age goalie save percentages with a curved decline tail.

    Args:
        goalie_base: Goalie baseline DataFrame indexed by age.
        age_counts: Series mapping age to qualifying goalie-season count.

    Returns:
        Baseline DataFrame with `Save %` reshaped into a gentle, non-linear
        late-career decline instead of a ruler-straight or upward sparse tail.
    """
    if 'Save %' not in goalie_base.columns or 34 not in goalie_base.index:
        return goalie_base

    goalie_base = goalie_base.reindex(goalie_base.index.union(range(35, 42)))
    start_age = 34
    start_val = float(goalie_base.loc[start_age, 'Save %'])
    prev_val = start_val

    recent_diffs = []
    for age in range(31, 35):
        if age not in goalie_base.index or (age - 1) not in goalie_base.index:
            continue
        prev = goalie_base.loc[age - 1, 'Save %']
        curr = goalie_base.loc[age, 'Save %']
        if pd.isna(prev) or pd.isna(curr):
            continue
        recent_diffs.append(float(curr) - float(prev))

    base_step = float(pd.Series(recent_diffs).median()) if recent_diffs else -0.14
    base_step = min(-0.06, max(-0.30, base_step))

    for age in range(start_age + 1, 42):
        year_offset = age - start_age
        curve_step = base_step * (1.0 + max(0, year_offset - 1) * 0.12)
        est_val = prev_val + curve_step
        curr = goalie_base.loc[age, 'Save %'] if age in goalie_base.index else float('nan')
        count = int(age_counts.get(age, 0))
        data_weight = min(1.0, count / 60.0)
        shaped = est_val if pd.isna(curr) else float(curr) * data_weight + est_val * (1 - data_weight)
        min_drop = min(0.18, 0.05 + 0.012 * max(0, year_offset - 1) + 0.03 * (1 - data_weight))
        shaped = min(shaped, prev_val - min_drop)
        shaped = max(shaped, prev_val - 0.45)
        goalie_base.loc[age, 'Save %'] = shaped
        prev_val = shaped

    return goalie_base


def _build_role_baseline(
    df: pd.DataFrame,
    stat_category: str,
) -> pd.DataFrame:
    """Build a single role-specific historical baseline DataFrame.

    Args:
        df: Historical seasons already filtered to one role bucket.
        stat_category: ``Skater`` or ``Goalie`` for role-specific shaping rules.

    Returns:
        Baseline DataFrame indexed by age. Empty when there is not enough data.
    """
    if df.empty:
        return pd.DataFrame()

    age_counts = df.groupby('Age').size()
    base = df.groupby('Age').quantile(0.75, numeric_only=True).sort_index()

    if stat_category == 'Goalie':
        if 'Save %' in base.columns:
            base['Save %'] = base['Save %'].clip(lower=80.0, upper=95.0)
        base = _apply_survivorship_decay(base, skip_cols={'Save %', 'GAA'})

        if (
            'Save %' in base.columns
            and all(age in base.index for age in (28, 29, 30, 31, 32))
        ):
            a28 = float(base.loc[28, 'Save %'])
            a32 = float(base.loc[32, 'Save %'])
            mids = [float(base.loc[age, 'Save %']) for age in (29, 30, 31)]
            if any(val > max(a28, a32) + 1.5 for val in mids):
                for age in (29, 30, 31):
                    t = (age - 28) / 4.0
                    base.loc[age, 'Save %'] = a28 + (a32 - a28) * t

        return _shape_goalie_save_tail(base, age_counts)

    base = _apply_survivorship_decay(base)
    return _shape_skater_late_tail(base, age_counts)


def _build_historical_baselines_from_df(df: pd.DataFrame) -> dict:
    """Build aggregate skater and goalie 75th-percentile age curves.

    Args:
        df: Historical seasons DataFrame from ``load_historical_data()``.

    Returns:
        Dict containing baseline DataFrames keyed by ``Skater`` and ``Goalie``.
    """
    if df.empty:
        return {}

    skater_df = df[(df['Position'] != 'G') & (df['GP'] >= 40)].copy()
    goalie_df = df[(df['Position'] == 'G') & (df['GP'] >= 20)].copy()

    return {
        'Skater': _build_role_baseline(skater_df, 'Skater'),
        'Goalie': _build_role_baseline(goalie_df, 'Goalie'),
    }


def build_historical_baselines(df: pd.DataFrame) -> dict:
    """Build baselines from an already-loaded historical DataFrame.

    Args:
        df: Historical seasons DataFrame from ``load_historical_data()``.

    Returns:
        Dict containing baseline DataFrames keyed by ``Skater`` and ``Goalie``.
    """
    return _build_historical_baselines_from_df(df)


@st.cache_data
def get_historical_baselines() -> dict:
    """Return cached historical baselines without hashing a DataFrame argument.

    Args:
        None.

    Returns:
        Dict containing baseline DataFrames keyed by ``Skater`` and ``Goalie``.
    """
    return _build_historical_baselines_from_df(load_historical_data())


def _build_team_baselines_from_df(all_team_df: pd.DataFrame) -> dict:
    """Compute 75th-percentile team baseline per SeasonYear for all TEAM_METRICS.

    Uses regular-season rows only (gameTypeId == 2).  Falls back to all rows if
    the gameTypeId column is absent.

    Args:
        all_team_df: Team-season DataFrame from load_all_team_seasons().

    Returns:
        Dict mapping int season_year to a nested dict of {metric: float_value}.
        Returns {} if all_team_df is empty.
    """
    if all_team_df.empty:
        return {}
    # Filter to regular season; fall back to all rows if column absent
    if "gameTypeId" in all_team_df.columns:
        reg = all_team_df[all_team_df["gameTypeId"] == 2].copy()
    else:
        reg = all_team_df.copy()
    if reg.empty:
        reg = all_team_df.copy()

    result = {}

    for sy, grp in reg.groupby("SeasonYear"):
        entry = {}
        for m in TEAM_METRICS:
            if m in grp.columns:
                vals = grp[m].dropna()
                entry[m] = float(vals.quantile(0.75)) if not vals.empty else None
        result[int(sy)] = entry
    return result


def build_team_baselines(all_team_df: pd.DataFrame) -> dict:
    """Build team baselines from an already-loaded team-season DataFrame.

    Args:
        all_team_df: Team-season DataFrame from ``load_all_team_seasons()``.

    Returns:
        Dict mapping int season_year to a nested dict of {metric: float_value}.
    """
    return _build_team_baselines_from_df(all_team_df)


@st.cache_data
def get_team_baselines() -> dict:
    """Return cached team baselines without hashing a DataFrame argument.

    Args:
        None.

    Returns:
        Dict mapping int season_year to a nested dict of {metric: float_value}.
    """
    return _build_team_baselines_from_df(load_all_team_seasons())
