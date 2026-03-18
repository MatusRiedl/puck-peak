"""
nhl.era — Era-adjustment functions for NHL scoring normalization.

Scoring levels have changed dramatically across NHL history due to rule changes,
expansion waves, equipment evolution, and goaltending technique shifts.  These
functions normalise raw stats to a common 2018+ baseline so that players from
different eras can be compared and KNN clones remain fair.

No Streamlit dependency — all functions are pure math on floats or DataFrames.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Private data tables (not imported by callers)
# ---------------------------------------------------------------------------

_GOALIE_ERA_LEAGUE_AVG_SV: dict[int, float] = {
    1967: 0.878,   # Original Six — stand-up goalies, no masks
    1979: 0.871,   # Expansion + WHA — offense-heavy, basic equipment
    1992: 0.873,   # Gretzky peak scoring — butterfly just emerging (Patrick Roy 1984+)
    1996: 0.890,   # Transitional — butterfly spreading, larger pads, obstruction era
    2004: 0.902,   # Dead puck — full butterfly, neutral-zone trap defense
    2012: 0.908,   # Post-lockout — interference crackdown, technique refinement
    2017: 0.912,   # Analytics era — systematic goalie training, peak equipment size
}
"""Historical league-average Save% by era threshold (ordered ascending by year)."""

_GOALIE_SV_MODERN_BASELINE: float = 0.9110
"""2018+ league-average Save% (equipment size limits enforced); the normalization target."""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_era_multiplier(year: int) -> float:
    """Return the scoring era multiplier for a given season year.

    Anchored to 2018+ as baseline (~3.05 GF/GP per team).
    Multiplier = baseline_GF / era_GF.  Values < 1.0 deflate high-scoring eras;
    values > 1.0 inflate low-scoring eras.

    Applied independently to Points, Goals, and Assists for skaters.

    Args:
        year: Season start year (e.g. 2019 for the 2019-20 season).

    Returns:
        Multiplier float between 0.80 and 1.15.
    """
    if year <= 1967:             return 1.00  # Original Six (~2.85 GF/GP — close to modern)
    if 1968 <= year <= 1979:     return 0.89  # Expansion + WHA era (~3.40 GF/GP)
    if 1980 <= year <= 1992:     return 0.80  # Gretzky/peak scoring era (~3.85 GF/GP)
    if 1993 <= year <= 1996:     return 0.90  # Transitional decline (~3.35 GF/GP)
    if 1997 <= year <= 2004:     return 1.15  # Dead puck era (~2.63 GF/GP)
    if 2005 <= year <= 2012:     return 1.06  # Post-lockout settling (~2.87 GF/GP)
    if 2013 <= year <= 2017:     return 1.12  # Analytics/trap era (~2.72 GF/GP)
    return 1.00                               # 2018+ — modern baseline


def get_goalie_era_sv_offset(year: int) -> float:
    """Return the additive Save% offset (0-1 scale) for goalie era normalisation.

    Preserves each goalie's deviation from their era's league average and expresses
    it in modern-era (2018+) terms.  Add the result to a raw Save% value (0-1 scale)
    to shift it to the modern baseline.

    Example: .890 in 1985 (era avg .873) → .890 + (.911 - .873) = .928 modern equivalent.

    Args:
        year: Season start year.

    Returns:
        Additive offset in 0-1 scale.  0.0 for seasons already at the modern baseline.
    """
    for threshold, avg in _GOALIE_ERA_LEAGUE_AVG_SV.items():
        if year <= threshold:
            return _GOALIE_SV_MODERN_BASELINE - avg
    return 0.0  # 2018+ — modern baseline, no adjustment needed


def metric_is_era_adjusted(
    metric: str,
    stat_category: str,
    do_era: bool,
    team_mode: bool = False,
) -> bool:
    """Return whether the active metric is actually era-adjusted."""
    if team_mode or not do_era:
        return False

    skater_era_metrics = {"Points", "Goals", "Assists", "PPG", "SH%"}
    goalie_era_metrics = {"Save %", "GAA", "Shutouts"}

    if stat_category == "Skater":
        return metric in skater_era_metrics
    if stat_category == "Goalie":
        return metric in goalie_era_metrics
    return False


def apply_era_to_hist(
    df: pd.DataFrame,
    do_era: bool,
    is_goalie: bool = False,
) -> pd.DataFrame:
    """Apply era adjustment to a historical parquet DataFrame.

    Returns a copy of df with era-adjusted stats matching the live player's
    adjustment state.  Used to keep KNN clone matching consistent with the
    displayed era-adjusted player curve.

    Skaters: Points, Goals, Assists multiplied by era multiplier.
    Goalies: GAA multiplied by era mult; Save% shifted by additive offset;
             Shutouts divided by era mult (inverse — harder to record in high-scoring eras).

    Args:
        df:        Historical seasons DataFrame with a 'SeasonYear' column.
        do_era:    If False, returns df unchanged.
        is_goalie: Set True to apply goalie-specific adjustments instead of skater ones.

    Returns:
        Era-adjusted DataFrame (copy).
    """
    if not do_era:
        return df
    d = df.copy()
    mult = d['SeasonYear'].apply(get_era_multiplier)
    if not is_goalie:
        for col in ['Points', 'Goals', 'Assists']:
            if col in d.columns:
                d[col] = d[col] * mult
    else:
        if 'GAA' in d.columns:
            d['GAA'] = (d['GAA'] * mult).clip(lower=0)
        if 'Save %' in d.columns:
            sv_offsets = d['SeasonYear'].apply(get_goalie_era_sv_offset)
            d['Save %'] = (d['Save %'] + sv_offsets * 100).clip(0, 100)
        if 'Shutouts' in d.columns:
            d['Shutouts'] = d['Shutouts'] / mult
    return d
