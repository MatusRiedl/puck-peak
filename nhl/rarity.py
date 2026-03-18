"""Age-rarity helpers for player Season Snapshot dialogs.

This module keeps the rarity feature separate from the main player pipeline so
projection and baseline code can continue to treat the historical parquet as
their old backbone while the dialog builds a richer one-off interpretation
layer. The public entry point returns a serializable payload for the dialog:
percentile, rank, sample size, optional skater role split, and a compact top-5
leaderboard from the same historical comparison pool.
"""

import pandas as pd
import streamlit as st

from nhl.data_loaders import get_player_identity_summary, load_historical_data
from nhl.era import apply_era_to_hist, metric_is_era_adjusted


_SNAPSHOT_SUM_COLS = (
    "GP",
    "Points",
    "Goals",
    "Assists",
    "PIM",
    "+/-",
    "Wins",
    "Shutouts",
    "Saves",
    "Shots",
    "TotalTOIMins",
    "WeightedSV",
    "WeightedGAA",
)
_SNAPSHOT_FIRST_COLS = (
    "Age",
    "PlayerID",
    "PositionCode",
    "NHLeMultiplier",
)
_SKATER_ROLE_FORWARD_CODES = {"C", "L", "R"}
_SKATER_ROLE_DEFENSE_CODES = {"D"}
_SKATER_RATE_METRICS = {"PPG", "SH%", "TOI"}
_GOALIE_RATE_METRICS = {"Save %", "GAA"}
_SUPPORTED_SKATER_METRICS = {"Points", "Goals", "Assists", "+/-", "GP", "PPG", "SH%", "PIM", "TOI"}
_SUPPORTED_GOALIE_METRICS = {"Save %", "GAA", "Shutouts", "Wins", "GP", "Saves"}
_RANKING_EPSILON = 1e-6


def _season_span_label(season_year: int | str | None) -> str:
    """Return one start-year season label like ``2022-23``."""
    try:
        year = int(season_year)
    except Exception:
        return ""
    return f"{year}-{str(year + 1)[2:]}"


def _normalize_position_code(value: object) -> str:
    """Normalize player position codes across payload variants."""
    clean_value = str(value or "").strip().upper()
    if clean_value.startswith("LW"):
        return "L"
    if clean_value.startswith("RW"):
        return "R"
    if clean_value.startswith("LD") or clean_value.startswith("RD"):
        return "D"
    return clean_value[:1]


def _build_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute rate columns from additive season totals."""
    if df.empty:
        return df

    d = df.copy()
    for col in ("GP", "Points", "Goals", "Shots", "TotalTOIMins", "WeightedSV", "WeightedGAA"):
        if col not in d.columns:
            d[col] = 0.0
    gp_denom = pd.to_numeric(d.get("GP", 0), errors="coerce").replace(0, float("nan"))
    shots_denom = pd.to_numeric(d.get("Shots", 0), errors="coerce").replace(0, float("nan"))

    d["PPG"] = pd.to_numeric(d.get("Points", 0), errors="coerce").div(gp_denom).fillna(0.0)
    d["TOI"] = pd.to_numeric(d.get("TotalTOIMins", 0), errors="coerce").div(gp_denom).fillna(0.0)
    d["SH%"] = (
        pd.to_numeric(d.get("Goals", 0), errors="coerce")
        .div(shots_denom)
        .mul(100.0)
        .fillna(0.0)
    )
    d["Save %"] = pd.to_numeric(d.get("WeightedSV", 0), errors="coerce").div(gp_denom).fillna(0.0)
    d["GAA"] = pd.to_numeric(d.get("WeightedGAA", 0), errors="coerce").div(gp_denom).fillna(0.0)
    return d


def collapse_player_snapshot_rows(season_rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse traded-player stints into one season/league/game-type row.

    The dialog works from raw season rows, which can contain multiple NHL stints
    for traded players at the same age. Rarity should compare that clicked
    season to the historical parquet the same way the parquet stores seasons:
    one additive row per player-season-age-position.
    """
    if season_rows is None or season_rows.empty:
        return pd.DataFrame()

    d = season_rows.copy()
    if "League" not in d.columns:
        d["League"] = ""
    if "GameType" not in d.columns:
        d["GameType"] = "Regular"
    if "SeasonYear" not in d.columns:
        d["SeasonYear"] = 0

    for col in _SNAPSHOT_SUM_COLS:
        if col not in d.columns:
            d[col] = 0.0
    for col in _SNAPSHOT_FIRST_COLS:
        if col not in d.columns:
            d[col] = ""

    agg_map = {col: "sum" for col in _SNAPSHOT_SUM_COLS}
    agg_map.update({col: "first" for col in _SNAPSHOT_FIRST_COLS})
    collapsed = (
        d.groupby(["SeasonYear", "League", "GameType"], as_index=False)
        .agg(agg_map)
        .sort_values(["SeasonYear", "League", "GameType"], kind="stable")
        .reset_index(drop=True)
    )
    return _build_rate_columns(collapsed)


def _supports_historical_metric(hist_df: pd.DataFrame, metric: str) -> bool:
    """Return whether the historical artifact can support the requested metric."""
    if hist_df.empty:
        return False
    if metric == "SH%":
        return "Shots" in hist_df.columns and pd.to_numeric(hist_df["Shots"], errors="coerce").fillna(0).gt(0).any()
    if metric == "TOI":
        return (
            "TotalTOIMins" in hist_df.columns
            and pd.to_numeric(hist_df["TotalTOIMins"], errors="coerce").fillna(0).gt(0).any()
        )
    return metric in hist_df.columns


def _metric_min_gp(metric: str, stat_category: str) -> int:
    """Return the minimum games-played threshold for comparable rate stats."""
    if stat_category == "Skater" and metric in _SKATER_RATE_METRICS:
        return 40
    if stat_category == "Goalie" and metric in _GOALIE_RATE_METRICS:
        return 20
    return 0


def _apply_skater_era_rates_if_needed(df: pd.DataFrame, metric: str, do_era: bool) -> pd.DataFrame:
    """Recompute skater rate stats after era-adjusting counting stats."""
    if not metric_is_era_adjusted(metric, "Skater", do_era):
        return df
    adjusted = apply_era_to_hist(df, True, is_goalie=False)
    return _build_rate_columns(adjusted)


def _apply_goalie_era_rates_if_needed(df: pd.DataFrame, metric: str, do_era: bool) -> pd.DataFrame:
    """Era-adjust goalie metrics only when the visible metric actually changes."""
    if not metric_is_era_adjusted(metric, "Goalie", do_era):
        return df
    return apply_era_to_hist(df, True, is_goalie=True)


def _resolve_role_label(position_code: object) -> str:
    """Return the skater peer bucket label for one clicked row."""
    clean_code = _normalize_position_code(position_code)
    if clean_code in _SKATER_ROLE_FORWARD_CODES:
        return "forwards"
    if clean_code in _SKATER_ROLE_DEFENSE_CODES:
        return "defensemen"
    return ""


def _filter_historical_pool(
    hist_df: pd.DataFrame,
    age: int,
    metric: str,
    stat_category: str,
    role_label: str = "",
) -> pd.DataFrame:
    """Return the comparable historical peer pool for one age and metric."""
    if hist_df.empty:
        return pd.DataFrame()

    d = hist_df.copy()
    if stat_category == "Goalie":
        d = d[d["Position"].astype(str).str.upper().eq("G")]
    else:
        d = d[~d["Position"].astype(str).str.upper().eq("G")]

    d = d[pd.to_numeric(d.get("Age"), errors="coerce").eq(int(age))]
    if d.empty:
        return d

    min_gp = _metric_min_gp(metric, stat_category)
    if min_gp > 0:
        d = d[pd.to_numeric(d.get("GP"), errors="coerce").fillna(0).ge(min_gp)]
    if d.empty:
        return d

    if role_label == "forwards":
        d = d[d["Position"].astype(str).str.upper().isin(_SKATER_ROLE_FORWARD_CODES)]
    elif role_label == "defensemen":
        d = d[d["Position"].astype(str).str.upper().isin(_SKATER_ROLE_DEFENSE_CODES)]
    return d


def _compute_ranking(metric_values: pd.Series, target_value: float, lower_is_better: bool) -> dict:
    """Return exact rank and midrank percentile for one target value."""
    values = pd.to_numeric(metric_values, errors="coerce").dropna()
    if values.empty or pd.isna(target_value):
        return {}

    if lower_is_better:
        strictly_better = int((values < (target_value - _RANKING_EPSILON)).sum())
        equal_count = int((values - target_value).abs().le(_RANKING_EPSILON).sum())
        opposite_side = int((values > (target_value + _RANKING_EPSILON)).sum())
    else:
        strictly_better = int((values > (target_value + _RANKING_EPSILON)).sum())
        equal_count = int((values - target_value).abs().le(_RANKING_EPSILON).sum())
        opposite_side = int((values < (target_value - _RANKING_EPSILON)).sum())

    sample_size = int(len(values))
    percentile = 100.0 * (opposite_side + (0.5 * equal_count)) / sample_size
    return {
        "rank": strictly_better + 1,
        "sample_size": sample_size,
        "percentile": percentile,
    }


def _build_unavailable_payload(season_row: dict, metric: str, reason: str, *, do_era: bool, stat_category: str) -> dict:
    """Return a consistent payload when rarity should not render."""
    try:
        age_value = int(pd.to_numeric(season_row.get("Age"), errors="coerce"))
    except Exception:
        age_value = 0
    return {
        "season_label": _season_span_label(season_row.get("SeasonYear")),
        "metric": metric,
        "value": None,
        "age": age_value,
        "percentile": None,
        "rank": None,
        "sample_size": 0,
        "role_label": "",
        "role_percentile": None,
        "role_rank": None,
        "role_sample_size": 0,
        "top_seasons": [],
        "is_era_adjusted": metric_is_era_adjusted(metric, stat_category, do_era),
        "unavailable_reason": reason,
    }


@st.cache_data(ttl=3600)
def _resolve_player_name(player_id: int) -> str:
    """Return a cached player display name for rarity leaderboard rows."""
    try:
        clean_player_id = int(player_id)
    except Exception:
        return ""
    if clean_player_id <= 0:
        return ""

    summary = get_player_identity_summary(clean_player_id)
    return str(summary.get("name", "") or "").strip()


def _build_top_seasons(pool: pd.DataFrame, metric: str, lower_is_better: bool, limit: int = 5) -> list[dict]:
    """Return the top historical seasons from the same comparison pool.

    The leaderboard intentionally follows the overall rarity pool that drives
    the main ``#rank of n`` line, not the optional skater role-split pool.
    Names come from the cached player identity helper so the app does not need a
    second historical names artifact.
    """
    if pool.empty or metric not in pool.columns:
        return []

    ascending = [lower_is_better, True, True]
    ranked = (
        pool.copy()
        .assign(
            _metric_value=pd.to_numeric(pool[metric], errors="coerce"),
            _season_year=pd.to_numeric(pool.get("SeasonYear"), errors="coerce").fillna(0).astype(int),
            _player_id=pd.to_numeric(pool.get("PlayerID"), errors="coerce").fillna(0).astype(int),
        )
        .dropna(subset=["_metric_value"])
        .sort_values(["_metric_value", "_season_year", "_player_id"], ascending=ascending, kind="stable")
        .head(limit)
    )
    if ranked.empty:
        return []

    top_rows: list[dict] = []
    for index, (_, row) in enumerate(ranked.iterrows(), start=1):
        player_id_value = pd.to_numeric(row.get("_player_id", 0), errors="coerce")
        player_id = 0 if pd.isna(player_id_value) else int(player_id_value)
        top_rows.append(
            {
                "display_rank": index,
                "player_id": player_id,
                "player_name": _resolve_player_name(player_id) or f"Player {player_id}",
                "season_label": _season_span_label(row.get("_season_year", 0)),
                "value": float(pd.to_numeric(row.get("_metric_value"), errors="coerce")),
            }
        )
    return top_rows


@st.cache_data
def get_age_rarity_summary(
    season_row: dict,
    metric: str,
    stat_category: str,
    do_era: bool,
) -> dict:
    """Return the dialog payload for one clicked historical season row.

    Args:
        season_row: Collapsed player season row, usually one ``SeasonYear +
            League + GameType`` row from ``collapse_player_snapshot_rows()``.
        metric: Active chart metric shown in the dialog.
        stat_category: ``Skater`` or ``Goalie``.
        do_era: Whether the chart is currently showing an era-adjusted metric.

    Returns:
        A serializable dict consumed by ``dialog.py``. The payload includes the
        main overall rarity result, optional skater role split, and a compact
        top-5 leaderboard from the same overall historical pool. If the click is
        ineligible, the payload includes ``unavailable_reason`` instead.
    """
    clean_row = dict(season_row or {})
    if not clean_row:
        return _build_unavailable_payload({}, metric, "Age rarity unavailable right now.", do_era=do_era, stat_category=stat_category)

    supported_metrics = _SUPPORTED_GOALIE_METRICS if stat_category == "Goalie" else _SUPPORTED_SKATER_METRICS
    if metric not in supported_metrics:
        return _build_unavailable_payload(
            clean_row,
            metric,
            "Age rarity is unavailable for this metric.",
            do_era=do_era,
            stat_category=stat_category,
        )

    if str(clean_row.get("League", "") or "").strip().upper() != "NHL":
        return _build_unavailable_payload(
            clean_row,
            metric,
            "Age rarity is shown only for NHL regular-season rows.",
            do_era=do_era,
            stat_category=stat_category,
        )
    if str(clean_row.get("GameType", "") or "").strip() != "Regular":
        return _build_unavailable_payload(
            clean_row,
            metric,
            "Age rarity is shown only for NHL regular-season rows.",
            do_era=do_era,
            stat_category=stat_category,
        )

    hist_df = load_historical_data()
    if hist_df.empty:
        return _build_unavailable_payload(
            clean_row,
            metric,
            "Age rarity unavailable right now.",
            do_era=do_era,
            stat_category=stat_category,
        )

    if not _supports_historical_metric(hist_df, metric):
        return _build_unavailable_payload(
            clean_row,
            metric,
            "Age rarity for this metric needs a refreshed historical parquet.",
            do_era=do_era,
            stat_category=stat_category,
        )

    row_df = _build_rate_columns(pd.DataFrame([clean_row]))
    if row_df.empty:
        return _build_unavailable_payload(
            clean_row,
            metric,
            "Age rarity unavailable right now.",
            do_era=do_era,
            stat_category=stat_category,
        )

    min_gp = _metric_min_gp(metric, stat_category)
    row_gp_value = pd.to_numeric(row_df.iloc[0].get("GP", 0), errors="coerce")
    row_gp = 0.0 if pd.isna(row_gp_value) else float(row_gp_value)
    if min_gp > 0 and row_gp < min_gp:
        return _build_unavailable_payload(
            clean_row,
            metric,
            f"Age rarity for {metric} requires at least {min_gp} GP.",
            do_era=do_era,
            stat_category=stat_category,
        )

    if stat_category == "Skater":
        row_df = _apply_skater_era_rates_if_needed(row_df, metric, do_era)
        hist_pool = _apply_skater_era_rates_if_needed(hist_df, metric, do_era)
    else:
        row_df = _apply_goalie_era_rates_if_needed(row_df, metric, do_era)
        hist_pool = _apply_goalie_era_rates_if_needed(hist_df, metric, do_era)

    try:
        age = int(pd.to_numeric(row_df.iloc[0].get("Age"), errors="coerce"))
    except Exception:
        age = int(clean_row.get("Age", 0) or 0)

    overall_pool = _filter_historical_pool(hist_pool, age=age, metric=metric, stat_category=stat_category)
    if overall_pool.empty or metric not in overall_pool.columns:
        return _build_unavailable_payload(
            clean_row,
            metric,
            "Age rarity unavailable right now.",
            do_era=do_era,
            stat_category=stat_category,
        )

    target_value = float(pd.to_numeric(row_df.iloc[0].get(metric), errors="coerce"))
    lower_is_better = metric == "GAA"
    overall = _compute_ranking(overall_pool[metric], target_value, lower_is_better=lower_is_better)
    if not overall:
        return _build_unavailable_payload(
            clean_row,
            metric,
            "Age rarity unavailable right now.",
            do_era=do_era,
            stat_category=stat_category,
        )

    payload = {
        "season_label": _season_span_label(clean_row.get("SeasonYear")),
        "metric": metric,
        "value": target_value,
        "age": age,
        "percentile": float(overall["percentile"]),
        "rank": int(overall["rank"]),
        "sample_size": int(overall["sample_size"]),
        "role_label": "",
        "role_percentile": None,
        "role_rank": None,
        "role_sample_size": 0,
        "top_seasons": _build_top_seasons(overall_pool, metric, lower_is_better=lower_is_better),
        "is_era_adjusted": metric_is_era_adjusted(metric, stat_category, do_era),
        "unavailable_reason": "",
    }

    if stat_category == "Skater":
        role_label = _resolve_role_label(row_df.iloc[0].get("PositionCode"))
        if role_label:
            role_pool = _filter_historical_pool(
                hist_pool,
                age=age,
                metric=metric,
                stat_category=stat_category,
                role_label=role_label,
            )
            role_ranking = _compute_ranking(role_pool[metric], target_value, lower_is_better=lower_is_better) if not role_pool.empty else {}
            if role_ranking:
                payload.update(
                    {
                        "role_label": role_label,
                        "role_percentile": float(role_ranking["percentile"]),
                        "role_rank": int(role_ranking["rank"]),
                        "role_sample_size": int(role_ranking["sample_size"]),
                    }
                )

    return payload
