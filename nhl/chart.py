"""
nhl.chart — Plotly chart rendering and JS bridge orchestration.

Assembles the final DataFrame from all processed pipelines, optionally adds
baseline data, builds the Plotly figure, renders a real toolbar row above the
chart, injects JS for responsive dtick / share-link handling, mounts a
Plotly-click bridge, and dispatches chart dialogs from first-click events.

Visual conventions (from CLAUDE.md):
    Real data:   solid colored line, filled markers
    Projection:  dotted player-colored line, open circle markers
    Baseline:    muted grey dashed line, visible round markers

Imports from project:
    nhl.constants — RATE_STATS, TEAM_RATE_STATS
    nhl.dialog    — show_season_details
"""

import colorsys
import hashlib
import json
from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from nhl.constants import (
    RATE_STATS, TEAM_RATE_STATS,
)
from nhl.dialog import show_season_details, show_team_game_details
from nhl.era import metric_is_era_adjusted
from nhl.ui_state import dialog_slot_available, mark_dialog_opened_this_run, session_state_get, session_state_set


X_AXIS_TICK_COLOR = "rgba(255, 255, 255, 0.80)"
Y_AXIS_TICK_COLOR = "rgba(255, 255, 255, 0.80)"
X_AXIS_CUE_COLOR = "rgba(255, 255, 255, 0.25)"
Y_AXIS_CUE_COLOR = "rgba(255, 255, 255, 0.25)"
BASELINE_LINE_DASH = "14px,10px"
BASELINE_LINE_COLOR = "rgba(190, 190, 190, 0.72)"
BASELINE_LINE_OPACITY = 0.65
BASELINE_MARKER_COLOR = "rgba(220, 220, 220, 0.92)"
PROJECTION_LINE_WIDTH = 2
PROJECTION_LINE_OPACITY = 0.90
PROJECTION_GLOW_OUTER_WIDTH = 5
PROJECTION_GLOW_OUTER_OPACITY = 0.16
PROJECTION_GLOW_INNER_WIDTH = 2
PROJECTION_GLOW_INNER_OPACITY = 0.26
PLAYER_COLOR_STATE_KEY = "player_chart_colors"
CLICKABLE_AGE_MARKER_SIZE = 9
CLICKABLE_AGE_MARKER_GLOW_SIZE = 16
CLICKABLE_AGE_MARKER_GLOW_OPACITY = 0.115
CLICKABLE_AGE_MARKER_OUTLINE = "rgba(255, 255, 255, 0.90)"
SEASON_MARKER_SIZE = 13
SEASON_MARKER_GLOW_SIZE = 24
SEASON_MARKER_GLOW_OPACITY = 0.101
SEASON_STEM_WIDTH = 2
SEASON_MARKER_OUTLINE = "rgba(255, 255, 255, 0.78)"
TRACE_COLOR_MIN_DISTANCE = {
    "Skater": 90.0,
    "Goalie": 80.0,
    "Team": 90.0,
}
DISTINCT_TRACE_COLOR_ATTEMPTS = 40
GOLDEN_RATIO_HUE_STEP = 0.618033988749895
CATEGORY_TRACE_STARTERS = {
    "Skater": [
        "#35D8FF",
        "#84FF2E",
        "#FF6B6B",
        "#C56BFF",
        "#FFC857",
        "#23E0B5",
        "#FF8A3D",
        "#F76FD6",
    ],
    "Goalie": [
        "#FF7A1A",
        "#FFE14A",
        "#FF5C8A",
        "#69C9FF",
        "#B67AFF",
        "#31D8A5",
        "#FFB347",
        "#F7FF6A",
    ],
    "Team": [
        "#FF5FD2",
        "#7AD7FF",
        "#FFD166",
        "#71FF70",
        "#FF7F63",
        "#B58CFF",
        "#2DE2C6",
        "#FF97C7",
    ],
}
CHART_CLICK_BRIDGE_COMPONENT_NAME = "comparison_chart_click_bridge"
CHART_CLICK_BRIDGE_MOUNT_KEY = "comparison_chart_click_bridge"
CHART_CLICK_BRIDGE_ANCHOR_ID = "comparison-main-plotly"
LAST_HANDLED_CHART_CLICK_NONCE_SESSION_KEY = "_last_handled_chart_click_nonce"
CHART_CLICK_BRIDGE_BIND_ATTEMPTS = 20
CHART_CLICK_BRIDGE_BIND_DELAY_MS = 150
CHART_HOVER_DISTANCE = 32
CHART_CLICK_BRIDGE_JS = f"""
export default function(component) {{
    const {{ data, setTriggerValue }} = component;

    const parseBridgeData = () => {{
        if (!data) {{
            return {{}};
        }}
        if (typeof data === 'string') {{
            try {{
                return JSON.parse(data);
            }} catch (err) {{
                return {{}};
            }}
        }}
        return (typeof data === 'object' && data !== null) ? data : {{}};
    }};

    const bridgeData = parseBridgeData();
    const chartInstanceId = String(bridgeData.chart_instance_id || '').trim();
    const anchorId = String(bridgeData.anchor_id || '').trim();
    const bindAttempts = Number(bridgeData.bind_attempts || 0) || {CHART_CLICK_BRIDGE_BIND_ATTEMPTS};
    const bindDelayMs = Number(bridgeData.bind_delay_ms || 0) || {CHART_CLICK_BRIDGE_BIND_DELAY_MS};

    function getCurrentTargetPlot(parent) {{
        const plots = Array.prototype.slice.call(parent.document.querySelectorAll('.js-plotly-plot'));
        if (!plots.length) return null;

        const anchor = anchorId ? parent.document.getElementById(anchorId) : null;
        if (!anchor || !anchor.compareDocumentPosition || !parent.Node) {{
            return plots[plots.length - 1];
        }}

        for (let i = 0; i < plots.length; i += 1) {{
            if (anchor.compareDocumentPosition(plots[i]) & parent.Node.DOCUMENT_POSITION_FOLLOWING) {{
                return plots[i];
            }}
        }}
        return plots[plots.length - 1];
    }}

    function normalizeCustomData(value) {{
        return Array.isArray(value) ? value : [];
    }}

    let lastEmittedAt = 0;
    const DEBOUNCE_MS = 500;
    function emitClick(payload) {{
        const now = Date.now();
        if (now - lastEmittedAt < DEBOUNCE_MS) return;
        lastEmittedAt = now;
        setTriggerValue('clicked', JSON.stringify(payload));
    }}

    let cleanup = null;
    let retryTimer = null;

    // Resolve the window that contains the Plotly chart.
    // On localhost the chart lives in window.parent; on Streamlit Cloud
    // the bridge iframe is a *sibling* of the app iframe, so we must
    // search through the parent's child iframes.
    let _appWindow = null;
    function getAppWindow() {{
        if (_appWindow) return _appWindow;
        const p = window.parent;
        try {{
            if (p.document.querySelectorAll('.js-plotly-plot').length) {{
                _appWindow = p;
                return p;
            }}
        }} catch(e) {{}}
        try {{
            const iframes = p.document.querySelectorAll('iframe');
            for (let i = 0; i < iframes.length; i++) {{
                try {{
                    const fw = iframes[i].contentWindow;
                    if (fw && fw !== window &&
                        fw.document.querySelectorAll('.js-plotly-plot').length) {{
                        _appWindow = fw;
                        return fw;
                    }}
                }} catch(e) {{}}
            }}
        }} catch(e) {{}}
        return p;
    }}

    function bind(attemptsLeft) {{
        const parent = getAppWindow();
        if (!chartInstanceId) return;

        const plot = getCurrentTargetPlot(parent);
        if (!plot || typeof plot.on !== 'function') {{
            if (attemptsLeft > 0) {{
                retryTimer = parent.setTimeout(function() {{
                    bind(attemptsLeft - 1);
                }}, bindDelayMs);
            }}
            return;
        }}

        if (plot.__nhlChartClickBridgeCleanup && plot.__nhlChartClickBridgeInstanceId !== chartInstanceId) {{
            plot.__nhlChartClickBridgeCleanup();
        }}

        if (plot.__nhlChartClickBridgeInstanceId === chartInstanceId && plot.__nhlChartClickBridgeCleanup) {{
            cleanup = plot.__nhlChartClickBridgeCleanup;
            return;
        }}

        const handler = function(event) {{
            const points = event && Array.isArray(event.points) ? event.points : [];
            if (!points.length) {{
                return;
            }}

            const point = points[0] || {{}};
            const traceName = String(
                (point.fullData && point.fullData.name)
                || (point.data && point.data.name)
                || ''
            ).trim();
            const payload = {{
                nonce: `${{Date.now()}}-${{Math.floor(Math.random() * 1000000)}}`,
                chart_instance_id: chartInstanceId,
                trace_name: traceName,
                x: point.x ?? null,
                y: point.y ?? null,
                customdata: normalizeCustomData(point.customdata),
                curve_number: Number.isInteger(point.curveNumber) ? point.curveNumber : null,
                point_number: Number.isInteger(point.pointNumber) ? point.pointNumber : null,
            }};
            emitClick(payload);
        }};

        plot.on('plotly_click', handler);

        // ---- Touch-tap proxy for mobile ----
        // Neither plotly_click nor plotly_hover fire reliably on mobile
        // because Plotly's drag handler consumes touch events.  The hover
        // tooltip DOES appear though, and Plotly always sets plot._hoverdata
        // when it shows the tooltip.  We detect taps via touch listeners and
        // read _hoverdata directly on touchend.
        let touchState = null;
        const TAP_MAX_DISTANCE = 15;
        const TAP_MAX_DURATION = 400;

        const plotArea = plot.querySelector('.nsewdrag') || plot;

        plotArea.addEventListener('touchstart', function(e) {{
            if (e.touches.length === 1) {{
                touchState = {{
                    startX: e.touches[0].clientX,
                    startY: e.touches[0].clientY,
                    startTime: Date.now(),
                }};
            }} else {{
                touchState = null;
            }}
        }}, {{ passive: true }});

        plotArea.addEventListener('touchmove', function(e) {{
            if (!touchState || e.touches.length !== 1) {{
                touchState = null;
                return;
            }}
            var dx = e.touches[0].clientX - touchState.startX;
            var dy = e.touches[0].clientY - touchState.startY;
            if (Math.sqrt(dx * dx + dy * dy) > TAP_MAX_DISTANCE) {{
                touchState = null;
            }}
        }}, {{ passive: true }});

        plotArea.addEventListener('touchend', function() {{
            if (!touchState) return;
            var elapsed = Date.now() - touchState.startTime;
            touchState = null;
            if (elapsed > TAP_MAX_DURATION) return;

            // Read Plotly's internal hover state (set when tooltip appears)
            var hoverData = plot._hoverdata;
            if (!hoverData || !hoverData.length) return;
            var point = hoverData[0];
            var traceName = String(
                (point.fullData && point.fullData.name)
                || (point.data && point.data.name)
                || ''
            ).trim();
            emitClick({{
                nonce: `${{Date.now()}}-${{Math.floor(Math.random() * 1000000)}}`,
                chart_instance_id: chartInstanceId,
                trace_name: traceName,
                x: point.x ?? null,
                y: point.y ?? null,
                customdata: normalizeCustomData(point.customdata),
                curve_number: Number.isInteger(point.curveNumber) ? point.curveNumber : null,
                point_number: Number.isInteger(point.pointNumber) ? point.pointNumber : null,
            }});
        }}, {{ passive: true }});

        const localCleanup = function() {{
            if (retryTimer) {{
                parent.clearTimeout(retryTimer);
                retryTimer = null;
            }}
            try {{
                if (typeof plot.removeListener === 'function') {{
                    plot.removeListener('plotly_click', handler);
                }} else if (typeof plot.off === 'function') {{
                    plot.off('plotly_click', handler);
                }}
            }} catch (err) {{
                // Ignore teardown races when Streamlit replaces the plot DOM.
            }}
            if (plot.__nhlChartClickBridgeCleanup === localCleanup) {{
                delete plot.__nhlChartClickBridgeCleanup;
                delete plot.__nhlChartClickBridgeInstanceId;
            }}
        }};

        plot.__nhlChartClickBridgeCleanup = localCleanup;
        plot.__nhlChartClickBridgeInstanceId = chartInstanceId;
        cleanup = localCleanup;
    }}

    bind(bindAttempts);

    return () => {{
        if (retryTimer) {{
            try {{ getAppWindow().clearTimeout(retryTimer); }} catch(e) {{}}
            retryTimer = null;
        }}
        if (cleanup) {{
            cleanup();
        }}
    }};
}}
"""
CHART_CLICK_BRIDGE = st.components.v2.component(
    CHART_CLICK_BRIDGE_COMPONENT_NAME,
    js=CHART_CLICK_BRIDGE_JS,
)


def _palette_for_category(stat_category: str) -> list[str]:
    """Return the chart color palette for the active stat category.

    Args:
        stat_category: One of ``Skater``, ``Goalie``, or ``Team``.

    Returns:
        list[str]: Ordered list of hex color strings for the colorway.
    """
    return list(CATEGORY_TRACE_STARTERS.get(stat_category, ["#4F8FFF"]))


def _rgb_to_hex(red: int, green: int, blue: int) -> str:
    """Convert integer RGB channels into an uppercase hex color string."""
    return f"#{red:02X}{green:02X}{blue:02X}"


def _hex_to_rgb(color: str | None) -> tuple[int, int, int] | None:
    """Convert one #RRGGBB string into integer RGB channels."""
    clean = str(color or "").strip()
    if not (clean.startswith("#") and len(clean) == 7):
        return None
    try:
        return (
            int(clean[1:3], 16),
            int(clean[3:5], 16),
            int(clean[5:7], 16),
        )
    except ValueError:
        return None


def _color_distance(color_a: str | None, color_b: str | None) -> float | None:
    """Return Euclidean RGB distance between two hex colors."""
    rgb_a = _hex_to_rgb(color_a)
    rgb_b = _hex_to_rgb(color_b)
    if rgb_a is None or rgb_b is None:
        return None
    return sum((a - b) ** 2 for a, b in zip(rgb_a, rgb_b)) ** 0.5


def _min_color_distance_to_assigned(candidate: str, assigned_colors: list[str]) -> float | None:
    """Return the smallest distance from one candidate color to assigned colors."""
    if not assigned_colors:
        return None

    distances = [
        distance
        for assigned in assigned_colors
        if (distance := _color_distance(candidate, assigned)) is not None
    ]
    return min(distances) if distances else None


def _build_seeded_trace_color(seed_text: str, category: str, attempt: int = 0) -> str:
    """Return one bright, deterministic, pseudo-random hex color for a trace."""
    digest = hashlib.sha256(f"{category}|{seed_text}|{attempt}".encode("utf-8")).digest()
    hue = (
        (int.from_bytes(digest[:2], "big") / 65535.0)
        + (attempt * GOLDEN_RATIO_HUE_STEP)
    ) % 1.0

    base_saturation = {
        "Skater": 0.78,
        "Goalie": 0.74,
        "Team": 0.82,
    }.get(category, 0.78)
    base_lightness = {
        "Skater": 0.60,
        "Goalie": 0.62,
        "Team": 0.58,
    }.get(category, 0.60)

    saturation = base_saturation + (((digest[2] / 255.0) - 0.5) * 0.14)
    lightness = base_lightness + (((digest[3] / 255.0) - 0.5) * 0.12)
    saturation = min(max(saturation, 0.68), 0.90)
    lightness = min(max(lightness, 0.50), 0.70)

    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return _rgb_to_hex(round(red * 255), round(green * 255), round(blue * 255))


def _pick_next_distinct_trace_color(
    seed_text: str,
    assigned_colors: list[str],
    category: str,
) -> str:
    """Pick a deterministic pseudo-random color that stays distinct from prior traces."""
    min_distance = float(TRACE_COLOR_MIN_DISTANCE.get(category, 85.0))
    starter_colors = _palette_for_category(category)
    if not assigned_colors and starter_colors:
        return starter_colors[0]

    best_candidate = starter_colors[0] if starter_colors else "#4F8FFF"
    best_distance = -1.0

    for candidate in starter_colors:
        candidate_min_distance = _min_color_distance_to_assigned(candidate, assigned_colors)
        if candidate_min_distance is None or candidate_min_distance >= min_distance:
            return candidate
        if candidate_min_distance > best_distance:
            best_candidate = candidate
            best_distance = candidate_min_distance

    for attempt in range(DISTINCT_TRACE_COLOR_ATTEMPTS):
        candidate = _build_seeded_trace_color(seed_text=seed_text, category=category, attempt=attempt)
        candidate_min_distance = _min_color_distance_to_assigned(candidate, assigned_colors)
        if candidate_min_distance is None:
            return candidate

        if candidate_min_distance >= min_distance:
            return candidate
        if candidate_min_distance > best_distance:
            best_candidate = candidate
            best_distance = candidate_min_distance

    return best_candidate


def _build_trace_color_map(final_df: pd.DataFrame, stat_category: str, team_mode: bool) -> dict[str, str]:
    """Build a stable real-trace color map in first-appearance order."""
    if final_df.empty or "Player" not in final_df.columns:
        return {}

    source_df = final_df[["Player", "BaseName"]].copy() if "BaseName" in final_df.columns else final_df[["Player"]].copy()
    if "BaseName" not in source_df.columns:
        source_df["BaseName"] = source_df["Player"]

    ordered_entries: list[tuple[str, str]] = []
    seen_players: set[str] = set()
    for row in source_df[["Player", "BaseName"]].itertuples(index=False, name=None):
        player_name = str(row[0])
        base_name = player_name if len(row) <= 1 or pd.isna(row[1]) else str(row[1])
        if player_name in seen_players:
            continue
        if "(Proj)" in player_name or _is_baseline_trace(player_name):
            continue
        seen_players.add(player_name)
        ordered_entries.append((player_name, base_name))

    trace_color_map: dict[str, str] = {}
    assigned_colors: list[str] = []
    for player_name, base_name in ordered_entries:
        selected_color = _pick_next_distinct_trace_color(
            seed_text=f"{base_name}|{player_name}",
            assigned_colors=assigned_colors,
            category=stat_category,
        )
        trace_color_map[player_name] = selected_color
        assigned_colors.append(selected_color)
    return trace_color_map


def _build_plotly_color_map(final_df: pd.DataFrame, trace_color_map: dict[str, str]) -> dict[str, str]:
    """Extend the real-trace color map with projection aliases for Plotly Express."""
    plotly_color_map = dict(trace_color_map)
    if final_df.empty or "Player" not in final_df.columns:
        return plotly_color_map

    for player_name in final_df["Player"].dropna():
        display_name = str(player_name)
        if "(Proj)" not in display_name:
            continue
        base_name = display_name.replace(" (Proj)", "")
        if base_name in trace_color_map:
            plotly_color_map[display_name] = trace_color_map[base_name]
    return plotly_color_map


def _store_player_chart_colors(player_colors: dict[str, str | None]) -> None:
    """Persist the active player-to-chart-color map for sibling UI panels.

    Args:
        player_colors: Mapping of real-player names to their assigned chart colors.

    Returns:
        None.
    """
    setattr(st.session_state, PLAYER_COLOR_STATE_KEY, dict(player_colors))


def _noop_chart_click_change() -> None:
    """Provide a stable callback for the chart click bridge."""


def _mount_chart_click_bridge(chart_instance_id: str) -> str | None:
    """Mount the chart click bridge and return its latest payload.

    Args:
        chart_instance_id: Stable identity for the visible chart instance.

    Returns:
        Latest serialized click payload, if the bridge has emitted one.
    """
    result = CHART_CLICK_BRIDGE(
        data=json.dumps(
            {
                "chart_instance_id": chart_instance_id,
                "anchor_id": CHART_CLICK_BRIDGE_ANCHOR_ID,
                "bind_attempts": CHART_CLICK_BRIDGE_BIND_ATTEMPTS,
                "bind_delay_ms": CHART_CLICK_BRIDGE_BIND_DELAY_MS,
            },
            separators=(",", ":"),
        ),
        key=CHART_CLICK_BRIDGE_MOUNT_KEY,
        on_clicked_change=_noop_chart_click_change,
    )
    return getattr(result, "clicked", None)


def _parse_chart_click_trigger(value, expected_chart_instance_id: str) -> dict | None:
    """Validate one bridge payload and normalize its point fields.

    Args:
        value: Raw bridge payload returned by the v2 component.
        expected_chart_instance_id: Active chart identity for this rerun.

    Returns:
        Normalized payload dict, or ``None`` when the trigger is invalid.
    """
    raw_value = str(value or "").strip()
    if not raw_value:
        return None

    try:
        payload = json.loads(raw_value)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    chart_instance_id = str(payload.get("chart_instance_id", "") or "").strip()
    nonce = str(payload.get("nonce", "") or "").strip()
    if chart_instance_id != str(expected_chart_instance_id or "").strip() or not nonce:
        return None

    custom_data = payload.get("customdata", [])
    if isinstance(custom_data, tuple):
        custom_data = list(custom_data)
    if not isinstance(custom_data, list):
        custom_data = []

    return {
        "nonce": nonce,
        "trace_name": str(payload.get("trace_name", "") or "").strip(),
        "point": {
            "x": payload.get("x"),
            "y": payload.get("y"),
            "customdata": custom_data,
            "curve_number": payload.get("curve_number"),
            "point_number": payload.get("point_number"),
        },
    }


def _dispatch_chart_click_point(
    point: dict,
    *,
    trace_name: str,
    team_mode: bool,
    games_mode: bool,
    is_single_season_team_games: bool,
    has_exact_game_custom_data: bool,
    metric: str,
    final_df: pd.DataFrame,
    raw_dfs_cache: list,
    season_type: str,
    selected_season: str | int,
    do_cumul: bool,
    ml_clones_dict: dict,
    historical_baselines: dict,
    stat_category: str,
    do_era: bool,
) -> bool:
    """Open the matching chart dialog for one validated Plotly click.

    Args:
        point: Normalized point payload from the JS bridge.
        trace_name: Plotly trace label for the clicked point.
        team_mode: Whether the chart is rendering team traces.
        games_mode: Whether the x-axis uses games played.
        is_single_season_team_games: Whether team selected-season game mode is active.
        has_exact_game_custom_data: Whether player game rows carry exact game metadata.
        metric: Active chart metric.
        final_df: Concatenated chart dataframe.
        raw_dfs_cache: Raw player frames used by dialogs.
        season_type: Active season scope.
        selected_season: Active chart season.
        do_cumul: Whether cumulative view is active.
        ml_clones_dict: Cached projection clone payloads.
        historical_baselines: Cached baseline payloads.
        stat_category: Active stat category.
        do_era: Whether era adjustment is active for the visible metric.

    Returns:
        True when a dialog was opened, else False.
    """
    clean_trace_name = str(trace_name or "").strip()
    if clean_trace_name.startswith("_"):
        return False

    custom_data = point.get("customdata", [])
    if isinstance(custom_data, tuple):
        custom_data = list(custom_data)
    if not isinstance(custom_data, list):
        custom_data = []

    if team_mode:
        if not is_single_season_team_games:
            return False
        try:
            clicked_game_number = int(point.get("x"))
        except Exception:
            return False

        clicked_team_abbr = str(custom_data[0]) if len(custom_data) > 0 else ""
        clicked_team_name = str(custom_data[1]) if len(custom_data) > 1 else ""
        clicked_game_date = str(custom_data[2]) if len(custom_data) > 2 else ""
        clicked_game_type = str(custom_data[3]) if len(custom_data) > 3 else ""
        clicked_opponent_abbr = str(custom_data[4]) if len(custom_data) > 4 else ""
        clicked_opponent_name = str(custom_data[5]) if len(custom_data) > 5 else ""
        clicked_home_road = str(custom_data[6]) if len(custom_data) > 6 else ""
        clicked_result = str(custom_data[7]) if len(custom_data) > 7 else ""
        try:
            clicked_goals_for = float(custom_data[8]) if len(custom_data) > 8 else 0.0
        except Exception:
            clicked_goals_for = 0.0
        try:
            clicked_goals_against = float(custom_data[9]) if len(custom_data) > 9 else 0.0
        except Exception:
            clicked_goals_against = 0.0
        try:
            clicked_game_id = int(custom_data[10]) if len(custom_data) > 10 and custom_data[10] not in (None, "") else None
        except Exception:
            clicked_game_id = None
        clicked_record_label = str(custom_data[11]) if len(custom_data) > 11 else ""

        show_team_game_details(
            team_name=clicked_team_name,
            team_abbr=clicked_team_abbr,
            metric=metric,
            val=point.get("y"),
            full_df=final_df,
            s_type=season_type,
            selected_season=selected_season,
            game_number=clicked_game_number,
            game_id=clicked_game_id,
            game_date=clicked_game_date,
            clicked_game_type=clicked_game_type,
            opponent_abbr=clicked_opponent_abbr,
            opponent_name=clicked_opponent_name,
            home_road_flag=clicked_home_road,
            result_label=clicked_result,
            goals_for=clicked_goals_for,
            goals_against=clicked_goals_against,
            record_label=clicked_record_label,
        )
        mark_dialog_opened_this_run()
        return True

    clicked_player_name = str(custom_data[1]) if len(custom_data) > 1 else clean_trace_name
    if _is_baseline_trace(clean_trace_name) or _is_baseline_trace(clicked_player_name):
        return False

    try:
        age_for_detail = int(custom_data[2]) if games_mode else point.get("x")
    except Exception:
        return False
    clicked_game_id = None
    clicked_game_date = None
    clicked_game_type = None
    clicked_game_number = None

    if has_exact_game_custom_data and len(custom_data) >= 6:
        try:
            clicked_game_number = int(point.get("x"))
        except Exception:
            clicked_game_number = None
        try:
            clicked_game_id = int(custom_data[3]) if custom_data[3] not in (None, "") else None
        except Exception:
            clicked_game_id = None
        clicked_game_date = str(custom_data[4] or "")
        clicked_game_type = str(custom_data[5] or "")

    show_season_details(
        player_name=clicked_player_name,
        age=age_for_detail,
        raw_dfs_list=raw_dfs_cache,
        metric=metric,
        val=point.get("y"),
        is_cumul=do_cumul,
        full_df=final_df,
        s_type=season_type,
        ml_clones_dict=ml_clones_dict,
        historical_baselines=historical_baselines,
        stat_category=stat_category,
        do_era=do_era,
        game_id=clicked_game_id,
        game_date=clicked_game_date,
        clicked_game_type=clicked_game_type,
        game_number=clicked_game_number,
    )
    mark_dialog_opened_this_run()
    return True


def _show_chart_dialog_from_trigger(
    trigger_value,
    chart_instance_id: str,
    *,
    suppress_dialogs: bool,
    team_mode: bool,
    games_mode: bool,
    is_single_season_team_games: bool,
    has_exact_game_custom_data: bool,
    metric: str,
    final_df: pd.DataFrame,
    raw_dfs_cache: list,
    season_type: str,
    selected_season: str | int,
    do_cumul: bool,
    ml_clones_dict: dict,
    historical_baselines: dict,
    stat_category: str,
    do_era: bool,
) -> bool:
    """Consume one chart-click bridge payload and open at most one dialog.

    Args:
        trigger_value: Latest bridge payload returned from the mounted component.
        chart_instance_id: Active chart identity for this rerun.
        suppress_dialogs: Whether chart dialogs must be dropped for this rerun.
        team_mode: Whether the chart is rendering team traces.
        games_mode: Whether the x-axis uses games played.
        is_single_season_team_games: Whether team selected-season game mode is active.
        has_exact_game_custom_data: Whether player game rows carry exact game metadata.
        metric: Active chart metric.
        final_df: Concatenated chart dataframe.
        raw_dfs_cache: Raw player frames used by dialogs.
        season_type: Active season scope.
        selected_season: Active chart season.
        do_cumul: Whether cumulative view is active.
        ml_clones_dict: Cached projection clone payloads.
        historical_baselines: Cached baseline payloads.
        stat_category: Active stat category.
        do_era: Whether era adjustment is active for the visible metric.

    Returns:
        True when a dialog was opened, else False.
    """
    parsed_trigger = _parse_chart_click_trigger(trigger_value, chart_instance_id)
    if parsed_trigger is None:
        return False

    nonce = parsed_trigger["nonce"]
    if session_state_get(LAST_HANDLED_CHART_CLICK_NONCE_SESSION_KEY) == nonce:
        return False
    session_state_set(LAST_HANDLED_CHART_CLICK_NONCE_SESSION_KEY, nonce)

    if suppress_dialogs or not dialog_slot_available():
        return False

    # The JS bridge emits one normalized point payload; Python keeps dialog routing centralized here.
    return _dispatch_chart_click_point(
        parsed_trigger["point"],
        trace_name=parsed_trigger["trace_name"],
        team_mode=team_mode,
        games_mode=games_mode,
        is_single_season_team_games=is_single_season_team_games,
        has_exact_game_custom_data=has_exact_game_custom_data,
        metric=metric,
        final_df=final_df,
        raw_dfs_cache=raw_dfs_cache,
        season_type=season_type,
        selected_season=selected_season,
        do_cumul=do_cumul,
        ml_clones_dict=ml_clones_dict,
        historical_baselines=historical_baselines,
        stat_category=stat_category,
        do_era=do_era,
    )


def _handle_native_chart_selection(
    selection_event,
    fig,
    *,
    suppress_dialogs: bool,
    team_mode: bool,
    games_mode: bool,
    is_single_season_team_games: bool,
    has_exact_game_custom_data: bool,
    metric: str,
    final_df: pd.DataFrame,
    raw_dfs_cache: list,
    season_type: str,
    selected_season: str | int,
    do_cumul: bool,
    ml_clones_dict: dict,
    historical_baselines: dict,
    stat_category: str,
    do_era: bool,
) -> bool:
    """Handle a native Plotly on_select event and open at most one dialog.

    Args:
        selection_event: Return value from st.plotly_chart with on_select="rerun".
        fig: The active Plotly figure (used to resolve trace names by index).
        suppress_dialogs: When True chart dialogs are suppressed this rerun.
        team_mode: Whether the chart is rendering team traces.
        games_mode: Whether the x-axis uses games played.
        is_single_season_team_games: Whether team selected-season game mode is active.
        has_exact_game_custom_data: Whether player game rows carry exact game metadata.
        metric: Active chart metric.
        final_df: Concatenated chart dataframe.
        raw_dfs_cache: Raw player frames used by dialogs.
        season_type: Active season scope.
        selected_season: Active chart season.
        do_cumul: Whether cumulative view is active.
        ml_clones_dict: Cached projection clone payloads.
        historical_baselines: Cached baseline payloads.
        stat_category: Active stat category.
        do_era: Whether era adjustment is active for the visible metric.

    Returns:
        True when a dialog was opened, else False.
    """
    # Harden payload access — Streamlit docs describe the return as "dict-like"
    points = []
    selection = getattr(selection_event, "selection", None) if selection_event else None
    if selection:
        points = selection.get("points", []) or []

    if not points:
        # Deselection (user clicked background): clear nonce so the same point
        # can be re-opened on the next click without a chart remount.
        session_state_set(LAST_HANDLED_CHART_CLICK_NONCE_SESSION_KEY, None)
        return False

    if len(points) != 1:
        # Unexpected multi-point payload — clear stale dedup state and bail.
        session_state_set(LAST_HANDLED_CHART_CLICK_NONCE_SESSION_KEY, None)
        return False

    pt = points[0]
    curve_number = pt.get("curve_number", 0)
    selection_key = f"{curve_number}|{pt.get('point_number')}|{pt.get('x')}|{pt.get('y')}"

    # Gate checks BEFORE writing the nonce — don't poison the click as "handled"
    # if we aren't actually going to handle it.
    if suppress_dialogs or not dialog_slot_available():
        return False

    if session_state_get(LAST_HANDLED_CHART_CLICK_NONCE_SESSION_KEY) == selection_key:
        return False
    session_state_set(LAST_HANDLED_CHART_CLICK_NONCE_SESSION_KEY, selection_key)

    # Resolve trace name from the figure (more reliable than payload fields)
    try:
        trace_name = fig.data[curve_number].name or ""
    except Exception:
        trace_name = ""

    custom_data = pt.get("customdata", [])
    if isinstance(custom_data, tuple):
        custom_data = list(custom_data)
    if not isinstance(custom_data, list):
        custom_data = []

    normalized_point = {
        "x": pt.get("x"),
        "y": pt.get("y"),
        "customdata": custom_data,
        "curve_number": curve_number,
        "point_number": pt.get("point_number"),
    }

    return _dispatch_chart_click_point(
        normalized_point,
        trace_name=trace_name,
        team_mode=team_mode,
        games_mode=games_mode,
        is_single_season_team_games=is_single_season_team_games,
        has_exact_game_custom_data=has_exact_game_custom_data,
        metric=metric,
        final_df=final_df,
        raw_dfs_cache=raw_dfs_cache,
        season_type=season_type,
        selected_season=selected_season,
        do_cumul=do_cumul,
        ml_clones_dict=ml_clones_dict,
        historical_baselines=historical_baselines,
        stat_category=stat_category,
        do_era=do_era,
    )


def _build_chart_glow_style(player_colors: dict) -> str:
    """Return a <style> block that illuminates the chart center with player trace colors.

    Uses broader radial gradients anchored at 50% 52% (vertical center-ish) that fade to
    transparent before fully reaching the edges. Returns an empty string when no colors are
    available.

    Args:
        player_colors: Mapping of player name to hex/rgb color string.

    Returns:
        str: HTML <style> block, or empty string if no colors.
    """
    colors = [c for c in player_colors.values() if c]
    if not colors:
        return ""

    # All gradients centered; broader ellipse keeps the glow subtle while spreading wider.
    glows = []
    for color in colors[:3]:
        strong = _with_alpha(color, 0.07)
        fade = _with_alpha(color, 0.0)
        glows.append(
            f"radial-gradient(ellipse 54% 42% at 50% 52%, {strong} 0%, {fade} 78%)"
        )

    bg = ", ".join(glows)

    # Outer edge glow: softly layered spread from the first two player colors.
    shadow_layers = []
    for color in colors[:2]:
        shadow_layers.append(f"0 0 32px {_with_alpha(color, 0.04)}")
        shadow_layers.append(f"0 0 64px {_with_alpha(color, 0.02)}")
    shadow = ", ".join(shadow_layers)

    return (
        "<style>"
        "div[data-testid='stPlotlyChart']{"
        f"background:{bg};"
        f"box-shadow:{shadow};"
        "border-radius:14px;"
        "overflow:hidden;"
        "}"
        "</style>"
    )


def _get_chart_context_label(
    team_mode: bool,
    games_mode: bool,
    selected_season: str | int = "All",
) -> str:
    """Return the x-context label for the chart toolbar title.

    Args:
        team_mode: True when the chart is rendering team comparisons.
        games_mode: True when the x-axis uses games played instead of age.
        selected_season: Selected chart-season value.

    Returns:
        str: Label describing the x context, such as Age, Games Played, or Game Number.
    """
    if team_mode and not games_mode:
        return "Season"
    if games_mode:
        if str(selected_season) != "All":
            return "Game Number"
        return "Games Played"
    return "Age"


def _get_chart_season_label(season_type: str) -> str:
    """Return the season descriptor for the chart toolbar title.

    Args:
        season_type: Selected season scope string from the UI.

    Returns:
        str: Human-readable season scope label.
    """
    season_labels = {
        "Regular": "Regular season",
        "Playoffs": "Playoffs",
        "Both": "Regular + playoffs",
    }
    return season_labels.get(season_type, season_type)


def _format_chart_season_label(value: str | int) -> str:
    """Format one chart-season selector value.

    Args:
        value: Raw season selector value.

    Returns:
        Human-readable label such as ``All`` or ``2024-25``.
    """
    if str(value) == "All":
        return "All"
    try:
        season_year = int(value)
        return f"{season_year}-{str(season_year + 1)[2:]}"
    except Exception:
        return str(value)


def _metric_is_era_adjusted(metric: str, stat_category: str, do_era: bool, team_mode: bool) -> bool:
    """Return whether the visible chart metric is actually era-adjusted.

    Args:
        metric: Selected y-axis metric.
        stat_category: Current category (`Skater`, `Goalie`, or `Team`).
        do_era: Whether the era toggle is enabled.
        team_mode: True when the chart is rendering team comparisons.

    Returns:
        bool: True only when the rendered metric reflects era-adjusted values.
    """
    return metric_is_era_adjusted(
        metric=metric,
        stat_category=stat_category,
        do_era=do_era,
        team_mode=team_mode,
    )


def _get_chart_era_label(metric: str, stat_category: str, do_era: bool, team_mode: bool) -> str:
    """Return a concise era-status label for the visible chart.

    Args:
        metric: Selected y-axis metric.
        stat_category: Current category (`Skater`, `Goalie`, or `Team`).
        do_era: Whether the era toggle is enabled.
        team_mode: True when the chart is rendering team comparisons.

    Returns:
        str: Era status text for player charts, or an empty string for team charts.
    """
    if team_mode:
        return ""
    return "Era adjusted" if _metric_is_era_adjusted(metric, stat_category, do_era, team_mode) else ""


def _build_chart_header(
    metric: str,
    team_mode: bool,
    games_mode: bool,
    season_type: str,
    stat_category: str,
    do_era: bool,
    selected_season: str | int = "All",
) -> str:
    """Build the chart toolbar title shown above the plot area.

    Args:
        metric: Selected y-axis metric.
        team_mode: True when the chart is rendering team comparisons.
        games_mode: True when the x-axis uses games played instead of age.
        season_type: Selected season scope string from the UI.
        stat_category: Current category (`Skater`, `Goalie`, or `Team`).
        do_era: Whether the era toggle is enabled.
        selected_season: Selected chart-season value.

    Returns:
        str: Title text such as ``Points by Age · Regular season · Era adjusted``.
    """
    x_label = _get_chart_context_label(
        team_mode=team_mode,
        games_mode=games_mode,
        selected_season=selected_season,
    )
    season_label = _get_chart_season_label(season_type)
    era_label = _get_chart_era_label(metric, stat_category, do_era, team_mode)
    header_verb = "at" if (not team_mode and games_mode and str(selected_season) != "All") else "by"
    header_parts = [f"{metric} {header_verb} {x_label}", season_label]
    if str(selected_season) != "All":
        header_parts.insert(1, _format_chart_season_label(selected_season))
    if era_label:
        header_parts.append(era_label)
    return " · ".join(header_parts)


def _build_chart_toolbar_markup(title: str, share_button_id: str, toolbar_id: str) -> str:
    """Build the HTML toolbar shown above the chart.

    Args:
        title: Visible chart title.
        share_button_id: DOM id for the copy-link button.
        toolbar_id: DOM id for the toolbar wrapper.

    Returns:
        str: Safe toolbar HTML.
    """
    safe_title = escape(title)
    return (
        f"<div id='{toolbar_id}' class='nhl-chart-toolbar'>"
        f"<div class='nhl-chart-toolbar__title'>{safe_title}</div>"
        f"<button id='{share_button_id}' type='button' class='nhl-chart-share-btn' aria-label='Copy share link'>"
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
        "<path d='M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71'/><path d='M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71'/></svg>"
        "<span class='nhl-chart-share-btn__label'>Copy link</span>"
        "</button>"
        "</div>"
    )


def _build_chart_axis_cue_annotations(
    metric: str,
    team_mode: bool,
    games_mode: bool,
    selected_season: str | int = "All",
) -> list[dict]:
    """Build subtle in-chart axis cue annotations.

    Args:
        metric: Selected y-axis metric.
        team_mode: True when the chart is rendering team comparisons.
        games_mode: True when the x-axis uses games played instead of age.
        selected_season: Selected chart-season value.

    Returns:
        list[dict]: Plotly annotation dictionaries for y/x context cues.
    """
    x_label = _get_chart_context_label(
        team_mode=team_mode,
        games_mode=games_mode,
        selected_season=selected_season,
    )
    return [
        dict(
            x=0.07,
            y=1.017,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            text=escape(metric),
            showarrow=False,
            font=dict(size=15, family="Arial Black", color=Y_AXIS_CUE_COLOR),
        ),
        dict(
            x=0.988,
            y=0.042,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="bottom",
            text=escape(x_label),
            showarrow=False,
            font=dict(size=15, family="Arial Black", color=X_AXIS_CUE_COLOR),
        ),
        dict(
            x=0.5,
            y=1.017,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            text="Click on chart for details",
            showarrow=False,
            font=dict(size=15, family="Arial", color=Y_AXIS_CUE_COLOR),
        ),
    ]


def _slugify_chart_export_name(title: str) -> str:
    """Convert the chart title into a stable download filename.

    Args:
        title: Human-readable chart title.

    Returns:
        str: Lowercase filesystem-friendly slug.
    """
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in title)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "nhl_age_chart"


def _is_baseline_trace(trace_name: str) -> bool:
    """Return whether a Plotly trace name represents a baseline series.

    Args:
        trace_name: Visible Plotly trace label.

    Returns:
        bool: True when the trace is a baseline line.
    """
    return "baseline" in trace_name.casefold()


def _apply_special_trace_styling(fig: go.Figure, player_colors: dict[str, str | None]) -> None:
    """Reapply baseline and projection styling after shared trace updates.

    Args:
        fig: Plotly figure to mutate in place.
        player_colors: Mapping of real-player names to their Plotly colors.

    Returns:
        None.
    """
    for trace in fig.data:
        if "(Proj)" in trace.name:
            player_name = trace.name.replace(" (Proj)", "")
            proj_color = player_colors.get(player_name) or "gray"
            trace.legendgroup = player_name
            trace.showlegend = False
            trace.line.dash = 'dot'
            trace.line.width = PROJECTION_LINE_WIDTH
            trace.line.color = _with_alpha(proj_color, PROJECTION_LINE_OPACITY)
            trace.marker.color = proj_color
            trace.marker.line.width = 0
            trace.marker.symbol = 'circle'
        elif _is_baseline_trace(trace.name):
            trace.legendgroup = trace.name
            trace.showlegend = False
            trace.line.dash = BASELINE_LINE_DASH
            trace.line.color = _with_alpha(BASELINE_LINE_COLOR, BASELINE_LINE_OPACITY)
            trace.line.width = 4
            trace.marker.size = 8
            trace.marker.color = BASELINE_MARKER_COLOR
            trace.marker.line.width = 0
            trace.marker.symbol = 'circle'
            trace.hovertemplate = str(getattr(trace, "hovertemplate", "")).replace(
                "<b>Click for details</b><br>", ""
            )
            trace.selected = dict(
                marker=dict(size=8, color=BASELINE_MARKER_COLOR, opacity=1.0)
            )
            trace.unselected = dict(marker=dict(opacity=1.0))


def _with_alpha(color: str | None, alpha: float) -> str:
    """Return one color string rewritten with a different alpha channel.

    Args:
        color: Plotly color string, usually hex or rgb/rgba.
        alpha: Target opacity between 0 and 1.

    Returns:
        str: RGBA color string with the requested opacity.
    """
    if not color:
        return f"rgba(255, 255, 255, {alpha:.3f})"

    clean = str(color).strip()
    if clean.startswith("#") and len(clean) == 7:
        red = int(clean[1:3], 16)
        green = int(clean[3:5], 16)
        blue = int(clean[5:7], 16)
        return f"rgba({red}, {green}, {blue}, {alpha:.3f})"

    if clean.startswith("rgb(") and clean.endswith(")"):
        rgb = [part.strip() for part in clean[4:-1].split(",")]
        if len(rgb) == 3:
            return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha:.3f})"

    if clean.startswith("rgba(") and clean.endswith(")"):
        rgba = [part.strip() for part in clean[5:-1].split(",")]
        if len(rgba) >= 3:
            return f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {alpha:.3f})"

    return clean


def _prepend_traces(fig: go.Figure, traces: list[go.Scatter]) -> None:
    """Insert background helper traces ahead of the visible data traces.

    Args:
        fig: Plotly figure to mutate in place.
        traces: Helper traces that should render behind the main series.

    Returns:
        None.
    """
    if not traces:
        return
    fig.add_traces(traces)
    fig.data = fig.data[-len(traces):] + fig.data[:-len(traces)]


def _build_marker_glow_traces(
    fig: go.Figure,
    player_colors: dict[str, str | None],
    glow_size: int,
    glow_opacity: float,
    trace_name: str,
    include_projection_traces: bool = False,
) -> list[go.Scatter]:
    """Build soft background glow traces behind clickable player markers.

    Args:
        fig: Figure holding the visible player traces.
        player_colors: Mapping of player names to their assigned colors.
        glow_size: Marker size for the glow-only helper traces.
        glow_opacity: Alpha value applied to the player color.
        trace_name: Internal helper-trace name.
        include_projection_traces: True to also glow projected player markers.

    Returns:
        list[go.Scatter]: Background glow-marker traces.
    """
    glow_traces: list[go.Scatter] = []
    for trace in fig.data:
        if trace.name.startswith("_") or _is_baseline_trace(trace.name):
            continue
        if "(Proj)" in trace.name and not include_projection_traces:
            continue

        player_name = trace.name.replace(" (Proj)", "")
        player_color = player_colors.get(player_name) or getattr(trace.marker, "color", None) or trace.line.color
        glow_traces.append(
            go.Scatter(
                x=trace.x,
                y=trace.y,
                mode="markers",
                marker=dict(
                    size=glow_size,
                    color=_with_alpha(player_color, glow_opacity),
                ),
                showlegend=False,
                legendgroup=trace.legendgroup or trace.name,
                hoverinfo="skip",
                name=trace_name,
            )
        )
    return glow_traces


def _build_single_season_marker_glow_traces(
    fig: go.Figure,
    player_colors: dict[str, str | None],
) -> list[go.Scatter]:
    """Build soft marker-glow traces for clickable single-season points.

    Args:
        fig: Figure holding the visible player traces.
        player_colors: Mapping of player names to their assigned colors.

    Returns:
        list[go.Scatter]: Background glow-marker traces.
    """
    return _build_marker_glow_traces(
        fig=fig,
        player_colors=player_colors,
        glow_size=SEASON_MARKER_GLOW_SIZE,
        glow_opacity=SEASON_MARKER_GLOW_OPACITY,
        trace_name="_season_marker_glow",
    )


def _build_clickable_age_marker_glow_traces(
    fig: go.Figure,
    player_colors: dict[str, str | None],
) -> list[go.Scatter]:
    """Build subtle glow traces for clickable player age-chart markers.

    Args:
        fig: Figure holding the visible player traces.
        player_colors: Mapping of player names to their assigned colors.

    Returns:
        list[go.Scatter]: Background glow-marker traces.
    """
    return _build_marker_glow_traces(
        fig=fig,
        player_colors=player_colors,
        glow_size=CLICKABLE_AGE_MARKER_GLOW_SIZE,
        glow_opacity=CLICKABLE_AGE_MARKER_GLOW_OPACITY,
        trace_name="_age_marker_glow",
        include_projection_traces=False,
    )


def _disable_click_selection_dimming(fig: go.Figure, include_projection_traces: bool = False) -> None:
    """Keep clickable chart selection from dimming the rest of the points.

    Args:
        fig: Figure holding the visible player traces.
        include_projection_traces: True to also preserve projection-point opacity.

    Returns:
        None.
    """
    for trace in fig.data:
        if trace.name.startswith("_") or _is_baseline_trace(trace.name):
            continue
        if "(Proj)" in trace.name and not include_projection_traces:
            continue

        base_marker = getattr(trace, "marker", None)
        base_size = getattr(base_marker, "size", None)
        base_color = getattr(base_marker, "color", None) or getattr(trace.line, "color", None)
        selected_marker = {"opacity": 1.0}
        if base_size is not None:
            selected_marker["size"] = base_size
        if base_color is not None:
            selected_marker["color"] = _with_alpha(base_color, 1.0)

        trace.selected = dict(marker=selected_marker)
        trace.unselected = dict(marker=dict(opacity=1.0))


def _build_single_season_lollipop_stem_traces(
    fig: go.Figure,
    player_colors: dict[str, str | None],
) -> list[go.Scatter]:
    """Build one vertical-stem trace per player for raw single-season mode.

    Args:
        fig: Figure holding the visible player traces.
        player_colors: Mapping of player names to their assigned colors.

    Returns:
        list[go.Scatter]: Background lollipop stem traces.
    """
    stem_traces: list[go.Scatter] = []
    for trace in fig.data:
        if trace.name.startswith("_") or "(Proj)" in trace.name or _is_baseline_trace(trace.name):
            continue

        stem_x: list[float | int | None] = []
        stem_y: list[float | int | None] = []
        for x_val, y_val in zip(trace.x if trace.x is not None else [], trace.y if trace.y is not None else []):
            if x_val is None or y_val is None or pd.isna(y_val):
                continue
            stem_x.extend([x_val, x_val, None])
            stem_y.extend([0, y_val, None])

        if not stem_x:
            continue

        player_color = player_colors.get(trace.name) or getattr(trace.marker, "color", None) or trace.line.color
        stem_traces.append(
            go.Scatter(
                x=stem_x,
                y=stem_y,
                mode="lines",
                line=dict(color=_with_alpha(player_color, 0.36), width=SEASON_STEM_WIDTH),
                showlegend=False,
                legendgroup=trace.legendgroup or trace.name,
                hoverinfo="skip",
                name="_season_lollipop_stems",
            )
        )
    return stem_traces


def render_chart(
    processed_dfs: list,
    metric: str,
    team_mode: bool,
    games_mode: bool,
    do_cumul: bool,
    do_base: bool,
    do_smooth: bool,
    stat_category: str,
    historical_baselines: dict,
    team_baselines: dict,
    raw_dfs_cache: list,
    ml_clones_dict: dict,
    season_type: str,
    sidebar_keys: dict,
    peak_info: dict = None,
    do_prime: bool = False,
    do_era: bool = False,
    selected_season: str | int = "All",
    share_params: dict | None = None,
    suppress_dialogs: bool = False,
) -> None:
    """Build the Plotly chart, optional baseline overlays, and click handling.

    Args:
        processed_dfs: Visible processed series ready for chart rendering.
        metric: Active y-axis metric.
        team_mode: Whether the chart is rendering team data instead of players.
        games_mode: Whether the x-axis is game number / games played.
        do_cumul: Whether cumulative counting stats are active.
        do_base: Whether baseline overlays should be rendered.
        do_smooth: Whether smoothing is active for eligible traces.
        stat_category: Active category (`Skater`, `Goalie`, or `Team`).
        historical_baselines: Cached historical baseline payloads.
        team_baselines: Cached team baseline payloads.
        raw_dfs_cache: Raw player dataframes used by Season Snapshot dialogs.
        ml_clones_dict: Cached projection clone payloads for dialogs.
        season_type: Active season scope (`Regular`, `Playoffs`, `Both`).
        sidebar_keys: Sidebar cache-busting values that feed the chart widget
            key so the JS click bridge stays aligned with the visible board.
        peak_info: Optional player peak-highlight payloads.
        do_prime: Whether prime-year highlighting is enabled.
        do_era: Whether era adjustment is active for the visible metric.
        selected_season: Canonical chart-season selection.
        share_params: Compact share-link params for the Copy link control.
        suppress_dialogs: When True, the chart consumes the current bridge
            click, records its nonce as handled, and skips opening a dialog.
            This guardrail keeps queued player-card or matchup-history modals
            from colliding with chart dialogs in the same rerun.
    """
    _store_player_chart_colors({})
    if peak_info is None:
        peak_info = {}
    if not processed_dfs:
        # Allow baseline-only render when Show Baseline is on in player mode
        if not (do_base and not games_mode and not team_mode):
            if team_mode:
                st.info("Add teams from the sidebar to compare their historical performance.")
            else:
                st.info("Add players from the sidebar to get started.")
            return

    final_df = pd.concat(processed_dfs, ignore_index=True) if processed_dfs else pd.DataFrame()

    # ------------------------------------------------------------------
    # Baseline overlay
    # ------------------------------------------------------------------
    if do_base and not games_mode:
        if team_mode:
            # Team baseline: 75th pct of all teams per season year
            base_data = []
            for _sy in sorted(team_baselines.keys()):
                _val = team_baselines[_sy].get(metric)
                if _val is not None and not pd.isna(_val):
                    base_data.append({
                        "SeasonYear": _sy,
                        metric:        _val,
                        "Player":      "NHL Team 75th Pct Baseline",
                        "BaseName":    "Baseline",
                    })
            if base_data:
                final_df = pd.concat([final_df, pd.DataFrame(base_data)], ignore_index=True)
        else:
            base_rows = []
            base_key = 'Goalie' if stat_category == 'Goalie' else 'Skater'
            base_df = historical_baselines.get(base_key)
            base_label = 'Reference baseline'
            counting_metrics = {
                'Points', 'Goals', 'Assists', 'Wins', 'Shutouts', 'GP', 'PIM', 'Saves', '+/-'
            }
            if base_df is not None and not base_df.empty and metric in base_df.columns:
                values = base_df[metric].sort_index().reindex(range(18, 41)).astype(float)
                values = values.interpolate(method='linear')
                cumulative_value = 0.0
                for age, val in values.items():
                    if pd.isna(val):
                        continue
                    plot_val = float(val)
                    if do_cumul and metric in counting_metrics:
                        cumulative_value += plot_val
                        plot_val = cumulative_value
                    base_rows.append({
                        'Age': int(age),
                        metric: plot_val,
                        'Player': base_label,
                        'BaseName': 'Baseline',
                    })

            if base_rows:
                final_df = pd.concat([final_df, pd.DataFrame(base_rows)], ignore_index=True)

    # Guard: baseline attempt may still leave final_df empty (e.g. switching to Goalie
    # mode with no goalies loaded and a metric that has no baseline data).
    if final_df.empty:
        st.info("Add players from the sidebar to get started.")
        return

    # Team season-year hover should display season spans (e.g., 2024 -> 24-25)
    if team_mode and not games_mode and "SeasonYear" in final_df.columns:
        def _season_span_label(start_year: float) -> str:
            """Format one season-start year into a short hockey-season span label."""
            try:
                sy = int(start_year)
                return f"{str(sy)[2:]}-{str(sy + 1)[2:]}"
            except Exception:
                return str(start_year)

        final_df["SeasonLabel"] = final_df["SeasonYear"].apply(_season_span_label)

    # ------------------------------------------------------------------
    # Determine x-axis column and custom_data columns
    # ------------------------------------------------------------------
    is_single_season_mode = games_mode and str(selected_season) != "All"
    is_single_season_player_games = is_single_season_mode and not team_mode
    is_single_season_team_games = is_single_season_mode and team_mode
    has_exact_game_custom_data = (
        is_single_season_player_games
        and all(col in final_df.columns for col in ["GameId", "GameDate", "GameType"])
    )
    has_exact_team_game_custom_data = (
        is_single_season_team_games
        and all(
            col in final_df.columns
            for col in [
                "GameId",
                "GameDate",
                "GameType",
                "OpponentAbbrev",
                "OpponentName",
                "HomeRoadFlag",
                "Goals",
                "GoalsAgainst",
                "ResultLabel",
                "RecordLabel",
            ]
        )
    )

    if team_mode and not games_mode:
        x_col       = "SeasonYear"
        custom_cols = ["BaseName", "Player", "SeasonLabel"]
    elif is_single_season_team_games and has_exact_team_game_custom_data:
        x_col = "CumGP"
        custom_cols = [
            "BaseName",
            "Player",
            "GameDate",
            "GameType",
            "OpponentAbbrev",
            "OpponentName",
            "HomeRoadFlag",
            "ResultLabel",
            "Goals",
            "GoalsAgainst",
            "GameId",
            "RecordLabel",
        ]
    elif games_mode:
        x_col       = "CumGP"
        if has_exact_game_custom_data:
            custom_cols = ["BaseName", "Player", "Age", "GameId", "GameDate", "GameType"]
        else:
            custom_cols = ["BaseName", "Player", "Age"] if not team_mode else ["BaseName", "Player"]
    else:
        x_col       = "Age"
        custom_cols = ["BaseName", "Player"]

    is_single_season_lollipop_mode = is_single_season_player_games and not do_cumul
    is_clickable_age_chart = not team_mode and not games_mode

    # ------------------------------------------------------------------
    # Chart widget key.
    # Keep the key stable for the visible board state so the JS click bridge
    # can use the same chart identity for nonce-tagged click payloads.
    # ------------------------------------------------------------------
    if team_mode:
        chart_key = (
            f"chart_team_{hash(str(st.session_state.teams))}"
            f"_{metric}_{st.session_state.do_smooth}_{st.session_state.x_axis_mode}"
            f"_{selected_season}_{season_type}"
        )
    else:
        player_names = [df['BaseName'].iloc[0] for df in processed_dfs if not df.empty]
        chart_key = (
            f"chart_{hash(str(player_names))}"
            f"_{metric}_{st.session_state.do_predict}_{st.session_state.do_smooth}"
            f"_{sidebar_keys.get('search_term', '')}"
            f"_{sidebar_keys.get('top_selected', '')}"
            f"_{sidebar_keys.get('team_abbr', '')}"
            f"_{sidebar_keys.get('roster_player', '')}"
            f"_{st.session_state.x_axis_mode}"
            f"_{selected_season}"
        )

    # ------------------------------------------------------------------
    # Data bounds for axis range constraints and JS pan clamping
    # ------------------------------------------------------------------
    _x_vals = final_df[x_col].dropna()
    _y_vals = final_df[metric].dropna()
    _x_pad  = (_x_vals.max() - _x_vals.min()) * 0.02 + 0.5
    _y_pad  = float(_y_vals.max()) * 0.05
    _x_min  = float(_x_vals.min()) - _x_pad
    _x_max  = float(_x_vals.max()) + _x_pad
    _y_min  = max(0.0, float(_y_vals.min()))
    _y_max  = float(_y_vals.max()) + _y_pad
    _is_age_mode   = (x_col == "Age")
    _is_games_mode = games_mode

    # Full data range for clamping
    _x_full_range = float(_x_vals.max() - _x_vals.min())

    # Initial 20-year zoom for team mode (users can double-click to see full history)
    _team_initial_range = None
    if team_mode and not games_mode and _x_full_range > 20:
        _max_year = float(_x_vals.max())
        _zoom_min = _max_year - 20
        _zoom_max = _max_year + (_x_pad * 0.5)  # Small padding on the right
        _team_initial_range = [_zoom_min, _zoom_max] if _zoom_min >= _x_min else None

    # Python-side dtick for team/season-year mode (applied immediately, before JS lands)
    # Use zoomed range size if applicable, otherwise full range
    if _team_initial_range:
        _x_range_size = _team_initial_range[1] - _team_initial_range[0]
    else:
        _x_range_size = _x_full_range

    if _x_range_size <= 25:
        _team_dtick = 2
    elif _x_range_size <= 50:
        _team_dtick = 5
    elif _x_range_size <= 100:
        _team_dtick = 10
    else:
        _team_dtick = 20

    chart_header = _build_chart_header(
        metric=metric,
        team_mode=team_mode,
        games_mode=games_mode,
        season_type=season_type,
        stat_category=stat_category,
        do_era=do_era,
        selected_season=selected_season,
    )
    chart_axis_cues = _build_chart_axis_cue_annotations(
        metric=metric,
        team_mode=team_mode,
        games_mode=games_mode,
        selected_season=selected_season,
    )
    trace_color_map = _build_trace_color_map(
        final_df=final_df,
        stat_category=stat_category,
        team_mode=team_mode,
    )
    plotly_color_map = _build_plotly_color_map(final_df, trace_color_map)

    # ------------------------------------------------------------------
    # Build Plotly figure
    # ------------------------------------------------------------------
    if is_single_season_lollipop_mode:
        fig = px.scatter(
            final_df,
            x           = x_col,
            y           = metric,
            color       = "Player",
            color_discrete_map = plotly_color_map,
            custom_data = custom_cols,
            template    = "plotly_dark",
        )
    else:
        fig = px.line(
            final_df,
            x           = x_col,
            y           = metric,
            color       = "Player",
            color_discrete_map = plotly_color_map,
            custom_data = custom_cols,
            markers     = True,
            template    = "plotly_dark",
            line_shape  = "spline" if do_smooth else "linear",
        )

    # Apply visual conventions per trace
    player_colors = dict(trace_color_map)  # Map player name -> explicit chart color
    proj_traces = []  # Store (x, y, color, legendgroup) for projection glow traces

    # First pass: wire legend groups and capture projection data
    for trace in fig.data:
        if "(Proj)" not in trace.name and not _is_baseline_trace(trace.name):
            assigned = player_colors.get(trace.name)
            if assigned:
                trace.line.color = assigned
                trace.marker.color = assigned
            trace.legendgroup = trace.name
        elif _is_baseline_trace(trace.name):
            trace.legendgroup = trace.name
    
    for trace in fig.data:
        if "(Proj)" in trace.name:
            # Extract player name from projection (e.g., "Sebastian Aho (Proj)" -> "Sebastian Aho")
            player_name = trace.name.replace(" (Proj)", "")
            proj_color = player_colors.get(player_name) if player_colors.get(player_name) else 'gray'
            proj_traces.append({
                'x': trace.x,
                'y': trace.y,
                'color': proj_color,
                'legendgroup': player_name,
            })

    _store_player_chart_colors(player_colors)

    _apply_special_trace_styling(fig, player_colors)

    # Add glow traces for projection lines (use player's color for each projection)
    for proj in proj_traces:
        if proj['x'] is not None and proj['y'] is not None:
            # Outer glow
            fig.add_trace(go.Scatter(
                x=proj['x'], y=proj['y'],
                mode='lines',
                line=dict(
                    color=_with_alpha(proj['color'], PROJECTION_GLOW_OUTER_OPACITY),
                    width=PROJECTION_GLOW_OUTER_WIDTH,
                    dash='dot',
                ),
                showlegend=False,
                legendgroup=proj['legendgroup'],
                hoverinfo='skip',
                name='_proj_glow_outer',
            ))
            # Inner glow
            fig.add_trace(go.Scatter(
                x=proj['x'], y=proj['y'],
                mode='lines',
                line=dict(
                    color=_with_alpha(proj['color'], PROJECTION_GLOW_INNER_OPACITY),
                    width=PROJECTION_GLOW_INNER_WIDTH,
                    dash='dot',
                ),
                showlegend=False,
                legendgroup=proj['legendgroup'],
                hoverinfo='skip',
                name='_proj_glow_inner',
            ))

    # Add exact peak-age/game highlights (single x-axis bucket only)
    if do_prime and not team_mode:
        for player_name, peak_data in peak_info.items():
            player_color = player_colors.get(player_name)
            peak_x = peak_data.get('x') if games_mode else peak_data.get('age')
            if player_color and peak_x is not None:
                fig.add_vrect(
                    x0=peak_x - 0.5,
                    x1=peak_x + 0.5,
                    fillcolor=player_color,
                    opacity=0.10,
                    layer="below",
                    line_width=0,
                )

    _active_palette = list(trace_color_map.values()) or _palette_for_category(stat_category)
    fig.update_layout(
        uirevision  = 'constant',
        colorway    = _active_palette,
        margin      = dict(l=6, r=6, t=18, b=6 if not team_mode else 14),
        height      = 430,
        font        = dict(size=16),
        hoverlabel  = dict(
            bgcolor     = "rgba(8, 13, 21, 0.94)",
            bordercolor = "rgba(255, 255, 255, 0.13)",
            font        = dict(size=16, family="Arial", color="rgba(255, 255, 255, 0.90)"),
            align       = "left",
            namelength  = 0,
        ),
        modebar     = dict(bgcolor="rgba(0,0,0,0)"),
        annotations = chart_axis_cues,
        title       = dict(text=""),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        showlegend  = False,
        hovermode   = "closest",
        hoverdistance = CHART_HOVER_DISTANCE,
        legend      = dict(
            title=None, orientation="h",
            yanchor="top", y=-0.20,
            xanchor="center", x=0.5,
            groupclick="togglegroup",
        ),
        clickmode   = 'event',
    )

    _val_fmt = (
        ".2f"
        if (team_mode and metric in TEAM_RATE_STATS) or (not team_mode and metric in RATE_STATS)
        else ".0f"
    )

    if team_mode and not games_mode:
        fig.update_layout(
            margin = dict(l=6, r=6, t=18, b=20),
        )
        fig.update_traces(
            connectgaps    = True,
            line           = dict(width=4, shape='spline', smoothing=0.6),
            marker         = dict(size=8),
            hovertemplate  = (
                f"<b>%{{customdata[1]}}</b><br>Season %{{customdata[2]}}<br>"
                f"%{{y:{_val_fmt}}} {metric}<extra></extra>"
            ),
        )
        fig.update_xaxes(
            title_text        = "",
            dtick             = _team_dtick,
            tickangle         = -45,
            automargin        = False,
            ticklabelposition = "inside",
            tickfont          = dict(size=16, family='Arial Black', color=X_AXIS_TICK_COLOR),
            range             = _team_initial_range,
        )
    elif games_mode:
        games_hover_label = "Game" if str(selected_season) != "All" else "Career Game"
        if is_single_season_player_games:
            fig.update_traces(
                connectgaps   = not is_single_season_lollipop_mode,
                line          = dict(width=0 if is_single_season_lollipop_mode else 4, shape='spline', smoothing=0.6),
                marker        = dict(
                    size=SEASON_MARKER_SIZE,
                    line=dict(width=1.6, color=SEASON_MARKER_OUTLINE),
                ),
                hovertemplate = (
                    f"<b>Click for details</b><br><br>"
                    f"<b>%{{customdata[1]}}</b><br>"
                    f"{games_hover_label if not team_mode else 'Season GP'} %{{x}}<br>"
                    f"%{{y:{_val_fmt}}} {metric}<extra></extra>"
                ),
            )
            if is_single_season_lollipop_mode:
                fig.update_traces(mode="markers")
        elif is_single_season_team_games:
            fig.update_traces(
                connectgaps   = True,
                line          = dict(width=4, shape='spline', smoothing=0.6),
                marker        = dict(
                    size=SEASON_MARKER_SIZE,
                    line=dict(width=1.6, color=SEASON_MARKER_OUTLINE),
                ),
                hovertemplate = (
                    "<b>Click for details</b><br><br>"
                    "<b>%{customdata[1]}</b><br>"
                    "%{customdata[2]} · %{customdata[3]}<br>"
                    "Opponent: %{customdata[5]} (%{customdata[6]})<br>"
                    "Result: %{customdata[7]} %{customdata[8]:.0f}-%{customdata[9]:.0f}<br>"
                    f"{games_hover_label} %{{x}}<br>"
                    f"%{{y:{_val_fmt}}} {metric}<extra></extra>"
                ),
            )
        else:
            fig.update_traces(
                connectgaps    = True,
                line           = dict(width=4, shape='spline', smoothing=0.6),
                marker         = dict(size=8),
                hovertemplate  = (
                    f"<b>%{{customdata[1]}}</b><br>"
                    f"{games_hover_label if not team_mode else 'Season GP'} %{{x}}<br>"
                    f"%{{y:{_val_fmt}}} {metric}<extra></extra>"
                ),
            )
        fig.update_xaxes(
            title_text        = "",
            tickangle         = 0,
            automargin        = False,
            ticklabelposition = "inside",
            tickfont          = dict(size=16, family='Arial Black', color=X_AXIS_TICK_COLOR),
        )
    else:
        fig.update_traces(
            connectgaps    = True,
            line           = dict(width=4, shape='spline', smoothing=0.6),
            marker         = dict(
                size=CLICKABLE_AGE_MARKER_SIZE,
                line=dict(width=1.35, color=CLICKABLE_AGE_MARKER_OUTLINE),
            ),
            hovertemplate  = (
                f"<b>Click for details</b><br><br><b>%{{customdata[1]}}</b><br>Age %{{x}}<br>"
                f"%{{y:{_val_fmt}}} {metric}<extra></extra>"
            ),
        )
        fig.update_xaxes(
            title_text        = "",
            tickangle         = 0,
            automargin        = False,
            ticklabelposition = "inside",
            tickfont          = dict(size=16, family='Arial Black', color=X_AXIS_TICK_COLOR),
        )

    fig.update_yaxes(
        title_text         = "",
        tickfont           = dict(size=16, family='Arial Black', color=Y_AXIS_TICK_COLOR),
        ticklabelposition  = "inside",
        automargin         = False,
    )

    # Percentage suffix for rate metrics shown as percentages
    if metric in ["Save %", "SH%", "Win%", "PP%"]:
        fig.update_yaxes(ticksuffix="%")
        if team_mode and not games_mode:
            fig.update_traces(
                hovertemplate=(
                    f"<b>%{{customdata[1]}}</b><br>Season %{{customdata[2]}}<br>%{{y:.1f}}% {metric}<extra></extra>"
                )
            )
        elif is_single_season_team_games:
            fig.update_traces(
                hovertemplate=(
                    "<b>Click for details</b><br><br>"
                    "<b>%{customdata[1]}</b><br>"
                    "%{customdata[2]} · %{customdata[3]}<br>"
                    "Opponent: %{customdata[5]} (%{customdata[6]})<br>"
                    "Result: %{customdata[7]} %{customdata[8]:.0f}-%{customdata[9]:.0f}<br>"
                    f"Game %{{x}}<br>%{{y:.1f}}% {metric}<extra></extra>"
                )
            )
        else:
            x_label = "Game" if games_mode and str(selected_season) != "All" else "Career Game" if games_mode else "Age"
            fig.update_traces(
                hovertemplate=(
                    (
                        f"<b>Click for details</b><br><br><b>%{{customdata[1]}}</b><br>{x_label} %{{x}}<br>%{{y:.1f}}% {metric}<extra></extra>"
                        if is_single_season_player_games or is_clickable_age_chart
                        else f"<b>%{{customdata[1]}}</b><br>{x_label} %{{x}}<br>%{{y:.1f}}% {metric}<extra></extra>"
                    )
                )
            )

    if is_single_season_player_games:
        helper_traces = _build_single_season_marker_glow_traces(fig, player_colors)
        if is_single_season_lollipop_mode:
            helper_traces = _build_single_season_lollipop_stem_traces(fig, player_colors) + helper_traces
        _prepend_traces(fig, helper_traces)
        _disable_click_selection_dimming(fig)
    elif is_single_season_team_games:
        _prepend_traces(fig, _build_single_season_marker_glow_traces(fig, player_colors))
        _disable_click_selection_dimming(fig)
    elif is_clickable_age_chart:
        _prepend_traces(fig, _build_clickable_age_marker_glow_traces(fig, player_colors))
        _disable_click_selection_dimming(fig, include_projection_traces=True)

    _apply_special_trace_styling(fig, player_colors)
    if not team_mode:
        fig.update_traces(showlegend=False)

    share_button_id = f"nhl-share-btn-{abs(hash(chart_key))}"
    toolbar_id = f"nhl-chart-toolbar-{abs(hash(chart_key))}"
    glow_style = _build_chart_glow_style(player_colors)
    st.markdown(
        _build_chart_toolbar_markup(chart_header, share_button_id, toolbar_id) + glow_style,
        unsafe_allow_html=True,
    )

    plotly_config = {
        "displayModeBar": True,
        "toImageButtonOptions": {
            "filename": _slugify_chart_export_name(chart_header),
            "format": "png",
            "scale": 2,
        },
        "modeBarButtonsToRemove": [
            "lasso2d", "select2d", "toggleSpikelines",
            "hoverCompareCartesian", "hoverClosestCartesian", "autoScale2d",
            "resetScale2d", "zoomIn2d", "zoomOut2d",
        ],
        "displaylogo": False,
    }

    st.markdown("<div id='comparison-main-plotly'></div>", unsafe_allow_html=True)
    st.markdown(
        "<style>"
        ".js-plotly-plot .hoverlayer .hovertext {"
        "  filter: drop-shadow(0 6px 18px rgba(0, 0, 0, 0.62));"
        "}"
        "</style>",
        unsafe_allow_html=True,
    )

    _native_selection = st.plotly_chart(
        fig,
        width           = "stretch",
        key             = chart_key,
        config          = plotly_config,
        on_select       = "rerun",
        selection_mode  = "points",
    )

    # ------------------------------------------------------------------
    # JS: responsive dtick + pan/zoom clamping
    # ------------------------------------------------------------------
    # Zoom range for JS (used for responsive dtick in season year mode)
    _x_zoom_min = _team_initial_range[0] if _team_initial_range else _x_min
    _x_zoom_max = _team_initial_range[1] if _team_initial_range else _x_max
    _share_params_json = json.dumps(share_params or {})
    _share_button_id_json = json.dumps(share_button_id)
    _toolbar_id_json = json.dumps(toolbar_id)
    _chart_instance_id_json = json.dumps(chart_key)
    _enable_player_trace_toggles = "true"

    components.html(f"""<script>
(function() {{
    var X_MIN = {_x_min:.4f};
    var X_MAX = {_x_max:.4f};
    var X_ZOOM_MIN = {_x_zoom_min:.4f};
    var X_ZOOM_MAX = {_x_zoom_max:.4f};
    var Y_MIN = {_y_min:.4f};
    var Y_MAX = {_y_max:.4f};
    var IS_AGE_MODE   = {'true' if _is_age_mode else 'false'};
    var IS_GAMES_MODE = {'true' if _is_games_mode else 'false'};
    var IS_SINGLE_SEASON_MODE = {'true' if str(selected_season) != 'All' else 'false'};
    var SHARE_PARAMS = {_share_params_json};
    var SHARE_BUTTON_ID = {_share_button_id_json};
    var TOOLBAR_ID = {_toolbar_id_json};
    var CHART_INSTANCE_ID = {_chart_instance_id_json};
    var ENABLE_PLAYER_TRACE_TOGGLES = {_enable_player_trace_toggles};

    function calcDtick(width, currentRange) {{
        var xRange = currentRange || (X_MAX - X_MIN);
        if (IS_AGE_MODE) {{
            var pixPerAge = width / xRange;
            if (pixPerAge >= 32) return 1;
            if (pixPerAge >= 16) return 2;
            if (pixPerAge >= 7)  return 5;
            return 10;
        }}
        if (IS_GAMES_MODE) {{
            if (IS_SINGLE_SEASON_MODE) {{
                var seasonTargetTicks = width >= 900 ? 9 : width >= 480 ? 6 : 4;
                var seasonDtick = xRange / seasonTargetTicks;
                if (seasonDtick <= 5) return 5;
                if (seasonDtick <= 10) return 10;
                if (seasonDtick <= 20) return 20;
                return 25;
            }}
            var targetTicks = width >= 900 ? 8 : width >= 480 ? 5 : 3;
            var rawDtick = xRange / targetTicks;
            if (rawDtick <= 100)  return 100;
            if (rawDtick <= 200)  return 200;
            if (rawDtick <= 300)  return 250;
            if (rawDtick <= 400)  return 400;
            if (rawDtick <= 750)  return 500;
            return Math.ceil(rawDtick / 500) * 500;
        }}
        // Season Year mode — use zoom range for initial calculation
        // 4-digit labels need ~50px each at typical chart widths
        var pixPerYear = width / xRange;
        if (pixPerYear >= 50) return 1;
        if (pixPerYear >= 25) return 2;
        if (pixPerYear >= 10) return 5;
        if (pixPerYear >= 5)  return 10;
        return 20;
    }}

    function calcResponsiveAxisTickFontSize(width) {{
        if (width <= 480) return 12;
        if (width <= 768) return 14;
        return 16;
    }}

    function calcResponsiveYAxisTickFontSize(width) {{
        if (width <= 480) return 11;
        if (width <= 768) return 13;
        return 16;
    }}

    function calcResponsiveYAxisTickFontFamily(width) {{
        if (width <= 768) return 'Arial';
        return 'Arial Black';
    }}

    function calcResponsiveChartHeight(width) {{
        if (width <= 480) return 340;
        if (width <= 768) return 360;
        return 430;
    }}

    function calcResponsiveYAxisCueX(width) {{
        if (width <= 480) return 0.094;
        if (width <= 768) return 0.076;
        return 0.056;
    }}

    function calcResponsiveYAxisCueY(width) {{
        if (width <= 480) return 1.000;
        if (width <= 768) return 1.006;
        return 1.012;
    }}

    function syncToolbarTitleOffset(plot, parent) {{
        var toolbar = parent.document.getElementById(TOOLBAR_ID);
        if (!toolbar) return;
        var title = toolbar.querySelector('.nhl-chart-toolbar__title');
        if (!title) return;

        var width = plot.offsetWidth || parent.innerWidth;
        if (width > 768) {{
            title.style.paddingLeft = '0px';
            return;
        }}

        var gutter = 0;
        if (plot._fullLayout && plot._fullLayout._size) {{
            gutter = plot._fullLayout._size.l || 0;
        }}
        title.style.paddingLeft = Math.max(0, Math.round(gutter)) + 'px';
    }}

    function getCurrentXRange(plot) {{
        // Get the current visible X-axis range from the plot layout
        if (plot.layout && plot.layout.xaxis) {{
            var axis = plot.layout.xaxis;
            // First try explicit range
            if (axis.range && axis.range.length === 2) {{
                return axis.range[1] - axis.range[0];
            }}
            // Fall back: compute from axis data bounds (works with auto-range)
            if (axis._fullRange && axis._fullRange.length === 2) {{
                return axis._fullRange[1] - axis._fullRange[0];
            }}
        }}
        // Ultimate fallback: use X_MAX - X_MIN from Python
        return X_MAX - X_MIN;
    }}

    function applySettings(plot, Plotly) {{
        var width = plot.offsetWidth || window.parent.innerWidth;
        // For season year mode with zoom, use zoom range for initial dtick
        // Otherwise use full data range (X_MAX - X_MIN) so all years display properly
        var initialRange = (!IS_AGE_MODE && !IS_GAMES_MODE && X_ZOOM_MAX > X_ZOOM_MIN)
            ? (X_ZOOM_MAX - X_ZOOM_MIN) : (X_MAX - X_MIN);
        var axisTickFontSize = calcResponsiveAxisTickFontSize(width);
        var yAxisTickFontSize = calcResponsiveYAxisTickFontSize(width);
        var updates = {{
            'xaxis.dtick': calcDtick(width, initialRange),
            'height': calcResponsiveChartHeight(width),
            'xaxis.tickfont.size': axisTickFontSize,
            'yaxis.tickfont.family': calcResponsiveYAxisTickFontFamily(width),
            'yaxis.tickfont.size': yAxisTickFontSize,
        }};
        updates['annotations[0].x'] = calcResponsiveYAxisCueX(width);
        updates['annotations[0].y'] = calcResponsiveYAxisCueY(width);
        updates['xaxis.tickangle'] = (IS_AGE_MODE || IS_GAMES_MODE) ? 0 : -45;
        Plotly.relayout(plot, updates).then(function() {{
            syncToolbarTitleOffset(plot, window.parent);
        }});

        // Clamp pan/zoom to data region and update dtick on zoom
        var _updating = false;
        plot.on('plotly_relayout', function(evt) {{
            if (_updating) return;

            // Handle clamping
            var clamps = {{}};
            var needsClamp  = false;
            var r0 = evt['xaxis.range[0]'], r1 = evt['xaxis.range[1]'];
            var y0 = evt['yaxis.range[0]'], y1 = evt['yaxis.range[1]'];
            if (r0 !== undefined && r0 < X_MIN) {{ clamps['xaxis.range[0]'] = X_MIN; needsClamp = true; }}
            if (r1 !== undefined && r1 > X_MAX) {{ clamps['xaxis.range[1]'] = X_MAX; needsClamp = true; }}
            if (y0 !== undefined && y0 < Y_MIN) {{ clamps['yaxis.range[0]'] = Y_MIN; needsClamp = true; }}
            if (y1 !== undefined && y1 > Y_MAX) {{ clamps['yaxis.range[1]'] = Y_MAX; needsClamp = true; }}
            if (needsClamp) {{ _updating = true; Plotly.relayout(plot, clamps); _updating = false; }}

            // For season year mode, update dtick based on current visible range
            // Handle both zoom (r0 or r1 defined) and double-click reset (both undefined)
            if (!IS_AGE_MODE && !IS_GAMES_MODE) {{
                // Check for double-click reset (both ranges are undefined)
                var isReset = (r0 === undefined && r1 === undefined);
                if (isReset || r0 !== undefined || r1 !== undefined) {{
                    var currentRange = getCurrentXRange(plot);
                    var newDtick = calcDtick(width, currentRange);
                    // Only update if dtick actually changed
                    if (plot.layout && plot.layout.xaxis && plot.layout.xaxis.dtick !== newDtick) {{
                        _updating = true;
                        Plotly.relayout(plot, {{'xaxis.dtick': newDtick}});
                        _updating = false;
                    }}
                }}
            }}
        }});
    }}

    function resolveLiveShareButton(parent) {{
        var btn = parent.document.getElementById(SHARE_BUTTON_ID);
        if (btn) return btn;

        var toolbar = parent.document.getElementById(TOOLBAR_ID);
        if (toolbar) {{
            var toolbarBtn = toolbar.querySelector('.nhl-chart-share-btn');
            if (toolbarBtn) return toolbarBtn;
        }}

        var buttons = parent.document.querySelectorAll('.nhl-chart-share-btn');
        if (buttons && buttons.length) {{
            return buttons[buttons.length - 1];
        }}

        return null;
    }}

    function bindShareButton(parent, attemptsLeft) {{
        var remainingAttempts = typeof attemptsLeft === 'number' ? attemptsLeft : 12;
        var btn = resolveLiveShareButton(parent);
        if (!btn) {{
            if (remainingAttempts > 0) {{
                setTimeout(function() {{
                    bindShareButton(parent, remainingAttempts - 1);
                }}, 150);
            }}
            return;
        }}
        var label = btn.querySelector('.nhl-chart-share-btn__label');

        function buildShareUrl(parent) {{
            var UrlCtor = parent.URL || URL;
            var SearchParamsCtor = parent.URLSearchParams || URLSearchParams;
            var url = new UrlCtor(parent.location.href);
            var searchParams = new SearchParamsCtor();

            Object.entries(SHARE_PARAMS).forEach(function(entry) {{
                var key = entry[0];
                var value = entry[1];
                if (value === null || value === undefined) return;
                var clean = String(value);
                if (!clean.length) return;
                searchParams.set(String(key), clean);
            }});

            url.search = searchParams.toString();
            return url.toString();
        }}

        function fallbackCopy(parent, url) {{
            var textArea = parent.document.createElement('textarea');
            textArea.value = url;
            textArea.setAttribute('readonly', '');
            textArea.style.position = 'fixed';
            textArea.style.left = '-9999px';
            parent.document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            var copied = false;
            try {{
                copied = parent.document.execCommand('copy');
            }} catch (err) {{
                copied = false;
            }}
            parent.document.body.removeChild(textArea);
            return copied;
        }}

        function showCopiedState() {{
            btn.classList.add('is-copied');
            if (label) label.textContent = 'Copied';
            setTimeout(function() {{
                btn.classList.remove('is-copied');
                if (label) label.textContent = 'Copy link';
            }}, 1400);
        }}

        function showCopyFailure() {{
            btn.classList.remove('is-copied');
            if (label) label.textContent = 'Copy failed';
            setTimeout(function() {{
                if (label) label.textContent = 'Copy link';
            }}, 1600);
        }}

        btn.onclick = function() {{
            var url = buildShareUrl(parent);
            var syncUrl = function() {{
                if (parent.history && parent.history.replaceState) {{
                    parent.history.replaceState(null, '', url);
                }}
            }};
            var succeed = function() {{
                syncUrl();
                showCopiedState();
            }};
            var fail = function() {{
                syncUrl();
                showCopyFailure();
            }};

            if (parent.navigator && parent.navigator.clipboard && parent.navigator.clipboard.writeText) {{
                parent.navigator.clipboard.writeText(url).then(succeed).catch(function() {{
                    if (fallbackCopy(parent, url)) {{
                        succeed();
                        return;
                    }}
                    fail();
                }});
                return;
            }}

            if (fallbackCopy(parent, url)) {{
                succeed();
                return;
            }}
            fail();
        }};
    }}

    function getChartTraceToggleState(parent) {{
        if (!parent.__nhlAgeChartTraceToggleState) {{
            parent.__nhlAgeChartTraceToggleState = {{}};
        }}
        if (!parent.__nhlAgeChartTraceToggleState[CHART_INSTANCE_ID]) {{
            parent.__nhlAgeChartTraceToggleState[CHART_INSTANCE_ID] = {{}};
        }}
        return parent.__nhlAgeChartTraceToggleState[CHART_INSTANCE_ID];
    }}

    function getCurrentTargetPlot(parent) {{
        var plots = Array.prototype.slice.call(parent.document.querySelectorAll('.js-plotly-plot'));
        if (!plots.length) return null;

        var anchor = parent.document.getElementById('comparison-main-plotly');
        if (!anchor || !anchor.compareDocumentPosition || !parent.Node) {{
            return plots[plots.length - 1];
        }}

        for (var i = 0; i < plots.length; i += 1) {{
            if (anchor.compareDocumentPosition(plots[i]) & parent.Node.DOCUMENT_POSITION_FOLLOWING) {{
                return plots[i];
            }}
        }}

        return plots[plots.length - 1];
    }}

    function getLegendGroupTraceIndices(plot, legendgroup) {{
        var indices = [];
        if (!plot || !plot.data) return indices;
        plot.data.forEach(function(trace, index) {{
            if (!trace) return;
            var traceGroup = String(trace.legendgroup || trace.name || '');
            if (traceGroup !== legendgroup) return;
            if (String(trace.name || '').toLowerCase().indexOf('baseline') !== -1) return;
            indices.push(index);
        }});
        return indices;
    }}

    function isLegendGroupVisible(plot, legendgroup) {{
        var indices = getLegendGroupTraceIndices(plot, legendgroup);
        if (!indices.length) return true;
        return indices.some(function(index) {{
            var trace = plot.data[index];
            return !trace || trace.visible === undefined || trace.visible === true;
        }});
    }}

    function syncPlayerTraceToggleButtons(parent, plot) {{
        var buttons = parent.document.querySelectorAll('[data-nhl-trace-toggle="1"]');
        var state = getChartTraceToggleState(parent);
        buttons.forEach(function(btn) {{
            var legendgroup = btn.getAttribute('data-legendgroup') || '';
            if (!legendgroup) return;
            var isVisible = isLegendGroupVisible(plot, legendgroup);
            if (state[legendgroup] === undefined) {{
                state[legendgroup] = isVisible;
            }}
            btn.setAttribute('aria-pressed', isVisible ? 'true' : 'false');
            btn.dataset.visible = isVisible ? 'true' : 'false';
            btn.classList.toggle('is-inactive', !isVisible);
        }});
    }}

    function setLegendGroupVisible(parent, Plotly, plot, legendgroup, isVisible) {{
        if (!plot || String(legendgroup || '').toLowerCase().indexOf('baseline') !== -1) {{
            return Promise.resolve();
        }}
        var indices = getLegendGroupTraceIndices(plot, legendgroup);
        if (!indices.length) return Promise.resolve();
        var state = getChartTraceToggleState(parent);
        state[legendgroup] = !!isVisible;
        return Plotly.restyle(plot, {{visible: isVisible ? true : 'legendonly'}}, indices).then(function() {{
            syncPlayerTraceToggleButtons(parent, plot);
        }});
    }}

    function applyStoredPlayerTraceVisibility(parent, Plotly, plot) {{
        var state = getChartTraceToggleState(parent);
        var updates = Object.keys(state).map(function(legendgroup) {{
            var indices = getLegendGroupTraceIndices(plot, legendgroup);
            if (!indices.length) return Promise.resolve();
            return Plotly.restyle(plot, {{visible: state[legendgroup] ? true : 'legendonly'}}, indices);
        }});
        if (!updates.length) {{
            syncPlayerTraceToggleButtons(parent, plot);
            return Promise.resolve();
        }}
        return Promise.all(updates).then(function() {{
            syncPlayerTraceToggleButtons(parent, plot);
        }});
    }}

    function bindPlayerTraceToggleButtons(parent, Plotly) {{
        if (!ENABLE_PLAYER_TRACE_TOGGLES) return;
        parent.document.querySelectorAll('[data-nhl-trace-toggle="1"]').forEach(function(btn) {{
            btn.dataset.boundChartInstance = CHART_INSTANCE_ID;
            btn.onclick = function(evt) {{
                evt.preventDefault();
                var legendgroup = btn.getAttribute('data-legendgroup') || '';
                if (!legendgroup) return;
                var currentPlot = getCurrentTargetPlot(parent);
                if (!currentPlot) return;
                var nextVisible = !isLegendGroupVisible(currentPlot, legendgroup);
                setLegendGroupVisible(parent, Plotly, currentPlot, legendgroup, nextVisible);
            }};
        }});

        var livePlot = getCurrentTargetPlot(parent);
        if (livePlot) {{
            syncPlayerTraceToggleButtons(parent, livePlot);
        }}
    }}

    function patchHoverLabelRects(targetPlot) {{
        if (!targetPlot || targetPlot._hoverRectObserver) return;
        var hoverLayer = targetPlot.querySelector('.hoverlayer');
        if (!hoverLayer) return;
        var observer = new MutationObserver(function(mutations) {{
            mutations.forEach(function(m) {{
                m.addedNodes.forEach(function(node) {{
                    if (!node.querySelectorAll) return;
                    node.querySelectorAll('rect').forEach(function(r) {{
                        r.setAttribute('rx', '18');
                        r.setAttribute('ry', '18');
                    }});
                    if (node.tagName && node.tagName.toLowerCase() === 'rect') {{
                        node.setAttribute('rx', '18');
                        node.setAttribute('ry', '18');
                    }}
                }});
            }});
        }});
        observer.observe(hoverLayer, {{ childList: true, subtree: true }});
        targetPlot._hoverRectObserver = observer;
    }}

    function init() {{
        var parent = window.parent;
        var Plotly = parent.Plotly;
        if (!Plotly) {{ setTimeout(init, 200); return; }}
        var plots = parent.document.querySelectorAll('.js-plotly-plot');
        if (!plots.length) {{ setTimeout(init, 200); return; }}
        plots.forEach(function(p) {{ applySettings(p, Plotly); }});
        bindShareButton(parent, 12);
        var targetPlot = getCurrentTargetPlot(parent);
        if (!targetPlot) {{ setTimeout(init, 200); return; }}
        patchHoverLabelRects(targetPlot);
        if (ENABLE_PLAYER_TRACE_TOGGLES) {{
            applyStoredPlayerTraceVisibility(parent, Plotly, targetPlot).then(function() {{
                bindPlayerTraceToggleButtons(parent, Plotly);
            }});
        }}

        parent.addEventListener('resize', function() {{
            parent.document.querySelectorAll('.js-plotly-plot').forEach(function(p) {{
                var range = getCurrentXRange(p);
                var width = p.offsetWidth || parent.innerWidth;
                Plotly.relayout(p, {{
                    'xaxis.dtick': calcDtick(width, range),
                    'height': calcResponsiveChartHeight(width),
                    'xaxis.tickangle': (IS_AGE_MODE || IS_GAMES_MODE) ? 0 : -45,
                    'xaxis.tickfont.size': calcResponsiveAxisTickFontSize(width),
                    'yaxis.tickfont.family': calcResponsiveYAxisTickFontFamily(width),
                    'yaxis.tickfont.size': calcResponsiveYAxisTickFontSize(width),
                    'annotations[0].x': calcResponsiveYAxisCueX(width),
                    'annotations[0].y': calcResponsiveYAxisCueY(width),
                }}).then(function() {{
                    syncToolbarTitleOffset(p, parent);
                }});
            }});
        }});
    }}

    setTimeout(init, 500);
}})();
</script>""", height=0)

    _handle_native_chart_selection(
        _native_selection,
        fig,
        suppress_dialogs=suppress_dialogs,
        team_mode=team_mode,
        games_mode=games_mode,
        is_single_season_team_games=is_single_season_team_games,
        has_exact_game_custom_data=has_exact_game_custom_data,
        metric=metric,
        final_df=final_df,
        raw_dfs_cache=raw_dfs_cache,
        season_type=season_type,
        selected_season=selected_season,
        do_cumul=do_cumul,
        ml_clones_dict=ml_clones_dict,
        historical_baselines=historical_baselines,
        stat_category=stat_category,
        do_era=do_era,
    )
