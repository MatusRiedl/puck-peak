"""Comparison panel rendering for chart details, current standings, and predictions."""

import colorsys
from dataclasses import dataclass
from html import escape
from typing import Callable
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from nhl.constants import ACTIVE_TEAMS, RATE_STATS, TEAM_BRAND_COLORS, TEAM_FOUNDED, TEAM_RATE_STATS
from nhl.data_loaders import (
    get_current_nhl_standings,
    get_player_career_rank,
    get_player_current_team,
    get_player_headshot,
    get_player_season_rank_map,
    get_team_season_rank_map,
    get_player_roster_info,
    get_team_season_summary,
    get_team_all_time_stats,
    load_win_prob_weights,
)
from nhl.dialog import (
    show_matchup_history,
    show_player_identity_details,
    show_team_identity_details,
)
from nhl.schedule import get_upcoming_games
from nhl.stanley_cup import build_stanley_cup_board
from nhl.ui_state import (
    dialog_slot_available,
    mark_dialog_opened_this_run,
    session_state_get,
    session_state_pop,
    session_state_set,
)

_TEAM_LOGO_URL = "https://assets.nhle.com/logos/nhl/svg/{abbr}_light.svg"
_TEAM_SHORT_NAMES = {
    "ANA": "Ducks",
    "BOS": "Bruins",
    "BUF": "Sabres",
    "CGY": "Flames",
    "CAR": "Hurricanes",
    "CHI": "Blackhawks",
    "COL": "Avalanche",
    "CBJ": "Blue Jackets",
    "DAL": "Stars",
    "DET": "Red Wings",
    "EDM": "Oilers",
    "FLA": "Panthers",
    "LAK": "Kings",
    "MIN": "Wild",
    "MTL": "Canadiens",
    "NSH": "Predators",
    "NJD": "Devils",
    "NYI": "Islanders",
    "NYR": "Rangers",
    "OTT": "Senators",
    "PHI": "Flyers",
    "PIT": "Penguins",
    "SJS": "Sharks",
    "SEA": "Kraken",
    "STL": "Blues",
    "TBL": "Lightning",
    "TOR": "Maple Leafs",
    "UTA": "Hockey Club",
    "VAN": "Canucks",
    "VGK": "Golden Knights",
    "WSH": "Capitals",
    "WPG": "Jets",
}
_DEFAULT_PANEL_TAB = "overview"
_DEFAULT_PLAYER_RANK_COLOR = "#4caf50"
_CARD_CONTEXT_TEXT_COLOR = "#b3b3b3"
_PROBABILITY_BAR_LIGHTNESS = 0.36
_PROBABILITY_BAR_SATURATION = 0.68
# Keep the right-rail prediction batch modest. Cold loads above this count
# trigger noticeably more NHL API rate-limit fallbacks, which surface as
# "Estimate unavailable." cards near the bottom of the list.
_PREDICTIONS_PANEL_MATCH_LIMIT = 8
_CATEGORY_TAB_KEYS = {
    "Skater": "panel_tab_skater",
    "Goalie": "panel_tab_goalie",
    "Team": "panel_tab_team",
}
_STANDINGS_WINDOW_COLORS = {
    "Atlantic": "#3b82f6",
    "Metropolitan": "#14b8a6",
    "Central": "#f97316",
    "Pacific": "#facc15",
}
_MATCHUP_HISTORY_QUERY_KEY = "mh"
_PENDING_MATCHUP_HISTORY_SESSION_KEY = "_pending_matchup_history"
_LAST_MATCHUP_HISTORY_TRIGGER_NONCE_SESSION_KEY = "_last_matchup_history_trigger_nonce"
_LAST_IDENTITY_CARD_TRIGGER_NONCE_SESSION_KEY = "_last_identity_card_trigger_nonce"
_MATCHUP_HISTORY_CLICK_BRIDGE_JS = """
export default function(component) {
    const { setTriggerValue } = component;

    const onClick = (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }

        const link = target.closest('.live-game-card-link[data-nhl-matchup-history="1"]');
        if (!link) {
            return;
        }

        const matchup = String(link.getAttribute('data-matchup-history') || '').trim();
        if (!matchup) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        const nonce = `${Date.now()}-${Math.floor(Math.random() * 1000000)}`;
        setTriggerValue('clicked', `${matchup}|${nonce}`);
    };

    document.addEventListener('click', onClick, true);

    return () => {
        document.removeEventListener('click', onClick, true);
    };
}
"""
_MATCHUP_HISTORY_CLICK_BRIDGE = st.components.v2.component(
    "comparison_matchup_history_click_bridge",
    js=_MATCHUP_HISTORY_CLICK_BRIDGE_JS,
)
_IDENTITY_CARD_CLICK_BRIDGE_JS = """
export default function(component) {
    const { setTriggerValue } = component;

    const getClickableShell = (target) => {
        if (!(target instanceof Element)) {
            return null;
        }
        return target.closest('.comparison-card-shell--clickable[data-nhl-identity-card="1"]');
    };

    const emitTrigger = (shell, event) => {
        if (!shell) {
            return;
        }
        const payload = String(shell.getAttribute('data-identity-card') || '').trim();
        if (!payload) {
            return;
        }
        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }
        const nonce = `${Date.now()}-${Math.floor(Math.random() * 1000000)}`;
        setTriggerValue('clicked', `${payload}|${nonce}`);
    };

    const onClick = (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        if (target.closest('[data-nhl-trace-toggle="1"]')) {
            return;
        }
        emitTrigger(getClickableShell(target), event);
    };

    const onKeyDown = (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        if (target.closest('[data-nhl-trace-toggle="1"]')) {
            return;
        }
        const shell = getClickableShell(target);
        if (!shell) {
            return;
        }
        if (event.key !== 'Enter' && event.key !== ' ') {
            return;
        }
        emitTrigger(shell, event);
    };

    document.addEventListener('click', onClick, true);
    document.addEventListener('keydown', onKeyDown, true);

    return () => {
        document.removeEventListener('click', onClick, true);
        document.removeEventListener('keydown', onKeyDown, true);
    };
}
"""
_IDENTITY_CARD_CLICK_BRIDGE = st.components.v2.component(
    "comparison_identity_card_click_bridge",
    js=_IDENTITY_CARD_CLICK_BRIDGE_JS,
)


def _normalize_probability_color(hex_color: str) -> str:
    """Normalize a bar color to the same muted intensity.

    Args:
        hex_color: Hex color string in #RRGGBB format.

    Returns:
        Muted hex color string with normalized lightness and saturation.
    """
    if len(hex_color) != 7 or not hex_color.startswith("#"):
        return hex_color
    try:
        red = int(hex_color[1:3], 16) / 255
        green = int(hex_color[3:5], 16) / 255
        blue = int(hex_color[5:7], 16) / 255
    except ValueError:
        return hex_color
    hue, _, _ = colorsys.rgb_to_hls(red, green, blue)
    norm_red, norm_green, norm_blue = colorsys.hls_to_rgb(
        hue,
        _PROBABILITY_BAR_LIGHTNESS,
        _PROBABILITY_BAR_SATURATION,
    )
    return "#{:02X}{:02X}{:02X}".format(
        round(norm_red * 255),
        round(norm_green * 255),
        round(norm_blue * 255),
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color to rgba() text for CSS custom properties."""
    clamped_alpha = max(0.0, min(alpha, 1.0))
    if len(hex_color) != 7 or not hex_color.startswith("#"):
        return f"rgba(255, 255, 255, {clamped_alpha:.3f})"
    try:
        red = int(hex_color[1:3], 16)
        green = int(hex_color[3:5], 16)
        blue = int(hex_color[5:7], 16)
    except ValueError:
        return f"rgba(255, 255, 255, {clamped_alpha:.3f})"
    return f"rgba({red}, {green}, {blue}, {clamped_alpha:.3f})"


def _get_team_probability_background(team_abbr: str, fallback_color: str) -> str:
    """Return a brand-matched solid color for one matchup bar segment.

    Args:
        team_abbr: Three-letter team abbreviation.
        fallback_color: Hex color used if the team is unknown.

    Returns:
        CSS color text safe for an inline style attribute.
    """
    primary_color = TEAM_BRAND_COLORS.get(str(team_abbr or "").upper(), (fallback_color, fallback_color))[0]
    return escape(_normalize_probability_color(primary_color), quote=True)


def _get_raw_team_color(team_abbr: str, fallback_color: str) -> str:
    """Return the raw (non-normalized) brand primary color for a team.

    Args:
        team_abbr: Three-letter NHL team abbreviation.
        fallback_color: Hex color used if the team is unknown.

    Returns:
        Raw hex color string for the team's primary brand color.
    """
    return TEAM_BRAND_COLORS.get(str(team_abbr or "").upper(), (fallback_color, fallback_color))[0]


def _build_probability_segment_style(
    width_pct: int,
    hex_color: str,
    *,
    is_leading: bool,
    is_tied: bool,
    edge_strength: float,
) -> str:
    """Build inline CSS vars for one probability bar segment."""
    if is_tied:
        opacity = 0.94
        brightness = 1.03
        saturation = 1.00
        sheen_alpha = 0.11
        glow_alpha = 0.10
    elif is_leading:
        opacity = 1.00
        brightness = 1.10 + (edge_strength * 0.16)
        saturation = 1.08 + (edge_strength * 0.18)
        sheen_alpha = 0.18 + (edge_strength * 0.06)
        glow_alpha = 0.16 + (edge_strength * 0.18)
    else:
        opacity = max(0.56, 0.82 - (edge_strength * 0.18))
        brightness = max(0.78, 0.92 - (edge_strength * 0.10))
        saturation = max(0.82, 0.96 - (edge_strength * 0.12))
        sheen_alpha = 0.05
        glow_alpha = 0.00
    style = (
        f"width:{width_pct}%;"
        f"--segment-color:{hex_color};"
        f"--segment-glow:{_hex_to_rgba(hex_color, glow_alpha)};"
        f"--segment-opacity:{opacity:.3f};"
        f"--segment-brightness:{brightness:.3f};"
        f"--segment-saturation:{saturation:.3f};"
        f"--segment-sheen:{sheen_alpha:.3f};"
    )
    return escape(style, quote=True)


def _build_probability_label_style(
    hex_color: str,
    *,
    is_leading: bool,
    is_tied: bool,
    edge_strength: float,
) -> str:
    """Build inline CSS vars for one probability label."""
    if is_tied:
        opacity = 0.95
        glow_alpha = 0.08
    elif is_leading:
        opacity = 1.00
        glow_alpha = 0.15 + (edge_strength * 0.10)
    else:
        opacity = max(0.62, 0.80 - (edge_strength * 0.12))
        glow_alpha = 0.00
    style = (
        f"--label-opacity:{opacity:.3f};"
        f"--label-glow:{_hex_to_rgba(hex_color, glow_alpha)};"
    )
    return escape(style, quote=True)


@dataclass(frozen=True)
class PanelTabSpec:
    """Definition of one comparison panel tab."""

    id: str
    label: str
    render_player: Callable[..., None]
    render_team: Callable[..., None]


def _season_span_label_from_id(season_id: int | None) -> str:
    """Convert seasonId (e.g. 20222023) to a short span (2022-23)."""
    if season_id is None:
        return "?"
    try:
        raw = int(season_id)
        raw_str = str(raw)
        if len(raw_str) >= 8:
            start = int(raw_str[:4])
        else:
            start = raw
        return f"{start}-{str(start + 1)[2:]}"
    except Exception:
        return "?"


def _format_chart_season_label(value: str | int) -> str:
    """Format one chart-season selector value for the comparison panel.

    Args:
        value: Raw season selector value.

    Returns:
        Human-readable label such as ``All`` or ``2024-25``.
    """
    if str(value) == "All":
        return "Whole career"
    try:
        season_year = int(value)
        return f"{season_year}-{str(season_year + 1)[2:]}"
    except Exception:
        return str(value)


def _is_selected_season_mode(selected_season: str | int) -> bool:
    """Return whether the comparison panel is in single-season mode.

    Args:
        selected_season: Selected chart-season value.

    Returns:
        True when one specific season is selected.
    """
    return str(selected_season) != "All"


def _build_selected_season_scope_label(selected_season: str | int, season_type: str) -> str:
    """Build the selected-season scope copy for Overview cards.

    Args:
        selected_season: Selected chart-season value.
        season_type: Regular, Playoffs, or Both.

    Returns:
        Compact scope label such as ``2024-25 · Regular season game log``.
    """
    season_scope_map = {
        "Regular": "Regular season",
        "Playoffs": "Playoffs",
        "Both": "Regular + playoffs",
    }
    return (
        f"{_season_span_label_from_id(selected_season)}"
        f" · {season_scope_map.get(season_type, season_type)} game log"
    )


def _build_selected_season_rank_label(
    selected_season: str | int,
    season_type: str,
    metric: str,
    rank: int,
) -> str:
    """Build the selected-season leaderboard copy for Overview cards.

    Args:
        selected_season: Selected chart-season value.
        season_type: Regular, Playoffs, or Both.
        metric: Active chart metric.
        rank: 1-based league rank for the selected player.

    Returns:
        Compact leaderboard label such as ``#2 in 2024-25 Points``.
    """
    season_prefix_map = {
        'Regular': '',
        'Playoffs': 'Playoff ',
        'Both': 'Combined ',
    }
    metric_label = f"{season_prefix_map.get(season_type, '')}{metric}".strip()
    return f"#{rank} in {_season_span_label_from_id(selected_season)} {metric_label}"


def _get_visible_stat_total(real_df: pd.DataFrame, column: str, use_last_visible: bool) -> int:
    """Return a visible stat total from the active comparison rows.

    Args:
        real_df: Non-projection rows for one player.
        column: Stat column to total.
        use_last_visible: Use the last visible cumulative value instead of a sum.

    Returns:
        Integer total for the requested stat column.
    """
    if column not in real_df.columns or real_df.empty:
        return 0
    values = pd.to_numeric(real_df[column], errors="coerce").dropna()
    if values.empty:
        return 0
    total = values.iloc[-1] if use_last_visible else values.sum()
    return int(round(float(total)))


def _format_peak_metric_value(metric: str, value: float | int | None) -> str:
    """Format one peak metric value for Overview copy.

    Args:
        metric: Active metric name.
        value: Peak metric value.

    Returns:
        Readable formatted value or ``?``.
    """
    if value is None or pd.isna(value):
        return "?"
    numeric_value = float(value)
    if metric in RATE_STATS or not numeric_value.is_integer():
        return f"{numeric_value:.2f}"
    return str(int(numeric_value))


def _get_category_tab_key(stat_category: str) -> str:
    """Return the session-state key that stores the active panel tab.

    Args:
        stat_category: Active stat category string.

    Returns:
        Session-state key for the current category's comparison tab.
    """
    return _CATEGORY_TAB_KEYS.get(stat_category, "panel_tab_skater")


def _sync_chart_season_picker() -> None:
    """Keep the chart-season widget and canonical state in sync.

    Args:
        None.

    Returns:
        None.
    """
    selected_season = st.session_state.get("_chart_season_picker", "All")
    st.session_state["_chart_season_picker"] = selected_season
    st.session_state["chart_season"] = selected_season


def _prime_chart_season_picker(chart_season_options: list[str | int]) -> None:
    """Seed the chart-season widget from canonical session state.

    Args:
        chart_season_options: Valid selectbox options for the current board.

    Returns:
        None.
    """
    current_chart_season = st.session_state.get("chart_season", "All")
    if current_chart_season not in chart_season_options:
        current_chart_season = "All"
        st.session_state["chart_season"] = current_chart_season
    if st.session_state.get("_chart_season_picker") != current_chart_season:
        st.session_state["_chart_season_picker"] = current_chart_season


def render_chart_season_picker(chart_season_options: list[str | int] | None) -> None:
    """Render the chart-season picker in the dedicated right-rail slot."""
    if not chart_season_options:
        return

    st.markdown("<div id='comparison-season-filter'></div>", unsafe_allow_html=True)
    _prime_chart_season_picker(chart_season_options)
    st.selectbox(
        "Chart season",
        options=chart_season_options,
        key="_chart_season_picker",
        on_change=_sync_chart_season_picker,
        format_func=_format_chart_season_label,
        label_visibility="collapsed",
    )


def get_panel_tab_ids() -> set[str]:
    """Return all registered detail tab IDs."""
    return {t.id for t in _DETAIL_PANEL_TABS}


def _get_visible_player_entries(processed_dfs: list, players: dict) -> list[tuple]:
    """Return visible player entries in stable board order."""
    return list(_iter_visible_players_for_category(processed_dfs, players))


def _render_card_grid(entries: list[tuple], render_card: Callable[..., None], empty_message: str) -> None:
    """Render comparison detail cards in a two-column alternating grid."""
    if not entries:
        st.info(empty_message)
        return

    card_columns = st.columns(2, gap="medium")
    for idx, entry in enumerate(entries):
        with card_columns[idx % 2]:
            render_card(*entry)


def _render_comparison_media_card(
    media_url: str | None,
    body_html: str,
    *,
    card_modifier_class: str = "",
    media_modifier_class: str = "",
    image_modifier_class: str = "",
    player_color: str | None = None,
    click_payload: str | None = None,
    click_label: str | None = None,
) -> None:
    """Render one comparison detail card with optional media and style modifiers."""
    outer_classes = "comparison-player-card"
    if card_modifier_class:
        outer_classes = f"{outer_classes} {card_modifier_class}"

    media_classes = "comparison-player-card__media"
    if media_modifier_class:
        media_classes = f"{media_classes} {media_modifier_class}"

    image_classes = "comparison-player-card__image"
    if image_modifier_class:
        image_classes = f"{image_classes} {image_modifier_class}"

    media_html = ""
    if media_url:
        safe_media_url = escape(media_url, quote=True)
        media_html = (
            f"<div class='{media_classes}'>"
            f"<img src='{safe_media_url}' class='{image_classes}' loading='lazy'>"
            "</div>"
        )
    else:
        outer_classes = "comparison-player-card comparison-player-card--no-image"

    card_style = ""
    if player_color:
        resolved = _resolve_chart_accent_color(player_color)
        card_style = escape(
            f"--pc-inset-glow:{_hex_to_rgba(resolved, 0.22)};"
            f"--pc-color-tint:{_hex_to_rgba(resolved, 0.08)};",
            quote=True,
        )

    card_html = (
        f"<div class='{outer_classes}' style='{card_style}'>"
        f"{media_html}"
        "<div class='comparison-player-card__body'>"
        f"{body_html or ''}"
        "</div>"
        "</div>"
    )
    if click_payload:
        safe_payload = escape(str(click_payload), quote=True)
        safe_label = escape(str(click_label or "Open details"), quote=True)
        card_html = (
            "<div class='comparison-card-shell comparison-card-shell--clickable' "
            "data-nhl-identity-card='1' "
            f"data-identity-card='{safe_payload}' "
            f"aria-label='{safe_label}' "
            f"title='{safe_label}' "
            "role='button' tabindex='0'>"
            f"{card_html}"
            "</div>"
        )

    st.markdown(
        card_html,
        unsafe_allow_html=True,
    )


def _render_player_media_card(
    image_url: str | None,
    body_html: str,
    *,
    player_color: str | None = None,
    click_payload: str | None = None,
    click_label: str | None = None,
) -> None:
    """Render a player detail card with a cutout-style player image."""
    _render_comparison_media_card(
        image_url,
        body_html,
        media_modifier_class="comparison-player-card__media--player",
        image_modifier_class="comparison-player-card__image--player-cutout",
        player_color=player_color,
        click_payload=click_payload,
        click_label=click_label,
    )


def _render_team_media_card(
    logo_url: str | None,
    body_html: str,
    *,
    player_color: str | None = None,
    click_payload: str | None = None,
    click_label: str | None = None,
) -> None:
    """Render a team detail card using the shared comparison card shell."""
    _render_comparison_media_card(
        logo_url,
        body_html,
        card_modifier_class="comparison-player-card--team",
        media_modifier_class="comparison-player-card__media--team",
        player_color=player_color,
        image_modifier_class="comparison-player-card__image--team-logo",
        click_payload=click_payload,
        click_label=click_label,
    )


def _render_player_card_grid(
    visible_players: list[tuple],
    render_card: Callable[..., None],
    empty_message: str,
) -> None:
    """Render player detail cards in two columns, alternating by selection order."""
    _render_card_grid(visible_players, render_card, empty_message)


def _get_empty_detail_message(entity_name: str, has_selection: bool, detail_label: str) -> str:
    """Return a consistent empty-state message for detail sections."""
    if has_selection:
        return (
            f"No selected {entity_name} match the current category and filters "
            f"for {detail_label}."
        )
    return f"Add {entity_name} from the sidebar to see {detail_label} here."


def _get_player_chart_colors() -> dict[str, str | None]:
    """Return the active chart color map for real chart traces.

    Args:
        None.

    Returns:
        Mapping of trace display names to the colors used on the chart.
    """
    session_state = getattr(st, "session_state", None)
    if session_state is None:
        return {}

    if hasattr(session_state, "get"):
        player_colors = session_state.get("player_chart_colors", {})
    else:
        player_colors = getattr(session_state, "player_chart_colors", {})

    return player_colors if isinstance(player_colors, dict) else {}


def _resolve_chart_accent_color(chart_color: str | None) -> str:
    """Return a safe comparison-card accent color with fallback."""
    return escape(str(chart_color or _DEFAULT_PLAYER_RANK_COLOR), quote=True)


def _build_colored_card_name(name: str, chart_color: str | None) -> str:
    """Return bold escaped name markup tinted to the active chart color."""
    safe_name = escape(str(name))
    safe_color = _resolve_chart_accent_color(chart_color)
    return f"<strong style='color:{safe_color};'>{safe_name}</strong>"


def _build_card_context_row(label: str, *, font_weight: int = 700) -> str:
    """Return muted comparison-card copy for season and all-time labels."""
    safe_label = escape(str(label))
    return (
        "<div class='comparison-card-context-row' style='font-size:14px;"
        f"color:{_CARD_CONTEXT_TEXT_COLOR};font-weight:{font_weight};'>{safe_label}</div>"
    )


def _build_card_stat_row(stats: list[tuple[str, str | int | float]]) -> str:
    """Return one compact stat row for comparison cards.

    Args:
        stats: Ordered ``(label, value)`` pairs for the compact top stat row.

    Returns:
        HTML markup that keeps each stat in its own consistent inline cell.
    """
    stat_items: list[str] = []
    for label, value in stats:
        safe_label = escape(str(label))
        safe_value = escape(str(value))
        stat_items.append(
            "<span class='comparison-card-stats__item'>"
            f"<span class='comparison-card-stats__label'>{safe_label}:&nbsp;</span>"
            f"<span class='comparison-card-stats__value'>{safe_value}</span>"
            "</span>"
        )
    return f"<div class='comparison-card-stats'>{''.join(stat_items)}</div>"


def _build_player_trace_toggle_button(
    player_name: str,
    player_color: str | None,
    *,
    compact: bool = False,
) -> str:
    """Return one icon-only HTML toggle wired to a player legend group."""
    safe_player_name = escape(str(player_name), quote=True)
    safe_color = _resolve_chart_accent_color(player_color)
    safe_title = escape(f"Toggle {player_name} on chart", quote=True)
    compact_class = " comparison-trace-toggle--compact" if compact else ""
    return (
        f"<button type='button' class='comparison-trace-toggle comparison-trace-toggle--icon-only{compact_class}' "
        "data-nhl-trace-toggle='1' "
        f"data-legendgroup='{safe_player_name}' "
        f"title='{safe_title}' aria-label='{safe_title}' aria-pressed='true' "
        f"style='--trace-toggle-color:{safe_color};'>"
        "<span class='comparison-trace-toggle__line' aria-hidden='true'></span>"
        "</button>"
    )


def _build_player_trace_toggle_markup(
    player_name: str,
    player_color: str | None,
) -> str:
    """Return one card-level icon toggle row for a comparison Overview card."""
    button_html = _build_player_trace_toggle_button(
        player_name=player_name,
        player_color=player_color,
        compact=False,
    )
    return (
        "<div class='comparison-trace-toggle-row'>"
        f"{button_html}"
        "</div>"
    )


def _iter_visible_players_for_category(processed_dfs: list, players: dict):
    """Yield selected players that still have visible rows in the active pipeline.

    Args:
        processed_dfs: Active processed DataFrames for the current chart category.
        players: Selected comparison players from session state.

    Returns:
        Iterator yielding ``(player_id, player_name, processed_df)`` tuples for
        players that still have non-projection rows in the active category.
    """
    proc_lookup: dict = {}
    for proc_df in processed_dfs:
        if proc_df.empty or "BaseName" not in proc_df.columns or "Player" not in proc_df.columns:
            continue
        base = proc_df["BaseName"].iloc[0]
        proc_lookup[base] = proc_df

    # Preserve insertion order from the selected players dict.
    for pid, name in players.items():
        proc_df = proc_lookup.get(name)
        if proc_df is None:
            continue
        real = proc_df[~proc_df["Player"].str.contains(r"\(Proj\)", na=False)]
        if real.empty:
            continue
        yield pid, name, proc_df


def _iter_visible_teams(processed_dfs: list, active_teams: dict):
    """Yield selected teams that still have visible rows in the active pipeline."""
    proc_lookup: dict[str, pd.DataFrame] = {}
    for proc_df in processed_dfs:
        if proc_df.empty or "BaseName" not in proc_df.columns or "Player" not in proc_df.columns:
            continue
        proc_lookup[str(proc_df["BaseName"].iloc[0])] = proc_df

    for abbr, full_name in active_teams.items():
        proc_df = proc_lookup.get(str(abbr))
        if proc_df is None or proc_df.empty:
            continue
        yield abbr, full_name, proc_df


def render_detail_tabs(
    processed_dfs: list,
    players: dict,
    teams: dict,
    peak_info: dict,
    metric: str,
    stat_category: str,
    season_type: str,
    team_mode: bool,
    selected_season: str | int = "All",
    do_cumul: bool = False,
) -> None:
    """Render the full-width Overview/Current Standings detail tabs below the chart."""
    identity_trigger_value = _mount_identity_card_click_bridge()
    tab_lookup = {tab.id: tab for tab in _DETAIL_PANEL_TABS}
    tab_key = _get_category_tab_key(stat_category)
    if tab_key not in st.session_state:
        st.session_state[tab_key] = _DEFAULT_PANEL_TAB

    if st.session_state[tab_key] not in tab_lookup:
        st.session_state[tab_key] = _DEFAULT_PANEL_TAB

    default_tab = tab_lookup.get(st.session_state.get(tab_key, _DEFAULT_PANEL_TAB), tab_lookup[_DEFAULT_PANEL_TAB])

    st.markdown("<div id='comparison-tabs'></div>", unsafe_allow_html=True)
    tab_containers = st.tabs(
        [tab_lookup[tab_id].label for tab_id in tab_lookup],
        default=default_tab.label,
    )

    for tab_id, tab_container in zip(tab_lookup, tab_containers):
        tab_spec = tab_lookup[tab_id]
        with tab_container:
            if team_mode:
                tab_spec.render_team(
                    active_teams=teams,
                    processed_dfs=processed_dfs,
                    metric=metric,
                    season_type=season_type,
                    selected_season=selected_season,
                    do_cumul=do_cumul,
                )
            else:
                tab_spec.render_player(
                    processed_dfs=processed_dfs,
                    players=players,
                    peak_info=peak_info,
                    metric=metric,
                    stat_category=stat_category,
                    season_type=season_type,
                    selected_season=selected_season,
                    do_cumul=do_cumul,
                )

    _show_identity_card_from_trigger(identity_trigger_value)


def _coerce_query_param_scalar(value) -> str:
    """Return one string value from Streamlit query-param storage."""
    if isinstance(value, (list, tuple)):
        return str(value[0] if value else "")
    return str(value or "")


def _noop_matchup_history_click_change() -> None:
    """Provide a stable callback for the JS trigger component."""


def _mount_matchup_history_click_bridge():
    """Mount the prediction-card JS bridge once and return the latest payload.

    The component's Streamlit key must stay unique per script run. `app.py`
    pre-mounts this bridge before the chart renders, then passes both the
    latest payload and an explicit "already mounted" flag into
    `render_predictions_panel()` so the predictions rail does not try to mount
    the same component a second time when no click has happened yet.
    """
    result = _MATCHUP_HISTORY_CLICK_BRIDGE(
        key="comparison_matchup_history_click_bridge",
        on_clicked_change=_noop_matchup_history_click_change,
    )
    return getattr(result, "clicked", None)


def _parse_matchup_history_request(value) -> tuple[str, str] | None:
    """Validate one matchup-history query-param payload."""
    raw_value = _coerce_query_param_scalar(value).strip().upper()
    if not raw_value:
        return None

    parts = [part.strip().upper() for part in raw_value.split(",")]
    if len(parts) != 2:
        return None

    away_abbr, home_abbr = parts
    if away_abbr not in ACTIVE_TEAMS or home_abbr not in ACTIVE_TEAMS or away_abbr == home_abbr:
        return None
    return away_abbr, home_abbr


def _parse_matchup_history_trigger(value) -> tuple[str, str, str] | None:
    """Validate one JS trigger payload of the form ``AWY,HOME|nonce``."""
    raw_value = _coerce_query_param_scalar(value).strip()
    if not raw_value or "|" not in raw_value:
        return None

    matchup_value, nonce = raw_value.split("|", 1)
    parsed_request = _parse_matchup_history_request(matchup_value)
    clean_nonce = str(nonce or "").strip()
    if parsed_request is None or not clean_nonce:
        return None

    away_abbr, home_abbr = parsed_request
    return away_abbr, home_abbr, clean_nonce


def _show_matchup_history_from_trigger(value) -> bool:
    """Open matchup history once for each unique JS trigger nonce."""
    parsed_trigger = _parse_matchup_history_trigger(value)
    if parsed_trigger is None:
        return False

    away_abbr, home_abbr, nonce = parsed_trigger
    if session_state_get(_LAST_MATCHUP_HISTORY_TRIGGER_NONCE_SESSION_KEY) == nonce:
        return False
    session_state_set(_LAST_MATCHUP_HISTORY_TRIGGER_NONCE_SESSION_KEY, nonce)

    if not dialog_slot_available():
        session_state_set(_PENDING_MATCHUP_HISTORY_SESSION_KEY, (away_abbr, home_abbr))
        return False

    show_matchup_history(
        away_abbr=away_abbr,
        home_abbr=home_abbr,
    )
    mark_dialog_opened_this_run()
    return True


def _consume_matchup_history_request() -> None:
    """Capture and clear one pending matchup-history request from the URL."""
    try:
        raw_value = dict(st.query_params).get(_MATCHUP_HISTORY_QUERY_KEY)
    except Exception:
        raw_value = None

    if raw_value is None:
        return

    parsed_request = _parse_matchup_history_request(raw_value)
    if parsed_request is not None:
        session_state_set(_PENDING_MATCHUP_HISTORY_SESSION_KEY, parsed_request)

    try:
        del st.query_params[_MATCHUP_HISTORY_QUERY_KEY]
    except Exception:
        pass


def has_pending_matchup_history_dialog_request(matchup_history_trigger_value: str | None = None) -> bool:
    """Return whether a matchup-history dialog is queued before chart rendering."""
    parsed_trigger = _parse_matchup_history_trigger(matchup_history_trigger_value)
    if parsed_trigger is not None:
        _, _, nonce = parsed_trigger
        if session_state_get(_LAST_MATCHUP_HISTORY_TRIGGER_NONCE_SESSION_KEY) != nonce:
            return True

    if session_state_get(_PENDING_MATCHUP_HISTORY_SESSION_KEY):
        return True

    try:
        raw_value = dict(st.query_params).get(_MATCHUP_HISTORY_QUERY_KEY)
    except Exception:
        raw_value = None
    return _parse_matchup_history_request(raw_value) is not None


def _show_pending_matchup_history_dialog() -> None:
    """Open the pending matchup-history dialog exactly once."""
    pending_request = session_state_pop(_PENDING_MATCHUP_HISTORY_SESSION_KEY, None)

    if not pending_request:
        return

    if not dialog_slot_available():
        session_state_set(_PENDING_MATCHUP_HISTORY_SESSION_KEY, pending_request)
        return

    away_abbr, home_abbr = pending_request
    show_matchup_history(
        away_abbr=away_abbr,
        home_abbr=home_abbr,
    )
    mark_dialog_opened_this_run()


def _noop_identity_card_click_change() -> None:
    """Provide a stable callback for the overview-card click bridge."""


def _mount_identity_card_click_bridge():
    """Mount the overview-card JS click bridge and return the latest payload."""
    result = _IDENTITY_CARD_CLICK_BRIDGE(
        key="comparison_identity_card_click_bridge",
        on_clicked_change=_noop_identity_card_click_change,
    )
    return getattr(result, "clicked", None)


def _parse_identity_card_request(value) -> tuple[str, str] | None:
    """Validate one identity-card payload like ``player:8478402``."""
    raw_value = _coerce_query_param_scalar(value).strip()
    if not raw_value or ":" not in raw_value:
        return None

    entity_kind, entity_value = raw_value.split(":", 1)
    clean_kind = str(entity_kind or "").strip().lower()
    clean_value = str(entity_value or "").strip()
    if clean_kind == "player":
        try:
            player_id = int(clean_value)
        except Exception:
            return None
        if player_id <= 0:
            return None
        return clean_kind, str(player_id)

    if clean_kind == "team":
        team_abbr = clean_value.upper()
        if team_abbr not in ACTIVE_TEAMS:
            return None
        return clean_kind, team_abbr

    return None


def _parse_identity_card_trigger(value) -> tuple[str, str, str] | None:
    """Validate one JS trigger payload of the form ``kind:value|nonce``."""
    raw_value = _coerce_query_param_scalar(value).strip()
    if not raw_value or "|" not in raw_value:
        return None

    payload_value, nonce = raw_value.split("|", 1)
    parsed_request = _parse_identity_card_request(payload_value)
    clean_nonce = str(nonce or "").strip()
    if parsed_request is None or not clean_nonce:
        return None

    entity_kind, entity_value = parsed_request
    return entity_kind, entity_value, clean_nonce


def _show_identity_card_from_trigger(value) -> bool:
    """Open player/team identity dialogs once for each unique trigger nonce."""
    parsed_trigger = _parse_identity_card_trigger(value)
    if parsed_trigger is None:
        return False

    entity_kind, entity_value, nonce = parsed_trigger
    if session_state_get(_LAST_IDENTITY_CARD_TRIGGER_NONCE_SESSION_KEY) == nonce:
        return False
    session_state_set(_LAST_IDENTITY_CARD_TRIGGER_NONCE_SESSION_KEY, nonce)

    if not dialog_slot_available():
        return False

    if entity_kind == "player":
        show_player_identity_details(int(entity_value))
        mark_dialog_opened_this_run()
        return True
    if entity_kind == "team":
        show_team_identity_details(entity_value)
        mark_dialog_opened_this_run()
        return True
    return False


def render_predictions_panel(
    share_params: dict | None = None,
    matchup_history_trigger_value: str | None = None,
    matchup_history_bridge_mounted: bool = False,
) -> None:
    """Render the dedicated Predictions panel in the desktop right rail.

    Args:
        share_params: Canonical URL params used to keep prediction-card links
            shareable without mutating the visible board state.
        matchup_history_trigger_value: Latest click payload from the JS bridge.
            This can legitimately be `None` even when the bridge is already
            mounted and idle.
        matchup_history_bridge_mounted: Whether the JS bridge was already
            mounted earlier in the same rerun. This prevents duplicate
            component-key errors when `app.py` pre-mounts the bridge so the
            chart can suppress stale Plotly dialogs while a matchup modal is
            pending.
    """
    trigger_value = matchup_history_trigger_value
    if not matchup_history_bridge_mounted:
        trigger_value = _mount_matchup_history_click_bridge()
    st.markdown("<div id='comparison-predictions-panel'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='comparison-panel-heading comparison-panel-heading--rail-title comparison-panel-heading--predictions'>Next matches prediction</div>",
        unsafe_allow_html=True,
    )
    _render_live_games_tab(share_params=share_params)
    if _show_matchup_history_from_trigger(trigger_value):
        return

    _consume_matchup_history_request()
    _show_pending_matchup_history_dialog()



def _build_live_game_card_html(game: dict) -> str:
    """Build the unified matchup card HTML for one upcoming game.

    Combines team logos, time/venue detail, and win-probability bar into a
    single ``.live-game-card`` div with team-color background illumination
    that scales with prediction confidence.

    Args:
        game: Normalized game dict returned by ``get_upcoming_games()``.

    Returns:
        HTML string for the complete game card.
    """
    away_abbr = game.get("away_abbr", "")
    home_abbr = game.get("home_abbr", "")
    away_logo = _TEAM_LOGO_URL.format(abbr=away_abbr)
    home_logo = _TEAM_LOGO_URL.format(abbr=home_abbr)
    away_short_name = _get_team_short_name(away_abbr, game.get("away_name", ""))
    home_short_name = _get_team_short_name(home_abbr, game.get("home_name", ""))

    detail_bits = [game["start_label_cest"]]
    if game.get("venue"):
        detail_bits.append(game["venue"])
    detail_text = escape(" • ".join(detail_bits))

    probability = game.get("pregame_win_prob")
    has_prob = isinstance(probability, dict)
    if has_prob:
        try:
            away_pct = int(probability.get("away_pct", 0))
            home_pct = int(probability.get("home_pct", 0))
        except Exception:
            has_prob = False

    if has_prob:
        away_pct = min(max(away_pct, 0), 100)
        home_pct = min(max(home_pct, 0), 100)
        away_background = _get_team_probability_background(away_abbr, "#4F8FFF")
        home_background = _get_team_probability_background(home_abbr, "#FF5C5C")
        away_raw = _get_raw_team_color(away_abbr, "#4F8FFF")
        home_raw = _get_raw_team_color(home_abbr, "#FF5C5C")
        edge_strength = min(abs(away_pct - home_pct) / 34.0, 1.0)
        is_tied = away_pct == home_pct
        away_leading = away_pct > home_pct
        home_leading = home_pct > away_pct
        panel_state = "tied"
        if away_leading:
            panel_state = "away-lead"
        elif home_leading:
            panel_state = "home-lead"

        # Card background tints — much stronger than the old probability bar tints
        away_card_alpha = 0.14 if is_tied else (0.30 + edge_strength * 0.14 if away_leading else 0.03)
        home_card_alpha = 0.14 if is_tied else (0.30 + edge_strength * 0.14 if home_leading else 0.03)
        away_card_tint = _hex_to_rgba(away_background, away_card_alpha)
        home_card_tint = _hex_to_rgba(home_background, home_card_alpha)

        # Inset glow from the winning team's raw brand color
        if is_tied:
            glow_alpha = 0.10
            glow_color = away_raw
        elif away_leading:
            glow_alpha = 0.18 + edge_strength * 0.14
            glow_color = away_raw
        else:
            glow_alpha = 0.18 + edge_strength * 0.14
            glow_color = home_raw
        card_inset_glow = _hex_to_rgba(glow_color, glow_alpha)

        away_bar_glow = _hex_to_rgba(
            away_background,
            0.08 if is_tied else (0.16 + edge_strength * 0.14 if away_leading else 0.0),
        )
        home_bar_glow = _hex_to_rgba(
            home_background,
            0.08 if is_tied else (0.16 + edge_strength * 0.14 if home_leading else 0.0),
        )
        bar_style = escape(
            f"--away-glow-width:{away_pct}%;"
            f"--home-glow-width:{home_pct}%;"
            f"--away-bar-glow:{away_bar_glow};"
            f"--home-bar-glow:{home_bar_glow};",
            quote=True,
        )
        away_label_state = "tied" if is_tied else ("leading" if away_leading else "trailing")
        home_label_state = "tied" if is_tied else ("leading" if home_leading else "trailing")
        away_segment_state = "tied" if is_tied else ("leading" if away_leading else "trailing")
        home_segment_state = "tied" if is_tied else ("leading" if home_leading else "trailing")
        away_label_style = _build_probability_label_style(
            away_background, is_leading=away_leading, is_tied=is_tied, edge_strength=edge_strength,
        )
        home_label_style = _build_probability_label_style(
            home_background, is_leading=home_leading, is_tied=is_tied, edge_strength=edge_strength,
        )
        away_segment_style = _build_probability_segment_style(
            away_pct, away_background, is_leading=away_leading, is_tied=is_tied, edge_strength=edge_strength,
        )
        home_segment_style = _build_probability_segment_style(
            home_pct, home_background, is_leading=home_leading, is_tied=is_tied, edge_strength=edge_strength,
        )
        away_short_esc = escape(away_short_name)
        home_short_esc = escape(home_short_name)
        model_label = escape(str(probability.get("model_label", "") or "").strip())
        goalie_label = escape(str(probability.get("goalie_label", "") or "").strip())
        playoff_note = ""
        if int(game.get("game_type", 0) or 0) == 3:
            playoff_note = "<div class='live-games-probability__meta live-games-probability__meta--playoff'>Regular-season calibrated model.</div>"
        meta_block = (
            "<div class='lgc-meta-popover'>"
            "<div class='lgc-meta'>"
            f"<div class='live-games-probability__meta'>{model_label}</div>"
            f"<div class='live-games-probability__meta'>{goalie_label}</div>"
            f"{playoff_note}"
            "</div>"
            "</div>"
        )

        prob_section = (
            "<div class='lgc-prob-section'>"
            "<div class='live-games-probability__labels'>"
            f"<span class='live-games-probability__label live-games-probability__label--away live-games-probability__label--{away_label_state}' style='{away_label_style}'>{away_short_esc} <strong>{away_pct}%</strong></span>"
            f"<span class='live-games-probability__label live-games-probability__label--home live-games-probability__label--{home_label_state}' style='{home_label_style}'><strong>{home_pct}%</strong> {home_short_esc}</span>"
            "</div>"
            f"<div class='live-games-probability__bar' style='{bar_style}'>"
            f"<span class='live-games-probability__segment live-games-probability__segment--away live-games-probability__segment--{away_segment_state}' style='{away_segment_style}'></span>"
            f"<span class='live-games-probability__segment live-games-probability__segment--home live-games-probability__segment--{home_segment_state}' style='{home_segment_style}'></span>"
            f"<span class='live-games-probability__divider' style='left:{away_pct}%;'></span>"
            "</div>"
            "</div>"
        )
        card_style = escape(
            f"--lgc-away-tint:{away_card_tint};"
            f"--lgc-home-tint:{home_card_tint};"
            f"--lgc-inset-glow:{card_inset_glow};",
            quote=True,
        )
    else:
        meta_block = ""
        prob_section = "<div class='lgc-prob-section live-games-probability--muted'>Estimate unavailable.</div>"
        card_style = ""
        panel_state = "no-prob"

    away_short_esc = escape(away_short_name)
    home_short_esc = escape(home_short_name)

    return (
        f"<div class='live-game-card live-game-card--{panel_state}' style='{card_style}'>"
        "<div class='lgc-header'>"
        "<div class='lgc-header__main'>"
        "<div class='lgc-matchup'>"
        f"<img src='{away_logo}' height='26' style='vertical-align:middle;'>"
        f"<strong>{away_short_esc}</strong>"
        "<span style='color:#aaa;font-size:13px;'>at</span>"
        f"<img src='{home_logo}' height='26' style='vertical-align:middle;'>"
        f"<strong>{home_short_esc}</strong>"
        "</div>"
        f"<div class='lgc-detail'>{detail_text}</div>"
        "</div>"
        f"{meta_block}"
        "</div>"
        f"{prob_section}"
        "</div>"
    )


def _build_live_game_card_href(game: dict, share_params: dict | None = None) -> str:
    """Build one self-link that preserves current shared app state."""
    away_abbr = str(game.get("away_abbr", "") or "").strip().upper()
    home_abbr = str(game.get("home_abbr", "") or "").strip().upper()
    if not away_abbr or not home_abbr:
        return "#"

    query_items: list[tuple[str, str]] = []
    for key, value in (share_params or {}).items():
        clean_key = str(key or "").strip()
        if not clean_key or value is None:
            continue
        query_items.append((clean_key, str(value)))
    query_items.append((_MATCHUP_HISTORY_QUERY_KEY, f"{away_abbr},{home_abbr}"))
    return f"?{urlencode(query_items)}"


def _build_live_game_card_link_html(game: dict, share_params: dict | None = None) -> str:
    """Wrap one prediction card in a full-card history link."""
    away_abbr = str(game.get("away_abbr", "") or "").strip().upper()
    home_abbr = str(game.get("home_abbr", "") or "").strip().upper()
    away_name = str(game.get("away_name", "") or game.get("away_abbr", "") or "").strip()
    home_name = str(game.get("home_name", "") or game.get("home_abbr", "") or "").strip()
    href = escape(_build_live_game_card_href(game, share_params=share_params), quote=True)
    title = escape(f"Open matchup history for {away_name} at {home_name}", quote=True)
    matchup_value = escape(f"{away_abbr},{home_abbr}", quote=True)
    return (
        "<div class='live-game-card-shell'>"
        f"<a class='live-game-card-link' href='{href}' aria-label='{title}' title='{title}' "
        "data-nhl-matchup-history='1' "
        f"data-matchup-history='{matchup_value}'></a>"
        f"{_build_live_game_card_html(game)}"
        "</div>"
    )


def _get_team_short_name(team_abbr: str, fallback_name: str) -> str:
    """Return the short display name for a team.

    Args:
        team_abbr: Three-letter NHL team abbreviation.
        fallback_name: Full team name to use if no short mapping exists.

    Returns:
        Team nickname without the city or market prefix.
    """
    return _TEAM_SHORT_NAMES.get(team_abbr, ACTIVE_TEAMS.get(team_abbr, fallback_name))


def _render_live_games_tab(share_params: dict | None = None) -> None:
    """Render the shared right-rail predictions cards for upcoming games."""
    upcoming_games = get_upcoming_games(limit=_PREDICTIONS_PANEL_MATCH_LIMIT)
    if not upcoming_games:
        st.info("No upcoming NHL games found right now.")
        return

    for game in upcoming_games:
        card_html = _build_live_game_card_link_html(game, share_params=share_params)
        st.markdown(card_html, unsafe_allow_html=True)


def _render_overview_player_card(
    pid,
    name,
    proc_df,
    peak_info: dict,
    metric: str,
    stat_category: str,
    season_type: str,
    selected_season: str | int,
    do_cumul: bool,
) -> None:
    """Render one player Overview card."""
    is_goalie = stat_category == "Goalie"
    season_mode = _is_selected_season_mode(selected_season)
    use_last_visible_total = season_mode and do_cumul
    player_colors = _get_player_chart_colors()
    player_color = player_colors.get(name)
    season_rank_map = (
        get_player_season_rank_map(stat_category, int(selected_season), season_type, metric)
        if season_mode
        else {}
    )
    rank_suffix_map = {"Goals": "Goals", "Assists": "Assists", "Points": "Points"}
    rank_suffix = "Wins" if is_goalie else rank_suffix_map.get(metric, "Points")

    real = proc_df[~proc_df["Player"].str.contains(r"\(Proj\)", na=False)]
    sort_cols = [col for col in ["CumGP", "Age", "SeasonYear"] if col in real.columns]
    if sort_cols:
        real = real.sort_values(sort_cols).reset_index(drop=True)

    headshot_url = get_player_headshot(int(pid))
    team_abbr = get_player_current_team(int(pid))
    logo_html = (
        f"<img src='{_TEAM_LOGO_URL.format(abbr=team_abbr)}' "
        f"height='18' style='vertical-align:middle;margin-left:6px;opacity:0.9;'>"
        if team_abbr
        else ""
    )

    career_gp = _get_visible_stat_total(real, "GP", use_last_visible_total)
    if is_goalie:
        career_w = _get_visible_stat_total(real, "Wins", use_last_visible_total)
        career_so = _get_visible_stat_total(real, "Shutouts", use_last_visible_total)
        career_sv = _get_visible_stat_total(real, "Saves", use_last_visible_total)
        stats_row = _build_card_stat_row(
            [("W", career_w), ("SO", career_so), ("SV", f"{career_sv:,}"), ("GP", career_gp)]
        )
    else:
        career_g = _get_visible_stat_total(real, "Goals", use_last_visible_total)
        career_a = _get_visible_stat_total(real, "Assists", use_last_visible_total)
        career_pt = _get_visible_stat_total(real, "Points", use_last_visible_total)
        stats_row = _build_card_stat_row(
            [("G", career_g), ("A", career_a), ("Pts", career_pt), ("GP", career_gp)]
        )

    scope_row = ""
    if season_mode:
        season_rank = season_rank_map.get(int(pid))
        season_label = (
            _build_selected_season_rank_label(selected_season, season_type, metric, season_rank)
            if season_rank is not None
            else _build_selected_season_scope_label(selected_season, season_type)
        )
        scope_row = _build_card_context_row(season_label)

    rank_row = ""
    if not season_mode:
        rank = get_player_career_rank(int(pid), stat_category, season_type, metric)
        if rank is not None:
            rank_row = _build_card_context_row(f"#{rank} in all-time {rank_suffix}")

    trace_toggle_row = _build_player_trace_toggle_markup(
        player_name=name,
        player_color=player_color,
    )

    peak = peak_info.get(name)
    best_row = ""
    if peak:
        metric_short_map = {"Points": "Pts", "Goals": "G", "Assists": "A"}
        metric_short = metric_short_map.get(metric, metric)
        if season_mode:
            peak_game = peak.get("game_number") or peak.get("x") or "?"
            peak_date = peak.get("game_date")
            peak_value = peak.get("raw_peak_val", peak.get("y"))
            peak_date_str = f" ({peak_date})" if peak_date else ""
            best_row = _build_card_context_row(
                f"Best game: #{peak_game}{peak_date_str}"
                f" -- {_format_peak_metric_value(metric, peak_value)} {metric_short}"
            )
        else:
            age = peak.get("age", "?")
            sy = peak.get("season_year")
            val = peak.get("y")
            peak_row_df = real[real["Age"] == age]
            peak_gp = (
                int(peak_row_df["GP"].iloc[0])
                if not peak_row_df.empty and "GP" in peak_row_df.columns
                else "?"
            )
            sy_str = f"{sy - 1}-{str(sy)[2:]}" if sy else "?"
            best_row = _build_card_context_row(
                f"Best at age {age} ({sy_str})"
                f" -- {_format_peak_metric_value(metric, val)} {metric_short} in {peak_gp} GP",
                font_weight=400,
            )

    roster_info = get_player_roster_info(int(pid))
    name_markup = _build_colored_card_name(name, player_color)
    if roster_info:
        pos = escape(str(roster_info["position"]))
        num = escape(str(roster_info["sweater_number"]))
        name_html = (
            f"<span style='color:#aaa;font-size:13px;'>[{pos}]</span> "
            f"{name_markup} "
            f"<span style='color:#aaa;font-size:13px;'>#{num}</span>"
        )
    else:
        name_html = name_markup

    _render_player_media_card(
        headshot_url,
        "<div style='line-height:1.4;margin:0;padding:0;'>"
        f"{name_html}{logo_html}<br>"
        f"{stats_row}"
        f"{scope_row}"
        f"{rank_row}"
        f"{best_row}"
        f"{trace_toggle_row}"
        "</div>",
        player_color=player_color,
        click_payload=f"player:{int(pid)}",
        click_label=f"Open player details for {name}",
    )


def _render_overview_players(
    processed_dfs: list,
    players: dict,
    peak_info: dict,
    metric: str,
    stat_category: str,
    season_type: str,
    selected_season: str | int = "All",
    do_cumul: bool = False,
) -> None:
    """Overview tab for player comparison cards."""
    visible_players = _get_visible_player_entries(processed_dfs, players)
    _render_player_card_grid(
        visible_players=visible_players,
        render_card=lambda pid, name, proc_df: _render_overview_player_card(
            pid=pid,
            name=name,
            proc_df=proc_df,
            peak_info=peak_info,
            metric=metric,
            stat_category=stat_category,
            season_type=season_type,
            selected_season=selected_season,
            do_cumul=do_cumul,
        ),
        empty_message=_get_empty_detail_message("players", bool(players), "overview details"),
    )


def _build_team_record_label(row: pd.Series) -> str:
    """Return a standard team record string from a season-progress row."""
    wins = int(round(float(row.get("Wins", 0) or 0)))
    losses = int(round(float(row.get("Losses", 0) or 0)))
    ot_losses = int(round(float(row.get("OTLosses", 0) or 0)))
    ties = int(round(float(row.get("Ties", 0) or 0)))
    if ot_losses > 0:
        return f"{wins}-{losses}-{ot_losses}"
    if ties > 0:
        return f"{wins}-{losses}-{ties}"
    return f"{wins}-{losses}"


def _build_team_streak_label(real_df: pd.DataFrame) -> str:
    """Return the closing streak label for a selected-season team trace."""
    if real_df.empty or "ResultCode" not in real_df.columns:
        return ""
    result_codes = [str(code or "").strip().upper() for code in real_df["ResultCode"].tolist() if str(code or "").strip()]
    if not result_codes:
        return ""
    last_code = result_codes[-1]
    streak_len = 0
    for code in reversed(result_codes):
        if code != last_code:
            break
        streak_len += 1
    if streak_len <= 0:
        return ""
    return f"Ended on {last_code}{streak_len}"


def _render_overview_teams(
    active_teams: dict,
    processed_dfs: list,
    metric: str,
    season_type: str = "Regular",
    selected_season: str | int = "All",
    do_cumul: bool = False,
) -> None:
    """Overview tab for franchise cards or selected-season team summaries."""
    del do_cumul
    season_mode = _is_selected_season_mode(selected_season)
    chart_colors = _get_player_chart_colors()
    team_stats = get_team_all_time_stats()
    season_rank_map = (
        get_team_season_rank_map(int(selected_season), season_type, metric)
        if season_mode
        else {}
    )
    season_summary_df = (
        get_team_season_summary(int(selected_season), season_type)
        if season_mode
        else pd.DataFrame()
    )
    season_summary_map = (
        season_summary_df.set_index("teamAbbrev").to_dict("index")
        if not season_summary_df.empty and "teamAbbrev" in season_summary_df.columns
        else {}
    )
    if not active_teams:
        st.info(_get_empty_detail_message("teams", False, "overview details"))
        return

    if season_mode:
        visible_teams = list(_iter_visible_teams(processed_dfs, active_teams))

        def _render_season_team_card(abbr: str, full_name: str, proc_df: pd.DataFrame) -> None:
            """Render one selected-season team overview card."""
            real = proc_df.sort_values(["CumGP", "GameDate", "GameId"], na_position="last").reset_index(drop=True)
            latest = real.iloc[-1]
            summary_stats = season_summary_map.get(abbr, {})
            founded = escape(str(TEAM_FOUNDED.get(abbr, "")))
            logo_url = _TEAM_LOGO_URL.format(abbr=abbr)
            team_color = chart_colors.get(full_name)
            record_label = str(latest.get("RecordLabel") or _build_team_record_label(latest))
            gp = int(round(float(latest.get("GP", 0) or 0)))
            points = int(round(float(latest.get("Points", 0) or 0)))
            goals = int(round(float(latest.get("Goals", 0) or 0)))
            goals_against = int(round(float(latest.get("GoalsAgainst", 0) or 0)))
            team_name_markup = _build_colored_card_name(full_name, team_color)
            team_name_html = (
                f"{team_name_markup}"
                f" <span style='color:#aaa;font-size:13px;'>{founded}</span>"
                if founded
                else team_name_markup
            )
            stats_row = _build_card_stat_row(
                [
                    ("Rec", record_label),
                    ("Pts", points),
                    ("GF", goals),
                    ("GA", goals_against),
                    ("GP", gp),
                ]
            )
            season_rank = season_rank_map.get(abbr)
            scope_label = (
                _build_selected_season_rank_label(selected_season, season_type, metric, season_rank)
                if season_rank is not None
                else _build_selected_season_scope_label(selected_season, season_type)
            )
            rank_row = _build_card_context_row(scope_label)

            extra_bits: list[str] = []
            if season_type == "Regular" and gp > 0:
                extra_bits.append(f"{int(round(points / gp * 82))}-pt pace")
            streak_label = _build_team_streak_label(real)
            if streak_label:
                extra_bits.append(streak_label)
            final_metric = summary_stats.get(metric)
            if final_metric is not None and pd.notna(final_metric):
                if metric in {"Win%", "PP%"}:
                    metric_value = f"{float(final_metric):.1f}%"
                elif metric in TEAM_RATE_STATS:
                    metric_value = f"{float(final_metric):.3f}"
                else:
                    metric_value = str(int(round(float(final_metric))))
                extra_bits.append(f"Final {escape(metric)}: {metric_value}")
            best_row = ""
            if extra_bits:
                best_row = _build_card_context_row(" | ".join(extra_bits), font_weight=400)
            trace_toggle_row = _build_player_trace_toggle_markup(
                player_name=full_name,
                player_color=team_color,
            )

            _render_team_media_card(
                logo_url,
                "<div style='line-height:1.45;margin:0;padding:0;'>"
                f"{team_name_html}<br>"
                f"{stats_row}"
                f"{rank_row}"
                f"{best_row}"
                f"{trace_toggle_row}"
                "</div>",
                player_color=team_color,
                click_payload=f"team:{abbr}",
                click_label=f"Open team details for {full_name}",
            )

        _render_card_grid(
            visible_teams,
            _render_season_team_card,
            _get_empty_detail_message("teams", True, "overview details"),
        )
    else:
        all_time_teams = [
            (abbr, full_name, team_stats[abbr])
            for abbr, full_name in active_teams.items()
            if abbr in team_stats
        ]

        def _render_all_time_team_card(abbr: str, full_name: str, stats: dict) -> None:
            """Render one all-time franchise overview card."""
            founded = escape(str(TEAM_FOUNDED.get(abbr, "")))
            logo_url = _TEAM_LOGO_URL.format(abbr=abbr)
            team_color = chart_colors.get(full_name)
            total_w = stats["total_wins"]
            total_pts = stats["total_points"]
            total_gf = stats["total_goals"]
            total_gp = stats["total_gp"]
            wins_rank = stats["wins_rank"]
            best_year = stats["best_year"]
            best_wins = stats["best_wins"]
            best_gp = stats["best_gp"]

            team_name_markup = _build_colored_card_name(full_name, team_color)
            name_html = (
                f"{team_name_markup}"
                f" <span style='color:#aaa;font-size:13px;'>{founded}</span>"
                if founded
                else team_name_markup
            )
            stats_row = _build_card_stat_row(
                [
                    ("W", f"{total_w:,}"),
                    ("Pts", f"{total_pts:,}"),
                    ("GF", f"{total_gf:,}"),
                    ("GP", f"{total_gp:,}"),
                ]
            )
            rank_row = _build_card_context_row(f"#{wins_rank} in franchise Wins")
            best_row = ""
            if best_year and best_wins is not None:
                sy_str = f"{best_year - 1}-{str(best_year)[2:]}"
                best_row = _build_card_context_row(
                    f"Best season: {sy_str} -- {best_wins} W in {best_gp} GP",
                    font_weight=400,
                )
            trace_toggle_row = _build_player_trace_toggle_markup(
                player_name=full_name,
                player_color=team_color,
            )

            _render_team_media_card(
                logo_url,
                "<div style='line-height:1.45;margin:0;padding:0;'>"
                f"{name_html}<br>"
                f"{stats_row}"
                f"{rank_row}"
                f"{best_row}"
                f"{trace_toggle_row}"
                "</div>",
                player_color=team_color,
                click_payload=f"team:{abbr}",
                click_label=f"Open team details for {full_name}",
            )

        _render_card_grid(
            all_time_teams,
            _render_all_time_team_card,
            _get_empty_detail_message("teams", True, "overview details"),
        )

@st.cache_data(ttl=3600)
def get_stanley_cup_board() -> dict:
    """Return one cached live current-standings board with a Cup favorite."""
    return build_stanley_cup_board(
        standings_df=get_current_nhl_standings(),
        artifact=load_win_prob_weights(),
        goalie_proxy_by_team=None,
    )


def _build_current_standings_board_markup(board: dict) -> str:
    """Return the league-wide standings board markup."""
    generated_at_label = escape(str(board.get("generated_at_label", "") or "").strip())
    favorite_team = board.get("favorite_team", {}) if isinstance(board, dict) else {}
    favorite_name = escape(str(favorite_team.get("team_name", "") or "").strip())
    favorite_summary = escape(str(favorite_team.get("summary_text", "") or "").strip())
    favorite_rank = favorite_team.get("rank")

    meta_bits: list[str] = []
    if generated_at_label:
        meta_bits.append(f"<span>{generated_at_label}</span>")
    if favorite_name:
        label_suffix = f" (model rank #{int(favorite_rank)})" if favorite_rank else ""
        meta_bits.append(
            "<span>"
            "<span class='stanley-cup-favorite-button-anchor'></span>"
            f"Cup pick: <strong>{favorite_name}</strong>{escape(label_suffix)}"
            "</span>"
        )

    division_markup: list[str] = []
    for division in board.get("divisions", []):
        division_name = str(division.get("division_name", "") or "").strip()
        conference_name = escape(str(division.get("conference_name", "") or "").strip())
        accent = _resolve_chart_accent_color(_STANDINGS_WINDOW_COLORS.get(division_name, "#60a5fa"))
        accent_soft = _hex_to_rgba(accent, 0.14)
        accent_glow = _hex_to_rgba(accent, 0.28)
        safe_division_name = escape(division_name)

        row_markup: list[str] = []
        for team in division.get("teams", []):
            team_abbr = str(team.get("team_abbr", "") or "").strip().upper()
            team_logo = escape(str(team.get("team_logo", "") or _TEAM_LOGO_URL.format(abbr=team_abbr)), quote=True)
            team_name = escape(str(team.get("team_common_name") or team.get("team_name") or team_abbr).strip())
            primary_brand = TEAM_BRAND_COLORS.get(team_abbr, ("#4f46e5", "#1d4ed8"))[0]
            favorite_style = ""
            favorite_badge = ""
            row_classes = "stanley-cup-row"
            if team.get("is_favorite"):
                row_classes = f"{row_classes} stanley-cup-row--favorite"
                favorite_style = escape(
                    (
                        f"--favorite-accent:{primary_brand};"
                        f"--favorite-accent-soft:{_hex_to_rgba(primary_brand, 0.18)};"
                        f"--favorite-accent-glow:{_hex_to_rgba(primary_brand, 0.34)};"
                    ),
                    quote=True,
                )
                favorite_badge = "<span class='stanley-cup-row-badge'>Cup pick</span>"

            row_markup.append(
                "<div class='comparison-card-shell comparison-card-shell--clickable "
                "stanley-cup-row-shell' "
                "data-nhl-identity-card='1' "
                f"data-identity-card='team:{escape(team_abbr, quote=True)}' "
                f"aria-label='Open team details for {escape(str(team.get('team_name') or team_abbr), quote=True)}' "
                f"title='Open team details for {escape(str(team.get('team_name') or team_abbr), quote=True)}' "
                "role='button' tabindex='0'>"
                f"<div class='{row_classes}' style='{favorite_style}'>"
                "<div class='stanley-cup-row-team'>"
                f"<img class='stanley-cup-team-logo' src='{team_logo}' alt='{team_abbr} logo' loading='lazy'>"
                f"<span class='stanley-cup-row-team-name'>{team_name}</span>"
                f"{favorite_badge}"
                "</div>"
                f"<div class='stanley-cup-row-value'>{int(team.get('games_played', 0) or 0)}</div>"
                f"<div class='stanley-cup-row-value'>{int(team.get('wins', 0) or 0)}</div>"
                f"<div class='stanley-cup-row-value'>{int(team.get('losses', 0) or 0)}</div>"
                f"<div class='stanley-cup-row-value'>{int(team.get('ot_losses', 0) or 0)}</div>"
                f"<div class='stanley-cup-row-value stanley-cup-row-value--pts'>{int(team.get('points', 0) or 0)}</div>"
                "</div>"
                "</div>"
            )

        division_markup.append(
            "<section class='stanley-cup-division-window' "
            f"style='--division-accent:{escape(accent, quote=True)};"
            f"--division-accent-soft:{escape(accent_soft, quote=True)};"
            f"--division-accent-glow:{escape(accent_glow, quote=True)};'>"
            "<div class='stanley-cup-division-header'>"
            f"<div class='stanley-cup-division-kicker'>{conference_name} Conference</div>"
            f"<div class='stanley-cup-division-heading'>{safe_division_name} Division</div>"
            "</div>"
            "<div class='stanley-cup-table-head'>"
            "<div class='stanley-cup-table-head__team'>Team</div>"
            "<div>GP</div>"
            "<div>W</div>"
            "<div>L</div>"
            "<div>OTL</div>"
            "<div>Pts</div>"
            "</div>"
            f"{''.join(row_markup)}"
            "</section>"
        )

    summary_markup = (
        f"<div class='stanley-cup-board-summary'>{favorite_summary}</div>"
        if favorite_summary
        else ""
    )

    return (
        "<div class='stanley-cup-board-shell'>"
        f"<div class='stanley-cup-board-meta'>{''.join(meta_bits)}</div>"
        f"{summary_markup}"
        "<div class='stanley-cup-board-grid'>"
        f"{''.join(division_markup)}"
        "</div>"
        "</div>"
    )


def _render_current_standings_shared() -> None:
    """Render the shared four-window live standings board."""
    board = get_stanley_cup_board()
    if not board.get("divisions"):
        st.info("Current NHL standings are unavailable right now.")
        return

    st.markdown(_build_current_standings_board_markup(board), unsafe_allow_html=True)


def _render_current_standings_players(
    processed_dfs: list,
    players: dict,
    peak_info: dict,
    metric: str,
    stat_category: str,
    season_type: str,
    selected_season: str | int = "All",
    do_cumul: bool = False,
) -> None:
    """Current standings tab for skater/goalie categories."""
    del processed_dfs, players, peak_info, metric, stat_category, season_type, selected_season, do_cumul
    _render_current_standings_shared()


def _render_current_standings_teams(
    active_teams: dict,
    processed_dfs: list,
    metric: str,
    season_type: str = "Regular",
    selected_season: str | int = "All",
    do_cumul: bool = False,
) -> None:
    """Current standings tab for team category."""
    del active_teams, processed_dfs, metric, season_type, selected_season, do_cumul
    _render_current_standings_shared()


_DETAIL_PANEL_TABS = (
    PanelTabSpec(
        id="overview",
        label="Overview",
        render_player=_render_overview_players,
        render_team=_render_overview_teams,
    ),
    PanelTabSpec(
        id="current-standings",
        label="Current Standings",
        render_player=_render_current_standings_players,
        render_team=_render_current_standings_teams,
    ),
)
