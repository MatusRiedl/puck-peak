"""
nhl.sidebar — Player and team sidebar UI for the Puck Peak app.

Renders the full sidebar, switching between player mode and team mode based on
st.session_state.stat_category.  Reads and writes session state directly (standard
Streamlit pattern for sidebar widgets).

Returns a sidebar_keys dict that the chart renderer uses for cache-busting the
chart widget key.  This avoids the chart renderer having to access sidebar widget
values from session state, keeping chart.py's dependency surface narrow.

Imports from project:
    nhl.constants    — ACTIVE_TEAMS
    nhl.data_loaders — search_player, search_local_players, get_top_50,
                       get_top_50_goalies, get_team_roster, get_player_headshot
"""

from html import escape

import streamlit as st
import streamlit.components.v1 as components

from nhl.constants import ACTIVE_TEAMS
from nhl.data_loaders import (
    get_player_headshot,
    get_team_roster,
    get_top_50,
    get_top_50_goalies,
    search_local_players,
    search_player,
)
from nhl.dialog import show_app_guide



_VALID_STAT_CATEGORIES = ("Skater", "Goalie", "Team")
_STAT_CATEGORY_LABELS = {
    "Skater": "⛸️ Skater",
    "Goalie": "🥅 Goalie",
    "Team": "🏒 Team",
}
_SUPPORT_URL = "https://ko-fi.com/iksperial"
_SUPPORT_LABEL = "Buy me a coffee"
_SUPPORT_SUBLABEL = "Support the development"
_SUPPORT_EMOJI = "☕️"


def _sanitize_stat_category(stat_category: str | None, fallback: str = "Skater") -> str:
    """Return a valid stat category for the sidebar selector.

    Args:
        stat_category: Raw category value from session state or the widget.
        fallback: Category to use when the raw value is missing or invalid.

    Returns:
        One of ``"Skater"``, ``"Goalie"``, or ``"Team"``.
    """
    if stat_category in _VALID_STAT_CATEGORIES:
        return stat_category
    if fallback in _VALID_STAT_CATEGORIES:
        return fallback
    return "Skater"


def _resolve_stat_category_selection(
    selected_category: str | None,
    current_category: str | None,
) -> str:
    """Resolve the effective category after a segmented-control interaction.

    Args:
        selected_category: Raw widget value, which may be ``None`` after a deselect.
        current_category: Current canonical app category before the click.

    Returns:
        A valid category string. Deselecting the active segment preserves the
        current category instead of allowing an empty app state.
    """
    safe_current = _sanitize_stat_category(current_category)
    return _sanitize_stat_category(selected_category, fallback=safe_current)


def _sync_stat_category_selection() -> None:
    """Keep the category widget and canonical app state in sync.

    Args:
        None.

    Returns:
        None.
    """
    resolved_category = _resolve_stat_category_selection(
        st.session_state.get("_stat_category_picker"),
        st.session_state.get("stat_category"),
    )
    st.session_state._stat_category_picker = resolved_category
    st.session_state.stat_category = resolved_category


def _format_stat_category_label(stat_category: str) -> str:
    """Return the emoji-decorated label for the category segmented control.

    Args:
        stat_category: Canonical internal category value.

    Returns:
        Display label with emoji while preserving plain internal values.
    """
    safe_category = _sanitize_stat_category(stat_category)
    return _STAT_CATEGORY_LABELS[safe_category]


@st.cache_data(ttl=300)
def _check_api_health() -> list:
    """Probe key NHL endpoints and return `(label, ok)` pairs."""
    probes = [
        ("Search",       "https://search.d3.nhle.com/api/v1/search/player?q=Mc&limit=1&culture=en-us"),
        ("Player Stats", "https://api-web.nhle.com/v1/player/8478402/landing"),
        ("Roster",       "https://api-web.nhle.com/v1/roster/EDM/current"),
        ("Team Stats",   "https://api.nhle.com/stats/rest/en/team/summary?limit=1"),
        ("Records",      "https://records.nhl.com/site/api/skater-career-scoring-regular-season"),
    ]
    try:
        import requests
    except Exception:
        return [(label, False) for label, _ in probes]
    results = []
    for label, url in probes:
        try:
            r = requests.get(url, timeout=3, stream=True)
            r.close()
            results.append((label, r.status_code < 400))
        except Exception:
            results.append((label, False))
    return results


def _build_support_button_markup(
    url: str = _SUPPORT_URL,
    label: str = _SUPPORT_LABEL,
    sublabel: str = _SUPPORT_SUBLABEL,
) -> str:
    """Build the sidebar support CTA markup.

    Args:
        url: Destination support URL.
        label: Visible CTA label.
        sublabel: Smaller helper text shown under the CTA label.

    Returns:
        Safe HTML markup for the sidebar support link.
    """
    safe_url = escape(url, quote=True)
    safe_label = escape(label)
    safe_sublabel = escape(sublabel)
    return (
        "<a class='sidebar-support-link' "
        f"href='{safe_url}' target='_blank' rel='noopener noreferrer'>"
        f"<span class='sidebar-support-link__emoji' aria-hidden='true'>{_SUPPORT_EMOJI}</span>"
        "<span class='sidebar-support-link__text'>"
        f"<span class='sidebar-support-link__label'>{safe_label}</span>"
        f"<span class='sidebar-support-link__sublabel'>{safe_sublabel}</span>"
        "</span>"
        "</a>"
    )


def _render_support_button() -> None:
    """Render the sidebar Ko-fi support CTA.

    Args:
        None.

    Returns:
        None.
    """
    st.markdown(_build_support_button_markup(), unsafe_allow_html=True)


def _render_ram_footer() -> None:
    """Render the sidebar RAM readout plus cached API health indicators."""
    rss_mb = "N/A"
    try:
        import os
        import psutil
        rss_mb = f"{psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.0f} MB"
    except Exception:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_mb = f"{int(line.split()[1]) / 1024:.0f} MB"
                        break
        except Exception:
            pass

    _render_support_button()

    with st.expander("App status", expanded=False):
        st.caption(f"RAM: {rss_mb}")
        try:
            statuses = _check_api_health()
            lines = ["**API Health** *(5 min cache)*"]
            for label, ok in statuses:
                dot = "🟢" if ok else "🟡"
                lines.append(f"{dot} {label}")
            st.caption("\n\n".join(lines))
        except Exception:
            st.caption("API Health: unavailable")


def _inject_no_keyboard() -> None:
    """Stop mobile browsers from popping the keyboard on selector-style dropdowns."""
    components.html(
        """
        <script>
        (function() {
            function fixInputs() {
                window.parent.document
                    .querySelectorAll('[data-baseweb="select"] input')
                    .forEach(function(el) {
                        el.setAttribute('inputmode', 'none');
                        el.setAttribute('readonly', 'readonly');
                    });
            }

            // Run immediately on load
            fixInputs();

            // Re-apply after every Streamlit rerun (DOM mutation)
            new MutationObserver(fixInputs).observe(
                window.parent.document.body,
                { childList: true, subtree: true }
            );
        })();
        </script>
        """,
        height=0,   # invisible — zero visual footprint
    )


def _inject_sidebar_overlay_dismiss() -> None:
    """Close the overlay sidebar when the user taps outside it on cramped screens.

    Args:
        None.

    Returns:
        None.
    """
    components.html(
        """
        <script>
        (function() {
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            const overlayMedia = parentWindow.matchMedia(
                '(max-width: 1400px), ((max-width: 1600px) and (max-aspect-ratio: 11/10))'
            );

            if (typeof parentWindow.__ppSidebarOverlayCleanup === 'function') {
                parentWindow.__ppSidebarOverlayCleanup();
            }

            let backdrop = null;
            let rafId = null;

            function getSidebar() {
                return doc.querySelector('section[data-testid="stSidebar"]');
            }

            function getCollapseButton() {
                return doc.querySelector('[data-testid="stSidebarCollapseButton"] button');
            }

            function isSidebarOpen(sidebar) {
                return !!sidebar && sidebar.getAttribute('aria-expanded') === 'true';
            }

            function removeBackdrop() {
                if (backdrop) {
                    backdrop.remove();
                    backdrop = null;
                }
            }

            function ensureBackdrop() {
                if (backdrop && doc.body.contains(backdrop)) {
                    return backdrop;
                }

                backdrop = doc.createElement('div');
                backdrop.setAttribute('data-testid', 'ppSidebarOverlayBackdrop');
                Object.assign(backdrop.style, {
                    position: 'fixed',
                    right: '0',
                    bottom: '0',
                    background: 'rgba(3, 8, 18, 0.22)',
                    zIndex: '1001',
                    pointerEvents: 'auto',
                    touchAction: 'manipulation',
                });
                backdrop.addEventListener('pointerdown', function(event) {
                    event.preventDefault();
                    event.stopPropagation();
                    const collapseButton = getCollapseButton();
                    if (collapseButton) {
                        collapseButton.click();
                    }
                });
                doc.body.appendChild(backdrop);
                return backdrop;
            }

            function updateBackdrop() {
                const sidebar = getSidebar();
                if (!overlayMedia.matches || !isSidebarOpen(sidebar)) {
                    removeBackdrop();
                    return;
                }

                const rect = sidebar.getBoundingClientRect();
                const nextBackdrop = ensureBackdrop();
                nextBackdrop.style.top = Math.max(0, rect.top) + 'px';
                nextBackdrop.style.left = Math.max(0, rect.right) + 'px';
            }

            function scheduleUpdate() {
                if (rafId !== null) {
                    parentWindow.cancelAnimationFrame(rafId);
                }
                rafId = parentWindow.requestAnimationFrame(function() {
                    rafId = null;
                    updateBackdrop();
                });
            }

            const observer = new MutationObserver(scheduleUpdate);
            observer.observe(doc.body, {
                attributes: true,
                childList: true,
                subtree: true,
                attributeFilter: ['aria-expanded', 'style', 'class'],
            });

            parentWindow.addEventListener('resize', scheduleUpdate, { passive: true });
            if (typeof overlayMedia.addEventListener === 'function') {
                overlayMedia.addEventListener('change', scheduleUpdate);
            } else if (typeof overlayMedia.addListener === 'function') {
                overlayMedia.addListener(scheduleUpdate);
            }

            parentWindow.__ppSidebarOverlayCleanup = function() {
                observer.disconnect();
                parentWindow.removeEventListener('resize', scheduleUpdate, { passive: true });
                if (typeof overlayMedia.removeEventListener === 'function') {
                    overlayMedia.removeEventListener('change', scheduleUpdate);
                } else if (typeof overlayMedia.removeListener === 'function') {
                    overlayMedia.removeListener(scheduleUpdate);
                }
                if (rafId !== null) {
                    parentWindow.cancelAnimationFrame(rafId);
                    rafId = null;
                }
                removeBackdrop();
            };

            scheduleUpdate();
        })();
        </script>
        """,
        height=0,
    )


def render_sidebar() -> dict:
    """Render the sidebar and return the widget-derived chart cache keys."""
    with st.sidebar:
        _inject_no_keyboard()   # Prevent mobile keyboard on dropdowns
        _inject_sidebar_overlay_dismiss()   # Close overlay sidebar when tapping outside it
        current_category = _sanitize_stat_category(st.session_state.get("stat_category"))
        st.session_state.stat_category = current_category
        if st.session_state.get("_stat_category_picker") != current_category:
            st.session_state._stat_category_picker = current_category


        st.markdown("<div class='faq-btn-anchor'></div>", unsafe_allow_html=True)
        if st.button(
            "FAQ",
            key="open_app_guide_sidebar",
            type="secondary",
            use_container_width=True,
            help="How this app works",
        ):
            show_app_guide()

        st.markdown(
            "<div class='comparison-panel-heading comparison-panel-heading--rail-title'"
            " style='margin:0.5rem auto 0.3rem;'>Category</div>",
            unsafe_allow_html=True,
        )
        st.segmented_control(
            "Category",
            options=_VALID_STAT_CATEGORIES,
            selection_mode="single",
            format_func=_format_stat_category_label,
            key="_stat_category_picker",
            on_change=_sync_stat_category_selection,
            label_visibility="collapsed",
            width="stretch",
        )
        if st.session_state.stat_category != "Team":
            return _render_player_sidebar()
        else:
            return _render_team_sidebar()


def _render_player_sidebar() -> dict:
    """Render the player-mode sidebar: search, top-50, roster, and player board.

    Three ways to add a player:
        1. Global search (D3 API + local records fallback).
        2. Top 50 all-time dropdown (auto-adds on selection).
        3. Active roster selector by team (auto-adds on selection).

    Writes to:
        st.session_state.players     — dict {pid: name} shared across Skater/Goalie modes
        st.session_state.search_ver  — incrementing int to reset the search box
        st.session_state.search_opts — current search result dict
        st.session_state.top50_ver   — incrementing int to reset the top-50 box
        st.session_state.top50_opts  — current top-50 dict for callback lookup
        st.session_state.roster_ver  — incrementing int to reset the roster box
        st.session_state.roster_opts — current roster dict for callback lookup

    Returns:
        Dict with sidebar keys for chart cache-busting.
    """
    search_term   = ""
    top_selected  = ""
    team_abbr     = ""
    roster_player = ""

    if 'search_ver' not in st.session_state:
        st.session_state.search_ver = 0
    if 'search_opts' not in st.session_state:
        st.session_state.search_opts = {}
    if 'top50_ver' not in st.session_state:
        st.session_state.top50_ver = 0
    if 'top50_opts' not in st.session_state:
        st.session_state.top50_opts = {}
    if 'roster_ver' not in st.session_state:
        st.session_state.roster_ver = 0
    if 'roster_opts' not in st.session_state:
        st.session_state.roster_opts = {}

    def _on_player_select():
        """Callback: add the selected player to the shared board and reset the search box."""
        ver  = st.session_state.search_ver
        sel  = st.session_state.get(f"_player_pick_{ver}")
        _SENT = "— select a player —"
        if not sel or sel == _SENT:
            return
        pid = st.session_state.search_opts.get(sel)
        if pid is None:
            return
        name = sel.split("] ")[-1] if "]" in sel else sel
        st.session_state.players[pid] = name
        st.session_state.search_ver  = ver + 1
        st.session_state.search_opts = {}

    def _on_top50_select():
        """Callback: add the selected top-50 player to the shared board and reset the dropdown."""
        ver  = st.session_state.top50_ver
        sel  = st.session_state.get(f"_top50_pick_{ver}")
        _SENT = "— select a player —"
        if not sel or sel == _SENT:
            return
        pid = st.session_state.top50_opts.get(sel)
        if pid is None:
            return
        name = sel.split(". ", 1)[-1].split(" (")[0]
        st.session_state.players[pid] = name
        st.session_state.top50_ver = ver + 1

    def _on_roster_select():
        """Callback: add the selected roster player to the shared board and reset the dropdown."""
        ver  = st.session_state.roster_ver
        sel  = st.session_state.get(f"_roster_pick_{ver}")
        _SENT = "— select a player —"
        if not sel or sel == _SENT:
            return
        pid = st.session_state.roster_opts.get(sel)
        if pid is None:
            return
        name = sel.split("] ")[-1] if "]" in sel else sel
        if " #" in name:
            name = name.split(" #")[0]
        st.session_state.players[pid] = name
        st.session_state.roster_ver = ver + 1

    st.markdown(
        "<div class='comparison-panel-heading comparison-panel-heading--rail-title'"
        " style='margin:0.5rem auto 0.3rem;'>Global Search</div>",
        unsafe_allow_html=True,
    )
    search_term = st.text_input(
        "Global Search",
        placeholder="e.g., McDavid, Crosby, Connor…",
        label_visibility="collapsed",
        key=f"search_input_{st.session_state.search_ver}",
    )

    opts = {}
    if search_term:
        results = search_player(search_term)
        for p in results:
            tm    = p.get('teamAbbrev')
            label = f"[{tm}] {p['name']}" if tm else p['name']
            opts[label] = int(p['playerId'])
        local = search_local_players(search_term, st.session_state.stat_category)
        for label, pid in local.items():
            if pid not in opts.values():
                opts[label] = pid
        # Active players (have [TEAM] prefix) first, retired/free agents below
        active_opts   = {k: v for k, v in opts.items() if k.startswith("[")}
        inactive_opts = {k: v for k, v in opts.items() if not k.startswith("[")}
        opts = {**active_opts, **inactive_opts}

    st.session_state.search_opts = opts

    if opts:
        _SENT = "— select a player —"
        st.selectbox(
            "Results:",
            [_SENT] + list(opts.keys()),
            key=f"_player_pick_{st.session_state.search_ver}",
            on_change=_on_player_select,
            label_visibility="collapsed",
        )
    elif search_term:
        st.caption("No players found")

    _is_goalie_mode = st.session_state.stat_category == "Goalie"
    if _is_goalie_mode:
        top_50_dict  = get_top_50_goalies()
        top_50_label = "Top 50 All-Time Goalies"
    else:
        current_metric = st.session_state.get("skater_metric", "Points")
        top_50_dict    = get_top_50(current_metric)
        top_50_label   = "Top 50 All-Time Skaters"
    _SENT = "— select a player —"
    st.session_state.top50_opts = top_50_dict
    st.markdown(
        f"<div class='comparison-panel-heading comparison-panel-heading--rail-title'"
        f" style='margin:0.5rem auto 0.3rem;'>{top_50_label}</div>",
        unsafe_allow_html=True,
    )
    top_selected = st.selectbox(
        top_50_label,
        [_SENT] + list(top_50_dict.keys()),
        key=f"_top50_pick_{st.session_state.top50_ver}",
        on_change=_on_top50_select,
        label_visibility="collapsed",
    )

    st.markdown(
        "<div class='comparison-panel-heading comparison-panel-heading--rail-title'"
        " style='margin:0.5rem auto 0.3rem;'>Active Rosters</div>",
        unsafe_allow_html=True,
    )
    team_abbr = st.selectbox(
        "Active Rosters",
        list(ACTIVE_TEAMS.keys()),
        format_func=lambda x: f"{x} - {ACTIVE_TEAMS[x]}",
        label_visibility="collapsed",
    )
    if team_abbr:
        st.markdown(
            f"<div style='text-align:center;margin-bottom:5px;'>"
            f"<img src='https://assets.nhle.com/logos/nhl/svg/{team_abbr}_light.svg' height='40'>"
            f"</div>",
            unsafe_allow_html=True,
        )
        roster = get_team_roster(team_abbr)
        if _is_goalie_mode:
            roster = {k: v for k, v in roster.items() if k.startswith("[G]")}
        else:
            roster = {k: v for k, v in roster.items() if not k.startswith("[G]")}
        if roster:
            st.session_state.roster_opts = roster
            roster_player = st.selectbox(
                "Select Player:",
                [_SENT] + list(roster.keys()),
                key=f"_roster_pick_{st.session_state.roster_ver}",
                on_change=_on_roster_select,
                label_visibility="collapsed",
            )

    st.markdown("---")
    if st.session_state.players:
        for pid, name in list(st.session_state.players.items()):
            c_name, c_btn = st.columns([8, 1], vertical_alignment="center", gap="small")
            with c_name:
                headshot = get_player_headshot(pid)
                safe_name = escape(str(name or ""))
                img_html = (
                    f"<span class='pp-skel-headshot-wrap'>"
                    f"<img src='{headshot}' loading='lazy' decoding='async' alt='{safe_name}'>"
                    f"</span>"
                    if headshot else ""
                )
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:0;'>"
                    f"{img_html}"
                    f"<div class='player-name'>{safe_name}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c_btn:
                if st.button("✖", key=f"drop_{pid}", type="secondary"):
                    del st.session_state.players[pid]
                    st.rerun()
    else:
        st.info("Board is empty")

    _render_ram_footer()

    return {
        "search_term":   search_term,
        "top_selected":  top_selected,
        "team_abbr":     team_abbr,
        "roster_player": roster_player,
    }


def _render_team_sidebar() -> dict:
    """Render the team-mode sidebar: team selector (auto-adds on selection), team board.

    The team logo is shown above the dropdown and reflects the current dropdown selection.
    Selecting a team immediately adds it to the board and resets the dropdown.

    Writes to:
        st.session_state.teams    — dict {abbr: name}
        st.session_state.team_ver — incrementing int to reset the team selectbox

    Returns:
        Dict with sidebar keys for chart cache-busting (only 'team_abbr' is relevant).
    """
    if 'team_ver' not in st.session_state:
        st.session_state.team_ver = 0

    _SENT = "— select a team —"
    _team_keys = list(ACTIVE_TEAMS.keys())

    def _on_team_select():
        """Callback: add the selected team to the board and reset the dropdown."""
        ver = st.session_state.team_ver
        sel = st.session_state.get(f"_team_pick_{ver}")
        if not sel or sel == _SENT:
            return
        st.session_state.teams[sel] = ACTIVE_TEAMS[sel]
        st.session_state.team_ver = ver + 1

    # Logo shown ABOVE the dropdown — reflects current dropdown selection
    _logo_abbr = st.session_state.get(f"_team_pick_{st.session_state.team_ver}", _SENT)
    if _logo_abbr and _logo_abbr != _SENT:
        st.markdown(
            f"<div style='text-align:center;margin-bottom:5px;'>"
            f"<img src='https://assets.nhle.com/logos/nhl/svg/{_logo_abbr}_light.svg' height='40'>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='comparison-panel-heading comparison-panel-heading--rail-title'"
        " style='margin:0.5rem auto 0.3rem;'>Team Comparison</div>",
        unsafe_allow_html=True,
    )
    _team_sel = st.selectbox(
        "Team Comparison",
        [_SENT] + _team_keys,
        format_func=lambda x: x if x == _SENT else f"{x} — {ACTIVE_TEAMS[x]}",
        key=f"_team_pick_{st.session_state.team_ver}",
        on_change=_on_team_select,
        label_visibility="collapsed",
    )

    st.markdown("---")

    if st.session_state.teams:
        for _abbr, _name in list(st.session_state.teams.items()):
            c_name, c_btn = st.columns([5, 1], vertical_alignment="center", gap="small")
            with c_name:
                _logo_url = f"https://assets.nhle.com/logos/nhl/svg/{_abbr}_light.svg"
                safe_team_name = escape(str(_name or ""))
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:0;'>"
                    f"<img src='{_logo_url}' style='width:32px;height:32px;"
                    f"object-fit:contain;flex-shrink:0;'>"
                    f"<div class='player-name'>{safe_team_name}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c_btn:
                if st.button("✖", key=f"drop_team_{_abbr}", type="secondary"):
                    del st.session_state.teams[_abbr]
                    st.rerun()
    else:
        st.info("Board is empty")

    _render_ram_footer()

    return {
        "search_term":   "",
        "top_selected":  "",
        "team_abbr":     _team_sel if _team_sel != _SENT else "",
        "roster_player": "",
    }

