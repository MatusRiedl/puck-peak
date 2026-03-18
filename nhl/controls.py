"""Top control surface for chart options, stat selection, and view toggles."""

from html import escape

import streamlit as st

from nhl.constants import RATE_STATS, TEAM_METRICS, TEAM_RATE_STATS, normalize_league_abbrev
from nhl.data_loaders import get_player_league_abbrevs


_CONTROL_PILL_SPECS: tuple[dict[str, str], ...] = (
    {"label": "Smooth", "state_key": "do_smooth"},
    {"label": "Proj", "state_key": "do_predict"},
    {"label": "Era", "state_key": "do_era"},
    {"label": "Cumul", "state_key": "do_cumul_toggle"},
    {"label": "Base", "state_key": "do_base"},
    {"label": "Prime", "state_key": "do_prime"},
)


def _get_control_pill_groups(
    team_mode: bool,
    games_mode: bool,
    season_mode: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    """Split toolbar pills into active and muted groups when needed.

    Args:
        team_mode: Whether Team mode is active.
        games_mode: Whether Games Played mode is active.
        season_mode: Whether single-season game-log mode is active.

    Returns:
        tuple[list[dict[str, str]], list[str]]: Active pill specs and any muted labels to show.
    """
    unavailable = set()
    hide_unavailable = False
    if team_mode and season_mode:
        unavailable.update({"Proj", "Era", "Cumul", "Base", "Prime"})
    elif team_mode:
        unavailable.update({"Proj", "Era", "Base", "Prime"})
    elif season_mode:
        unavailable.update({"Proj", "Era", "Base", "Prime"})
        hide_unavailable = True
    elif games_mode:
        unavailable.update({"Proj", "Base"})

    available_specs = [spec for spec in _CONTROL_PILL_SPECS if spec["label"] not in unavailable]
    unavailable_labels = [] if hide_unavailable else [
        spec["label"] for spec in _CONTROL_PILL_SPECS if spec["label"] in unavailable
    ]
    return available_specs, unavailable_labels


def _selected_control_pills_from_state(available_specs: list[dict[str, str]]) -> list[str]:
    """Read the currently enabled toolbar pills from session state.

    Args:
        available_specs: Active pill specs for the current mode.

    Returns:
        list[str]: Labels that should render as selected.
    """
    return [
        spec["label"]
        for spec in available_specs
        if bool(st.session_state.get(spec["state_key"], False))
    ]


def _prepare_control_pills_widget_state(available_specs: list[dict[str, str]]) -> None:
    """Keep the pills widget state aligned with the active mode.

    Args:
        available_specs: Active pill specs for the current mode.

    Returns:
        None.
    """
    available_labels = [spec["label"] for spec in available_specs]
    next_selected = _selected_control_pills_from_state(available_specs)
    next_scope = tuple(available_labels)
    current_scope = st.session_state.get("_controls_option_pills_scope")
    current_value = st.session_state.get("_controls_option_pills")

    if current_scope != next_scope:
        st.session_state["_controls_option_pills"] = next_selected
        st.session_state["_controls_option_pills_scope"] = next_scope
        return

    if not isinstance(current_value, list):
        st.session_state["_controls_option_pills"] = next_selected
        return

    invalid_labels = [label for label in current_value if label not in available_labels]
    if invalid_labels:
        st.session_state["_controls_option_pills"] = [
            label for label in current_value if label in available_labels
        ]


def _sync_control_bool_state(available_specs: list[dict[str, str]], selected_labels) -> None:
    """Write selected pills back into the existing boolean session keys.

    Args:
        available_specs: Active pill specs for the current mode.
        selected_labels: Pills selected by the widget.

    Returns:
        None.
    """
    if selected_labels is None:
        selected_lookup = set()
    elif isinstance(selected_labels, str):
        selected_lookup = {selected_labels}
    else:
        selected_lookup = set(selected_labels)

    for spec in available_specs:
        st.session_state[spec["state_key"]] = spec["label"] in selected_lookup


def render_controls() -> tuple:
    """Render the controls expander and return `(metric, do_cumul)`."""
    team_mode = st.session_state.stat_category == "Team"
    season_mode = st.session_state.get("chart_season", "All") != "All"
    games_mode = st.session_state.x_axis_mode == "Games Played" or season_mode

    with st.popover("⚙️ Metric Selections"):
        # ------------------------------------------------------------------
        # Row 1: Compact pill toolbar
        # ------------------------------------------------------------------
        _cumul_rate_set = TEAM_RATE_STATS if team_mode else RATE_STATS
        _available_pills, _unavailable_pills = _get_control_pill_groups(
            team_mode,
            games_mode,
            season_mode,
        )

        st.markdown("<div id='controls-row1'></div>", unsafe_allow_html=True)
        _prepare_control_pills_widget_state(_available_pills)
        _selected_pills = st.pills(
            "View options",
            options=[spec["label"] for spec in _available_pills],
            selection_mode="multi",
            key="_controls_option_pills",
            label_visibility="collapsed",
            width="stretch",
        )
        _sync_control_bool_state(_available_pills, _selected_pills)
        if team_mode and season_mode:
            st.session_state.do_smooth = False

        if _unavailable_pills:
            _muted_pills = "".join(
                f"<span class='controls-pill controls-pill--disabled'>{escape(label)}</span>"
                for label in _unavailable_pills
            )
            st.markdown(
                "<div class='controls-toolbar-muted'>"
                "<span class='controls-toolbar-muted__label'>Unavailable:</span>"
                f"{_muted_pills}"
                "</div>",
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------------
        # Row 2: X axis | Metric | Season Type | Leagues dropdowns
        # ------------------------------------------------------------------
        _x_opts = (
            ["Season Year", "Games Played"]
            if st.session_state.stat_category == "Team"
            else ["Age", "Games Played"]
        )

        st.markdown("<div id='controls-dropdowns'></div>", unsafe_allow_html=True)
        c_xaxis, c_metric, c_season, c_league = st.columns(
            [1.5, 1.5, 1.5, 2], vertical_alignment="top"
        )

        with c_xaxis:
            st.selectbox(
                "X axis",
                _x_opts,
                key="x_axis_mode",
                disabled=season_mode,
                help=(
                    "Season Year: plot by NHL season (teams). "
                    "Age: plot by player age. "
                    "Games Played: cumulative game number. "
                    "Single-season chart mode overrides this to individual games."
                ),
            )

        with c_metric:
            if st.session_state.stat_category == "Skater":
                metric = st.selectbox(
                    "Metric",
                    ["Points", "Goals", "Assists", "+/-", "GP", "PPG", "SH%", "PIM", "TOI"],
                    key="skater_metric",
                    help=(
                        "+/-: Plus/Minus Differential | GP: Games Played | "
                        "PPG: Points Per Game | SH%: Shooting Percentage | "
                        "PIM: Penalty Minutes | TOI: Time on Ice (Avg Mins)"
                    ),
                )
            elif st.session_state.stat_category == "Goalie":
                metric = st.selectbox(
                    "Metric",
                    ["Save %", "GAA", "Shutouts", "Wins", "GP", "Saves"],
                    key="goalie_metric",
                    help=(
                        "Save %: Save Percentage | GAA: Goals Against Average | "
                        "GP: Games Played | Saves: Total Saves"
                    ),
                )
            else:
                metric = st.selectbox(
                    "Metric",
                    TEAM_METRICS,
                    key="team_metric",
                    help=(
                        "Points: standings pts | Wins: wins per season | Win%: pts pct | "
                        "Goals: team GF | GF/G: goals for/game | GA/G: goals against/game | "
                        "PP%: power play % | PPG: team scoring pts/game (est.)"
                    ),
                )

        with c_season:
            st.selectbox("Season Type", ["Regular", "Playoffs", "Both"], key="season_type")

        with c_league:
            if season_mode:
                st.multiselect(
                    "Leagues",
                    options=["NHL"],
                    key="league_filter",
                    disabled=True,
                    help="Single-season mode uses NHL game logs only.",
                )
            elif team_mode:
                st.multiselect(
                    "Leagues",
                    options=["NHL"],
                    default=["NHL"],
                    disabled=True,
                    help="Unavailable in Team mode.",
                )
            else:
                # Dynamic league universe from currently loaded players.
                _player_ids: set[int] = set()
                for _board_key in ("players", "skater_players", "goalie_players"):
                    _board = st.session_state.get(_board_key, {}) or {}
                    if isinstance(_board, dict):
                        for _pid in _board.keys():
                            try:
                                _player_ids.add(int(_pid))
                            except Exception:
                                continue

                _league_set = {"NHL"}
                for _pid in _player_ids:
                    for _lg in get_player_league_abbrevs(_pid):
                        if _lg:
                            _league_set.add(_lg)

                _non_nhl_sorted = sorted(
                    (lg for lg in _league_set if normalize_league_abbrev(lg) != "NHL"),
                    key=lambda s: s.upper(),
                )
                _league_options = ["NHL"] + _non_nhl_sorted

                # Keep current selection valid against dynamic options via normalized match.
                _current_selection = st.session_state.get("league_filter")
                if _current_selection is None:
                    _current_selection = ["NHL"]
                _norm_to_display: dict[str, str] = {}
                for _opt in _league_options:
                    _norm = normalize_league_abbrev(_opt)
                    if _norm and _norm not in _norm_to_display:
                        _norm_to_display[_norm] = _opt
                _resolved_selection: list[str] = []
                for _sel in _current_selection:
                    _mapped = _norm_to_display.get(normalize_league_abbrev(_sel))
                    if _mapped and _mapped not in _resolved_selection:
                        _resolved_selection.append(_mapped)
                st.session_state.league_filter = _resolved_selection

                st.multiselect(
                    "Leagues",
                    options=_league_options,
                    key="league_filter",
                    help=(
                        "NHL is available but optional. Additional options are discovered from "
                        "seasonTotals leagueAbbrev values of currently loaded players. "
                        "With Era on, non-NHL skater Points, Goals, and Assists are "
                        "league-normalized; with Era off, league scoring stays raw. GP stays raw."
                    ),
                )
                _non_nhl = [
                    l for l in (st.session_state.league_filter or [])
                    if normalize_league_abbrev(l) != 'NHL'
                ]
                if _non_nhl:
                    if st.session_state.do_era:
                        st.caption(f"League-normalized with Era on: {', '.join(_non_nhl)}")
                    else:
                        st.caption(f"Raw league scoring with Era off: {', '.join(_non_nhl)}")

        # Captions for toggle context (rendered after metric is resolved)
        if st.session_state.do_cumul_toggle and metric in _cumul_rate_set:
            st.caption(f"⚠️ Cumulative disabled — {metric} is a rate stat.")
        if games_mode and not season_mode:
            _gm_note = (
                "ℹ️ Cumulative & Baseline unavailable in Games mode."
                if team_mode
                else "ℹ️ Projection & Baseline unavailable in Games mode."
            )
            st.caption(_gm_note)
        _ERA_GOALIE_STATS = {'Save %', 'GAA', 'Shutouts'}
        if (
            st.session_state.do_era
            and st.session_state.stat_category == 'Goalie'
            and metric not in _ERA_GOALIE_STATS
        ):
            st.caption(
                f"ℹ️ Era adjust for goalies applies to Save %, GAA, and Shutouts. "
                f"{metric} is not era-adjusted."
            )

        # Resolve do_cumul: False when metric is a rate stat; games_mode honours the toggle
        do_cumul = (
            st.session_state.do_cumul_toggle
            and metric not in _cumul_rate_set
            and not (team_mode and season_mode)
        )

    return metric, do_cumul
