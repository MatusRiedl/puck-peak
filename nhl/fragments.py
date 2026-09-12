"""
nhl.fragments — `@st.fragment` wrappers around the three slow render panels.

Fragments exist solely so that *post-load* widget interactions inside the
chart, detail tabs, or predictions panel rerun only that scoped block — not
the entire app. They do not own the load lifecycle: app.py creates empty
slots, runs the pipeline, then mounts the appropriate fragment into the
slot.

Keeping these wrappers thin and module-level so Streamlit can register the
fragment scopes properly.
"""

import streamlit as st

from nhl.chart import render_chart
from nhl.comparison import render_detail_tabs, render_predictions_panel
from nhl.dialog import show_app_guide
from nhl.ui_state import begin_dialog_run, dialog_slot_available, mark_dialog_opened_this_run


@st.fragment
def chart_fragment(**kwargs) -> None:
    """Render the main Plotly chart inside an isolated fragment scope.

    Args:
        **kwargs: Forwarded verbatim to `nhl.chart.render_chart`.
    """
    begin_dialog_run("chart")
    render_chart(**kwargs)


@st.fragment
def detail_tabs_fragment(**kwargs) -> None:
    """Render the Overview / Current Standings tabs inside a fragment scope.

    Args:
        **kwargs: Forwarded verbatim to `nhl.comparison.render_detail_tabs`.
    """
    begin_dialog_run("detail_tabs")
    render_detail_tabs(**kwargs)


@st.fragment
def predictions_fragment(**kwargs) -> None:
    """Render the right-rail predictions panel inside a fragment scope.

    Args:
        **kwargs: Forwarded verbatim to `nhl.comparison.render_predictions_panel`.
    """
    begin_dialog_run("predictions")
    render_predictions_panel(**kwargs)


@st.fragment
def faq_button_fragment() -> None:
    """Render the sidebar FAQ button and its guide dialog in an isolated scope.

    At top-level script scope this button forced a full script rerun — CSS
    re-inject, the whole player/team pipeline, a rebuilt Plotly figure — purely to
    open a modal that reads none of that. Fragment-scoped, the click reruns nothing
    but this button.

    This is also the call site that used to skip the one-dialog mutex entirely; the
    gate below brings it in line with the five dialogs opened from chart.py and
    comparison.py.
    """
    begin_dialog_run("faq")
    st.markdown("<div class='faq-btn-anchor'></div>", unsafe_allow_html=True)
    if st.button(
        "FAQ",
        key="open_app_guide_sidebar",
        type="secondary",
        use_container_width=True,
        help="How this app works",
    ):
        if dialog_slot_available():
            mark_dialog_opened_this_run()
            show_app_guide()
