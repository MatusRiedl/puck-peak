"""
nhl.fragments — `@st.fragment` wrappers around the three slow render panels.

Fragments exist solely so that *post-load* widget interactions inside the
chart, detail tabs, or predictions panel rerun only that scoped block — not
the entire app. They do not own the cold-load lifecycle: app.py paints
skeletons first, runs the pipeline, then mounts the appropriate fragment
into the swapped slot.

Keeping these wrappers thin and module-level so Streamlit can register the
fragment scopes properly.
"""

import streamlit as st

from nhl.chart import render_chart
from nhl.comparison import render_detail_tabs, render_predictions_panel


@st.fragment
def chart_fragment(**kwargs) -> None:
    """Render the main Plotly chart inside an isolated fragment scope.

    Args:
        **kwargs: Forwarded verbatim to `nhl.chart.render_chart`.
    """
    render_chart(**kwargs)


@st.fragment
def detail_tabs_fragment(**kwargs) -> None:
    """Render the Overview / Current Standings tabs inside a fragment scope.

    Args:
        **kwargs: Forwarded verbatim to `nhl.comparison.render_detail_tabs`.
    """
    render_detail_tabs(**kwargs)


@st.fragment
def predictions_fragment(**kwargs) -> None:
    """Render the right-rail predictions panel inside a fragment scope.

    Args:
        **kwargs: Forwarded verbatim to `nhl.comparison.render_predictions_panel`.
    """
    render_predictions_panel(**kwargs)
