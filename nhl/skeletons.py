"""
nhl.skeletons — Static HTML skeleton loaders for paint-first rendering.

Generates lightweight shimmer placeholders that paint before any data fetch
runs, so the user sees page structure within ~100ms instead of a blank page
during cold-cache loads. Real content is mounted later via slot.empty() +
slot.container() swaps in app.py.

All blocks share the `pp-skel-block` CSS class (defined in nhl.styles); the
shimmer animation honors prefers-reduced-motion automatically.
"""


def chart_skeleton(height_px: int = 480) -> str:
    """Return HTML for a chart-shaped shimmer placeholder.

    Args:
        height_px: Approximate pixel height matching the real Plotly chart
            so the skeleton-to-real swap causes no layout drift.

    Returns:
        A single HTML string with a chart-aspect grey block plus a small
        toolbar strip above it.
    """
    return (
        "<div class='pp-skel-wrap'>"
        f"  <div class='pp-skel-block pp-skel-chart-toolbar'></div>"
        f"  <div class='pp-skel-block pp-skel-chart' style='height:{int(height_px)}px;'></div>"
        "</div>"
    )


def detail_tabs_skeleton() -> str:
    """Return HTML for the Overview / Current Standings tab section placeholder.

    The block mimics a pill row plus a three-row card grid so the eye lands
    on the same structural shape that will appear once data resolves.

    Returns:
        A single HTML string with a tab-pill strip and three stacked card
        rows.
    """
    cards = "".join(
        "<div class='pp-skel-block pp-skel-card'></div>" for _ in range(3)
    )
    return (
        "<div class='pp-skel-wrap'>"
        "  <div class='pp-skel-tabs'>"
        "    <div class='pp-skel-block pp-skel-pill'></div>"
        "    <div class='pp-skel-block pp-skel-pill'></div>"
        "  </div>"
        f"  <div class='pp-skel-grid'>{cards}</div>"
        "</div>"
    )


def predictions_skeleton(count: int = 4) -> str:
    """Return HTML for the right-rail predictions panel placeholder.

    Args:
        count: Number of game-card rows to render.

    Returns:
        A single HTML string with a heading bar and N stacked game-card
        rows.
    """
    games = "".join(
        "<div class='pp-skel-block pp-skel-game'></div>" for _ in range(int(count))
    )
    return (
        "<div class='pp-skel-wrap'>"
        "  <div class='pp-skel-block pp-skel-heading'></div>"
        f"  <div class='pp-skel-stack'>{games}</div>"
        "</div>"
    )
