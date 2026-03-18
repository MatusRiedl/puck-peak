"""Background cache warmers for non-active categories."""

import threading
from typing import Callable


def _preload_in_thread(target: Callable, name: str) -> None:
    """Start a daemon thread for a cache-warming target."""
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()


def preload_goalie_data() -> None:
    """Warm cached goalie lookup data in background threads."""
    from nhl.data_loaders import get_clone_details_map, get_id_to_name_map

    _preload_in_thread(lambda: get_id_to_name_map("Goalie"), "preload_goalie_names")
    _preload_in_thread(lambda: get_clone_details_map("Goalie"), "preload_goalie_details")


def preload_team_data() -> None:
    """Warm cached team-season data in a background thread."""
    from nhl.data_loaders import load_all_team_seasons

    _preload_in_thread(load_all_team_seasons, "preload_team_seasons")


def preload_all_categories(current_category: str = "Skater") -> None:
    """Warm caches for categories other than the one currently being viewed."""
    # Preload Goalie data if not currently viewing Goalies
    if current_category != "Goalie":
        preload_goalie_data()

    # Preload Team data if not currently viewing Teams
    if current_category != "Team":
        preload_team_data()

    # Note: Skater data is loaded on-demand since that's the default category.
    # If we start on Goalie or Team mode, Skater data will be preloaded too.
    if current_category != "Skater":
        from nhl.data_loaders import get_clone_details_map, get_id_to_name_map

        _preload_in_thread(lambda: get_id_to_name_map("Skater"), "preload_skater_names")
        _preload_in_thread(lambda: get_clone_details_map("Skater"), "preload_skater_details")
