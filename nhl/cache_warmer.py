"""Optional background cache warmer for shared NHL cache."""

import logging
import os
import random
import threading
import time
from collections.abc import Callable

from nhl.data_loaders import (
    get_clone_details_map,
    get_current_nhl_standings,
    get_id_to_name_map,
    get_player_landing,
    get_team_roster,
    get_top_50,
    get_top_50_goalies,
    load_all_team_seasons,
)
from nhl.schedule import get_featured_players, get_live_or_recent_game, get_upcoming_games

log = logging.getLogger("nhl.cache_warmer")

_ENABLED_ENV = "PUCKPEAK_CACHE_WARMER_ENABLED"
_LIVE_INTERVAL_ENV = "PUCKPEAK_CACHE_WARMER_LIVE_INTERVAL_SECONDS"
_SEASONAL_INTERVAL_ENV = "PUCKPEAK_CACHE_WARMER_SEASONAL_INTERVAL_SECONDS"
_HISTORICAL_INTERVAL_ENV = "PUCKPEAK_CACHE_WARMER_HISTORICAL_INTERVAL_SECONDS"

_DEFAULT_LIVE_INTERVAL_SECONDS = 300
_DEFAULT_SEASONAL_INTERVAL_SECONDS = 3600
_DEFAULT_HISTORICAL_INTERVAL_SECONDS = 21600

_SEEDED_TEAMS = ("EDM", "PIT", "WSH", "COL", "TOR", "NYR")
_SEEDED_PLAYERS = (
    8478402,  # Connor McDavid
    8471675,  # Sidney Crosby
    8471214,  # Alex Ovechkin
    8477492,  # Nathan MacKinnon
    8479318,  # Auston Matthews
    8478048,  # Igor Shesterkin
)

_warmer_started = False
_warmer_lock = threading.Lock()


def _read_bool_env(name: str, default: bool = False) -> bool:
    """Parse a boolean-ish environment flag with safe fallback."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    normalized = str(raw_value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False

    log.warning("Invalid boolean env %s=%r, using default %s", name, raw_value, default)
    return default


def _read_int_env(name: str, default: int, minimum: int = 1) -> int:
    """Parse an integer environment variable with validation."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        parsed = int(str(raw_value).strip())
    except (TypeError, ValueError):
        log.warning("Invalid integer env %s=%r, using default %s", name, raw_value, default)
        return default

    if parsed < minimum:
        log.warning(
            "Out-of-range env %s=%r, using default %s (minimum %s)",
            name,
            raw_value,
            default,
            minimum,
        )
        return default

    return parsed


def _run_safe_task(task_name: str, task_fn: Callable[[], object]) -> object | None:
    """Run one warm-up task and swallow all exceptions."""
    try:
        return task_fn()
    except Exception:
        log.exception("Cache warmer task failed: %s", task_name)
        return None


def _run_live_cycle() -> None:
    """Warm live schedule paths used by the default home surfaces."""
    matchup = _run_safe_task("get_live_or_recent_game", get_live_or_recent_game)
    if isinstance(matchup, tuple) and len(matchup) == 2:
        home_abbr, away_abbr = matchup
        _run_safe_task(
            f"get_featured_players:{home_abbr}:{away_abbr}",
            lambda: get_featured_players(home_abbr, away_abbr),
        )

    _run_safe_task(
        "get_upcoming_games:8:14",
        lambda: get_upcoming_games(limit=8, days_ahead=14),
    )


def _run_seasonal_cycle() -> None:
    """Warm high-traffic seasonal entry points and a small seed set."""
    _run_safe_task("load_all_team_seasons", load_all_team_seasons)
    _run_safe_task("get_current_nhl_standings", get_current_nhl_standings)

    for team_abbr in _SEEDED_TEAMS:
        _run_safe_task(
            f"get_team_roster:{team_abbr}",
            lambda team_abbr=team_abbr: get_team_roster(team_abbr),
        )

    for player_id in _SEEDED_PLAYERS:
        _run_safe_task(
            f"get_player_landing:{player_id}",
            lambda player_id=player_id: get_player_landing(player_id),
        )


def _run_historical_cycle() -> None:
    """Warm historical lookups that are shared across many user flows."""
    _run_safe_task('get_id_to_name_map:"Skater"', lambda: get_id_to_name_map("Skater"))
    _run_safe_task('get_clone_details_map:"Skater"', lambda: get_clone_details_map("Skater"))
    _run_safe_task('get_id_to_name_map:"Goalie"', lambda: get_id_to_name_map("Goalie"))
    _run_safe_task('get_clone_details_map:"Goalie"', lambda: get_clone_details_map("Goalie"))
    _run_safe_task('get_top_50:"Points"', lambda: get_top_50("Points"))
    _run_safe_task('get_top_50:"Goals"', lambda: get_top_50("Goals"))
    _run_safe_task('get_top_50:"Assists"', lambda: get_top_50("Assists"))
    _run_safe_task("get_top_50_goalies", get_top_50_goalies)


def _warmer_loop(
    tier_name: str,
    interval_seconds: int,
    cycle_fn: Callable[[], None],
    jitter_range_seconds: tuple[int, int],
) -> None:
    """Run one tier's cache-warming loop forever on a daemon thread."""
    jitter_low, jitter_high = jitter_range_seconds
    if jitter_high > 0:
        startup_delay = random.uniform(jitter_low, jitter_high)
        if startup_delay > 0:
            time.sleep(startup_delay)

    while True:
        try:
            cycle_fn()
        except Exception:
            log.exception("Cache warmer loop failed for tier %s", tier_name)
        time.sleep(interval_seconds)


def _start_tier_thread(
    tier_name: str,
    interval_seconds: int,
    cycle_fn: Callable[[], None],
    jitter_range_seconds: tuple[int, int],
) -> None:
    """Start one daemon thread for a warming tier."""
    thread = threading.Thread(
        target=_warmer_loop,
        name=f"cache-warmer-{tier_name}",
        args=(tier_name, interval_seconds, cycle_fn, jitter_range_seconds),
        daemon=True,
    )
    thread.start()


def start_background_warmer() -> bool:
    """Start the background cache warmer once per process.

    Returns:
        True when a new set of warmer threads was started, else False.
    """
    global _warmer_started

    if not _read_bool_env(_ENABLED_ENV, default=False):
        return False

    with _warmer_lock:
        if _warmer_started:
            return False

        live_interval = _read_int_env(_LIVE_INTERVAL_ENV, _DEFAULT_LIVE_INTERVAL_SECONDS)
        seasonal_interval = _read_int_env(
            _SEASONAL_INTERVAL_ENV,
            _DEFAULT_SEASONAL_INTERVAL_SECONDS,
        )
        historical_interval = _read_int_env(
            _HISTORICAL_INTERVAL_ENV,
            _DEFAULT_HISTORICAL_INTERVAL_SECONDS,
        )

        _start_tier_thread("live", live_interval, _run_live_cycle, (0, 15))
        _start_tier_thread("seasonal", seasonal_interval, _run_seasonal_cycle, (15, 60))
        _start_tier_thread("historical", historical_interval, _run_historical_cycle, (60, 180))
        _warmer_started = True

    log.info(
        "Started cache warmer threads with live=%ss seasonal=%ss historical=%ss",
        live_interval,
        seasonal_interval,
        historical_interval,
    )
    return True


__all__ = ["start_background_warmer"]
