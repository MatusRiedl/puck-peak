"""Compact URL encoding and decoding for shareable app state."""

import re

from nhl.constants import ACTIVE_TEAMS

_VALID_SKATER_METRICS = {"Points", "Goals", "Assists", "+/-", "GP", "PPG", "SH%", "PIM", "TOI"}
_VALID_GOALIE_METRICS = {"Save %", "GAA", "Shutouts", "Wins", "GP", "Saves"}
_VALID_TEAM_METRICS   = {"Points", "Wins", "Win%", "Goals", "GF/G", "GA/G", "PP%", "PPG"}

_CAT_ENCODE = {"Skater": "S", "Goalie": "G", "Team": "T"}
_CAT_DECODE = {v: k for k, v in _CAT_ENCODE.items()}

_XM_ENCODE  = {"Age": "A", "Games Played": "GP", "Season Year": "SY"}
_XM_DECODE  = {v: k for k, v in _XM_ENCODE.items()}

_BOOL_PARAMS = {
    "sm":  "do_smooth",
    "pr":  "do_predict",
    "era": "do_era",
    "cu":  "do_cumul_toggle",
    "bl":  "do_base",
    "pf":  "do_prime",
}

_BOOL_DEFAULTS = {
    "do_smooth": False,
    "do_predict": True,
    "do_era": False,
    "do_cumul_toggle": False,
    "do_base": True,
    "do_prime": True,
}

_METRIC_DEFAULTS = {
    "skater_metric": "Points",
    "goalie_metric": "Save %",
    "team_metric": "Points",
}

_VALID_PANEL_TABS = {"overview", "current-standings"}
_LEGACY_PANEL_TAB_ALIASES = {
    "trophies": "overview",
    "stanley_cup": "current-standings",
}

_PLAYER_ID_PATTERN = re.compile(r"[0-9]{1,12}")
_TEAM_ABBR_PATTERN = re.compile(r"[A-Z]{2,4}")
_MAX_SHARED_DISPLAY_NAME_LENGTH = 120


def _sanitize_panel_tab(value: str) -> str:
    """Validate and normalize comparison-panel tab IDs from session/URL.

    Args:
        value: Raw panel-tab identifier from session state or the query string.

    Returns:
        A safe lowercase tab key, or ``"overview"`` if the value is invalid.
    """
    if value is None:
        return "overview"
    v = str(value).strip().lower()
    if not v:
        return "overview"
    if len(v) > 32:
        return "overview"
    if not re.fullmatch(r"[a-z0-9_-]+", v):
        return "overview"
    v = _LEGACY_PANEL_TAB_ALIASES.get(v, v)
    if v not in _VALID_PANEL_TABS:
        return "overview"
    return v


def _default_x_axis_mode(stat_category: str) -> str:
    """Return the implicit default X-axis mode for a stat category.

    Args:
        stat_category: Current chart category.

    Returns:
        ``"Season Year"`` for Team mode and ``"Age"`` otherwise.
    """
    return "Season Year" if stat_category == "Team" else "Age"


def _sanitize_chart_season(value) -> str | int:
    """Validate the chart-top season selector value from session or URL.

    Args:
        value: Raw selector value.

    Returns:
        ``"All"`` or a safe four-digit season start year.
    """
    if value is None:
        return "All"
    clean = str(value).strip()
    if not clean or clean.lower() == "all":
        return "All"
    if not re.fullmatch(r"[0-9]{4}", clean):
        return "All"
    year = int(clean)
    if year < 1900 or year > 2100:
        return "All"
    return year


def _coerce_player_param_key(value: object) -> str | None:
    """Return one normalized player ID string when the shared-link key is valid."""
    clean_value = str(value or "").strip()
    if not _PLAYER_ID_PATTERN.fullmatch(clean_value):
        return None
    try:
        numeric_value = int(clean_value)
    except ValueError:
        return None
    if numeric_value <= 0:
        return None
    return str(numeric_value)


def _coerce_team_param_key(value: object) -> str | None:
    """Return one normalized team abbreviation when the shared-link key is valid."""
    clean_value = str(value or "").strip().upper()
    if not _TEAM_ABBR_PATTERN.fullmatch(clean_value):
        return None
    return clean_value


def _sanitize_shared_display_name(value: object) -> str:
    """Normalize untrusted shared-link display text into inert plain text.

    The shared-link query string can contain legacy ``pid|name`` and
    ``abbr|name`` payloads. These display names are treated as untrusted input,
    so control characters are removed, HTML tags are stripped, and whitespace is
    normalized before the value is stored in session state.
    """
    raw_value = str(value or "")
    filtered_value = "".join(ch for ch in raw_value if ch.isprintable() or ch.isspace())
    filtered_value = re.sub(r"<[^>]*>", "", filtered_value)
    filtered_value = filtered_value.replace("<", "").replace(">", "")
    filtered_value = re.sub(r"\s+", " ", filtered_value).strip()
    return filtered_value[:_MAX_SHARED_DISPLAY_NAME_LENGTH]


def _resolve_shared_player_names(
    players: dict[str, str],
    id_to_name_lookup: dict[str, str],
) -> dict[str, str]:
    """Resolve shared-link player state to canonical or sanitized display names."""
    resolved_players: dict[str, str] = {}
    for player_id, display_name in (players or {}).items():
        clean_player_id = _coerce_player_param_key(player_id)
        if not clean_player_id:
            continue

        canonical_name = str(id_to_name_lookup.get(clean_player_id, "") or "").strip()
        if canonical_name:
            resolved_players[clean_player_id] = canonical_name
            continue

        fallback_name = _sanitize_shared_display_name(display_name)
        if fallback_name and fallback_name != clean_player_id:
            resolved_players[clean_player_id] = fallback_name
            continue

        resolved_players[clean_player_id] = f"Unknown (ID {clean_player_id})"

    return resolved_players


def _resolve_shared_team_names(
    teams: dict[str, str],
    active_teams: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve shared-link team state to canonical or sanitized display names."""
    active_team_lookup = active_teams or ACTIVE_TEAMS
    resolved_teams: dict[str, str] = {}
    for team_abbr, display_name in (teams or {}).items():
        clean_team_abbr = _coerce_team_param_key(team_abbr)
        if not clean_team_abbr:
            continue

        canonical_name = str(active_team_lookup.get(clean_team_abbr, "") or "").strip()
        if canonical_name:
            resolved_teams[clean_team_abbr] = canonical_name
            continue

        fallback_name = _sanitize_shared_display_name(display_name)
        resolved_teams[clean_team_abbr] = fallback_name or clean_team_abbr

    return resolved_teams


def _parse_player_params(raw_value: str) -> dict[str, str]:
    """Parse short or legacy player query params into session-state shape.

    Args:
        raw_value: Semicolon-separated player entries from the URL.

    Returns:
        Dict mapping player ID strings to either the decoded name or the raw ID
        placeholder when the short ID-only format is used.
    """
    players: dict[str, str] = {}
    for entry in str(raw_value or "").split(";"):
        clean_entry = str(entry).strip()
        if not clean_entry:
            continue
        if "|" in clean_entry:
            pid, name = clean_entry.split("|", 1)
            clean_player_id = _coerce_player_param_key(pid)
            if clean_player_id:
                sanitized_name = _sanitize_shared_display_name(name)
                players[clean_player_id] = sanitized_name or clean_player_id
            continue
        clean_player_id = _coerce_player_param_key(clean_entry)
        if clean_player_id:
            players[clean_player_id] = clean_player_id
    return players


def _parse_team_params(raw_value: str) -> dict[str, str]:
    """Parse short or legacy team query params into session-state shape.

    Args:
        raw_value: Semicolon-separated team entries from the URL.

    Returns:
        Dict mapping team abbreviations to the decoded name or the raw
        abbreviation placeholder when the short format is used.
    """
    teams: dict[str, str] = {}
    for entry in str(raw_value or "").split(";"):
        clean_entry = str(entry).strip()
        if not clean_entry:
            continue
        if "|" in clean_entry:
            abbr, name = clean_entry.split("|", 1)
            clean_team_abbr = _coerce_team_param_key(abbr)
            if clean_team_abbr:
                sanitized_name = _sanitize_shared_display_name(name)
                teams[clean_team_abbr] = sanitized_name or clean_team_abbr
            continue
        clean_team_abbr = _coerce_team_param_key(clean_entry)
        if clean_team_abbr:
            teams[clean_team_abbr] = clean_team_abbr
    return teams


def encode_state_to_params(ss) -> dict:
    """Convert current session state into a compact URL params dict.

    Args:
        ss: ``st.session_state`` or any dict-like mapping containing app state.

    Returns:
        Dict mapping URL param keys to encoded string values, omitting entries
        that still match the app defaults.
    """
    params = {}

    stat_category = ss.get("stat_category", "Skater")
    if stat_category != "Skater":
        params["cat"] = _CAT_ENCODE.get(stat_category, "S")

    if ss.get("skater_metric") not in (None, _METRIC_DEFAULTS["skater_metric"]):
        params["sk_m"] = ss["skater_metric"]
    if ss.get("goalie_metric") not in (None, _METRIC_DEFAULTS["goalie_metric"]):
        params["go_m"] = ss["goalie_metric"]
    if ss.get("team_metric") not in (None, _METRIC_DEFAULTS["team_metric"]):
        params["tm_m"] = ss["team_metric"]

    season_type = ss.get("season_type", "Regular")
    if season_type != "Regular":
        params["sp"] = season_type

    chart_season = _sanitize_chart_season(ss.get("chart_season", "All"))
    if chart_season != "All":
        params["cs"] = str(chart_season)

    x_axis_mode = ss.get("x_axis_mode", _default_x_axis_mode(stat_category))
    season_forces_games = (
        chart_season != "All"
        and x_axis_mode == "Games Played"
    )
    if x_axis_mode != _default_x_axis_mode(stat_category) and not season_forces_games:
        params["xm"] = _XM_ENCODE.get(x_axis_mode, "A")

    league_filter = ss.get("league_filter") or ["NHL"]
    if league_filter != ["NHL"]:
        params["lg"] = ",".join(league_filter)

    for url_key, ss_key in _BOOL_PARAMS.items():
        default_value = _BOOL_DEFAULTS[ss_key]
        current_value = bool(ss.get(ss_key, default_value))
        if current_value != default_value:
            params[url_key] = "1" if current_value else "0"

    panel_tab_skater = _sanitize_panel_tab(ss.get("panel_tab_skater", "overview"))
    if panel_tab_skater != "overview":
        params["pt_s"] = panel_tab_skater

    panel_tab_goalie = _sanitize_panel_tab(ss.get("panel_tab_goalie", "overview"))
    if panel_tab_goalie != "overview":
        params["pt_g"] = panel_tab_goalie

    panel_tab_team = _sanitize_panel_tab(ss.get("panel_tab_team", "overview"))
    if panel_tab_team != "overview":
        params["pt_t"] = panel_tab_team

    pl = ss.get("players") or {}
    if pl:
        params["pl"] = ";".join(str(pid).strip() for pid in pl if str(pid).strip())

    tm = ss.get("teams") or {}
    if tm:
        params["tm"] = ";".join(str(abbr).strip().upper() for abbr in tm if str(abbr).strip())

    return params


def apply_params_to_state(params: dict, ss) -> None:
    """Apply known URL params into session state without overwriting missing defaults."""
    if not params:
        return

    if "cat" in params:
        ss["stat_category"] = _CAT_DECODE.get(params["cat"], "Skater")

    if "sk_m" in params and params["sk_m"] in _VALID_SKATER_METRICS:
        ss["skater_metric"] = params["sk_m"]
    if "go_m" in params and params["go_m"] in _VALID_GOALIE_METRICS:
        ss["goalie_metric"] = params["go_m"]
    if "tm_m" in params and params["tm_m"] in _VALID_TEAM_METRICS:
        ss["team_metric"] = params["tm_m"]

    if "sp" in params and params["sp"] in ("Regular", "Playoffs", "Both"):
        ss["season_type"] = params["sp"]

    if "cs" in params:
        ss["chart_season"] = _sanitize_chart_season(params["cs"])

    if "xm" in params:
        ss["x_axis_mode"] = _XM_DECODE.get(params["xm"], "Age")

    if "lg" in params and params["lg"]:
        leagues: list[str] = []
        for lg in params["lg"].split(","):
            clean = str(lg).strip()
            if clean and clean not in leagues:
                leagues.append(clean)
        if leagues:
            ss["league_filter"] = leagues

    for url_key, ss_key in _BOOL_PARAMS.items():
        if url_key in params:
            ss[ss_key] = params[url_key] == "1"

    if "pt_s" in params:
        ss["panel_tab_skater"] = _sanitize_panel_tab(params["pt_s"])
    if "pt_g" in params:
        ss["panel_tab_goalie"] = _sanitize_panel_tab(params["pt_g"])
    if "pt_t" in params:
        ss["panel_tab_team"] = _sanitize_panel_tab(params["pt_t"])

    _players = {}
    for _key in ("sk", "go", "pl"):
        if _key in params and params[_key]:
            _players.update(_parse_player_params(params[_key]))
    if _players:
        ss["players"] = _players

    if "tm" in params and params["tm"]:
        teams = _parse_team_params(params["tm"])
        if teams:
            ss["teams"] = teams
