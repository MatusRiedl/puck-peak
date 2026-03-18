"""Cached parquet and NHL API loaders.

Keep the silent fallback behavior. The APIs are undocumented and occasionally
weird.
"""

import json
import os
from datetime import date, datetime

import pandas as pd
import requests  # Used only by discover_all_leagues (audit helper)
import streamlit as st

from nhl.api import get_client
from nhl.cache import get_cache, T1_TTL, T2_DEFAULT_TTL, effective_ttl
from nhl.constants import (
    ACTIVE_TEAMS,
    CURRENT_SEASON_YEAR,
    NHLE_DEFAULT_MULTIPLIER,
    NHLE_MULTIPLIERS,
    PLAYER_GAME_LOG_URL,
    SEASON_GOALIE_SUMMARY_URL,
    SEASON_SKATER_SUMMARY_URL,
    SEARCH_URL,
    STATS_URL,
    ROSTER_URL,
    TEAM_FOUNDED,
    TEAM_LIST_URL,
    TEAM_STATS_URL,
    TEAM_METRICS,
    TEAM_LINEAGES,
    normalize_league_abbrev,
)
from nhl.win_prob import validate_model_artifact


_TEAM_ALIAS_TO_ACTIVE = {
    alias: active_abbr
    for active_abbr, aliases in TEAM_LINEAGES.items()
    for alias in aliases
}


def _normalize_historical_goalie_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce historical goalie `SavePct` values into sane 0-1 scale."""
    if df.empty or 'SavePct' not in df.columns or 'Position' not in df.columns:
        return df

    d = df.copy()
    goalie_mask = d['Position'].astype(str).str.upper().eq('G')
    if not goalie_mask.any():
        return d

    save_pct = pd.to_numeric(d.loc[goalie_mask, 'SavePct'], errors='coerce')
    over_one = save_pct > 1.5
    if over_one.any():
        save_pct.loc[over_one] = save_pct.loc[over_one] / 100.0

    d.loc[goalie_mask, 'SavePct'] = save_pct.clip(lower=0.0, upper=1.0).fillna(0.0)
    return d


def _canonical_team_abbrev(team_abbr: str | None) -> str:
    """Return the active-team abbreviation for a historical franchise alias."""
    clean_abbr = str(team_abbr or "").strip().upper()
    if not clean_abbr:
        return ""
    return _TEAM_ALIAS_TO_ACTIVE.get(clean_abbr, clean_abbr)


def _payload_text(value: object) -> str:
    """Return a clean string from plain or NHL nested-name payload values."""
    if isinstance(value, dict):
        for key in ("default", "fr", "cs", "de", "es", "fi", "sk", "sv"):
            text = str(value.get(key, "") or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _format_season_span(season_year: int | None) -> str:
    """Format one NHL start year like ``2024`` into ``2024-25``."""
    if season_year is None:
        return ""
    try:
        year = int(season_year)
    except Exception:
        return ""
    return f"{year}-{str(year + 1)[2:]}"


def _format_season_id_span(season_id: int | None) -> str:
    """Format one NHL seasonId like ``20242025`` into ``2024-25``."""
    if season_id is None:
        return ""
    try:
        raw_value = int(season_id)
    except Exception:
        return ""

    raw_text = str(raw_value)
    start_year = raw_value
    if len(raw_text) >= 8:
        try:
            start_year = int(raw_text[:4])
        except Exception:
            return ""
    return _format_season_span(start_year)


def _parse_iso_date(value: str) -> datetime | None:
    """Parse a ``YYYY-MM-DD`` date string when possible."""
    clean_value = str(value or "").strip()
    if len(clean_value) < 10:
        return None
    try:
        return datetime.strptime(clean_value[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _calculate_age_from_birth_date(value: str) -> int | None:
    """Return the current age for a birth date string."""
    birth_dt = _parse_iso_date(value)
    if birth_dt is None:
        return None
    today = date.today()
    age = today.year - birth_dt.year
    if (today.month, today.day) < (birth_dt.month, birth_dt.day):
        age -= 1
    return max(age, 0)


def _format_height_label(height_inches, height_cm) -> str:
    """Format height in imperial and metric units when available."""
    try:
        total_inches = int(height_inches)
    except Exception:
        total_inches = 0
    try:
        centimeters = int(height_cm)
    except Exception:
        centimeters = 0

    parts: list[str] = []
    if total_inches > 0:
        feet = total_inches // 12
        inches = total_inches % 12
        parts.append(f"{feet}'{inches}\"")
    if centimeters > 0:
        parts.append(f"{centimeters} cm")
    return " / ".join(parts)


def _format_weight_label(weight_lb, weight_kg) -> str:
    """Format weight in imperial and metric units when available."""
    try:
        pounds = int(weight_lb)
    except Exception:
        pounds = 0
    try:
        kilograms = int(weight_kg)
    except Exception:
        kilograms = 0

    parts: list[str] = []
    if pounds > 0:
        parts.append(f"{pounds} lb")
    if kilograms > 0:
        parts.append(f"{kilograms} kg")
    return " / ".join(parts)


def _build_record_label(wins: int, losses: int, ot_losses: int, ties: int = 0) -> str:
    """Return a standard NHL record label."""
    if ot_losses > 0:
        return f"{wins}-{losses}-{ot_losses}"
    if ties > 0:
        return f"{wins}-{losses}-{ties}"
    return f"{wins}-{losses}"


def _format_draft_summary(draft_details: object) -> str:
    """Normalize draft details into one compact sentence."""
    if not isinstance(draft_details, dict):
        return "Undrafted"

    try:
        draft_year = int(draft_details.get("year", 0) or 0)
    except Exception:
        draft_year = 0
    team_abbr = str(draft_details.get("teamAbbrev", "") or "").strip().upper()

    try:
        draft_round = int(draft_details.get("round", 0) or 0)
    except Exception:
        draft_round = 0
    try:
        pick_in_round = int(draft_details.get("pickInRound", 0) or 0)
    except Exception:
        pick_in_round = 0
    try:
        overall_pick = int(draft_details.get("overallPick", 0) or 0)
    except Exception:
        overall_pick = 0

    if draft_year <= 0 and not team_abbr and overall_pick <= 0:
        return "Undrafted"

    parts: list[str] = []
    if draft_year > 0:
        parts.append(str(draft_year))
    if team_abbr:
        parts.append(team_abbr)
    if draft_round > 0:
        round_text = f"Round {draft_round}"
        if pick_in_round > 0:
            round_text += f", pick {pick_in_round}"
        parts.append(round_text)
    elif pick_in_round > 0:
        parts.append(f"Pick {pick_in_round}")
    if overall_pick > 0:
        parts.append(f"{overall_pick} overall")
    return " | ".join(parts) if parts else "Undrafted"


def _format_lineage_segment(segment: dict) -> str:
    """Format one franchise-identity stint for display."""
    start_year = segment.get("start_year")
    end_year = segment.get("end_year")
    name = str(segment.get("name", "") or segment.get("abbr", "") or "").strip()
    if not name:
        return ""

    start_label = _format_season_span(start_year)
    if not start_label:
        return name

    if end_year is None:
        end_label = "present"
    else:
        try:
            clean_end_year = int(end_year)
        except Exception:
            clean_end_year = None
        if clean_end_year is None:
            end_label = "present"
        elif clean_end_year >= CURRENT_SEASON_YEAR:
            end_label = "present"
        elif clean_end_year == int(start_year):
            end_label = ""
        else:
            end_label = _format_season_span(clean_end_year)

    if not end_label:
        return f"{name} ({start_label})"
    return f"{name} ({start_label} to {end_label})"


# ---------------------------------------------------------------------------
# Parquet / historical data
# ---------------------------------------------------------------------------

@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """Load the historical parquet and add the derived columns the app expects."""
    try:
        if os.path.exists("nhl_historical_seasons.parquet"):
            df = pd.read_parquet("nhl_historical_seasons.parquet")
            if "Position" not in df.columns:
                df["Position"] = "S"
            for col in ("GP", "Points", "Goals", "SavePct"):
                if col not in df.columns:
                    df[col] = 0.0
            if "Shots" not in df.columns:
                df["Shots"] = 0.0
            if "TotalTOIMins" not in df.columns:
                df["TotalTOIMins"] = 0.0

            # Some historical parquet variants label all rows as skaters.
            # When no explicit goalie rows exist, infer goalies from
            # goalie-only counting stats so downstream Position=='G' filters work.
            _pos = df["Position"].astype(str).str.upper()
            if (_pos == "G").sum() == 0:
                goalie_mask = pd.Series(False, index=df.index)
                for col in ("Saves", "Wins", "Shutouts"):
                    if col in df.columns:
                        goalie_mask = goalie_mask | df[col].fillna(0).gt(0)
                if goalie_mask.any():
                    df.loc[goalie_mask, "Position"] = "G"

            df = _normalize_historical_goalie_rates(df)

            gp_denom = pd.to_numeric(df.get('GP', 0), errors='coerce').replace(0, float('nan'))
            shots_denom = pd.to_numeric(df.get('Shots', 0), errors='coerce').replace(0, float('nan'))
            df['PPG'] = pd.to_numeric(df.get('Points', 0), errors='coerce').div(gp_denom).fillna(0.0)
            df['Save %'] = pd.to_numeric(df.get('SavePct', 0), errors='coerce').mul(100.0).fillna(0.0)
            df['SH%'] = (
                pd.to_numeric(df.get('Goals', 0), errors='coerce')
                .div(shots_denom)
                .mul(100.0)
                .fillna(0.0)
            )
            df['TOI'] = pd.to_numeric(df.get('TotalTOIMins', 0), errors='coerce').div(gp_denom).fillna(0.0)
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data
def load_win_prob_weights() -> dict:
    """Load the exported win-probability model artifact from disk."""
    try:
        if not os.path.exists("win_prob_weights.json"):
            return {}
        with open("win_prob_weights.json", "r", encoding="utf-8") as infile:
            payload = json.load(infile)
        return validate_model_artifact(payload)
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_all_team_seasons() -> pd.DataFrame:
    """Fetch all team-season records from the NHL stats REST API.

    Fetches regular-season and playoff rows separately (gameTypeId 2 and 3),
    then concatenates them into one table with an explicit gameTypeId column.
    teamAbbrev (triCode) is joined from the separate team-list endpoint.
    Cached hourly so active-season rows do not go stale during the season.

    Returns:
        DataFrame with one row per team-season, or an empty DataFrame on failure.
    """
    try:
        client = get_client()

        # Build teamId -> triCode map from team list
        team_list_data = client.get(
            url=TEAM_LIST_URL,
            cache_key="team_list",
            ttl=T1_TTL,
            timeout=15,
        )
        if team_list_data is None:
            return pd.DataFrame()
        team_list = team_list_data.get("data", [])
        id_to_tricode = {
            t["id"]: t["triCode"]
            for t in team_list
            if "id" in t and "triCode" in t
        }

        def _fetch_team_summary_by_type(game_type_id: int) -> pd.DataFrame:
            """Fetch team/summary rows filtered by game type."""
            try:
                resp = client.get(
                    url=TEAM_STATS_URL,
                    params={"limit": -1, "cayenneExp": f"gameTypeId={game_type_id}"},
                    cache_key=f"team_summary:{game_type_id}",
                    ttl=T2_DEFAULT_TTL,
                    timeout=30,
                )
                if resp is None:
                    return pd.DataFrame()
                rows = resp.get("data", [])
                if not rows:
                    return pd.DataFrame()
                dfx = pd.DataFrame(rows)
                dfx["gameTypeId"] = game_type_id
                return dfx
            except Exception:
                return pd.DataFrame()

        # Fetch regular season and playoffs separately.
        reg_df = _fetch_team_summary_by_type(2)
        ply_df = _fetch_team_summary_by_type(3)
        if reg_df.empty and ply_df.empty:
            return pd.DataFrame()
        df = pd.concat([d for d in (reg_df, ply_df) if not d.empty], ignore_index=True)

        # Attach teamAbbrev from the triCode map
        df["teamAbbrev"] = df["teamId"].map(id_to_tricode)
        df["FranchiseAbbrev"] = df["teamAbbrev"].apply(_canonical_team_abbrev)
        # Derive columns
        df["SeasonYear"] = df["seasonId"] // 10000
        df["GP"]     = df["gamesPlayed"]
        df["Wins"]   = df["wins"]
        df["Points"] = df["points"]
        if "pointPct" in df.columns:
            df["Win%"] = (df["pointPct"] * 100).round(1)
        else:
            df["Win%"] = (df["points"] / (df["gamesPlayed"] * 2) * 100).round(1)
        df["GF/G"]  = df["goalsForPerGame"].round(3)
        df["GA/G"]  = df["goalsAgainstPerGame"].round(3)
        df["Goals"] = df["goalsFor"]
        df["Losses"] = df["losses"]
        df["OTLosses"] = df["otLosses"]
        df["Ties"] = df["ties"]
        df["PPG"]   = (df["goalsFor"] / df["gamesPlayed"] * 2.7).round(3)
        df["PP%"]   = (
            (df["powerPlayPct"] * 100).round(1)
            if "powerPlayPct" in df.columns
            else float("nan")
        )

        keep = [
            "teamId", "teamFullName", "teamAbbrev", "FranchiseAbbrev", "seasonId", "gameTypeId",
            "SeasonYear", "GP", "Wins", "Losses", "OTLosses", "Ties", "Points", "Win%",
            "Goals", "GF/G", "GA/G", "PPG", "PP%",
        ]
        return df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# NHL records API helpers
# ---------------------------------------------------------------------------

def _paginate_records(base_url: str) -> list:
    """Fetch all pages from an NHL records endpoint.

    The default page size for records.nhl.com is ~25 rows; this function uses
    page_size=500 to minimise round-trips.

    Args:
        base_url: Records endpoint URL without query parameters.

    Returns:
        List of raw record dicts from all pages combined.
    """
    client = get_client()
    all_data = []
    start = 0
    page_size = 500
    while True:
        try:
            payload = client.get(
                url=base_url,
                params={"start": start, "limit": page_size},
                timeout=15,
            )
            if payload is None:
                break
            page = payload.get('data', [])
            all_data.extend(page)
            if len(page) < page_size:
                break
            start += page_size
        except Exception:
            break
    return all_data


@st.cache_data(ttl=3600)
def fetch_all_time_records(category: str, s_type: str) -> list:
    """Fetch career records for skaters or goalies from records.nhl.com.

    Combines regular-season and playoff data when s_type == 'Both' by summing
    numeric fields per player.

    Args:
        category: 'Skater' or 'Goalie'.
        s_type:   'Regular', 'Playoffs', or 'Both'.

    Returns:
        List of record dicts, each containing playerId and career stat fields.
        Returns [] on network failure.
    """
    cache_key = f"records:{category}:{s_type}"
    cache = get_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        if category == "Skater":
            reg_url = "https://records.nhl.com/site/api/skater-career-scoring-regular-season"
            ply_url = "https://records.nhl.com/site/api/skater-career-scoring-playoff"
        else:
            reg_url = "https://records.nhl.com/site/api/goalie-career-stats"
            ply_url = "https://records.nhl.com/site/api/goalie-career-playoff-stats"

        reg_data = _paginate_records(reg_url)
        if s_type == "Regular":
            cache.set(cache_key, reg_data, ttl=T1_TTL)
            return reg_data

        ply_data = _paginate_records(ply_url)
        if s_type == "Playoffs":
            cache.set(cache_key, ply_data, ttl=T1_TTL)
            return ply_data

        # s_type == "Both" — combine by summing numeric fields per player
        combined = {}
        for r in reg_data:
            combined[r['playerId']] = r.copy()
        for p in ply_data:
            pid = p['playerId']
            if pid in combined:
                for k in ['points', 'goals', 'assists', 'gamesPlayed',
                          'penaltyMinutes', 'wins', 'shutouts', 'saves', 'plusMinus']:
                    if k in p and k in combined[pid]:
                        combined[pid][k] += p[k]
            else:
                combined[pid] = p.copy()
        result = list(combined.values())
        cache.set(cache_key, result, ttl=T1_TTL)
        return result
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_top_50(metric: str = "Points") -> dict:
    """Fetch the top 50 all-time NHL skaters ranked by the specified career counting stat.

    Only Points, Goals, and Assists are supported sort keys. All other metric values
    fall back to Points ranking as the least-surprising default.

    Args:
        metric: Stat label to sort by. One of 'Points', 'Goals', 'Assists', or any
            other skater metric string (non-matching values default to 'Points').

    Returns:
        Dict mapping display label (e.g. '1. Wayne Gretzky (2857 P)') to playerId int.
        Falls back to a hardcoded 4-player dict (no stat suffix) if the API call fails.
    """
    _SORT_MAP   = {"Points": "points", "Goals": "goals", "Assists": "assists"}
    _SUFFIX_MAP = {"Points": "P",      "Goals": "G",     "Assists": "A"}
    sort_key = _SORT_MAP.get(metric, "points")
    suffix   = _SUFFIX_MAP.get(metric, "P")
    try:
        res = get_client().get(
            url="https://records.nhl.com/site/api/skater-career-scoring-regular-season",
            params={"sort": sort_key, "dir": "DESC", "limit": 100},
            cache_key=f"top50_skater:{metric}",
            ttl=T1_TTL,
            timeout=5,
        )
        if res is None:
            raise ValueError("API returned None")
        players = {}
        added_ids = set()
        count = 1
        for p in res.get('data', []):
            pid = int(p['playerId'])
            if pid not in added_ids:
                stat_val = p.get(sort_key, 0)
                base     = f"{count}. {p.get('firstName', '')} {p.get('lastName', '')}".strip()
                name     = f"{base} ({stat_val} {suffix})"
                players[name] = pid
                added_ids.add(pid)
                count += 1
                if count > 50:
                    break
        if players:
            return players
    except Exception:
        pass
    return {
        "1. Wayne Gretzky": 8447400,
        "2. Jaromir Jagr":  8448208,
        "3. Sidney Crosby": 8471675,
        "4. Alexander Ovechkin": 8471214,
    }


@st.cache_data(ttl=3600)
def get_top_50_goalies() -> dict:
    """Fetch the top 50 all-time NHL goalies ranked by career regular-season wins.

    Returns:
        Dict mapping display label ('1. Martin Brodeur') to playerId int.
        Falls back to a hardcoded 4-player dict if the API call fails.
    """
    try:
        res = get_client().get(
            url="https://records.nhl.com/site/api/goalie-career-stats",
            params={"sort": "wins", "dir": "DESC", "limit": 100},
            cache_key="top50_goalie",
            ttl=T1_TTL,
            timeout=5,
        )
        if res is None:
            raise ValueError("API returned None")
        players = {}
        added_ids = set()
        count = 1
        for p in res.get('data', []):
            pid = int(p['playerId'])
            if pid not in added_ids:
                name = f"{count}. {p.get('firstName', '')} {p.get('lastName', '')}".strip()
                players[name] = pid
                added_ids.add(pid)
                count += 1
                if count > 50:
                    break
        if players:
            return players
    except Exception:
        pass
    return {
        "1. Martin Brodeur":    8455710,
        "2. Patrick Roy":       8451033,
        "3. Marc-Andre Fleury": 8471679,
        "4. Roberto Luongo":    8466141,
    }


# ---------------------------------------------------------------------------
# Player search
# ---------------------------------------------------------------------------

def _normalize_search_results(payload: object) -> list[dict]:
    """Return only valid player rows from the NHL search payload."""
    if isinstance(payload, dict):
        rows = payload.get('data', [])
    elif isinstance(payload, list):
        rows = payload
    else:
        return []

    clean_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        try:
            player_id = int(row.get('playerId'))
        except (TypeError, ValueError):
            continue

        name = str(row.get('name') or '').strip()
        if not name:
            first = str(row.get('firstName', '') or '').strip()
            last = str(row.get('lastName', '') or '').strip()
            name = f"{first} {last}".strip()
        if not name:
            continue

        clean_rows.append(
            {
                'playerId': player_id,
                'name': name,
                'teamAbbrev': str(row.get('teamAbbrev', '') or '').strip(),
            }
        )
    return clean_rows


def _name_matches_query(full_name: str, query: str) -> bool:
    """Match one- or multi-token queries against player-name prefixes."""
    name_parts = [part for part in full_name.lower().split() if part]
    query_parts = [part for part in query.lower().split() if part]
    if not name_parts or not query_parts:
        return False
    return all(any(name_part.startswith(q) for name_part in name_parts) for q in query_parts)

@st.cache_data(ttl=3600)
def search_player(query: str) -> list:
    """Search the D3 NHL player endpoint. Returns `[]` on empty query or failure."""
    if not query:
        return []
    try:
        normalized_query = query.strip().lower()
        payload = get_client().get(
            url=SEARCH_URL,
            params={"culture": "en-us", "limit": 40, "q": query},
            cache_key=f"search:{normalized_query}",
            ttl=T2_DEFAULT_TTL,
            timeout=5,
        )
        if payload is None:
            return []
        return _normalize_search_results(payload)
    except Exception:
        return []


def search_local_players(query: str, category: str) -> dict:
    """Fill D3 search gaps by matching first or last names from local records."""
    q = query.lower().strip()
    if len(q) < 2:
        return {}
    id_map = get_id_to_name_map(category)
    details = get_clone_details_map(category)
    results = {}
    for pid, full_name in id_map.items():
        parts = full_name.lower().split()
        if len(parts) < 2:
            continue
        if _name_matches_query(full_name, q):
            team = (details.get(pid) or {}).get('team', '') or ''
            if not team:
                # clone_details_map often has no team for active players because
                # the records API omits activeTeamAbbrevs.  Fall back to a D3
                # search by last name — search_player() is cached so no extra
                # latency on repeats.
                last_name = full_name.split()[-1]
                for r in search_player(last_name):
                    if int(r.get('playerId', 0)) == pid:
                        team = r.get('teamAbbrev', '') or ''
                        break
            label = f"[{team}] {full_name}" if team else full_name
            results[label] = pid
        if len(results) >= 20:
            break
    return results


# ---------------------------------------------------------------------------
# Roster & headshot
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_team_roster(team_abbr: str) -> dict:
    """Fetch the current NHL roster for a team.

    Args:
        team_abbr: Three-letter team abbreviation (e.g. 'EDM').

    Returns:
        Dict mapping '[POS] First Last #NN' to int playerId, sorted alphabetically.
        Jersey number is omitted only if the API does not return it.
        Returns {} on failure.
    """
    try:
        res = get_client().get(
            url=ROSTER_URL.format(team_abbr),
            cache_key=f"roster:{team_abbr}",
            ttl=T2_DEFAULT_TTL,
            timeout=10,
        )
        if res is None:
            return {}
        players = {}
        pos_map = {'C': 'C', 'L': 'LW', 'R': 'RW', 'D': 'D', 'G': 'G'}
        for pos_group in ['forwards', 'defensemen', 'goalies']:
            for p in res.get(pos_group, []):
                raw_pos   = p.get('positionCode', '?')
                clean_pos = pos_map.get(raw_pos, raw_pos)
                num       = p.get('sweaterNumber', '')
                base      = f"[{clean_pos}] {p['firstName']['default']} {p['lastName']['default']}"
                name      = f"{base} #{num}" if num else base
                players[name] = int(p['id'])
        return dict(sorted(players.items()))
    except Exception:
        return {}


@st.cache_data(ttl=7200)
def get_player_landing(player_id: int) -> dict:
    """Return the NHL player landing payload for a single player.

    This is the shared source for player metadata helpers so the app fetches the
    landing endpoint once per player instead of once per wrapper.

    Args:
        player_id: Numeric NHL player ID.

    Returns:
        Parsed landing payload dict, or {} if the request fails.
    """
    res = get_client().get(
        url=STATS_URL.format(player_id),
        cache_key=f"player_landing:{player_id}",
        ttl=7200,
        timeout=5,
    )
    return res if isinstance(res, dict) else {}


def get_player_headshot(player_id: int) -> str:
    """Return the headshot URL from the cached player landing payload.

    Args:
        player_id: Numeric NHL player ID.

    Returns:
        URL string, or '' if no headshot is available.
    """
    res = get_player_landing(player_id)
    return str(res.get('headshot', '') or '')


def get_player_current_team(player_id: int) -> str:
    """Return the current team abbreviation from the cached landing payload.

    Active players return a tricode (e.g. 'EDM'). Retired players and free
    agents return an empty string, which callers should treat as "no logo".

    Args:
        player_id: Numeric NHL player ID.

    Returns:
        Three-letter team abbreviation string, or '' if inactive/unavailable.
    """
    res = get_player_landing(player_id)
    return str(res.get('currentTeamAbbrev', '') or '')


def get_player_roster_info(player_id: int) -> dict:
    """Return active-player position and sweater number, or `{}` if unavailable."""
    _POS_MAP = {'C': 'C', 'L': 'LW', 'R': 'RW', 'D': 'D', 'G': 'G'}
    try:
        res = get_player_landing(player_id)
        if not res.get('currentTeamAbbrev'):
            return {}
        raw_pos = res.get('position', '')
        num = res.get('sweaterNumber')
        if not raw_pos or num is None:
            return {}
        return {
            'position': _POS_MAP.get(raw_pos, raw_pos),
            'sweater_number': int(num),
        }
    except Exception:
        return {}


def get_player_hero_image(player_id: int) -> str:
    """Return the player hero image, falling back to headshot when needed."""
    res = get_player_landing(player_id)
    return str(res.get('heroImage', res.get('headshot', '')) or '')


# ---------------------------------------------------------------------------
# Awards / trophies
# ---------------------------------------------------------------------------

def get_player_awards(player_id: int) -> list:
    """Return the player's awards list from the cached landing payload.

    Args:
        player_id: Numeric NHL player ID.

    Returns:
        List of award dicts (possibly empty). Returns [] on failure.
    """
    awards = get_player_landing(player_id).get('awards', [])
    return awards if isinstance(awards, list) else []


def _summarize_player_awards(awards: list[dict]) -> list[dict]:
    """Aggregate raw landing-page awards into one trophy summary list."""
    summary: dict[str, dict[str, int | None]] = {}
    for award in awards:
        if not isinstance(award, dict):
            continue

        trophy_name = _payload_text(award.get("trophy"))
        if not trophy_name:
            continue

        season_ids: list[int] = []
        seasons = award.get("seasons")
        if isinstance(seasons, list):
            for season in seasons:
                if not isinstance(season, dict):
                    continue
                try:
                    season_ids.append(int(season.get("seasonId")))
                except Exception:
                    continue

        wins_here = len(season_ids) if season_ids else 1
        item = summary.setdefault(trophy_name, {"count": 0, "latest": None})
        item["count"] = int(item.get("count", 0) or 0) + wins_here
        if season_ids:
            latest_season = max(season_ids)
            current_latest = item.get("latest")
            if current_latest is None or latest_season > int(current_latest):
                item["latest"] = latest_season

    rows = [
        {
            "trophy": trophy_name,
            "count": int(data.get("count", 0) or 0),
            "latest": data.get("latest"),
            "latest_label": _format_season_id_span(data.get("latest")),
        }
        for trophy_name, data in summary.items()
    ]
    rows.sort(
        key=lambda row: (
            -int(row.get("count", 0) or 0),
            -int(row.get("latest", 0) or 0),
            str(row.get("trophy", "") or ""),
        )
    )
    return rows


@st.cache_data(ttl=3600)
def get_player_identity_summary(player_id: int) -> dict:
    """Return normalized player identity details for the overview-card modal."""
    try:
        clean_player_id = int(player_id)
    except Exception:
        return {}
    if clean_player_id <= 0:
        return {}

    payload = get_player_landing(clean_player_id)
    if not isinstance(payload, dict) or not payload:
        return {}

    first_name = _payload_text(payload.get("firstName"))
    last_name = _payload_text(payload.get("lastName"))
    full_name = f"{first_name} {last_name}".strip() or str(clean_player_id)

    birth_date = str(payload.get("birthDate", "") or "").strip()
    birth_dt = _parse_iso_date(birth_date)
    birth_date_label = birth_dt.strftime("%b %d, %Y") if birth_dt is not None else ""
    birth_year = birth_dt.year if birth_dt is not None else None

    birthplace_parts = [
        _payload_text(payload.get("birthCity")),
        _payload_text(payload.get("birthStateProvince")),
        str(payload.get("birthCountry", "") or "").strip().upper(),
    ]
    birthplace = ", ".join(part for part in birthplace_parts if part)

    position = str(payload.get("position", "") or "").strip().upper()
    shot_value = str(payload.get("shootsCatches", "") or "").strip().upper()
    shot_label = "Catches" if position == "G" else "Shoots"

    season_totals = payload.get("seasonTotals", []) or []
    first_nhl_season = None
    debut_team = ""
    debut_rows: list[dict] = []
    for season in season_totals:
        if not isinstance(season, dict):
            continue
        if normalize_league_abbrev(season.get("leagueAbbrev", "")) != "NHL":
            continue
        if str(season.get("gameTypeId", "")).strip() not in {"2", "3"}:
            continue
        season_str = str(season.get("season", "") or "").strip()
        if len(season_str) < 4:
            continue
        try:
            season_year = int(season_str[:4])
        except Exception:
            continue
        if first_nhl_season is None or season_year < first_nhl_season:
            first_nhl_season = season_year
            debut_rows = [season]
        elif season_year == first_nhl_season:
            debut_rows.append(season)

    if debut_rows:
        debut_rows.sort(
            key=lambda row: (
                0 if str(row.get("gameTypeId", "")).strip() == "2" else 1,
                int(row.get("sequence", 0) or 0),
                _payload_text(row.get("teamName")),
            )
        )
        debut_row = debut_rows[0]
        debut_team = (
            _payload_text(debut_row.get("teamName"))
            or _payload_text(debut_row.get("teamCommonName"))
            or _payload_text(debut_row.get("teamPlaceNameWithPreposition"))
        )

    honors: list[str] = []
    if bool(payload.get("inHHOF")):
        honors.append("Hockey Hall of Fame")
    if bool(payload.get("inTop100AllTime")):
        honors.append("NHL Top 100")
    trophies = _summarize_player_awards(get_player_awards(clean_player_id))

    return {
        "player_id": clean_player_id,
        "name": full_name,
        "birth_date": birth_date_label,
        "birth_year": birth_year,
        "age": _calculate_age_from_birth_date(birth_date),
        "birthplace": birthplace,
        "shot_label": shot_label,
        "shot_value": shot_value,
        "height": _format_height_label(
            payload.get("heightInInches"),
            payload.get("heightInCentimeters"),
        ),
        "weight": _format_weight_label(
            payload.get("weightInPounds"),
            payload.get("weightInKilograms"),
        ),
        "draft": _format_draft_summary(payload.get("draftDetails")),
        "first_nhl_season": first_nhl_season,
        "first_nhl_season_label": _format_season_span(first_nhl_season),
        "debut_team": debut_team,
        "honors": honors,
        "trophies": trophies,
    }


@st.cache_data(ttl=3600)
def get_current_nhl_standings() -> pd.DataFrame:
    """Return a normalized current NHL standings table for team detail surfaces."""
    try:
        payload = get_client().get(
            url="https://api-web.nhle.com/v1/standings/now",
            cache_key="standings",
            ttl=T2_DEFAULT_TTL,
            timeout=15,
        )
    except Exception:
        return pd.DataFrame()

    if not isinstance(payload, dict):
        return pd.DataFrame()

    rows = payload.get("standings", []) or []
    standings_timestamp = str(payload.get("standingsDateTimeUtc", "") or "").strip()

    pp_summary = get_team_season_summary(CURRENT_SEASON_YEAR, "Regular")
    pp_pct_by_team: dict[str, float] = {}
    if not pp_summary.empty and "teamAbbrev" in pp_summary.columns and "PP%" in pp_summary.columns:
        valid_pp = pp_summary[["teamAbbrev", "PP%"]].dropna(subset=["teamAbbrev"])
        pp_pct_by_team = {
            str(row["teamAbbrev"]).strip().upper(): float(row["PP%"])
            for _, row in valid_pp.iterrows()
            if str(row["teamAbbrev"]).strip()
        }

    normalized_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        team_abbr = _payload_text(row.get("teamAbbrev")).upper()
        if not team_abbr:
            continue

        games_played = int(row.get("gamesPlayed", 0) or 0)
        wins = int(row.get("wins", 0) or 0)
        losses = int(row.get("losses", 0) or 0)
        ot_losses = int(row.get("otLosses", 0) or 0)
        ties = int(row.get("ties", 0) or 0)
        points = int(row.get("points", 0) or 0)
        goal_diff = int(row.get("goalDifferential", 0) or 0)
        l10_games_played = int(row.get("l10GamesPlayed", 0) or 0)
        l10_wins = int(row.get("l10Wins", 0) or 0)
        l10_losses = int(row.get("l10Losses", 0) or 0)
        l10_ot_losses = int(row.get("l10OtLosses", 0) or 0)
        l10_ties = int(row.get("l10Ties", 0) or 0)
        l10_points = int(row.get("l10Points", 0) or 0)

        normalized_rows.append(
            {
                "teamAbbrev": team_abbr,
                "teamName": _payload_text(row.get("teamName")) or ACTIVE_TEAMS.get(team_abbr, team_abbr),
                "teamCommonName": _payload_text(row.get("teamCommonName")),
                "teamLogo": str(row.get("teamLogo", "") or "").strip(),
                "conferenceName": str(row.get("conferenceName", "") or "").strip(),
                "divisionName": str(row.get("divisionName", "") or "").strip(),
                "gamesPlayed": games_played,
                "wins": wins,
                "losses": losses,
                "otLosses": ot_losses,
                "ties": ties,
                "points": points,
                "goalDifferential": goal_diff,
                "goalDiffPerGame": (goal_diff / games_played) if games_played > 0 else 0.0,
                "pointPctg": float(row.get("pointPctg", 0.0) or 0.0),
                "regulationWins": int(row.get("regulationWins", 0) or 0),
                "regulationPlusOtWinPctg": float(row.get("regulationPlusOtWinPctg", 0.0) or 0.0),
                "streakCode": str(row.get("streakCode", "") or "").strip(),
                "streakCount": int(row.get("streakCount", 0) or 0),
                "leagueSequence": int(row.get("leagueSequence", 0) or 0),
                "conferenceSequence": int(row.get("conferenceSequence", 0) or 0),
                "divisionSequence": int(row.get("divisionSequence", 0) or 0),
                "l10GamesPlayed": l10_games_played,
                "l10Wins": l10_wins,
                "l10Losses": l10_losses,
                "l10OtLosses": l10_ot_losses,
                "l10Ties": l10_ties,
                "l10Points": l10_points,
                "l10GoalDifferential": int(row.get("l10GoalDifferential", 0) or 0),
                "recordLabel": _build_record_label(wins, losses, ot_losses, ties),
                "l10RecordLabel": (
                    f"{l10_wins}-{l10_losses}-{l10_ot_losses}"
                    if l10_games_played > 0
                    else ""
                ),
                "l10PointPctg": (l10_points / (l10_games_played * 2.0)) if l10_games_played > 0 else 0.0,
                "standingsDateTimeUtc": standings_timestamp,
                "PP%": pp_pct_by_team.get(team_abbr, float("nan")),
            }
        )

    if not normalized_rows:
        return pd.DataFrame()

    standings_df = pd.DataFrame(normalized_rows)
    if "leagueSequence" in standings_df.columns:
        standings_df = standings_df.sort_values(
            ["leagueSequence", "teamAbbrev"],
            kind="stable",
        ).reset_index(drop=True)
    return standings_df


@st.cache_data(ttl=3600)
def get_team_trophy_summary() -> dict:
    """Return team trophy summary keyed by tricode (Stanley Cup count + latest season).

    Source endpoint:
        records.nhl.com franchise-team-totals
        records.nhl.com franchise-season-results

    Returns:
        Dict of {
            triCode: {
                'stanley_cups': int,
                'latest_cup_season': int | None,
                'cup_seasons': list[int],
                'cup_labels': list[str],
            }
        }
        for active NHL teams.
        Returns {} on failure.
    """
    try:
        client = get_client()
        totals_data = client.get(
            url="https://records.nhl.com/site/api/franchise-team-totals",
            params={"limit": 2000},
            cache_key="franchise_team_totals",
            ttl=T1_TTL,
            timeout=15,
        )
        if totals_data is None:
            return {}
        totals_rows = totals_data.get("data", [])
        result: dict = {}
        for row in totals_rows:
            if not isinstance(row, dict):
                continue
            if not row.get("activeTeam"):
                continue
            if int(row.get("gameTypeId", 0) or 0) != 2:
                continue
            tri = str(row.get("triCode", "")).strip().upper()
            if not tri:
                continue
            cups = row.get("cups")
            try:
                cups_int = int(cups) if cups is not None else 0
            except Exception:
                cups_int = 0
            result[tri] = {
                "stanley_cups": cups_int,
                "latest_cup_season": None,
                "cup_seasons": [],
                "cup_labels": [],
            }

        # Derive latest Stanley Cup season per team from SCF-winning rows.
        # Some seasons appear twice (gameTypeId 2 and 3), so dedupe by seasonId.
        seasons_data = client.get(
            url="https://records.nhl.com/site/api/franchise-season-results",
            params={"limit": 5000},
            cache_key="franchise_season_results",
            ttl=T1_TTL,
            timeout=20,
        )
        if seasons_data is None:
            return {}
        season_rows = seasons_data.get("data", [])
        cup_seasons: dict[str, set[int]] = {}
        for row in season_rows:
            if not isinstance(row, dict):
                continue
            tri = str(row.get("triCode", "")).strip().upper()
            if not tri:
                continue
            if str(row.get("seriesAbbrev", "")).strip().upper() != "SCF":
                continue
            if str(row.get("decision", "")).strip().upper() != "W":
                continue
            sid = row.get("seasonId")
            try:
                sid_int = int(sid)
            except Exception:
                continue
            cup_seasons.setdefault(tri, set()).add(sid_int)

        for tri, data in result.items():
            seasons = cup_seasons.get(tri)
            if seasons:
                ordered_seasons = sorted({int(season_id) for season_id in seasons}, reverse=True)
                data["cup_seasons"] = ordered_seasons
                data["cup_labels"] = [
                    label
                    for label in (_format_season_id_span(season_id) for season_id in ordered_seasons)
                    if label
                ]
                data["latest_cup_season"] = max(ordered_seasons)
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Per-player raw stats
# ---------------------------------------------------------------------------

def discover_all_leagues(sample_player_ids: list[str]) -> dict[str, int]:
    """Audit helper that counts observed `leagueAbbrev` values for sample players."""
    counts: dict[str, int] = {}
    for pid in sample_player_ids:
        try:
            res = requests.get(STATS_URL.format(int(pid)), timeout=15).json()
        except Exception:
            continue
        for season in res.get('seasonTotals', []) or []:
            league = str(season.get('leagueAbbrev', '')).strip()
            if not league:
                continue
            counts[league] = counts.get(league, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def get_player_league_abbrevs(player_id: int) -> list[str]:
    """Return unique leagueAbbrev values from cached seasonTotals rows.

    Args:
        player_id: Numeric NHL player ID.

    Returns:
        Sorted list of unique non-empty league abbreviations.
    """
    season_totals = get_player_landing(player_id).get('seasonTotals', []) or []
    leagues = {
        str(season.get('leagueAbbrev', '')).strip()
        for season in season_totals
        if isinstance(season, dict) and str(season.get('leagueAbbrev', '')).strip()
    }
    return sorted(leagues)


def _toi_to_minutes(toi_str: str) -> float:
    """Convert an ``MM:SS`` TOI string into decimal minutes.

    Args:
        toi_str: Time-on-ice string from the NHL APIs.

    Returns:
        Decimal minutes, or 0.0 when parsing fails.
    """
    try:
        parts = str(toi_str or '0:00').split(':')
        if len(parts) != 2:
            return 0.0
        return int(parts[0]) + int(parts[1]) / 60.0
    except Exception:
        return 0.0


def _season_year_to_id(season_year: int) -> int | None:
    """Convert a start year like 2024 into an NHL seasonId like 20242025.

    Args:
        season_year: Four-digit NHL season start year.

    Returns:
        Combined seasonId integer, or ``None`` if the year is invalid.
    """
    try:
        year = int(season_year)
    except Exception:
        return None
    if year < 1900 or year > CURRENT_SEASON_YEAR:
        return None
    return int(f"{year}{year + 1}")


def _weighted_team_metric(grp: pd.DataFrame, col: str) -> float:
    """Return a GP-weighted team metric average, or NaN when unavailable."""
    if col not in grp.columns or "GP" not in grp.columns:
        return float("nan")
    valid = grp[[col, "GP"]].dropna()
    if valid.empty:
        return float("nan")
    total_gp = float(valid["GP"].sum())
    if total_gp <= 0:
        return float(valid[col].mean())
    return float((valid[col] * valid["GP"]).sum() / total_gp)


@st.cache_data(ttl=3600)
def get_team_available_nhl_seasons(team_abbr: str) -> list[int]:
    """Return descending NHL season start years available for one team."""
    clean_abbr = _canonical_team_abbrev(team_abbr)
    if not clean_abbr:
        return []
    df = load_all_team_seasons()
    if df.empty or "SeasonYear" not in df.columns:
        return []
    if "FranchiseAbbrev" not in df.columns:
        df = df.copy()
        df["FranchiseAbbrev"] = df["teamAbbrev"].apply(_canonical_team_abbrev)
    franchise_col = "FranchiseAbbrev"
    seasons = pd.to_numeric(
        df.loc[df[franchise_col] == clean_abbr, "SeasonYear"],
        errors="coerce",
    ).dropna()
    return sorted({int(year) for year in seasons.tolist()}, reverse=True)


@st.cache_data(ttl=3600)
def get_team_identity_summary(team_abbr: str) -> dict:
    """Return normalized team identity details for the overview-card modal."""
    clean_abbr = _canonical_team_abbrev(team_abbr)
    if not clean_abbr:
        return {}

    team_name = ACTIVE_TEAMS.get(clean_abbr, clean_abbr)
    joined_nhl_year = TEAM_FOUNDED.get(clean_abbr)
    current_identity_since_year = None
    total_nhl_seasons = 0
    lineage: list[dict] = []

    all_team_df = load_all_team_seasons()
    if not all_team_df.empty and "SeasonYear" in all_team_df.columns:
        team_history = all_team_df.copy()
        if "gameTypeId" in team_history.columns:
            team_history = team_history[team_history["gameTypeId"] == 2].copy()
        if "FranchiseAbbrev" not in team_history.columns and "teamAbbrev" in team_history.columns:
            team_history["FranchiseAbbrev"] = team_history["teamAbbrev"].apply(_canonical_team_abbrev)

        franchise_rows = team_history[
            team_history.get("FranchiseAbbrev", pd.Series(index=team_history.index, dtype="object")) == clean_abbr
        ].copy()

        if not franchise_rows.empty:
            season_years = pd.to_numeric(franchise_rows["SeasonYear"], errors="coerce").dropna()
            if not season_years.empty:
                season_values = sorted({int(year) for year in season_years.tolist()})
                total_nhl_seasons = len(season_values)
                if joined_nhl_year is None:
                    joined_nhl_year = season_values[0]

            if "teamAbbrev" in franchise_rows.columns:
                current_years = pd.to_numeric(
                    franchise_rows.loc[
                        franchise_rows["teamAbbrev"].astype(str).str.upper() == clean_abbr,
                        "SeasonYear",
                    ],
                    errors="coerce",
                ).dropna()
                if not current_years.empty:
                    current_identity_since_year = int(current_years.min())

                for abbr, abbr_rows in franchise_rows.groupby("teamAbbrev", sort=False):
                    clean_segment_abbr = str(abbr or "").strip().upper()
                    if not clean_segment_abbr:
                        continue
                    segment_years = pd.to_numeric(abbr_rows["SeasonYear"], errors="coerce").dropna()
                    if segment_years.empty:
                        continue
                    segment_name = ""
                    if "teamFullName" in abbr_rows.columns:
                        name_rows = abbr_rows[["SeasonYear", "teamFullName"]].dropna(subset=["teamFullName"])
                        if not name_rows.empty:
                            name_rows = name_rows.sort_values("SeasonYear", kind="stable")
                            segment_name = str(name_rows.iloc[-1]["teamFullName"] or "").strip()
                    if not segment_name:
                        segment_name = ACTIVE_TEAMS.get(clean_segment_abbr, clean_segment_abbr)
                    lineage.append(
                        {
                            "abbr": clean_segment_abbr,
                            "name": segment_name,
                            "start_year": int(segment_years.min()),
                            "end_year": int(segment_years.max()),
                        }
                    )

    if not lineage:
        fallback_year = int(joined_nhl_year) if joined_nhl_year is not None else None
        lineage = [
            {
                "abbr": clean_abbr,
                "name": team_name,
                "start_year": fallback_year,
                "end_year": None,
            }
        ]

    lineage.sort(
        key=lambda segment: (
            int(segment.get("start_year") or 0),
            str(segment.get("abbr", "") or ""),
        )
    )
    lineage_label = " -> ".join(
        segment_text
        for segment_text in (_format_lineage_segment(segment) for segment in lineage)
        if segment_text
    )

    conference_name = ""
    division_name = ""
    team_trophies = get_team_trophy_summary().get(clean_abbr, {})
    stanley_cup_seasons = [
        int(season_id)
        for season_id in (team_trophies.get("cup_seasons", []) or [])
        if str(season_id).strip()
    ]
    stanley_cup_labels = [
        str(label or "").strip()
        for label in (team_trophies.get("cup_labels", []) or [])
        if str(label or "").strip()
    ]
    standings_df = get_current_nhl_standings()
    if not standings_df.empty and "teamAbbrev" in standings_df.columns:
        team_rows = standings_df[standings_df["teamAbbrev"] == clean_abbr]
        if not team_rows.empty:
            standings_row = team_rows.iloc[0]
            conference_name = str(standings_row.get("conferenceName", "") or "").strip()
            division_name = str(standings_row.get("divisionName", "") or "").strip()

    return {
        "team_abbr": clean_abbr,
        "team_name": team_name,
        "joined_nhl_year": joined_nhl_year,
        "joined_nhl_label": _format_season_span(joined_nhl_year),
        "current_identity_since_year": current_identity_since_year,
        "current_identity_since_label": _format_season_span(current_identity_since_year),
        "conference_name": conference_name,
        "division_name": division_name,
        "stanley_cup_count": int(team_trophies.get("stanley_cups", 0) or 0),
        "stanley_cup_seasons": stanley_cup_seasons,
        "stanley_cup_labels": stanley_cup_labels,
        "lineage": lineage,
        "lineage_label": lineage_label,
        "total_nhl_seasons": total_nhl_seasons,
    }


def _fetch_team_game_summary_rows(team_id: int, season_id: int, game_type_id: int) -> list[dict]:
    """Fetch raw per-game team summary rows for one team-season and game type."""
    try:
        season_year = int(season_id) // 10000
        payload = get_client().get(
            url=TEAM_STATS_URL,
            params={
                "limit": -1,
                "start": 0,
                "sort": "gameDate",
                "isGame": "true",
                "cayenneExp": (
                    f"teamId={int(team_id)} and seasonId={int(season_id)} "
                    f"and gameTypeId={int(game_type_id)}"
                ),
            },
            cache_key=f"team_game_summary:{team_id}:{season_id}:{game_type_id}",
            ttl=effective_ttl(season_year),
            timeout=10,
        )
        if payload is None:
            return []
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        return [row for row in rows if isinstance(row, dict)]
    except Exception:
        return []


def _normalize_team_game_log_rows(
    rows: list,
    season_year: int,
    team_abbr: str,
    team_name: str,
    game_type: str,
    game_type_id: int,
) -> list[dict]:
    """Normalize one team-season game log into the app's team schema."""
    normalized_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        try:
            game_id = int(row.get("gameId", 0) or 0)
        except Exception:
            game_id = 0

        goals_for = float(row.get("goalsFor", 0) or 0)
        goals_against = float(row.get("goalsAgainst", 0) or 0)
        wins = float(row.get("wins", 0) or 0)
        losses = float(row.get("losses", 0) or 0)
        ot_losses = float(row.get("otLosses", 0) or 0)
        ties = float(row.get("ties", 0) or 0)
        points = float(row.get("points", 0) or 0)
        point_pct = pd.to_numeric(row.get("pointPct"), errors="coerce")
        power_play_pct = pd.to_numeric(row.get("powerPlayPct"), errors="coerce")
        home_road_flag = str(row.get("homeRoad", "") or "").strip().upper()
        opponent_abbr = str(row.get("opponentTeamAbbrev", "") or "").strip().upper()
        opponent_name = ACTIVE_TEAMS.get(opponent_abbr, opponent_abbr)

        if wins > 0:
            result_code = "W"
        elif ot_losses > 0:
            result_code = "OTL"
        elif ties > 0:
            result_code = "T"
        else:
            result_code = "L"

        normalized_rows.append(
            {
                "SeasonYear": int(season_year),
                "gameTypeId": int(game_type_id),
                "GameType": game_type,
                "GameDate": str(row.get("gameDate", "") or ""),
                "GameId": game_id,
                "GP": 1.0,
                "Wins": wins,
                "Losses": losses,
                "OTLosses": ot_losses,
                "Ties": ties,
                "Points": points,
                "Goals": goals_for,
                "GoalsAgainst": goals_against,
                "Win%": float(point_pct * 100.0) if pd.notna(point_pct) else float("nan"),
                "GF/G": goals_for,
                "GA/G": goals_against,
                "PP%": float(power_play_pct * 100.0) if pd.notna(power_play_pct) else float("nan"),
                "PPG": goals_for * 2.7,
                "TeamAbbrev": team_abbr,
                "TeamName": team_name,
                "OpponentAbbrev": opponent_abbr,
                "OpponentName": opponent_name,
                "HomeRoadFlag": home_road_flag,
                "ResultCode": result_code,
                "ResultLabel": result_code,
            }
        )
    return normalized_rows


@st.cache_data(ttl=3600)
def get_team_season_game_log(team_abbr: str, season_year: int) -> pd.DataFrame:
    """Fetch one NHL season of normalized per-game rows for a team."""
    clean_abbr = _canonical_team_abbrev(team_abbr)
    season_id = _season_year_to_id(season_year)
    if not clean_abbr or season_id is None:
        return pd.DataFrame()

    try:
        season_df = load_all_team_seasons()
        if season_df.empty:
            return pd.DataFrame()
        if "FranchiseAbbrev" not in season_df.columns:
            season_df = season_df.copy()
            season_df["FranchiseAbbrev"] = season_df["teamAbbrev"].apply(_canonical_team_abbrev)
        franchise_col = "FranchiseAbbrev"
        team_rows = season_df[
            (season_df[franchise_col] == clean_abbr)
            & (season_df["SeasonYear"] == int(season_year))
        ]
        if team_rows.empty or "teamId" not in team_rows.columns:
            return pd.DataFrame()

        team_id = int(team_rows["teamId"].dropna().iloc[0])
        team_name = ACTIVE_TEAMS.get(clean_abbr, clean_abbr)

        data: list[dict] = []
        for game_type_id, game_type_label in ((2, "Regular"), (3, "Playoffs")):
            raw_rows = _fetch_team_game_summary_rows(team_id, season_id, game_type_id)
            data.extend(
                _normalize_team_game_log_rows(
                    rows=raw_rows,
                    season_year=int(season_year),
                    team_abbr=clean_abbr,
                    team_name=team_name,
                    game_type=game_type_label,
                    game_type_id=game_type_id,
                )
            )

        if not data:
            return pd.DataFrame()

        return (
            pd.DataFrame(data)
            .sort_values(["GameDate", "GameId", "gameTypeId"])
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_team_season_summary(season_year: int, season_type: str) -> pd.DataFrame:
    """Return one-row-per-team summary rows for one NHL season selection."""
    df = load_all_team_seasons()
    if df.empty or "SeasonYear" not in df.columns or "teamAbbrev" not in df.columns:
        return pd.DataFrame()
    if "FranchiseAbbrev" not in df.columns:
        df = df.copy()
        df["FranchiseAbbrev"] = df["teamAbbrev"].apply(_canonical_team_abbrev)

    season_df = df[df["SeasonYear"] == int(season_year)].copy()
    if season_df.empty:
        return pd.DataFrame()

    if season_type == "Regular":
        season_df = season_df[season_df["gameTypeId"] == 2]
    elif season_type == "Playoffs":
        season_df = season_df[season_df["gameTypeId"] == 3]

    if season_df.empty:
        return pd.DataFrame()

    if season_type == "Both":
        rows: list[dict] = []
        group_cols = ["FranchiseAbbrev"]
        if "teamId" in season_df.columns:
            group_cols.append("teamId")

        for keys, grp in season_df.groupby(group_cols, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            team_abbr = str(keys[0])
            team_name = ACTIVE_TEAMS.get(team_abbr, team_abbr)
            team_id = keys[1] if len(keys) > 1 else None
            gp = float(pd.to_numeric(grp.get("GP"), errors="coerce").fillna(0).sum())
            goals = float(pd.to_numeric(grp.get("Goals"), errors="coerce").fillna(0).sum())
            points = float(pd.to_numeric(grp.get("Points"), errors="coerce").fillna(0).sum())
            wins = float(pd.to_numeric(grp.get("Wins"), errors="coerce").fillna(0).sum())
            losses = float(pd.to_numeric(grp.get("Losses"), errors="coerce").fillna(0).sum())
            ot_losses = float(pd.to_numeric(grp.get("OTLosses"), errors="coerce").fillna(0).sum())
            ties = float(pd.to_numeric(grp.get("Ties"), errors="coerce").fillna(0).sum())
            win_pct = (points / (gp * 2.0) * 100.0) if gp > 0 else _weighted_team_metric(grp, "Win%")
            gf_g = (goals / gp) if gp > 0 else _weighted_team_metric(grp, "GF/G")
            ga_g = _weighted_team_metric(grp, "GA/G")
            ppg = ((goals / gp) * 2.7) if gp > 0 else _weighted_team_metric(grp, "PPG")
            pp_pct = _weighted_team_metric(grp, "PP%")
            rows.append(
                {
                    "teamAbbrev": team_abbr,
                    "teamFullName": team_name,
                    "teamId": team_id,
                    "SeasonYear": int(season_year),
                    "GP": int(round(gp)),
                    "Wins": int(round(wins)),
                    "Losses": int(round(losses)),
                    "OTLosses": int(round(ot_losses)),
                    "Ties": int(round(ties)),
                    "Points": int(round(points)),
                    "Goals": int(round(goals)),
                    "Win%": round(float(win_pct), 1) if pd.notna(win_pct) else float("nan"),
                    "GF/G": round(float(gf_g), 3) if pd.notna(gf_g) else float("nan"),
                    "GA/G": round(float(ga_g), 3) if pd.notna(ga_g) else float("nan"),
                    "PP%": round(float(pp_pct), 1) if pd.notna(pp_pct) else float("nan"),
                    "PPG": round(float(ppg), 3) if pd.notna(ppg) else float("nan"),
                }
            )
        return pd.DataFrame(rows).reset_index(drop=True)

    season_df["teamAbbrev"] = season_df["FranchiseAbbrev"]
    if "teamFullName" in season_df.columns:
        season_df["teamFullName"] = season_df["teamAbbrev"].map(ACTIVE_TEAMS).fillna(season_df["teamFullName"])
    return season_df.reset_index(drop=True)


@st.cache_data(ttl=3600)
def get_team_season_rank_map(
    season_year: int,
    season_type: str,
    metric: str,
) -> dict[str, int]:
    """Return team abbreviation to 1-based league rank for one season metric."""
    leaderboard = get_team_season_summary(season_year, season_type)
    if leaderboard.empty or metric not in leaderboard.columns or "teamAbbrev" not in leaderboard.columns:
        return {}

    ranked = leaderboard[["teamAbbrev", metric]].copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    ranked = ranked.dropna(subset=["teamAbbrev", metric])
    if ranked.empty:
        return {}

    ascending = metric == "GA/G"
    ranked = ranked.sort_values([metric, "teamAbbrev"], ascending=[ascending, True]).reset_index(drop=True)

    rank_map: dict[str, int] = {}
    previous_value = None
    current_rank = 0
    for index, (team_abbr, value) in enumerate(
        ranked[["teamAbbrev", metric]].itertuples(index=False, name=None),
        start=1,
    ):
        numeric_value = float(value)
        if previous_value is None or numeric_value != previous_value:
            current_rank = index
            previous_value = numeric_value
        rank_map[str(team_abbr)] = current_rank
    return rank_map


def _seconds_to_minutes(raw_seconds: object) -> float:
    """Convert NHL summary API seconds fields into decimal minutes.

    Args:
        raw_seconds: Numeric seconds value from the NHL stats API.

    Returns:
        Decimal minutes, or ``0.0`` when parsing fails.
    """
    try:
        return float(raw_seconds or 0.0) / 60.0
    except Exception:
        return 0.0


def _fetch_season_summary_rows(category: str, season_id: int, game_type_id: str) -> list[dict]:
    """Fetch raw season-summary rows for one category and game type.

    Args:
        category: ``Skater`` or ``Goalie``.
        season_id: Combined NHL seasonId such as ``20242025``.
        game_type_id: NHL game-type ID string like ``'2'`` or ``'3'``.

    Returns:
        Raw summary rows, or an empty list on failure.
    """
    endpoint = SEASON_GOALIE_SUMMARY_URL if category == 'Goalie' else SEASON_SKATER_SUMMARY_URL
    try:
        season_year = int(season_id) // 10000
        payload = get_client().get(
            url=endpoint,
            params={
                'limit': -1,
                'start': 0,
                'sort': 'playerId',
                'cayenneExp': f'seasonId={season_id} and gameTypeId={int(game_type_id)}',
            },
            cache_key=f"season_summary:{category}:{season_id}:{game_type_id}",
            ttl=effective_ttl(season_year),
            timeout=10,
        )
        if payload is None:
            return []
        rows = payload.get('data', []) if isinstance(payload, dict) else []
        return [row for row in rows if isinstance(row, dict)]
    except Exception:
        return []


def _build_skater_season_leaderboard(rows: list[dict]) -> pd.DataFrame:
    """Normalize skater season-summary rows into app metric columns.

    Args:
        rows: Raw rows from the NHL skater summary endpoint.

    Returns:
        One row per player with merged counting and derived rate stats.
    """
    normalized_rows: list[dict] = []
    for row in rows:
        try:
            player_id = int(row.get('playerId', 0) or 0)
        except Exception:
            player_id = 0
        if player_id <= 0:
            continue
        gp = float(row.get('gamesPlayed', 0) or 0)
        normalized_rows.append({
            'playerId': player_id,
            'playerName': str(row.get('skaterFullName', '') or row.get('lastName', '') or '').strip(),
            'GP': gp,
            'Points': float(row.get('points', 0) or 0),
            'Goals': float(row.get('goals', 0) or 0),
            'Assists': float(row.get('assists', 0) or 0),
            'PIM': float(row.get('penaltyMinutes', 0) or 0),
            '+/-': float(row.get('plusMinus', 0) or 0),
            'Shots': float(row.get('shots', 0) or 0),
            'TotalTOIMins': _seconds_to_minutes(row.get('timeOnIcePerGame', 0)) * gp,
        })

    if not normalized_rows:
        return pd.DataFrame()

    df = pd.DataFrame(normalized_rows)
    leaderboard = (
        df.groupby('playerId', as_index=False)
        .agg({
            'playerName': 'first',
            'GP': 'sum',
            'Points': 'sum',
            'Goals': 'sum',
            'Assists': 'sum',
            'PIM': 'sum',
            '+/-': 'sum',
            'Shots': 'sum',
            'TotalTOIMins': 'sum',
        })
    )
    gp_denom = leaderboard['GP'].where(leaderboard['GP'].ne(0))
    shots_denom = leaderboard['Shots'].where(leaderboard['Shots'].ne(0))
    leaderboard['PPG'] = leaderboard['Points'].div(gp_denom).fillna(0.0)
    leaderboard['SH%'] = leaderboard['Goals'].mul(100.0).div(shots_denom).fillna(0.0)
    leaderboard['TOI'] = leaderboard['TotalTOIMins'].div(gp_denom).fillna(0.0)
    return leaderboard


def _build_goalie_season_leaderboard(rows: list[dict]) -> pd.DataFrame:
    """Normalize goalie season-summary rows into app metric columns.

    Args:
        rows: Raw rows from the NHL goalie summary endpoint.

    Returns:
        One row per goalie with merged counting and derived rate stats.
    """
    normalized_rows: list[dict] = []
    for row in rows:
        try:
            player_id = int(row.get('playerId', 0) or 0)
        except Exception:
            player_id = 0
        if player_id <= 0:
            continue
        normalized_rows.append({
            'playerId': player_id,
            'playerName': str(row.get('goalieFullName', '') or row.get('lastName', '') or '').strip(),
            'GP': float(row.get('gamesPlayed', 0) or 0),
            'Wins': float(row.get('wins', 0) or 0),
            'Shutouts': float(row.get('shutouts', 0) or 0),
            'Saves': float(row.get('saves', 0) or 0),
            'GoalsAgainst': float(row.get('goalsAgainst', 0) or 0),
            'ShotsAgainst': float(row.get('shotsAgainst', 0) or 0),
            'TotalTOIMins': _seconds_to_minutes(row.get('timeOnIce', 0)),
        })

    if not normalized_rows:
        return pd.DataFrame()

    df = pd.DataFrame(normalized_rows)
    leaderboard = (
        df.groupby('playerId', as_index=False)
        .agg({
            'playerName': 'first',
            'GP': 'sum',
            'Wins': 'sum',
            'Shutouts': 'sum',
            'Saves': 'sum',
            'GoalsAgainst': 'sum',
            'ShotsAgainst': 'sum',
            'TotalTOIMins': 'sum',
        })
    )
    shots_against_denom = leaderboard['ShotsAgainst'].where(leaderboard['ShotsAgainst'].ne(0))
    toi_denom = leaderboard['TotalTOIMins'].where(leaderboard['TotalTOIMins'].ne(0))
    leaderboard['Save %'] = leaderboard['Saves'].mul(100.0).div(shots_against_denom).fillna(0.0)
    leaderboard['GAA'] = leaderboard['GoalsAgainst'].mul(60.0).div(toi_denom).fillna(0.0)
    return leaderboard


@st.cache_data(ttl=3600)
def get_season_leaderboard(category: str, season_year: int, season_type: str) -> pd.DataFrame:
    """Build a merged NHL season leaderboard for comparison-card rank text.

    Args:
        category: ``Skater`` or ``Goalie``.
        season_year: Four-digit NHL season start year.
        season_type: ``Regular``, ``Playoffs``, or ``Both``.

    Returns:
        Leaderboard DataFrame keyed by ``playerId``.
    """
    season_id = _season_year_to_id(season_year)
    if season_id is None or category not in {'Skater', 'Goalie'}:
        return pd.DataFrame()

    game_type_ids = {
        'Regular': ['2'],
        'Playoffs': ['3'],
        'Both': ['2', '3'],
    }.get(season_type)
    if not game_type_ids:
        return pd.DataFrame()

    rows: list[dict] = []
    for game_type_id in game_type_ids:
        rows.extend(_fetch_season_summary_rows(category, season_id, game_type_id))

    if category == 'Goalie':
        return _build_goalie_season_leaderboard(rows)
    return _build_skater_season_leaderboard(rows)


@st.cache_data(ttl=3600)
def get_player_season_rank_map(
    category: str,
    season_year: int,
    season_type: str,
    metric: str,
) -> dict[int, int]:
    """Return playerId -> league rank for one selected season metric.

    Args:
        category: ``Skater`` or ``Goalie``.
        season_year: Four-digit NHL season start year.
        season_type: ``Regular``, ``Playoffs``, or ``Both``.
        metric: Active chart metric.

    Returns:
        Dict mapping player IDs to 1-based league ranks.
    """
    leaderboard = get_season_leaderboard(category, season_year, season_type)
    if leaderboard.empty or metric not in leaderboard.columns or 'playerId' not in leaderboard.columns:
        return {}

    ranked = leaderboard[['playerId', metric]].copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors='coerce')
    ranked['playerId'] = pd.to_numeric(ranked['playerId'], errors='coerce')
    ranked = ranked.dropna(subset=['playerId', metric])
    if ranked.empty:
        return {}

    ascending = metric == 'GAA'
    ranked = ranked.sort_values([metric, 'playerId'], ascending=[ascending, True]).reset_index(drop=True)

    rank_map: dict[int, int] = {}
    previous_value = None
    current_rank = 0
    for index, (player_id, value) in enumerate(ranked[['playerId', metric]].itertuples(index=False, name=None), start=1):
        value = float(value)
        if previous_value is None or value != previous_value:
            current_rank = index
            previous_value = value
        rank_map[int(player_id)] = current_rank
    return rank_map


def _normalize_player_game_log_rows(
    rows: list,
    season_year: int,
    birth_year: int,
    game_type: str,
) -> list[dict]:
    """Normalize NHL player game-log rows into the app's raw-stats schema.

    Args:
        rows: Raw ``gameLog`` list from the NHL endpoint.
        season_year: Start year of the selected NHL season.
        birth_year: Player birth year from the landing payload.
        game_type: ``Regular`` or ``Playoffs``.

    Returns:
        List of normalized per-game row dicts.
    """
    age = season_year - birth_year
    normalized_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        def _payload_text(value: object) -> str:
            """Return a clean string from plain or NHL nested-name payload values."""
            if isinstance(value, dict):
                return str(value.get('default', '') or '').strip()
            return str(value or '').strip()

        toi_mins = _toi_to_minutes(row.get('toi', '0:00'))
        shots_against = float(row.get('shotsAgainst', 0) or 0)
        goals_against = float(row.get('goalsAgainst', 0) or 0)
        raw_save_pct = float(row.get('savePctg', 0.0) or 0.0)
        save_pct = raw_save_pct / 100.0 if raw_save_pct > 1.5 else raw_save_pct
        calc_saves = max(0.0, shots_against - goals_against) if shots_against > 0 else 0.0
        game_gaa = (goals_against * 60.0 / toi_mins) if toi_mins > 0 else 0.0

        try:
            game_id = int(row.get('gameId', 0) or 0)
        except Exception:
            game_id = 0

        normalized_rows.append({
            'League': 'NHL',
            'Age': age,
            'SeasonYear': season_year,
            'GameType': game_type,
            'GP': 1,
            'Points': float(row.get('points', 0) or 0),
            'Goals': float(row.get('goals', 0) or 0),
            'Assists': float(row.get('assists', 0) or 0),
            'PIM': float(row.get('pim', 0) or 0),
            '+/-': float(row.get('plusMinus', 0) or 0),
            'Shots': float(row.get('shots', 0) or 0),
            'TotalTOIMins': toi_mins,
            'Wins': 1.0 if str(row.get('decision', '')).upper() == 'W' else 0.0,
            'Shutouts': float(row.get('shutouts', 0) or 0),
            'Saves': calc_saves,
            'WeightedSV': save_pct * 100.0,
            'WeightedGAA': game_gaa,
            'NHLeMultiplier': 1.0,
            'GameDate': str(row.get('gameDate', '') or ''),
            'GameId': game_id,
            'TeamAbbrev': str(row.get('teamAbbrev', '') or '').strip().upper(),
            'OpponentAbbrev': str(row.get('opponentAbbrev', '') or '').strip().upper(),
            'HomeRoadFlag': str(row.get('homeRoadFlag', '') or '').strip().upper(),
            'TeamName': (
                _payload_text(row.get('teamName'))
                or _payload_text(row.get('commonName'))
            ),
            'OpponentName': (
                _payload_text(row.get('opponentTeamName'))
                or _payload_text(row.get('opponentCommonName'))
            ),
        })
    return normalized_rows


@st.cache_data(ttl=3600)
def get_player_available_nhl_seasons(player_id: int) -> list[int]:
    """Return the player's available NHL season start years.

    Args:
        player_id: Numeric NHL player ID.

    Returns:
        Descending list of NHL season start years seen in ``seasonTotals``.
    """
    season_totals = get_player_landing(player_id).get('seasonTotals', []) or []
    seasons: set[int] = set()
    for season in season_totals:
        if not isinstance(season, dict):
            continue
        if normalize_league_abbrev(season.get('leagueAbbrev', '')) != 'NHL':
            continue
        if str(season.get('gameTypeId', '')) not in {'2', '3'}:
            continue
        season_str = str(season.get('season', '')).strip()
        if len(season_str) < 4:
            continue
        try:
            season_year = int(season_str[:4])
        except Exception:
            continue
        if 1900 <= season_year <= CURRENT_SEASON_YEAR:
            seasons.add(season_year)
    return sorted(seasons, reverse=True)


@st.cache_data(ttl=3600)
def get_player_season_game_log(
    player_id: int,
    base_name: str,
    season_year: int,
) -> tuple:
    """Fetch one NHL season of normalized per-game rows for a player.

    Args:
        player_id: Numeric NHL player ID.
        base_name: Display name already selected in the UI.
        season_year: Start year of the selected season.

    Returns:
        ``(DataFrame, base_name, position_code)`` matching ``get_player_raw_stats``.
    """
    try:
        res = get_player_landing(player_id)
        birth_date = str(res.get('birthDate', '2000'))
        birth_year = int(birth_date[:4]) if len(birth_date) >= 4 else 2000
        position = str(res.get('position', 'S') or 'S')
        season_id = _season_year_to_id(season_year)
        if season_id is None:
            return pd.DataFrame(), base_name, position

        client = get_client()
        ttl = effective_ttl(int(season_year))
        data: list[dict] = []
        for game_type_id, game_type_label in (('2', 'Regular'), ('3', 'Playoffs')):
            payload = client.get(
                url=PLAYER_GAME_LOG_URL.format(int(player_id), season_id, game_type_id),
                cache_key=f"player_gamelog:{player_id}:{season_year}:{game_type_id}",
                ttl=ttl,
                timeout=10,
            )
            if payload is None:
                continue
            rows = payload.get('gameLog', []) if isinstance(payload, dict) else []
            data.extend(
                _normalize_player_game_log_rows(
                    rows=rows,
                    season_year=int(season_year),
                    birth_year=birth_year,
                    game_type=game_type_label,
                )
            )

        if not data:
            return pd.DataFrame(), base_name, position

        df = pd.DataFrame(data).sort_values(['GameDate', 'GameId']).reset_index(drop=True)
        df["PlayerID"] = int(player_id)
        df["PositionCode"] = str(position or "S")
        return df, base_name, position
    except Exception:
        return pd.DataFrame(), base_name, 'S'


@st.cache_data(ttl=7200)
def get_player_raw_stats(
    player_id: int,
    base_name: str,
) -> tuple:
    """Fetch raw player seasons across all leagues and both game types.

    Keeps unknown leagues, stores NHLe multipliers for downstream use, and
    preserves the goalie-safe saves fallback.
    """
    try:
        res = get_player_landing(player_id)
        birth_date = str(res.get('birthDate', '2000'))
        birth_year = int(birth_date[:4]) if len(birth_date) >= 4 else 2000
        position   = res.get('position', 'S')
        data = []

        for s in res.get('seasonTotals', []):
            league_raw = str(s.get('leagueAbbrev', '')).strip()
            league_key = normalize_league_abbrev(league_raw)
            nhle_mult  = NHLE_MULTIPLIERS.get(league_key, NHLE_DEFAULT_MULTIPLIER)
            game_type = str(s.get('gameTypeId', ''))
            if game_type in ['2', '3']:
                season_str = str(s.get('season', ''))
                season_year = int(season_str[:4]) if len(season_str) >= 4 else 2000
                age = season_year - birth_year
                gp  = max(s.get('gamesPlayed', 1), 1)

                toi_str = str(s.get('avgToi', '0:00'))
                try:
                    parts   = toi_str.split(':')
                    toi_val = int(parts[0]) + int(parts[1]) / 60.0 if len(parts) == 2 else 0
                except Exception:
                    toi_val = 0

                # FIX #5: Robust Saves calculation.
                raw_saves = s.get('saves')
                if raw_saves is not None and raw_saves > 0:
                    calc_saves = raw_saves
                else:
                    sa = s.get('shotsAgainst', 0) or 0
                    ga = s.get('goalsAgainst', 0) or 0
                    calc_saves = max(0, sa - ga) if sa > 0 else 0

                data.append({
                    "PlayerID":   int(player_id),
                    "PositionCode": str(position or 'S'),
                    "League":     league_raw,
                    "Age":        age,
                    "SeasonYear": season_year,
                    "GameType":   "Regular" if game_type == '2' else "Playoffs",
                    "GP":         gp,
                    "Points":     s.get('points', 0),
                    "Goals":      s.get('goals', 0),
                    "Assists":    s.get('assists', 0),
                    "PIM":        s.get('pim', 0) or s.get('penaltyMinutes', 0),
                    "+/-":        s.get('plusMinus', 0),
                    "Shots":      s.get('shots', 0),
                    "TotalTOIMins": toi_val * gp,
                    "Wins":       s.get('wins', 0),
                    "Shutouts":   s.get('shutouts', 0),
                    "Saves":      calc_saves,
                    "WeightedSV": float(s.get('savePctg', 0.0)) * 100 * gp,
                    "WeightedGAA": float(s.get('goalsAgainstAvg', 0.0)) * gp,
                    "NHLeMultiplier": nhle_mult,
                })
        return pd.DataFrame(data), base_name, position
    except Exception:
        return pd.DataFrame(), base_name, 'S'


# ---------------------------------------------------------------------------
# Records-based lookup maps
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_id_to_name_map(category: str) -> dict:
    """Build a player ID -> full name map from the records API.

    Cached for 1 hour so newly active players appear without a restart.

    Args:
        category: 'Skater' or 'Goalie'.

    Returns:
        Dict mapping int playerId to 'First Last' name string.
    """
    records = fetch_all_time_records(category, "Regular")
    return {
        int(r['playerId']): f"{r.get('firstName', '')} {r.get('lastName', '')}".strip()
        for r in records
    }


@st.cache_data(ttl=3600)
def get_clone_details_map(category: str) -> dict:
    """Build the player-career lookup used by the KNN clone dialog."""
    records = fetch_all_time_records(category, "Regular")
    details = {}
    for r in records:
        pid  = int(r['playerId'])
        team = r.get('lastTeamAbbrev', '') or r.get('activeTeamAbbrevs', '') or ''
        if ',' in str(team):
            team = str(team).split(',')[-1].strip()
        gp = int(r.get('gamesPlayed', 0) or 0)

        # Only overwrite if this row has more games played (filters out franchise fragments)
        if pid not in details or gp > details[pid]['gp']:
            if category == "Skater":
                details[pid] = {
                    'name': f"{r.get('firstName', '')} {r.get('lastName', '')}".strip(),
                    'team': team,
                    'gp':   gp,
                    'pts':  int(r.get('points', 0) or 0),
                    'g':    int(r.get('goals', 0) or 0),
                    'a':    int(r.get('assists', 0) or 0),
                    'pm':   int(r.get('plusMinus', 0) or 0),
                }
            else:
                details[pid] = {
                    'name': f"{r.get('firstName', '')} {r.get('lastName', '')}".strip(),
                    'team': team,
                    'gp':   gp,
                    'w':    int(r.get('wins', 0) or 0),
                    'sv':   int(r.get('saves', 0) or 0),
                    'so':   int(r.get('shutouts', 0) or 0),
                }
    return details


# ---------------------------------------------------------------------------
# All-time ranking
# ---------------------------------------------------------------------------

def get_all_time_rank(
    category: str,
    s_type: str,
    metric: str,
    value: float,
) -> int | None:
    """Estimate where a career total would rank on the all-time NHL list."""
    records = fetch_all_time_records(category, s_type)
    if not records:
        return None
    key_map = {
        "Points":  "points",
        "Goals":   "goals",
        "Assists": "assists",
        "+/-":     "plusMinus",
        "GP":      "gamesPlayed",
        "PIM":     "penaltyMinutes",
        "Wins":    "wins",
        "Shutouts":"shutouts",
        "Saves":   "saves",
    }
    key = key_map.get(metric)
    if not key:
        return None
    records = sorted(
        [r for r in records if r.get(key) is not None],
        key=lambda x: x.get(key, 0),
        reverse=True,
    )
    for i, record in enumerate(records):
        if value >= record.get(key, 0):
            return i + 1
    return len(records) + 1


def get_player_career_rank(pid: int, category: str, s_type: str, metric: str = "Points") -> int | None:
    """Look up a player's exact all-time rank by player ID instead of by value."""
    records = fetch_all_time_records(category, s_type)
    if not records:
        return None
    _RANK_KEY_MAP = {"Points": "points", "Goals": "goals", "Assists": "assists"}
    rank_key = "wins" if category == "Goalie" else _RANK_KEY_MAP.get(metric, "points")
    sorted_records = sorted(
        [r for r in records if r.get(rank_key) is not None],
        key=lambda x: x.get(rank_key, 0),
        reverse=True,
    )
    # Deduplicate by playerId (API may return multiple stints per player);
    # keep first (highest-value) occurrence so rank matches the Top 50 dropdown.
    seen_ids: set = set()
    deduped: list = []
    for r in sorted_records:
        pid_r = int(r.get('playerId', -1))
        if pid_r not in seen_ids:
            seen_ids.add(pid_r)
            deduped.append(r)
    for i, r in enumerate(deduped):
        if int(r.get('playerId', -1)) == pid:
            return i + 1
    return None


@st.cache_data(ttl=3600)
def get_team_all_time_stats() -> dict:
    """Compute all-time franchise stats for each NHL team from historical data.

    Uses regular-season records only (gameTypeId == 2). Computes career totals,
    all-time wins rank (1 = most wins), and best single season by wins.

    Returns:
        Dict mapping active team abbreviation (str) to a stats dict with keys:
            total_wins (int), total_gp (int), total_points (int), total_goals (int),
            wins_rank (int), best_year (int | None), best_wins (int | None),
            best_gp (int | None).
    """
    df = load_all_team_seasons()
    if df.empty:
        return {}
    if "FranchiseAbbrev" not in df.columns:
        df = df.copy()
        df["FranchiseAbbrev"] = df["teamAbbrev"].apply(_canonical_team_abbrev)
    reg = df[df['gameTypeId'] == 2].copy()

    totals = reg.groupby('FranchiseAbbrev', as_index=False).agg(
        total_wins=('Wins', 'sum'),
        total_gp=('GP', 'sum'),
        total_points=('Points', 'sum'),
        total_goals=('Goals', 'sum'),
    )
    totals = totals.sort_values('total_wins', ascending=False).reset_index(drop=True)
    totals['wins_rank'] = range(1, len(totals) + 1)

    best = (
        reg.sort_values('Wins', ascending=False)
           .groupby('FranchiseAbbrev', as_index=False)
           .first()[['FranchiseAbbrev', 'SeasonYear', 'Wins', 'GP']]
           .rename(columns={'SeasonYear': 'best_year', 'Wins': 'best_wins', 'GP': 'best_gp'})
    )

    merged = totals.merge(best, on='FranchiseAbbrev', how='left')
    result: dict = {}
    for _, row in merged.iterrows():
        result[row['FranchiseAbbrev']] = {
            'total_wins':   int(row['total_wins']),
            'total_gp':     int(row['total_gp']),
            'total_points': int(row['total_points']),
            'total_goals':  int(row['total_goals']),
            'wins_rank':    int(row['wins_rank']),
            'best_year':    int(row['best_year'])  if pd.notna(row.get('best_year'))  else None,
            'best_wins':    int(row['best_wins'])  if pd.notna(row.get('best_wins'))  else None,
            'best_gp':      int(row['best_gp'])    if pd.notna(row.get('best_gp'))    else None,
        }
    return result
