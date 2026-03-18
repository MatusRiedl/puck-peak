"""Chart click dialogs for season snapshots, projections, baselines, and help."""

from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st

from nhl.constants import ACTIVE_TEAMS, NHLE_DEFAULT_MULTIPLIER, NHLE_MULTIPLIERS, TEAM_BRAND_COLORS
from nhl.data_loaders import (
    get_all_time_rank,
    get_player_identity_summary,
    get_team_identity_summary,
)
from nhl.era import get_era_multiplier
from nhl.rarity import collapse_player_snapshot_rows, get_age_rarity_summary
from nhl.schedule import get_game_details, get_matchup_history


BASELINE_LABEL_TO_KEY = {
    'Skater 75th Percentile Baseline': 'Skater',
    'Goalie 75th Percentile Baseline': 'Goalie',
}

_ERA_EXPLAINER_BANDS: tuple[tuple[str, int], ...] = (
    ("<= 1967", 1967),
    ("1968-79", 1975),
    ("1980-92", 1985),
    ("1993-96", 1995),
    ("1997-2004", 2001),
    ("2005-12", 2010),
    ("2013-17", 2015),
    ("2018+", 2022),
)

_TEAM_LOGO_URL = "https://assets.nhle.com/logos/nhl/svg/{abbr}_light.svg"
_DIALOG_WEEKDAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_DIALOG_MONTH_ABBR = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
_TEAM_SHORT_NAMES = {
    "ANA": "Ducks",
    "BOS": "Bruins",
    "BUF": "Sabres",
    "CAR": "Hurricanes",
    "CBJ": "Blue Jackets",
    "CGY": "Flames",
    "CHI": "Blackhawks",
    "COL": "Avalanche",
    "DAL": "Stars",
    "DET": "Red Wings",
    "EDM": "Oilers",
    "FLA": "Panthers",
    "LAK": "Kings",
    "MIN": "Wild",
    "MTL": "Canadiens",
    "NJD": "Devils",
    "NSH": "Predators",
    "NYI": "Islanders",
    "NYR": "Rangers",
    "OTT": "Senators",
    "PHI": "Flyers",
    "PIT": "Penguins",
    "SEA": "Kraken",
    "SJS": "Sharks",
    "STL": "Blues",
    "TBL": "Lightning",
    "TOR": "Leafs",
    "UTA": "Utah",
    "VAN": "Canucks",
    "VGK": "Knights",
    "WPG": "Jets",
    "WSH": "Caps",
}
_DEFAULT_PLAYER_DIALOG_ACCENT = "#4F8FFF"
_DEFAULT_DIALOG_VALUE_COLOR = "#EAF1FF"


def _get_team_short_name(team_abbr: str, fallback_name: str) -> str:
    """Return the short display name for a team.

    Args:
        team_abbr: Three-letter NHL team abbreviation.
        fallback_name: Full team name to shorten when no mapping exists.

    Returns:
        Compact team nickname for tight matchup-card layouts.
    """
    mapped_name = _TEAM_SHORT_NAMES.get(team_abbr, ACTIVE_TEAMS.get(team_abbr, fallback_name))
    clean_name = str(mapped_name or fallback_name or team_abbr).strip()
    parts = clean_name.split()
    if len(parts) <= 1:
        return clean_name
    return ' '.join(parts[-2:]) if clean_name in {"Blue Jackets", "Red Wings"} else parts[-1]


def _parse_iso_date(value: str) -> datetime | None:
    """Parse the leading ``YYYY-MM-DD`` portion of a date string when present."""
    clean_value = str(value or "").strip()
    if len(clean_value) < 10:
        return None

    try:
        return datetime.strptime(clean_value[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _format_matchup_history_start_label(game: dict) -> str:
    """Add a compact year suffix to matchup-history date labels."""
    raw_label = str(game.get("start_label_cest", "") or "").strip()
    parsed_game_date = _parse_iso_date(str(game.get("game_date", "") or ""))
    parsed_label_date = _parse_iso_date(raw_label)

    if parsed_label_date is not None:
        year_suffix = f"'{parsed_game_date:%y}" if parsed_game_date is not None else f"'{parsed_label_date:%y}"
        return (
            f"{_DIALOG_WEEKDAY_ABBR[parsed_label_date.weekday()]} "
            f"{parsed_label_date.day:02d} "
            f"{_DIALOG_MONTH_ABBR[parsed_label_date.month - 1]} "
            f"{year_suffix}"
        )

    if not raw_label or raw_label == "Time TBD":
        if parsed_game_date is None:
            return raw_label
        return (
            f"{_DIALOG_WEEKDAY_ABBR[parsed_game_date.weekday()]} "
            f"{parsed_game_date.day:02d} "
            f"{_DIALOG_MONTH_ABBR[parsed_game_date.month - 1]} "
            f"'{parsed_game_date:%y}"
        )

    if parsed_game_date is not None:
        year_suffix = f"'{parsed_game_date:%y}"
        if year_suffix not in raw_label:
            if ", " in raw_label:
                label_head, label_tail = raw_label.split(", ", 1)
                return f"{label_head} {year_suffix}, {label_tail}"
            return f"{raw_label} {year_suffix}"

    return raw_label


def _format_matchup_history_status_label(status_label: str) -> str:
    """Drop redundant ``Final`` labels while keeping OT/SO context."""
    clean_label = str(status_label or "").strip()
    if not clean_label:
        return ""

    upper_label = clean_label.upper()
    if upper_label == "FINAL":
        return ""
    if upper_label.startswith("FINAL/"):
        return clean_label.split("/", 1)[-1]
    return clean_label


def _matchup_history_sort_key(game: dict) -> tuple[str, int]:
    """Return a descending-safe sort key for matchup-history cards."""
    raw_game_date = str(game.get("game_date", "") or "").strip()
    raw_start_label = str(game.get("start_label_cest", "") or "").strip()
    sortable_date = raw_game_date
    if not sortable_date and _parse_iso_date(raw_start_label) is not None:
        sortable_date = raw_start_label[:10]

    try:
        game_id = int(game.get("game_id", 0) or 0)
    except Exception:
        game_id = 0

    return sortable_date, game_id


def _get_raw_player_df(raw_dfs_list: list, clean_name: str):
    """Return the raw dataframe for the clicked player, if present."""
    for df in raw_dfs_list:
        if not df.empty and 'BaseName' in df.columns and df['BaseName'].iloc[0] == clean_name:
            return df
    return None


def _format_snapshot_metric_value(metric: str, value: float | int | None) -> str:
    """Format one Season Snapshot metric value for rarity copy."""
    if value is None or pd.isna(value):
        return "?"

    numeric_value = float(value)
    if metric in {"Save %", "SH%"}:
        return f"{numeric_value:.2f}%"
    if metric in {"PPG", "GAA", "TOI"}:
        return f"{numeric_value:.2f}"
    if numeric_value.is_integer():
        return f"{int(numeric_value):,}"
    return f"{numeric_value:.2f}"


def _format_percentile_label(percentile: float | int | None) -> str:
    """Format percentile text without rounding sub-100 values up to 100.0."""
    if percentile is None or pd.isna(percentile):
        return "?"

    bounded = max(0.0, min(100.0, float(percentile)))
    if bounded >= 100.0:
        return "100.0"

    for decimals in range(1, 5):
        rounded = round(bounded, decimals)
        if rounded < 100.0:
            return f"{rounded:.{decimals}f}"

    return "99.9999"


def _render_age_rarity_callout(rarity_summary: dict) -> None:
    """Render the age-rarity callout block under career subtotals."""
    if not rarity_summary:
        return

    unavailable_reason = str(rarity_summary.get("unavailable_reason", "") or "").strip()
    if unavailable_reason:
        st.caption(unavailable_reason)
        return

    metric = str(rarity_summary.get("metric", "") or "").strip()
    season_label = str(rarity_summary.get("season_label", "") or "").strip()
    age = int(rarity_summary.get("age", 0) or 0)
    value_label = escape(_format_snapshot_metric_value(metric, rarity_summary.get("value")))
    percentile = float(rarity_summary.get("percentile", 0.0) or 0.0)
    percentile_label = _format_percentile_label(percentile)
    rank = int(rarity_summary.get("rank", 0) or 0)
    sample_size = int(rarity_summary.get("sample_size", 0) or 0)
    role_label = str(rarity_summary.get("role_label", "") or "").strip()
    is_era_adjusted = bool(rarity_summary.get("is_era_adjusted", False))

    heading = "Era-adjusted age rarity" if is_era_adjusted else "Age rarity"
    main_line = (
        f"{value_label} {escape(metric)} in {escape(season_label)} was a "
        f"{percentile_label}th-percentile NHL age-{age} season "
        f"(#{rank:,} of {sample_size:,})."
    )

    role_line = ""
    if role_label:
        role_percentile = float(rarity_summary.get("role_percentile", 0.0) or 0.0)
        role_percentile_label = _format_percentile_label(role_percentile)
        role_rank = int(rarity_summary.get("role_rank", 0) or 0)
        role_sample_size = int(rarity_summary.get("role_sample_size", 0) or 0)
        role_line = (
            f"Among {escape(role_label)}: {role_percentile_label}th percentile "
            f"(#{role_rank:,} of {role_sample_size:,})."
        )

    top_seasons = rarity_summary.get("top_seasons", []) or []
    top_rows_html = ""
    if top_seasons:
        top_lines = []
        for row in top_seasons:
            display_rank = int(row.get("display_rank", 0) or 0)
            player_name = escape(str(row.get("player_name", "") or "").strip())
            top_season_label = escape(str(row.get("season_label", "") or "").strip())
            top_value = escape(_format_snapshot_metric_value(metric, row.get("value")))
            top_lines.append(
                f"{display_rank}. {player_name} ({top_season_label}) - {top_value} {escape(metric)}"
            )
        if top_lines:
            top_rows_html = (
                "<div style='margin-top:8px;'>"
                f"<b>Top NHL age-{age} seasons in this pool:</b><br>"
                f"{'<br>'.join(top_lines)}"
                "</div>"
            )

    body_html = (
        f"<div style='background-color:#2b2442;border-left:4px solid #a477ff;"
        f"padding:10px 14px;border-radius:4px;margin-bottom:8px;'>"
        f"<b>{escape(heading)}:</b> {main_line}"
        f"{f'<br>{role_line}' if role_line else ''}"
        f"{top_rows_html}"
        "</div>"
    )
    st.markdown(body_html, unsafe_allow_html=True)


def _resolve_real_game_row(df, age: int, s_type: str, game_id: int | None, game_date: str | None, clicked_game_type: str | None):
    """Resolve the clicked real-data row, preferring exact game identifiers."""
    filtered_df = df[df['GameType'] == s_type] if s_type != "Both" else df
    if clicked_game_type and s_type == "Both" and 'GameType' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['GameType'] == clicked_game_type]

    if game_id and 'GameId' in filtered_df.columns:
        game_match = filtered_df[filtered_df['GameId'] == int(game_id)]
        if not game_match.empty:
            return game_match.iloc[-1]

    if game_date and 'GameDate' in filtered_df.columns:
        date_match = filtered_df[filtered_df['GameDate'] == str(game_date)]
        if not date_match.empty:
            return date_match.iloc[-1]

    age_match = filtered_df[filtered_df['Age'] == age]
    if not age_match.empty:
        return age_match.iloc[-1]
    return None


def _build_matchup_context(game_row, score_details: dict) -> dict:
    """Merge score-endpoint data with preserved game-log matchup metadata."""
    team_abbr = str(game_row.get('TeamAbbrev', '') or '').strip().upper()
    opponent_abbr = str(game_row.get('OpponentAbbrev', '') or '').strip().upper()
    home_road_flag = str(game_row.get('HomeRoadFlag', '') or '').strip().upper()
    team_name = str(game_row.get('TeamName', '') or '').strip() or ACTIVE_TEAMS.get(team_abbr, team_abbr)
    opponent_name = str(game_row.get('OpponentName', '') or '').strip() or ACTIVE_TEAMS.get(opponent_abbr, opponent_abbr)

    if home_road_flag == 'H':
        fallback_away_abbr, fallback_away_name = opponent_abbr, opponent_name
        fallback_home_abbr, fallback_home_name = team_abbr, team_name
    else:
        fallback_away_abbr, fallback_away_name = team_abbr, team_name
        fallback_home_abbr, fallback_home_name = opponent_abbr, opponent_name

    return {
        'away_abbr': str(score_details.get('away_abbr', '') or fallback_away_abbr),
        'away_name': str(score_details.get('away_name', '') or fallback_away_name),
        'away_score': score_details.get('away_score'),
        'home_abbr': str(score_details.get('home_abbr', '') or fallback_home_abbr),
        'home_name': str(score_details.get('home_name', '') or fallback_home_name),
        'home_score': score_details.get('home_score'),
        'venue': str(score_details.get('venue', '') or ''),
        'start_label_cest': str(score_details.get('start_label_cest', '') or ''),
        'status_label': str(score_details.get('status_label', '') or ''),
    }


def _build_matchup_card_html_legacy_v1(game: dict) -> str:
    """Render a compact matchup card with logos, score, and venue/time details."""
    away_abbr = escape(str(game.get('away_abbr', '') or ''))
    home_abbr = escape(str(game.get('home_abbr', '') or ''))
    away_name = escape(str(game.get('away_name', '') or away_abbr))
    home_name = escape(str(game.get('home_name', '') or home_abbr))
    away_short_name = escape(_get_team_short_name(away_abbr, away_name))
    home_short_name = escape(_get_team_short_name(home_abbr, home_name))
    away_logo = _TEAM_LOGO_URL.format(abbr=away_abbr) if away_abbr else ''
    home_logo = _TEAM_LOGO_URL.format(abbr=home_abbr) if home_abbr else ''
    away_score = game.get('away_score')
    home_score = game.get('home_score')
    away_won = away_score is not None and home_score is not None and away_score > home_score
    home_won = away_score is not None and home_score is not None and home_score > away_score

    def _team_block(abbr: str, short_name: str, logo: str, side_label: str, align: str = 'left') -> str:
        """Render one team column with an explicit home/away label."""
        text_align = 'right' if align == 'right' else 'left'
        direction = 'row-reverse' if align == 'right' else 'row'
        return (
            f"<div style='display:flex;align-items:center;gap:8px;min-width:0;flex:1 1 0;flex-direction:{direction};overflow:hidden;'>"
            f"<img src='{logo}' height='38'>"
            f"<div style='text-align:{text_align};min-width:0;'>"
            f"<div style='font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#8b949e;font-weight:700;'>{side_label}</div>"
            f"<div style='display:flex;align-items:baseline;gap:7px;justify-content:{'flex-end' if align == 'right' else 'flex-start'};white-space:nowrap;'>"
            f"<div style='font-size:19px;font-weight:800;line-height:1.0;'>{abbr or short_name}</div>"
            f"<div style='font-size:14px;color:#b7bcc2;font-weight:600;line-height:1.0;overflow:hidden;text-overflow:ellipsis;'>{short_name}</div>"
            "</div>"
            "</div>"
            "</div>"
        )

    def _score_html(value, did_win: bool) -> str:
        """Return styled score markup with winner emphasis."""
        color = '#ffffff' if did_win else '#8b949e'
        return f"<div style='font-size:32px;font-weight:800;color:{color};line-height:1.0;'>{value if value is not None else '—'}</div>"

    detail_bits = [bit for bit in (game.get('status_label'), game.get('start_label_cest'), game.get('venue')) if bit]
    detail_html = escape(' • '.join(detail_bits)) if detail_bits else 'Matchup details unavailable'

    return (
        "<div style='background:#231f20;border:1px solid #343434;border-radius:14px;padding:14px 16px;margin:10px 0 12px 0;'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:nowrap;'>"
        f"{_team_block(away_abbr, away_short_name, away_logo, 'Away')}"
        f"<div style='display:flex;align-items:center;gap:12px;flex:0 0 auto;padding:0 4px;'>{_score_html(away_score, away_won)}<div style='font-size:18px;color:#8b949e;font-weight:700;'>@</div>{_score_html(home_score, home_won)}</div>"
        f"{_team_block(home_abbr, home_short_name, home_logo, 'Home', align='right')}"
        "</div>"
        f"<div style='margin-top:10px;font-size:13px;color:#b7bcc2;'>{detail_html}</div>"
        "</div>"
    )


def _build_matchup_card_html_legacy_v2(game: dict, compact_layout: bool = False) -> str:
    """Render one matchup card, with an optional tighter history layout."""
    away_abbr_raw = str(game.get('away_abbr', '') or '')
    home_abbr_raw = str(game.get('home_abbr', '') or '')
    away_name_raw = str(game.get('away_name', '') or away_abbr_raw)
    home_name_raw = str(game.get('home_name', '') or home_abbr_raw)
    away_abbr = escape(away_abbr_raw)
    home_abbr = escape(home_abbr_raw)
    away_name = escape(away_name_raw)
    home_name = escape(home_name_raw)
    away_short_name = escape(_get_team_short_name(away_abbr_raw, away_name_raw))
    home_short_name = escape(_get_team_short_name(home_abbr_raw, home_name_raw))
    away_logo = _TEAM_LOGO_URL.format(abbr=away_abbr_raw) if away_abbr_raw else ''
    home_logo = _TEAM_LOGO_URL.format(abbr=home_abbr_raw) if home_abbr_raw else ''
    away_score = game.get('away_score')
    home_score = game.get('home_score')
    away_won = away_score is not None and home_score is not None and away_score > home_score
    home_won = away_score is not None and home_score is not None and home_score > away_score

    def _team_block(abbr: str, short_name: str, logo: str, side_label: str, align: str = 'left') -> str:
        """Render one team column with an explicit home/away label."""
        text_align = 'right' if align == 'right' else 'left'
        direction = 'row-reverse' if align == 'right' else 'row'
        team_label = abbr or short_name
        if compact_layout:
            name_html = f"<div style='font-size:22px;font-weight:800;line-height:1.0;'>{team_label}</div>"
        else:
            name_html = (
                f"<div style='display:flex;align-items:baseline;gap:7px;justify-content:{'flex-end' if align == 'right' else 'flex-start'};white-space:nowrap;'>"
                f"<div style='font-size:19px;font-weight:800;line-height:1.0;'>{team_label}</div>"
                f"<div style='font-size:14px;color:#b7bcc2;font-weight:600;line-height:1.0;overflow:hidden;text-overflow:ellipsis;'>{short_name}</div>"
                "</div>"
            )
        return (
            f"<div style='display:flex;align-items:center;gap:8px;min-width:0;flex:1 1 0;flex-direction:{direction};overflow:hidden;'>"
            f"<img src='{logo}' height='38'>"
            f"<div style='text-align:{text_align};min-width:0;'>"
            f"<div style='font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#8b949e;font-weight:700;'>{side_label}</div>"
            f"{name_html}"
            "</div>"
            "</div>"
        )

    def _score_html(value, did_win: bool) -> str:
        """Return styled score markup with winner emphasis."""
        color = '#ffffff' if did_win else '#8b949e'
        return f"<div style='font-size:32px;font-weight:800;color:{color};line-height:1.0;'>{value if value is not None else 'â€”'}</div>"

    if compact_layout:
        detail_bits = [
            bit
            for bit in (
                _format_matchup_history_status_label(str(game.get('status_label', '') or '')),
                _format_matchup_history_start_label(game),
                game.get('venue'),
            )
            if bit
        ]
    else:
        detail_bits = [bit for bit in (game.get('status_label'), game.get('start_label_cest'), game.get('venue')) if bit]
    detail_html = escape(' • '.join(detail_bits)) if detail_bits else 'Matchup details unavailable'

    return (
        "<div style='background:#231f20;border:1px solid #343434;border-radius:14px;padding:14px 16px;margin:10px 0 12px 0;'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:nowrap;'>"
        f"{_team_block(away_abbr, away_short_name, away_logo, 'Away')}"
        f"<div style='display:flex;align-items:center;gap:12px;flex:0 0 auto;padding:0 4px;'>{_score_html(away_score, away_won)}<div style='font-size:18px;color:#8b949e;font-weight:700;'>@</div>{_score_html(home_score, home_won)}</div>"
        f"{_team_block(home_abbr, home_short_name, home_logo, 'Home', align='right')}"
        "</div>"
        f"<div style='margin-top:10px;font-size:13px;color:#b7bcc2;'>{detail_html}</div>"
        "</div>"
    )

def _build_matchup_card_html(game: dict, compact_layout: bool = False) -> str:
    """Render one matchup card, with an optional tighter history layout."""
    away_abbr_raw = str(game.get('away_abbr', '') or '')
    home_abbr_raw = str(game.get('home_abbr', '') or '')
    away_name_raw = str(game.get('away_name', '') or away_abbr_raw)
    home_name_raw = str(game.get('home_name', '') or home_abbr_raw)
    away_abbr = escape(away_abbr_raw)
    home_abbr = escape(home_abbr_raw)
    away_short_name = escape(_get_team_short_name(away_abbr_raw, away_name_raw))
    home_short_name = escape(_get_team_short_name(home_abbr_raw, home_name_raw))
    away_logo = _TEAM_LOGO_URL.format(abbr=away_abbr_raw) if away_abbr_raw else ''
    home_logo = _TEAM_LOGO_URL.format(abbr=home_abbr_raw) if home_abbr_raw else ''
    away_score = game.get('away_score')
    home_score = game.get('home_score')
    away_won = away_score is not None and home_score is not None and away_score > home_score
    home_won = away_score is not None and home_score is not None and home_score > away_score

    def _team_block(abbr: str, short_name: str, logo: str, side_label: str, align: str = 'left') -> str:
        """Render one team column with an explicit home/away label."""
        text_align = 'right' if align == 'right' else 'left'
        direction = 'row-reverse' if align == 'right' else 'row'
        team_label = abbr or short_name
        if compact_layout:
            name_html = f"<div style='font-size:22px;font-weight:800;line-height:1.0;'>{team_label}</div>"
        else:
            name_html = (
                f"<div style='display:flex;align-items:baseline;gap:7px;justify-content:{'flex-end' if align == 'right' else 'flex-start'};white-space:nowrap;'>"
                f"<div style='font-size:19px;font-weight:800;line-height:1.0;'>{team_label}</div>"
                f"<div style='font-size:14px;color:#b7bcc2;font-weight:600;line-height:1.0;overflow:hidden;text-overflow:ellipsis;'>{short_name}</div>"
                "</div>"
            )
        return (
            f"<div style='display:flex;align-items:center;gap:8px;min-width:0;flex:1 1 0;flex-direction:{direction};overflow:hidden;'>"
            f"<img src='{logo}' height='38'>"
            f"<div style='text-align:{text_align};min-width:0;'>"
            f"<div style='font-size:11px;letter-spacing:0.08em;text-transform:uppercase;color:#8b949e;font-weight:700;'>{side_label}</div>"
            f"{name_html}"
            "</div>"
            "</div>"
        )

    def _score_html(value, did_win: bool) -> str:
        """Return styled score markup with winner emphasis."""
        color = '#ffffff' if did_win else '#8b949e'
        return f"<div style='font-size:32px;font-weight:800;color:{color};line-height:1.0;'>{value if value is not None else '-'}</div>"

    if compact_layout:
        detail_bits = [
            bit
            for bit in (
                _format_matchup_history_status_label(str(game.get('status_label', '') or '')),
                _format_matchup_history_start_label(game),
                game.get('venue'),
            )
            if bit
        ]
    else:
        detail_bits = [bit for bit in (game.get('status_label'), game.get('start_label_cest'), game.get('venue')) if bit]
    detail_html = escape(' | '.join(detail_bits)) if detail_bits else 'Matchup details unavailable'

    return (
        "<div style='background:#231f20;border:1px solid #343434;border-radius:14px;padding:14px 16px;margin:10px 0 12px 0;'>"
        "<div style='display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:nowrap;'>"
        f"{_team_block(away_abbr, away_short_name, away_logo, 'Away')}"
        f"<div style='display:flex;align-items:center;gap:12px;flex:0 0 auto;padding:0 4px;'>{_score_html(away_score, away_won)}<div style='font-size:18px;color:#8b949e;font-weight:700;'>@</div>{_score_html(home_score, home_won)}</div>"
        f"{_team_block(home_abbr, home_short_name, home_logo, 'Home', align='right')}"
        "</div>"
        f"<div style='margin-top:10px;font-size:13px;color:#b7bcc2;'>{detail_html}</div>"
        "</div>"
    )



def _format_game_toi(total_toi_mins: float) -> str:
    """Convert stored total TOI minutes into a mm:ss display string."""
    try:
        total_seconds = max(0, int(round(float(total_toi_mins or 0) * 60)))
    except Exception:
        total_seconds = 0
    return f"{total_seconds // 60}:{total_seconds % 60:02d}"


def _build_player_game_stat_frame(game_row, stat_category: str) -> pd.DataFrame:
    """Build the one-row exact-game stat table for the dialog."""
    if stat_category == 'Skater':
        row = {
            'Date': str(game_row.get('GameDate', '') or ''),
            'Type': str(game_row.get('GameType', '') or ''),
            'G': int(round(float(game_row.get('Goals', 0) or 0))),
            'A': int(round(float(game_row.get('Assists', 0) or 0))),
            'Pts': int(round(float(game_row.get('Points', 0) or 0))),
            '+/-': int(round(float(game_row.get('+/-', 0) or 0))),
            'Shots': int(round(float(game_row.get('Shots', 0) or 0))),
            'TOI': _format_game_toi(game_row.get('TotalTOIMins', 0)),
        }
    else:
        row = {
            'Date': str(game_row.get('GameDate', '') or ''),
            'Type': str(game_row.get('GameType', '') or ''),
            'W': int(round(float(game_row.get('Wins', 0) or 0))),
            'Saves': int(round(float(game_row.get('Saves', 0) or 0))),
            'SO': int(round(float(game_row.get('Shutouts', 0) or 0))),
            'Save %': round(float(game_row.get('WeightedSV', 0) or 0), 1),
            'GAA': round(float(game_row.get('WeightedGAA', 0) or 0), 2),
            'TOI': _format_game_toi(game_row.get('TotalTOIMins', 0)),
        }
    return pd.DataFrame([row])


def _resolve_team_game_row(
    full_df,
    team_abbr: str,
    game_id: int | None,
    game_date: str | None,
    clicked_game_type: str | None,
):
    """Return the selected team game row, preferring exact game identifiers."""
    if full_df is None or getattr(full_df, "empty", True):
        return None

    filtered_df = full_df.copy()
    if team_abbr and 'BaseName' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['BaseName'] == team_abbr]
    if clicked_game_type and 'GameType' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['GameType'] == clicked_game_type]

    if game_id and 'GameId' in filtered_df.columns:
        game_match = filtered_df[filtered_df['GameId'] == int(game_id)]
        if not game_match.empty:
            return game_match.iloc[-1]

    if game_date and 'GameDate' in filtered_df.columns:
        date_match = filtered_df[filtered_df['GameDate'] == str(game_date)]
        if not date_match.empty:
            return date_match.iloc[-1]

    return filtered_df.iloc[-1] if not filtered_df.empty else None


def _build_team_game_stat_frame(game_row, metric: str, val: float) -> pd.DataFrame:
    """Build the one-row team game snapshot table for the dialog."""
    home_road_flag = str(game_row.get('HomeRoadFlag', '') or '').strip().upper()
    site_label = 'Home' if home_road_flag == 'H' else 'Road' if home_road_flag == 'R' else ''
    opponent_name = str(game_row.get('OpponentName', '') or game_row.get('OpponentAbbrev', '') or '')
    metric_value = float(val) if val is not None else float(game_row.get(metric, 0) or 0)
    if metric in {'Win%', 'PP%'}:
        metric_display = f"{metric_value:.1f}%"
    elif metric in {'GF/G', 'GA/G', 'PPG'}:
        metric_display = f"{metric_value:.3f}"
    else:
        metric_display = str(int(round(metric_value)))

    game_goals_for = int(round(float(game_row.get('GameGoalsFor', game_row.get('Goals', 0)) or 0)))
    game_goals_against = int(round(float(game_row.get('GameGoalsAgainst', game_row.get('GoalsAgainst', 0)) or 0)))

    row = {
        'Date': str(game_row.get('GameDate', '') or ''),
        'Type': str(game_row.get('GameType', '') or ''),
        'Site': site_label,
        'Opponent': opponent_name,
        'Result': str(game_row.get('ResultLabel', '') or ''),
        'Score': f'{game_goals_for}-{game_goals_against}',
        'Record': str(game_row.get('RecordLabel', '') or ''),
        'Pts': int(round(float(game_row.get('Points', 0) or 0))),
        'W': int(round(float(game_row.get('Wins', 0) or 0))),
        'GF': int(round(float(game_row.get('Goals', 0) or 0))),
        'GA': int(round(float(game_row.get('GoalsAgainst', 0) or 0))),
        metric: metric_display,
    }
    return pd.DataFrame([row])


def _render_projection_guide_tab() -> None:
    """Render the projection explainer content.

    Args:
        None.

    Returns:
        None.
    """
    st.markdown("#### Basics")
    st.markdown(
        "- **Solid line:** real performance already logged.\n"
        "- **Dotted line:** projected continuation of the current career path.\n"
        "- **Dashed line:** strong historical benchmark, not a player-specific forecast."
    )

    st.markdown("#### How projection works")
    proj_skater_col, proj_goalie_col = st.columns(2)
    with proj_skater_col:
        st.markdown(
            "**Skaters**\n\n"
            "Compared only against historical skaters with similar early and mid-career "
            "shapes, then extended using how those comparable careers aged."
        )
    with proj_goalie_col:
        st.markdown(
            "**Goalies**\n\n"
            "Compared only against historical goalies. Their projection rules stay "
            "separate because goalie aging and volatility behave differently."
        )

    ml_col, rules_col = st.columns(2)
    with ml_col:
        st.markdown(
            "**What is ML-ish**\n\n"
            "The nearest-match layer. The app uses historical similarity to find the "
            "closest career paths and borrow aging behavior from them."
        )
    with rules_col:
        st.markdown(
            "**What is not ML**\n\n"
            "The guardrails. Caps, smoothing, presentation logic, and late-career "
            "stability rules are deliberate system design, not black-box learning."
        )


def _render_baseline_guide_tab() -> None:
    """Render the baseline explainer content.

    Args:
        None.

    Returns:
        None.
    """
    st.markdown(
        "The dashed baseline line is your historical benchmark. It is there so you can see "
        "whether a player is tracking below, around, or above a strong historical standard."
    )

    base_skater_col, base_goalie_col = st.columns(2)
    with base_skater_col:
        st.markdown(
            "**Skater baseline**\n\n"
            "A skater-only 75th percentile age curve built from historical player seasons. "
            "Think of it as: what does a really strong skater age track usually look like?"
        )
    with base_goalie_col:
        st.markdown(
            "**Goalie baseline**\n\n"
            "A separate goalie-only 75th percentile curve. Goalies get their own baseline "
            "because goalie stats age differently from skater stats."
        )

    st.markdown("#### How to read it")
    st.markdown(
        "- above the baseline usually means an exceptional track for that age\n"
        "- around the baseline means the player is in strong company\n"
        "- below the baseline does not mean bad — just below that specific historical bar"
    )


def _render_era_adjust_skaters_guide_tab() -> None:
    """Render the skater era-adjustment explainer content.

    Args:
        None.

    Returns:
        None.
    """
    st.markdown(
        "Era adjust normalizes NHL scoring to a modern baseline so cross-era comparisons "
        "stop cheating. It fixes the obvious problem: a raw point total from the 80s and "
        "a raw point total from the dead-puck era are not the same accomplishment."
    )

    st.markdown("#### What changes")
    st.markdown(
        "- **Raw stats:** the original season totals for **Goals**, **Assists**, and **Points**.\n"
        "- **Era-adjusted stats:** those same NHL totals after the app applies the era multiplier."
    )

    st.markdown("#### Multipliers used")
    st.markdown(
        "\n".join(
            f"- **{label}:** `{get_era_multiplier(sample_year):.2f}`"
            for label, sample_year in _ERA_EXPLAINER_BANDS
        )
    )

    st.markdown("#### Leagues")
    st.markdown(
        "When **Era** is on, the app does not pretend every league scores like the NHL. "
        "Non-NHL skater seasons get a **league multiplier** first so outside-league production "
        "is translated into a rough NHL-equivalent level."
    )
    st.markdown(
        "The multiplier is looked up from the season's league code: "
        f"**NHL = {NHLE_MULTIPLIERS['NHL']:.2f}**, **KHL = {NHLE_MULTIPLIERS['KHL']:.2f}**, "
        f"**SHL = {NHLE_MULTIPLIERS['SHL']:.2f}**, **AHL = {NHLE_MULTIPLIERS['AHL']:.2f}**, and unknown "
        f"or rare leagues fall back to **{NHLE_DEFAULT_MULTIPLIER:.2f}**."
    )
    st.markdown(
        "With **Era off**, the chart shows raw league scoring. With **Era on**, it only scales "
        "**Points**, **Goals**, and **Assists**. GP and the other non-scoring stats stay raw. "
        "After that, skater **era adjust** only runs on **NHL rows**, so the app does not double-count the adjustment."
    )


def _render_era_adjust_goalies_guide_tab() -> None:
    """Render the goalie era-adjustment explainer content.

    Args:
        None.

    Returns:
        None.
    """
    st.markdown(
        "Goalie era adjust uses separate logic because goalie stats are not the same beast as skater scoring. "
        "The app tries to move goalie seasons into a modern context without pretending Save %, GAA, and Shutouts all behave the same way."
    )

    st.markdown("#### What changes")
    st.markdown(
        "- **Save %:** shifted toward the modern 2018+ environment while keeping how far above or below league average the goalie was.\n"
        "- **GAA:** scaled by the era multiplier so goals-against sits in the same modern scoring context.\n"
        "- **Shutouts:** moved the opposite way, because shutouts were harder to pile up in high-scoring eras."
    )

    goalie_sv_col, goalie_misc_col = st.columns(2)
    with goalie_sv_col:
        st.markdown(
            "**Save % logic**\n\n"
            "This is not the skater method. The app compares a goalie's Save % to that era's league average, then shifts it toward the modern baseline."
        )
    with goalie_misc_col:
        st.markdown(
            "**GAA + Shutouts logic**\n\n"
            "GAA gets scaled into the modern scoring environment, while Shutouts move inversely so old high-scoring eras do not get unfairly punished."
        )

def _render_smoothing_guide_tab() -> None:
    """Render the smoothing explainer content.

    Args:
        None.

    Returns:
        None.
    """
    st.markdown(
        "Smoothing is a display aid. It makes the visible curve easier to read without pretending "
        "the raw season-to-season noise is the real story."
    )

    smooth_math_col, smooth_visual_col = st.columns(2)
    with smooth_math_col:
        st.markdown(
            "**What the app does**\n\n"
            "When **Smooth** is on, the selected metric becomes a **3-season rolling average**. "
            "At the start of a career it uses whatever history exists, so the early points still render cleanly."
        )
    with smooth_visual_col:
        st.markdown(
            "**Why it looks calmer**\n\n"
            "The chart also switches to a curved line shape, so the trend reads more like a career arc "
            "and less like a seismograph having a bad day."
        )

    st.markdown("#### What smoothing helps with")
    st.markdown(
        "- softens one-year spikes and dips\n"
        "- makes long careers easier to scan\n"
        "- visually reduces harsh jumps between neighboring seasons"
    )


@st.dialog("How This App Works")
def show_app_guide() -> None:
    """Show a concise methodology guide without exposing the full recipe.

    Args:
        None.

    Returns:
        None.
    """
    st.markdown(
        "This app turns NHL careers into age curves so you can compare real results, "
        "historical baselines, and a forward-looking projection in one place."
    )

    projection_tab, baseline_tab, era_adjust_skaters_tab, era_adjust_goalies_tab, smoothing_tab = st.tabs(["Projection", "Baseline", "Era adjust skaters", "Era adjust goalies", "Smoothing"])
    with projection_tab:
        _render_projection_guide_tab()
    with baseline_tab:
        _render_baseline_guide_tab()
    with era_adjust_skaters_tab:
        _render_era_adjust_skaters_guide_tab()
    with era_adjust_goalies_tab:
        _render_era_adjust_goalies_guide_tab()
    with smoothing_tab:
        _render_smoothing_guide_tab()


@st.dialog("Season Snapshot", on_dismiss="rerun")
def show_season_details(
    player_name: str,
    age: int,
    raw_dfs_list: list,
    metric: str,
    val: float,
    is_cumul: bool,
    full_df,
    s_type: str,
    ml_clones_dict: dict,
    historical_baselines: dict,
    stat_category: str,
    do_era: bool,
    game_id: int | None = None,
    game_date: str | None = None,
    clicked_game_type: str | None = None,
    game_number: int | None = None,
) -> None:
    """Render the correct click dialog for a real point, projection, or baseline."""
    age = int(age)
    clean_name    = player_name.replace(" (Proj)", "")
    baseline_key  = BASELINE_LABEL_TO_KEY.get(clean_name)
    is_baseline   = baseline_key is not None
    is_projection = "(Proj)" in player_name
    is_real       = not is_baseline and not is_projection

    if game_number is not None:
        st.markdown(f"### {player_name} — Game {int(game_number)} · Age {age}")
    else:
        st.markdown(f"### {player_name} at Age {age}")

    # ── CASE 3: BASELINE LINE CLICK ────────────────────────────────────
    if is_baseline:
        base_df = historical_baselines.get(baseline_key)
        if base_df is not None and not base_df.empty and age in base_df.index:
            if stat_category == "Skater":
                b_gp  = int(round(base_df.loc[age, 'GP']))    if 'GP'      in base_df.columns else 0
                b_pts = int(round(base_df.loc[age, 'Points'])) if 'Points'  in base_df.columns else 0
                b_g   = int(round(base_df.loc[age, 'Goals']))  if 'Goals'   in base_df.columns else 0
                b_a   = int(round(base_df.loc[age, 'Assists']))if 'Assists' in base_df.columns else 0
                b_pm  = int(round(base_df.loc[age, '+/-']))   if '+/-'     in base_df.columns else 0
                st.markdown(
                    f"<div style='background-color:#2b2b2b;border-left:4px solid rgba(255,255,255,0.4);"
                    f"padding:10px 14px;border-radius:4px;margin-bottom:8px;'>"
                    f"<b>75th Percentile at Age {age}:</b> {b_gp} GP | {b_pts} Pts | "
                    f"{b_g} G | {b_a} A | {b_pm} +/-</div>",
                    unsafe_allow_html=True,
                )
            else:
                b_gp = int(round(base_df.loc[age, 'GP']))       if 'GP'       in base_df.columns else 0
                b_w  = int(round(base_df.loc[age, 'Wins']))     if 'Wins'     in base_df.columns else 0
                b_sv = int(round(base_df.loc[age, 'Saves']))    if 'Saves'    in base_df.columns else 0
                b_so = int(round(base_df.loc[age, 'Shutouts'])) if 'Shutouts' in base_df.columns else 0
                st.markdown(
                    f"<div style='background-color:#2b2b2b;border-left:4px solid rgba(255,255,255,0.4);"
                    f"padding:10px 14px;border-radius:4px;margin-bottom:8px;'>"
                    f"<b>75th Percentile at Age {age}:</b> {b_gp} GP | {b_w} W | "
                    f"{b_sv} Saves | {b_so} SO</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.write("No baseline data available for this age.")
        return  # No further content for baseline clicks

    # ── SHARED: Career Totals (blue) — shown for both real & projection ─
    player_raw_df = _get_raw_player_df(raw_dfs_list, clean_name)
    if player_raw_df is not None:
        display_df_career = player_raw_df[player_raw_df['GameType'] == s_type] if s_type != "Both" else player_raw_df
        career_gp         = int(display_df_career['GP'].sum())
        label             = "Reg+Playoffs" if s_type == "Both" else s_type
        if stat_category == "Skater":
            career_pts = int(display_df_career['Points'].sum())
            career_g   = int(display_df_career['Goals'].sum())
            career_a   = int(display_df_career['Assists'].sum())
            career_pm  = int(display_df_career['+/-'].sum())
            st.info(
                f"**Career Totals ({label}):** {career_gp} GP | {career_pts} Pts | "
                f"{career_g} G | {career_a} A | {career_pm} +/-"
            )
        else:
            career_w  = int(display_df_career['Wins'].sum())
            career_so = int(display_df_career['Shutouts'].sum())
            career_sv = int(display_df_career['Saves'].sum())
            st.info(
                f"**Career Totals ({label}):** {career_gp} GP | {career_w} W | "
                f"{career_sv} Saves | {career_so} SO"
            )

    # ── CASE 1: REAL DATA LINE CLICK ───────────────────────────────────
    if is_real:
        if player_raw_df is None:
            st.write("No player data available for this click.")
            return

        if game_id or game_date:
            game_row = _resolve_real_game_row(
                player_raw_df,
                age=age,
                s_type=s_type,
                game_id=game_id,
                game_date=game_date,
                clicked_game_type=clicked_game_type,
            )
            if game_row is not None:
                score_details = get_game_details(
                    str(game_row.get('GameDate', '') or game_date or ''),
                    int(game_row.get('GameId', 0) or game_id or 0),
                )
                matchup_context = _build_matchup_context(game_row, score_details)
                st.markdown(_build_matchup_card_html(matchup_context), unsafe_allow_html=True)
                st.markdown("**Player stat line**")
                st.dataframe(
                    _build_player_game_stat_frame(game_row, stat_category),
                    hide_index=True,
                    use_container_width=True,
                )
                return

        # Career subtotals up to clicked age (orange)
        sub_df = player_raw_df[player_raw_df['Age'] <= age]
        if s_type != "Both":
            sub_df = sub_df[sub_df['GameType'] == s_type]
        if not sub_df.empty:
            s_gp = int(sub_df['GP'].sum())
            if stat_category == "Skater":
                s_pts = int(sub_df['Points'].sum())
                s_g   = int(sub_df['Goals'].sum())
                s_a   = int(sub_df['Assists'].sum())
                s_pm  = int(sub_df['+/-'].sum())
                st.warning(
                    f"**Career Subtotals (to Age {age}):** {s_gp} GP | "
                    f"{s_pts} Pts | {s_g} G | {s_a} A | {s_pm} +/-"
                )
            else:
                s_w  = int(sub_df['Wins'].sum())
                s_so = int(sub_df['Shutouts'].sum())
                s_sv = int(sub_df['Saves'].sum())
                st.warning(
                    f"**Career Subtotals (to Age {age}):** {s_gp} GP | "
                    f"{s_w} W | {s_sv} Saves | {s_so} SO"
                )

        # Season detail table
        season_data = player_raw_df[player_raw_df['Age'] == age]
        if s_type != "Both":
            season_data = season_data[season_data['GameType'] == s_type]
        season_snapshot_df = collapse_player_snapshot_rows(season_data)

        if not season_snapshot_df.empty:
            rarity_row_df = season_snapshot_df[
                season_snapshot_df['League'].astype(str).str.upper().eq('NHL')
                & season_snapshot_df['GameType'].astype(str).eq('Regular')
            ]
            rarity_summary = (
                get_age_rarity_summary(
                    season_row=rarity_row_df.iloc[0].to_dict(),
                    metric=metric,
                    stat_category=stat_category,
                    do_era=do_era,
                )
                if not rarity_row_df.empty
                else {
                    "unavailable_reason": "Age rarity is shown only for NHL regular-season rows.",
                }
            )
            _render_age_rarity_callout(rarity_summary)

            cols_to_show = (
                ['SeasonYear', 'League', 'GameType', 'GP', 'Points', 'Goals', 'Assists', '+/-']
                if stat_category == "Skater"
                else ['SeasonYear', 'League', 'GameType', 'GP', 'Wins', 'Saves', 'Shutouts']
            )
            display_df = season_snapshot_df[cols_to_show].copy()
            for col in display_df.columns:
                if col not in ['SeasonYear', 'League', 'GameType']:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0).astype(int)
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        return  # End of real data click

    # ── CASE 2: PROJECTION LINE CLICK ──────────────────────────────────
    if is_projection:
        counting_stats = [
            'Points', 'Goals', 'Assists', 'Wins', 'Shutouts', 'GP', 'PIM', 'Saves', '+/-'
        ]

        # Projected career totals up to clicked age
        player_data = full_df[full_df['BaseName'] == clean_name]
        player_data = player_data[player_data['Age'] <= age].drop_duplicates(
            subset=['Age'], keep='last'
        )

        if metric in counting_stats:
            career_total = val if is_cumul else player_data[metric].sum()
            rank = get_all_time_rank(stat_category, s_type, metric, career_total)
            if rank:
                st.success(
                    f"🏆 **At Age {age}:** Estimated **{int(career_total)}** career "
                    f"{metric} → **#{rank} All-Time** in NHL history."
                )

        # ML Projection Clones — single column with team + career stats
        clones = ml_clones_dict.get(clean_name, []) or []
        if clones:
            st.markdown("---")
            st.markdown("**Nearest Historical Matches:**")
            is_skater_mode = stat_category == "Skater"

            if is_skater_mode:
                stat_headers = (
                    "<th style='text-align:right; padding:4px;'>GP</th>"
                    "<th style='text-align:right; padding:4px;'>Pts</th>"
                    "<th style='text-align:right; padding:4px;'>G</th>"
                    "<th style='text-align:right; padding:4px;'>A</th>"
                )
            else:
                stat_headers = (
                    "<th style='text-align:right; padding:4px;'>GP</th>"
                    "<th style='text-align:right; padding:4px;'>W</th>"
                    "<th style='text-align:right; padding:4px;'>Saves</th>"
                    "<th style='text-align:right; padding:4px;'>SO</th>"
                )

            table_html  = "<table style='width:100%; font-size:13px; border-collapse:collapse;'>"
            table_html += (
                "<tr style='border-bottom:1px solid #444;'>"
                f"<th style='text-align:left; padding:4px;'>Player</th>{stat_headers}"
                "</tr>"
            )

            for c in clones:
                tm = f"[{c['team']}] " if c.get('team') and c['team'] != '—' else ""
                yrs = c.get('years', '')
                yr_str = f" ({yrs})" if yrs else ""
                table_html += "<tr style='border-bottom:1px solid #333;'>"
                table_html += f"<td style='padding:3px 4px; white-space:nowrap;'>{tm}{c['name']}{yr_str}</td>"
                if is_skater_mode:
                    table_html += (
                        f"<td style='text-align:right; padding:3px 4px;'>{c.get('gp', 0)}</td>"
                        f"<td style='text-align:right; padding:3px 4px;'>{c.get('pts', 0)}</td>"
                        f"<td style='text-align:right; padding:3px 4px;'>{c.get('g', 0)}</td>"
                        f"<td style='text-align:right; padding:3px 4px;'>{c.get('a', 0)}</td>"
                    )
                else:
                    table_html += (
                        f"<td style='text-align:right; padding:3px 4px;'>{c.get('gp', 0)}</td>"
                        f"<td style='text-align:right; padding:3px 4px;'>{c.get('w', 0)}</td>"
                        f"<td style='text-align:right; padding:3px 4px;'>{c.get('sv', 0)}</td>"
                        f"<td style='text-align:right; padding:3px 4px;'>{c.get('so', 0)}</td>"
                    )
                table_html += "</tr>"
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)


@st.dialog("Team Game Snapshot", on_dismiss="rerun")
def show_team_game_details(
    team_name: str,
    team_abbr: str,
    metric: str,
    val: float,
    full_df,
    s_type: str,
    selected_season: str | int,
    game_number: int,
    game_id: int | None = None,
    game_date: str | None = None,
    clicked_game_type: str | None = None,
    opponent_abbr: str | None = None,
    opponent_name: str | None = None,
    home_road_flag: str | None = None,
    result_label: str | None = None,
    goals_for: float | int | None = None,
    goals_against: float | int | None = None,
    record_label: str | None = None,
) -> None:
    """Render the dialog for one selected-season team game point."""
    del s_type
    try:
        season_year = int(selected_season)
        season_label = f"{season_year}-{str(season_year + 1)[2:]}"
    except Exception:
        season_label = str(selected_season)

    type_label = clicked_game_type or "Game"
    st.markdown(f"### {team_name} — Game {int(game_number)} · {season_label} · {type_label}")

    game_row = _resolve_team_game_row(
        full_df=full_df,
        team_abbr=team_abbr,
        game_id=game_id,
        game_date=game_date,
        clicked_game_type=clicked_game_type,
    )
    if game_row is None:
        st.write("No team data available for this click.")
        return

    game_row = game_row.copy()
    if home_road_flag:
        game_row['HomeRoadFlag'] = home_road_flag
    if opponent_abbr:
        game_row['OpponentAbbrev'] = opponent_abbr
    if opponent_name:
        game_row['OpponentName'] = opponent_name
    if result_label:
        game_row['ResultLabel'] = result_label
    if record_label:
        game_row['RecordLabel'] = record_label
    if goals_for is not None:
        game_row['GameGoalsFor'] = goals_for
    if goals_against is not None:
        game_row['GameGoalsAgainst'] = goals_against
    if team_abbr:
        game_row['TeamAbbrev'] = team_abbr
    if team_name:
        game_row['TeamName'] = team_name

    score_details = get_game_details(str(game_date or game_row.get('GameDate', '') or ''), int(game_id or 0))
    matchup_context = _build_matchup_context(game_row, score_details)
    st.markdown(_build_matchup_card_html(matchup_context), unsafe_allow_html=True)
    st.markdown("**Team snapshot**")
    st.dataframe(
        _build_team_game_stat_frame(game_row, metric, val),
        hide_index=True,
        use_container_width=True,
    )


def _render_identity_rows(
    rows: list[tuple[str, str]],
    *,
    columns: int = 2,
    label_color: str = "#FFFFFF",
    value_color: str = "#FFFFFF",
) -> bool:
    """Render compact label/value rows for identity dialogs."""
    clean_rows = [
        (str(label or "").strip(), str(value or "").strip())
        for label, value in rows
        if str(label or "").strip() and str(value or "").strip()
    ]
    if not clean_rows:
        return False

    column_count = max(1, min(int(columns), len(clean_rows)))
    if column_count == 1:
        for label, value in clean_rows:
            st.markdown(
                _build_identity_row_html(
                    label,
                    value,
                    label_color=label_color,
                    value_color=value_color,
                ),
                unsafe_allow_html=True,
            )
        return True

    layout_columns = st.columns(column_count)
    for idx, (label, value) in enumerate(clean_rows):
        with layout_columns[idx % column_count]:
            st.markdown(
                _build_identity_row_html(
                    label,
                    value,
                    label_color=label_color,
                    value_color=value_color,
                ),
                unsafe_allow_html=True,
            )
    return True


def _get_player_chart_colors() -> dict[str, str | None]:
    """Return the active chart colors saved for the current comparison view."""
    session_state = getattr(st, "session_state", None)
    if session_state is None:
        return {}
    if hasattr(session_state, "get"):
        player_colors = session_state.get("player_chart_colors", {})
    else:
        player_colors = getattr(session_state, "player_chart_colors", {})
    return player_colors if isinstance(player_colors, dict) else {}


def _resolve_dialog_color(color: str | None, *, fallback: str) -> str:
    """Return a safe CSS color string for dialog markup."""
    clean_color = str(color or "").strip() or fallback
    return escape(clean_color, quote=True)


def _get_player_identity_accent_color(player_name: str) -> str:
    """Return the active chart color for one player identity dialog."""
    clean_name = str(player_name or "").strip()
    player_colors = _get_player_chart_colors()
    accent_color = player_colors.get(clean_name)

    if not accent_color and clean_name:
        target_name = clean_name.casefold()
        for candidate_name, candidate_color in player_colors.items():
            if str(candidate_name or "").strip().casefold() == target_name:
                accent_color = candidate_color
                break

    return _resolve_dialog_color(accent_color, fallback=_DEFAULT_PLAYER_DIALOG_ACCENT)


def _get_team_identity_accent_color(team_abbr: str, team_name: str) -> str:
    """Return the accent color for one team identity dialog.

    Checks the active chart session state first (keyed by full team name), then
    falls back to the team's primary brand color, then to the default accent.

    Args:
        team_abbr: Three-letter NHL abbreviation used for brand-color lookup.
        team_name: Full display name used to match chart session state.

    Returns:
        CSS-safe hex color string.
    """
    player_colors = _get_player_chart_colors()
    clean_name = str(team_name or "").strip()
    accent_color = player_colors.get(clean_name)
    if not accent_color and clean_name:
        target_name = clean_name.casefold()
        for candidate_name, candidate_color in player_colors.items():
            if str(candidate_name or "").strip().casefold() == target_name:
                accent_color = candidate_color
                break
    if not accent_color:
        brand = TEAM_BRAND_COLORS.get(str(team_abbr or "").strip().upper())
        if brand:
            accent_color = brand[0]
    return _resolve_dialog_color(accent_color, fallback=_DEFAULT_PLAYER_DIALOG_ACCENT)


def _get_team_brand_color(team_abbr: str, fallback: str = _DEFAULT_PLAYER_DIALOG_ACCENT) -> str:
    """Return the primary brand color for a team abbreviation.

    Args:
        team_abbr: Three-letter NHL abbreviation.
        fallback: Color used when the team is unknown.

    Returns:
        CSS-safe hex color string.
    """
    brand = TEAM_BRAND_COLORS.get(str(team_abbr or "").strip().upper())
    return escape(brand[0] if brand else fallback, quote=True)


def _build_identity_row_html(
    label: str,
    value: str,
    *,
    label_color: str,
    value_color: str,
) -> str:
    """Return one compact dialog row with clearer label/value contrast."""
    safe_label = escape(str(label or "").strip())
    safe_value = escape(str(value or "").strip()).replace("\n", "<br>")
    safe_label_color = _resolve_dialog_color(label_color, fallback="#FFFFFF")
    safe_value_color = _resolve_dialog_color(value_color, fallback=_DEFAULT_DIALOG_VALUE_COLOR)
    return (
        "<div style='margin:0 0 1.15rem 0;'>"
        f"<div style='font-weight:700;font-size:1.02rem;color:{safe_label_color};"
        "margin-bottom:0.2rem;line-height:1.25;'>"
        f"{safe_label}</div>"
        f"<div style='color:{safe_value_color};line-height:1.45;'>{safe_value}</div>"
        "</div>"
    )


def _render_identity_section(title: str, body: str, *, accent_color: str) -> None:
    """Render a titled identity section using the active player accent color."""
    st.markdown(
        _build_identity_row_html(
            title,
            body,
            label_color=accent_color,
            value_color=_DEFAULT_DIALOG_VALUE_COLOR,
        ),
        unsafe_allow_html=True,
    )


@st.dialog("Player Details")
def show_player_identity_details(player_id: int) -> None:
    """Render the overview-card identity modal for one player or goalie."""
    summary = get_player_identity_summary(player_id)
    if not summary:
        st.info("Player details unavailable right now.")
        return

    player_name = str(summary.get("name", "Player") or "Player").strip() or "Player"
    accent_color = _get_player_identity_accent_color(player_name)
    st.markdown(
        (
            "<div style='margin:0 0 0.95rem 0;'>"
            f"<div style='font-size:1.72rem;font-weight:800;line-height:1.15;"
            f"color:{accent_color};'>{escape(player_name)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    born_bits: list[str] = []
    birth_date = str(summary.get("birth_date", "") or "").strip()
    if birth_date:
        born_bits.append(birth_date)
    age = summary.get("age")
    if age is not None:
        born_bits.append(f"Age {int(age)}")

    rows = [
        ("Born", " | ".join(born_bits)),
        ("Birthplace", str(summary.get("birthplace", "") or "").strip()),
        (str(summary.get("shot_label", "") or "").strip(), str(summary.get("shot_value", "") or "").strip()),
        ("Height", str(summary.get("height", "") or "").strip()),
        ("Weight", str(summary.get("weight", "") or "").strip()),
        ("Draft", str(summary.get("draft", "") or "").strip()),
        ("First NHL season", str(summary.get("first_nhl_season_label", "") or "").strip()),
        ("Debut team", str(summary.get("debut_team", "") or "").strip()),
    ]
    has_content = _render_identity_rows(
        rows,
        columns=2,
        label_color=accent_color,
        value_color=_DEFAULT_DIALOG_VALUE_COLOR,
    )

    honors = summary.get("honors", []) or []
    honors_text = " | ".join(str(item or "").strip() for item in honors if str(item or "").strip())
    if honors_text:
        if has_content:
            st.markdown("---")
        _render_identity_section("Honors", honors_text, accent_color=accent_color)
        has_content = True

    trophy_lines: list[str] = []
    for trophy_row in summary.get("trophies", []) or []:
        if not isinstance(trophy_row, dict):
            continue
        trophy_name = str(trophy_row.get("trophy", "") or "").strip()
        if not trophy_name:
            continue
        try:
            count = int(trophy_row.get("count", 0) or 0)
        except Exception:
            count = 0
        if count <= 0:
            continue
        latest_label = str(trophy_row.get("latest_label", "") or "").strip()
        latest_suffix = f" (latest {latest_label})" if latest_label else ""
        trophy_lines.append(f"- {trophy_name}: x{count}{latest_suffix}")
    if trophy_lines:
        if has_content:
            st.markdown("---")
        _render_identity_section("Trophies", "\n".join(trophy_lines), accent_color=accent_color)
        has_content = True

    if not has_content:
        st.info("Player details unavailable right now.")


@st.dialog("Team Details")
def show_team_identity_details(team_abbr: str) -> None:
    """Render the overview-card identity modal for one team."""
    summary = get_team_identity_summary(team_abbr)
    if not summary:
        st.info("Team details unavailable right now.")
        return

    team_name = str(summary.get("team_name", team_abbr) or team_abbr).strip()
    accent_color = _get_team_identity_accent_color(team_abbr, team_name)
    st.markdown(
        (
            "<div style='margin:0 0 0.95rem 0;'>"
            f"<div style='font-size:1.72rem;font-weight:800;line-height:1.15;"
            f"color:{accent_color};'>{escape(team_name)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    joined_label = str(summary.get("joined_nhl_label", "") or "").strip()
    if not joined_label:
        joined_year = summary.get("joined_nhl_year")
        joined_label = str(joined_year) if joined_year is not None else ""

    current_identity_label = str(summary.get("current_identity_since_label", "") or "").strip()
    if current_identity_label == joined_label:
        current_identity_label = ""

    conference_name = str(summary.get("conference_name", "") or "").strip()
    division_name = str(summary.get("division_name", "") or "").strip()
    conference_division = " / ".join(part for part in [conference_name, division_name] if part)

    total_nhl_seasons = summary.get("total_nhl_seasons")
    total_nhl_seasons_label = str(int(total_nhl_seasons)) if total_nhl_seasons else ""
    stanley_cup_count = int(summary.get("stanley_cup_count", 0) or 0)
    stanley_cup_count_label = str(stanley_cup_count) if stanley_cup_count > 0 else ""

    rows = [
        ("Joined NHL", joined_label),
        ("Current identity since", current_identity_label),
        ("Conference / division", conference_division),
        ("Total NHL seasons", total_nhl_seasons_label),
        ("Stanley Cups", stanley_cup_count_label),
    ]
    rendered_rows = _render_identity_rows(
        rows,
        columns=2,
        label_color=accent_color,
        value_color=_DEFAULT_DIALOG_VALUE_COLOR,
    )

    lineage_label = str(summary.get("lineage_label", "") or "").strip()
    stanley_cup_labels = [
        str(label or "").strip()
        for label in (summary.get("stanley_cup_labels", []) or [])
        if str(label or "").strip()
    ]
    stanley_cup_years_text = " | ".join(stanley_cup_labels)
    if lineage_label:
        if rendered_rows:
            st.markdown("---")
        _render_identity_section("Franchise lineage", lineage_label, accent_color=accent_color)
        rendered_rows = True

    if stanley_cup_years_text:
        if rendered_rows:
            st.markdown("---")
        _render_identity_section("Stanley Cup wins", stanley_cup_years_text, accent_color=accent_color)
        rendered_rows = True

    if not rendered_rows and not lineage_label and not stanley_cup_years_text:
        st.info("Team details unavailable right now.")


@st.dialog("Matchup History")
def show_matchup_history(
    away_abbr: str,
    home_abbr: str,
    limit: int = 10,
) -> None:
    """Render the latest head-to-head meetings for one upcoming matchup."""
    clean_away_abbr = str(away_abbr or "").strip().upper()
    clean_home_abbr = str(home_abbr or "").strip().upper()
    away_color = _get_team_brand_color(clean_away_abbr)
    home_color = _get_team_brand_color(clean_home_abbr)

    history_games = get_matchup_history(
        away_abbr=clean_away_abbr,
        home_abbr=clean_home_abbr,
        limit=limit,
    )
    if history_games:
        history_games = sorted(history_games, key=_matchup_history_sort_key, reverse=True)

    # Tally wins for the header sub-block.
    away_wins = home_wins = ties = 0
    for game in (history_games or []):
        away_score = game.get("away_score")
        home_score = game.get("home_score")
        try:
            a, h = int(away_score), int(home_score)
        except Exception:
            continue
        if a > h:
            if str(game.get("away_abbr", "") or "").strip().upper() == clean_away_abbr:
                away_wins += 1
            elif str(game.get("away_abbr", "") or "").strip().upper() == clean_home_abbr:
                home_wins += 1
        elif h > a:
            if str(game.get("home_abbr", "") or "").strip().upper() == clean_away_abbr:
                away_wins += 1
            elif str(game.get("home_abbr", "") or "").strip().upper() == clean_home_abbr:
                home_wins += 1
        else:
            ties += 1

    total = away_wins + home_wins + ties
    if total > 0:
        ties_line = (
            f"<br><span style='color:#8b949e;'>{ties} tie{'s' if ties != 1 else ''}</span>"
            if ties else ""
        )
        sub_html = (
            f"<div style='font-size:0.9rem;color:#8b949e;margin:0.35rem 0 0.85rem 0;line-height:1.7;'>"
            f"In the last {total} matchups:<br>"
            f"<span style='color:#ffffff;font-weight:700;'>{escape(clean_away_abbr)} won {away_wins}</span><br>"
            f"<span style='color:#ffffff;font-weight:700;'>{escape(clean_home_abbr)} won {home_wins}</span>"
            f"{ties_line}</div>"
        )
    else:
        sub_html = (
            f"<div style='font-size:0.9rem;color:#8b949e;margin:0.35rem 0 0.85rem 0;'>"
            f"Last {int(limit)} meetings</div>"
        )

    st.markdown(
        (
            "<div style='font-size:1.45rem;font-weight:800;line-height:1.2;margin:0 0 0 0;'>"
            f"<span style='color:{away_color};'>{escape(clean_away_abbr)}</span>"
            "<span style='color:#8b949e;font-weight:600;'> vs </span>"
            f"<span style='color:{home_color};'>{escape(clean_home_abbr)}</span>"
            "</div>"
        ) + sub_html,
        unsafe_allow_html=True,
    )

    if not history_games:
        st.info("No completed matchup history available right now.")
        return

    for game in history_games:
        st.markdown(_build_matchup_card_html(game, compact_layout=True), unsafe_allow_html=True)
