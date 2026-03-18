"""
nhl.constants — Shared configuration constants for the Puck Peak app.

All URL strings, lookup tables, rate-stat sets, NHLe multipliers, stat caps/floors,
and the dynamic current-season year live here.  Nothing in this module imports from
any other project module, making it safe to import from every other module without
risk of circular dependencies.
"""

from datetime import datetime
import re
import unicodedata

# ---------------------------------------------------------------------------
# API endpoint templates
# ---------------------------------------------------------------------------

SEARCH_URL    = "https://search.d3.nhle.com/api/v1/search/player"
"""D3 player-search endpoint — full-name prefix matching, returns up to 40 results."""

STATS_URL     = "https://api-web.nhle.com/v1/player/{}/landing"
"""Per-player stats endpoint; format with player_id to get seasonTotals."""

PLAYER_GAME_LOG_URL = "https://api-web.nhle.com/v1/player/{}/game-log/{}/{}"
"""Per-player game-log endpoint; format with player_id, season_id, and gameTypeId."""

SEASON_SKATER_SUMMARY_URL = "https://api.nhle.com/stats/rest/en/skater/summary"
"""Season skater-summary endpoint; query with seasonId and gameTypeId filters."""

SEASON_GOALIE_SUMMARY_URL = "https://api.nhle.com/stats/rest/en/goalie/summary"
"""Season goalie-summary endpoint; query with seasonId and gameTypeId filters."""

ROSTER_URL    = "https://api-web.nhle.com/v1/roster/{}/current"
"""Current-roster endpoint; format with team abbreviation (e.g. 'EDM')."""

TEAM_STATS_URL = "https://api.nhle.com/stats/rest/en/team/summary"
"""Team-season summary endpoint; returns all team-season rows with limit=-1."""

TEAM_LIST_URL  = "https://api.nhle.com/stats/rest/en/team"
"""Team-list endpoint; used to build teamId -> triCode lookup map."""

# ---------------------------------------------------------------------------
# Active NHL franchises (as of 2024-25 season)
# ---------------------------------------------------------------------------

ACTIVE_TEAMS = {
    "ANA": "Anaheim Ducks",       "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",      "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",  "CBJ": "Columbus Blue Jackets",
    "DAL": "Dallas Stars",        "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",     "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",   "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",  "NSH": "Nashville Predators",
    "NJD": "New Jersey Devils",   "NYI": "New York Islanders",
    "NYR": "New York Rangers",    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins",
    "SJS": "San Jose Sharks",     "SEA": "Seattle Kraken",
    "STL": "St. Louis Blues",     "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs", "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks",   "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}
"""32-team abbreviation -> full name mapping used across sidebar and pipeline."""

TEAM_LINEAGES: dict[str, tuple[str, ...]] = {
    "CAR": ("CAR", "HFD"),
    "CGY": ("CGY", "AFM"),
    "COL": ("COL", "QUE"),
    "DAL": ("DAL", "MNS"),
    "NJD": ("NJD", "CLR", "KCS"),
    "UTA": ("UTA", "ARI", "PHX", "WIN"),
    "WPG": ("WPG", "ATL"),
}
"""Current active team -> historical abbreviation aliases for franchise lineage lookups."""

TEAM_BRAND_COLORS: dict[str, tuple[str, str]] = {
    "ANA": ("#F47A38", "#B9975B"),
    "BOS": ("#FFB81C", "#111111"),
    "BUF": ("#003087", "#FFB81C"),
    "CGY": ("#C8102E", "#F1BE48"),
    "CAR": ("#CC0000", "#111111"),
    "CHI": ("#CF0A2C", "#111111"),
    "COL": ("#6F263D", "#236192"),
    "CBJ": ("#002654", "#CE1126"),
    "DAL": ("#006847", "#8F8F8C"),
    "DET": ("#CE1126", "#A2AAAD"),
    "EDM": ("#041E42", "#FF4C00"),
    "FLA": ("#041E42", "#C8102E"),
    "LAK": ("#A2AAAD", "#111111"),
    "MIN": ("#154734", "#A6192E"),
    "MTL": ("#AF1E2D", "#192168"),
    "NSH": ("#FFB81C", "#041E42"),
    "NJD": ("#CE1126", "#111111"),
    "NYI": ("#00539B", "#F47D30"),
    "NYR": ("#0038A8", "#CE1126"),
    "OTT": ("#C52032", "#C2912C"),
    "PHI": ("#F74902", "#111111"),
    "PIT": ("#FCB514", "#111111"),
    "SJS": ("#006D75", "#EA7200"),
    "SEA": ("#68A2B9", "#001628"),
    "STL": ("#002F87", "#FCB514"),
    "TBL": ("#0050A4", "#A2AAAD"),
    "TOR": ("#0055B8", "#A2AAAD"),
    "UTA": ("#6CA6D8", "#041E42"),
    "VAN": ("#00843D", "#00205B"),
    "VGK": ("#B4975A", "#333F48"),
    "WSH": ("#041E42", "#C8102E"),
    "WPG": ("#6CA6D8", "#041E42"),
}
"""Team abbreviation -> (gradient start, gradient end) colors for compact matchup UI."""

TEAM_FOUNDED: dict[str, int] = {
    "ANA": 1993, "BOS": 1924, "BUF": 1970, "CGY": 1972,
    "CAR": 1979, "CHI": 1926, "COL": 1972, "CBJ": 2000,
    "DAL": 1967, "DET": 1926, "EDM": 1979, "FLA": 1993,
    "LAK": 1967, "MIN": 2000, "MTL": 1917, "NSH": 1998,
    "NJD": 1974, "NYI": 1972, "NYR": 1926, "OTT": 1992,
    "PHI": 1967, "PIT": 1967, "SJS": 1991, "SEA": 2021,
    "STL": 1967, "TBL": 1992, "TOR": 1917, "UTA": 1979,
    "VAN": 1970, "VGK": 2017, "WSH": 1974, "WPG": 1999,
}
"""NHL franchise founding year (year the current org entered the NHL)."""

# ---------------------------------------------------------------------------
# Stat classification sets
# ---------------------------------------------------------------------------

RATE_STATS = {'PPG', 'Save %', 'GAA', 'SH%', 'TOI'}
"""
Stats that require *mean* (not sum) aggregation during pivot/groupby.
These are never cumulatively summed and their KNN uses mean aggfunc.
"""

TEAM_RATE_STATS = {'GF/G', 'GA/G', 'Win%', 'PP%', 'PPG'}
"""Team-mode rate stats — same semantics as RATE_STATS but for team pipeline."""

TEAM_METRICS = ["Points", "Wins", "Win%", "Goals", "GF/G", "GA/G", "PP%", "PPG"]
"""Ordered list of selectable metrics in team mode."""

ML_SUPPORTED_METRICS = [
    'Points', 'Goals', 'Assists', '+/-', 'PPG',
    'PIM', 'TOI', 'Wins', 'Shutouts', 'Saves', 'Save %', 'GAA',
]
"""
Metrics the KNN engine can project.
GP is intentionally excluded — survivorship bias makes KNN unreliable for games played.
GP projection uses the dedicated 4-phase durability curve instead.
TOI is supported only through the KNN path because the historical parquet now
has additive time-on-ice coverage, but the linear fallback remains intentionally
disabled for that metric.
"""

KNN_ONLY_PROJECTION_METRICS = {'TOI'}
"""
Metrics that may project only through the KNN engine.
If clone matching cannot produce a credible forecast, the app hides the
projection instead of falling back to the generic linear extrapolator.
"""

NO_PROJECTION_METRICS = {'GP', 'SH%'}
"""
Metrics for which the Forecast projection line is suppressed entirely.
Neither KNN nor the linear fallback runs for these metrics.
GP: survivorship bias corrupts both skater and goalie projections.
SH%: linear extrapolation yields implausible results for skaters and the app
does not currently maintain a separate KNN-only path for it.
"""

# ---------------------------------------------------------------------------
# NHLe (NHL Equivalency) multipliers — 2024 Bacon/Chatel model values
# ---------------------------------------------------------------------------

def normalize_league_abbrev(league: str) -> str:
    """Normalize NHL API league abbreviations for stable lookup keys.

    The NHL API returns mixed casing and occasional diacritics/punctuation variants
    (for example 'Liiga' vs 'LIIGA', 'Schüler-BL').  This function canonicalizes
    values so key matching is robust.
    """
    text = str(league or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.upper()


NHLE_DEFAULT_MULTIPLIER: float = 0.35
"""Safe fallback NHLe multiplier for unrecognized/rare leagues."""


NHLE_MULTIPLIERS = {
    'NHL':   1.0,
    'KHL':   0.77,
    'SHL':   0.57,
    'AHL':   0.39,
    'NLA':   0.46,
    'NL':    0.46,  # Swiss top division alias/rename (NLA -> NL)
    'LIIGA': 0.44,
    'NCAA':  0.19,
    'OHL':   0.14,
    'WHL':   0.14,
    'QMJHL': 0.11,
    'RUS-KHL': 0.77,  # API variant observed for KHL seasons
}
"""
League -> scoring-translation multiplier.  Applied to Points, Goals, Assists for
non-NHL rows. Unknown leagues should use NHLE_DEFAULT_MULTIPLIER at runtime.
Keys are normalized with normalize_league_abbrev().
"""

# Exhaustive league keys discovered from a curated 24-player multi-league sample.
# Most tournament/youth tags do not have stable published NHLe translations, so they
# default to NHLE_DEFAULT_MULTIPLIER while still being explicitly represented.
_DISCOVERED_API_LEAGUES = [
    '4 NATIONS', 'AYMBHL', 'CAN-CUP', 'CC', 'CHAMPIONS HL', 'CZECH',
    'CZECH U16', 'CZECH U18', 'CZECH U20', 'CZECH-JR.', 'CZECH2', 'CZECH3',
    'CZECHIA', 'CZECHIA2', 'CZR-U17', 'CZR-U18', 'CZREP', 'CZREP-2',
    'CZREP-JR.', 'DEL', 'DNL', 'DNL U20', 'ECHL', 'EHT', 'EJC-A',
    'ELITE JR. A', 'ELITE NOVIZEN', 'EUROLIGA', 'EYOF', 'FIN-JR.',
    'FIN-U18', 'FINLAND', 'FINLAND-2', 'GERMAN-2', 'GERMANY U16 2',
    'HOCKEYALLSVENSKAN', 'ITALY', 'IVAN HLINKA MEMORIAL', 'JR. A SM-LIIGA',
    'JR. B SM-SARJA', 'JR. C I-DIVISIOONA', 'JR. C SM-SARJA',
    'JR. C SM-SARJA Q', 'LIIGA', 'M-CUP', 'MESTIS', 'MHL', 'MIDGET',
    'MINI A', 'MINOR-SK', 'NAPHL 14U', 'NLB', 'OG', 'OGQ', 'OJHL',
    'OLYMPICS', 'QC INT PW', 'RUSSIA', 'RUSSIA-2', 'RUSSIA-3',
    'RUSSIA-JR.', 'SBHL', 'SCHULER-BL', 'SJHL', 'SLOVAK-2', 'SLOVAK-JR.',
    'SLOVAKIA', 'SMHL', 'SVK-U18', 'SWE-JR.', 'SWE-U18', 'SWEDEN',
    'SWEDEN-2', 'SWISS-6', 'SWISS-JR.', 'SWISS-U17', 'T1EBHL',
    'T1EHL 16U', 'TEL-CUP', 'TOP NOVIZEN', 'U-17', 'U-18', 'U15-ELIT',
    'U15-TOP', 'U16 SM-SARJA', 'U17-ELIT', 'U17-TOP', 'U18 SM-SARJA',
    'U20 I-DIVISIOONA', 'U20-ELIT', 'U20 SM-SARJA', 'USHL', 'VHL',
    'W-CUP', 'WC', 'WC-A', 'WCUP', 'WEC-A', 'WJ18-A', 'WJC-18',
    'WJC-18 D1A', 'WJC-20', 'WJC-20 D1A', 'WJC-A', 'WSI U14',
]
for _lg in _DISCOVERED_API_LEAGUES:
    NHLE_MULTIPLIERS.setdefault(_lg, NHLE_DEFAULT_MULTIPLIER)
del _lg

# ---------------------------------------------------------------------------
# Dynamic current season year
# ---------------------------------------------------------------------------

_now = datetime.now()
CURRENT_SEASON_YEAR: int = _now.year if _now.month >= 9 else _now.year - 1
"""
Start year of the current NHL season (e.g. 2024 for the 2024-25 season).
Computed once at import time from the system clock; NHL seasons start in October,
so January-August still belong to the season that started the prior calendar year.
"""

# ---------------------------------------------------------------------------
# Stat caps and floors (applied to every projected year)
# ---------------------------------------------------------------------------

STAT_CAPS: dict[str, float] = {
    "Points":  155,
    "Goals":    70,
    "Assists": 105,
    "+/-":      60,
    "GP":       82,   # skaters; goalie cap overridden to 65 inline in knn_engine
    "PPG":       1.9,
    "SH%":      25,
    "PIM":     150,
    "TOI":      28,
    "Save %":   93.5,
    "GAA":       1.8,  # NOTE: GAA cap is a *floor* — no projection below 1.8
    "Wins":     45,
    "Shutouts": 10,
    "Saves":  2000,
}
"""
Per-metric upper bounds applied during projection.
GAA is special: its cap acts as a floor (no goalie can project below 1.8 GAA).
"""

STAT_FLOORS: dict[str, float] = {
    "+/-": -60,
}
"""
Per-metric lower bounds applied during projection.
+/- has a symmetric floor (-60) to mirror its ceiling (60).
"""

# ---------------------------------------------------------------------------
# Chart color palettes — one per stat category
# ---------------------------------------------------------------------------

SKATER_COLORS: list[str] = [
    "#4FC3F7",  # electric blue
    "#76FF03",  # neon lime
    "#29B6F6",  # sky blue
    "#00E5A8",  # neon mint
    "#00E5FF",  # neon cyan
    "#B2FF59",  # lime glow
    "#40C4FF",  # bright azure
    "#64FFDA",  # aqua mint
    "#18FFFF",  # bright cyan
    "#AEEA00",  # chartreuse
]
"""Alternating neon blue/green palette for skater traces."""

GOALIE_COLORS: list[str] = [
    "#FFB300",  # neon amber
    "#FF5252",  # neon red
    "#FFD740",  # bright gold
    "#FF6E40",  # coral orange
    "#FFAB40",  # vivid orange
    "#FF4081",  # hot pink accent
    "#FFC400",  # golden yellow
    "#FF7043",  # sunset orange
    "#FF8A65",  # bright salmon
    "#F50057",  # raspberry neon
]
"""Alternating neon warm palette for goalie traces with stronger separation."""

TEAM_COLORS: list[str] = [
    "#7C4DFF",  # vivid violet
    "#536DFE",  # electric indigo
    "#B388FF",  # bright lavender
    "#448AFF",  # cobalt blue
    "#9575CD",  # soft violet
    "#5C6BC0",  # indigo blue
    "#7E57C2",  # amethyst
    "#64B5F6",  # cool neon blue
    "#651FFF",  # deep electric violet
    "#7986CB",  # periwinkle
]
"""Alternating neon blue-violet palette for team traces."""
