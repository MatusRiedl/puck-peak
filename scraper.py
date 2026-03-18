import requests
import pandas as pd
import concurrent.futures
import time
from datetime import datetime

SKATER_RECORDS_URL  = "https://records.nhl.com/site/api/skater-career-scoring-regular-season"
GOALIE_RECORDS_URL  = "https://records.nhl.com/site/api/goalie-career-stats"
PLAYER_API_URL      = "https://api-web.nhle.com/v1/player/{}/landing"
REQUEST_HEADERS     = {"User-Agent": "puck-peak/1.0"}
REQUEST_TIMEOUT     = 15
PLAYER_FETCH_WORKERS = 10

# NHL stats REST API — returns all players for a specific season (limit=-1 = no cap).
# seasonId format: start_year + end_year, e.g. 20242025 for 2024-25.
SEASON_SKATER_URL   = "https://api.nhle.com/stats/rest/en/skater/summary?limit=-1&start=0&sort=points&cayenneExp=seasonId={sid}"
SEASON_GOALIE_URL   = "https://api.nhle.com/stats/rest/en/goalie/summary?limit=-1&start=0&sort=wins&cayenneExp=seasonId={sid}"


def _normalize_save_pct(
    raw_save_pct: float | int | None,
    saves: float,
    shots_against: float,
) -> float:
    """Normalize NHL API save percentage values to 0-1 scale.

    The NHL API usually returns goalie savePctg in 0-1 scale, but some traded
    season rows and historical edge cases can surface percent-scale values or
    malformed values that disagree with the saves / shots against counts.

    Args:
        raw_save_pct: Raw savePctg value from the NHL API seasonTotals row.
        saves: Season saves value for the same row.
        shots_against: Season shots-against value for the same row.

    Returns:
        Save percentage in 0-1 scale, clamped to [0.0, 1.0].
    """
    try:
        raw_val = float(raw_save_pct or 0.0)
    except Exception:
        raw_val = 0.0

    if raw_val > 1.5:
        raw_val = raw_val / 100.0

    computed_val = None
    if shots_against > 0:
        computed_val = saves / shots_against

    if computed_val is not None:
        if raw_val <= 0 or raw_val > 1 or abs(raw_val - computed_val) > 0.015:
            raw_val = computed_val

    return max(0.0, min(raw_val, 1.0))


def _get_json_with_retries(
    url: str,
    timeout: int = REQUEST_TIMEOUT,
    max_attempts: int = 5,
    base_sleep: float = 0.75,
) -> dict | list:
    """Fetch JSON from an NHL endpoint with light retry and backoff.

    The NHL public APIs occasionally rate-limit with HTTP 429 or return a blank
    body during transient failures. Retrying here prevents the scraper from
    silently dropping most historical players.

    Args:
        url: Endpoint URL to request.
        timeout: Per-request timeout in seconds.
        max_attempts: Maximum number of attempts before failing.
        base_sleep: Base backoff interval in seconds.

    Returns:
        Parsed JSON payload as a dict or list.

    Raises:
        requests.HTTPError: If a non-retryable HTTP status persists.
        ValueError: If JSON decoding keeps failing after retries.
    """
    last_error = None

    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, timeout=timeout, headers=REQUEST_HEADERS)
            if resp.status_code == 429:
                retry_after = resp.headers.get('Retry-After')
                try:
                    sleep_for = float(retry_after)
                except Exception:
                    sleep_for = base_sleep * (attempt + 1)
                time.sleep(sleep_for)
                continue

            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_sleep * (attempt + 1))

    if last_error is not None:
        raise last_error
    raise ValueError(f"Failed to fetch JSON from {url}")


def _collapse_traded_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple team stints into one player-season row.

    Counting stats are summed. Goalie rate stats are rebuilt from aggregate
    numerators after the groupby so traded goalies do not get impossible SavePct
    or GAA values from summing per-team rate stats.

    Args:
        df: Raw season DataFrame with one row per API seasonTotals stint.

    Returns:
        DataFrame with one row per player, season, age, and position.
    """
    if df.empty:
        return df

    agg_cols = [
        'GP', 'Points', 'Goals', 'Assists', 'PIM', '+/-', 'Wins', 'Shutouts',
        'Saves', 'GoalsAgainst', 'WeightedGAA', 'Shots', 'TotalTOIMins',
    ]
    collapsed = (
        df.groupby(['PlayerID', 'SeasonYear', 'Age', 'Position'], as_index=False)[agg_cols]
        .sum()
    )

    faced = collapsed['Saves'] + collapsed['GoalsAgainst']
    collapsed['SavePct'] = (collapsed['Saves'] / faced.where(faced > 0)).fillna(0.0)
    collapsed['GAA'] = (
        collapsed['WeightedGAA'] / collapsed['GP'].where(collapsed['GP'] > 0)
    ).fillna(0.0)

    return collapsed.drop(columns=['GoalsAgainst', 'WeightedGAA'])


def _toi_to_minutes(toi_str: str) -> float:
    """Convert an ``MM:SS`` TOI string into decimal minutes."""
    try:
        parts = str(toi_str or '0:00').split(':')
        if len(parts) != 2:
            return 0.0
        return int(parts[0]) + int(parts[1]) / 60.0
    except Exception:
        return 0.0


def get_all_season_ids():
    """Return all NHL season IDs from 1917-18 through the most recently completed season."""
    now = datetime.now()
    end_year = now.year if now.month >= 9 else now.year - 1
    return [f"{y}{y + 1}" for y in range(1917, end_year + 1)]


def _fetch_all_records(base_url):
    """Paginate through an NHL records endpoint until all results are collected.
    The API defaults to ~25 results per call; without pagination the scraper would
    only capture the first page — that's why historical seasons showed 9 players."""
    ids = set()
    start = 0
    page_size = 500   # large page to minimise round-trips
    while True:
        try:
            url  = f"{base_url}?start={start}&limit={page_size}"
            resp = _get_json_with_retries(url)
            page = resp.get('data', [])
            for p in page:
                ids.add(int(p['playerId']))
            # Stop when this page is smaller than the requested limit (last page).
            if len(page) < page_size:
                break
            start += page_size
        except Exception as e:
            print(f"  Warning: records page failed (start={start}): {e}")
            break
    return ids


def get_all_player_ids():
    """Collect all NHL player IDs from two sources:
    1. Career records endpoints (all-time historical players) — paginated.
    2. Season summary endpoints for every NHL season (catches players absent
       from career leaderboards and preserves historical roster coverage).
    """
    print("Fetching master list of all NHL players in history...")
    ids = set()

    # --- Source 1: all-time career records (paginated) ---
    for url in [SKATER_RECORDS_URL, GOALIE_RECORDS_URL]:
        ids |= _fetch_all_records(url)

    print(f"  Career records: {len(ids)} players")

    # --- Source 2: all historical season summaries (catches players missing from career records) ---
    # The career records endpoint is a scoring leaderboard — role players with modest career
    # totals may be absent. Sweeping every season ensures complete roster coverage.
    recent_ids = set()
    all_sids = get_all_season_ids()
    print(f"  Sweeping {len(all_sids)} seasons for additional player IDs...")
    for sid in all_sids:
        for url_tpl in [SEASON_SKATER_URL, SEASON_GOALIE_URL]:
            try:
                url = url_tpl.format(sid=sid)
                data = _get_json_with_retries(url).get('data', [])
                for p in data:
                    pid = p.get('playerId')
                    if pid:
                        recent_ids.add(int(pid))
            except Exception as e:
                print(f"  Warning: season {sid} fetch failed: {e}")

    print(f"  Season-sweep additions: {len(recent_ids - ids)} new players")
    ids |= recent_ids
    print(f"  Total unique player IDs: {len(ids)}")
    return list(ids)


def fetch_player_data(player_id):
    """Fetch raw NHL regular-season rows for one player.

    Args:
        player_id: Numeric NHL player ID.

    Returns:
        List of dict rows ready for DataFrame construction. Returns an empty list
        on request or parsing failure.
    """
    try:
        res = _get_json_with_retries(PLAYER_API_URL.format(player_id))
        birth_date = str(res.get('birthDate', '2000'))
        birth_year = int(birth_date[:4]) if len(birth_date) >= 4 else 2000
        position = str(res.get('positionCode') or res.get('position') or 'S').upper()

        seasons = []
        for s in res.get('seasonTotals', []):
            league    = str(s.get('leagueAbbrev', '')).strip().upper()
            game_type = str(s.get('gameTypeId', ''))

            # Only NHL Regular Season data for the baseline model
            if league == 'NHL' and game_type == '2':
                season_str  = str(s.get('season', ''))
                season_year = int(season_str[:4]) if len(season_str) >= 4 else 2000
                age         = season_year - birth_year
                gp          = max(s.get('gamesPlayed', 1), 1)
                toi_mins    = _toi_to_minutes(str(s.get('avgToi', '0:00')))
                shots_against = max(float(s.get('shotsAgainst', 0) or 0), 0.0)
                goals_against = max(float(s.get('goalsAgainst', 0) or 0), 0.0)
                raw_saves = s.get('saves', None)
                saves = (
                    max(float(raw_saves), 0.0)
                    if raw_saves is not None
                    else max(0.0, shots_against - goals_against)
                )
                save_pct = _normalize_save_pct(
                    raw_save_pct=s.get('savePctg', 0.0),
                    saves=saves,
                    shots_against=shots_against,
                )

                # Store RAW Points — no era adjustment baked in.
                # The app applies era adjustment on demand using its 8-period multiplier.
                raw_pts = s.get('points', 0)

                seasons.append({
                    "PlayerID":   player_id,
                    "Age":        age,
                    "SeasonYear": season_year,
                    "Position":   position,
                    "GP":         gp,
                    "Points":     raw_pts,           # RAW (no era adjustment)
                    "Goals":      s.get('goals', 0),
                    "Assists":    s.get('assists', 0),
                    "PIM":        s.get('pim', 0) or s.get('penaltyMinutes', 0),
                    "+/-":        s.get('plusMinus', 0),
                    "Shots":      s.get('shots', 0) or 0,
                    "TotalTOIMins": toi_mins * gp,
                    "Wins":       s.get('wins', 0),
                    "Shutouts":   s.get('shutouts', 0),
                    "Saves":      saves,
                    "GoalsAgainst": goals_against,
                    "WeightedGAA": float(s.get('goalsAgainstAvg', 0.0)) * gp,
                    "SavePct":    save_pct,
                    "GAA":        float(s.get('goalsAgainstAvg', 0.0)),
                })
        return seasons
    except Exception:
        return []


def main():
    """Scrape historical player seasons and export the parquet artifact."""
    player_ids = get_all_player_ids()
    all_seasons_data = []

    print("Initiating multi-threaded scraping... This will take a couple of minutes.")
    start_time = time.time()

    # Keep concurrency high enough to finish quickly, but below the point where
    # the landing endpoint starts rate-limiting the entire run.
    with concurrent.futures.ThreadPoolExecutor(max_workers=PLAYER_FETCH_WORKERS) as executor:
        results = list(executor.map(fetch_player_data, player_ids))

    for res in results:
        if res:
            all_seasons_data.extend(res)

    df = pd.DataFrame(all_seasons_data)

    # Collapse traded-player duplicates (same player, same season, same age) into one row.
    # Rate stats must be rebuilt after aggregation. Summing SavePct or GAA creates
    # impossible goalie rows for traded seasons.
    df = _collapse_traded_seasons(df)

    elapsed = round(time.time() - start_time, 1)
    print(f"Scraping complete in {elapsed} seconds.")
    print(f"Total valid NHL season-rows recorded: {len(df)}")
    print(f"Unique players: {df['PlayerID'].nunique()}")
    print(f"Season range: {df['SeasonYear'].min()} – {df['SeasonYear'].max()}")

    # Save as highly-compressed Parquet
    df.to_parquet("nhl_historical_seasons.parquet", index=False, compression="snappy")
    print("Saved to nhl_historical_seasons.parquet")


if __name__ == "__main__":
    main()

