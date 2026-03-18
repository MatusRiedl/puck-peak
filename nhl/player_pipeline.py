"""Per-player processing pipeline for chart-ready age or games-played curves."""

import pandas as pd

from nhl.constants import (
    KNN_ONLY_PROJECTION_METRICS,
    ML_SUPPORTED_METRICS,
    NHLE_DEFAULT_MULTIPLIER,
    NHLE_MULTIPLIERS,
    NO_PROJECTION_METRICS,
    RATE_STATS,
    normalize_league_abbrev,
)
from nhl.data_loaders import get_player_raw_stats, get_player_season_game_log
from nhl.era import apply_era_to_hist, get_era_multiplier, get_goalie_era_sv_offset
from nhl.knn_engine import run_knn_projection, run_linear_fallback

# Projection eligibility thresholds (tweak if you want)
MIN_SEASONS_FOR_PROJ = 2
MIN_CAREER_GP_FOR_PROJ_SKATER = 82   # < full season => no projection
MIN_CAREER_GP_FOR_PROJ_GOALIE = 25
TOI_PROJECTION_COVERAGE_START_YEAR = 1997
MIN_TOI_SEASONS_FOR_PROJ = 3
MIN_TOI_GP_FOR_PROJ = 120
MIN_TOI_HIST_GP = 40


def _filter_toi_projection_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the modern TOI-bearing age rows eligible for clone matching."""
    if df is None or df.empty:
        return pd.DataFrame()
    required_cols = {"SeasonYear", "TotalTOIMins", "GP"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    d = df.copy()
    season_year = pd.to_numeric(d["SeasonYear"], errors="coerce")
    total_toi = pd.to_numeric(d["TotalTOIMins"], errors="coerce").fillna(0.0)
    gp = pd.to_numeric(d["GP"], errors="coerce").fillna(0.0)
    return d[
        season_year.ge(TOI_PROJECTION_COVERAGE_START_YEAR)
        & total_toi.gt(0)
        & gp.gt(0)
    ].copy()


def _can_project_toi(df: pd.DataFrame, stat_category: str) -> bool:
    """Return whether the active curve has enough modern TOI history to forecast."""
    if stat_category != "Skater":
        return False

    usable = _filter_toi_projection_rows(df)
    if usable.empty:
        return False

    all_ages = pd.to_numeric(df.get("Age"), errors="coerce")
    usable_ages = pd.to_numeric(usable.get("Age"), errors="coerce")
    if all_ages.dropna().empty or usable_ages.dropna().empty:
        return False
    if int(usable_ages.max()) != int(all_ages.max()):
        return False

    usable_seasons = int(pd.to_numeric(usable.get("SeasonYear"), errors="coerce").dropna().nunique())
    usable_gp = float(pd.to_numeric(usable.get("GP"), errors="coerce").fillna(0.0).sum())
    return usable_seasons >= MIN_TOI_SEASONS_FOR_PROJ and usable_gp >= MIN_TOI_GP_FOR_PROJ


def _filter_toi_knn_hist(hist_df: pd.DataFrame) -> pd.DataFrame:
    """Return the historical skater TOI pool with modern nonzero coverage only."""
    if hist_df is None or hist_df.empty:
        return pd.DataFrame()
    required_cols = {"SeasonYear", "TotalTOIMins", "GP"}
    if not required_cols.issubset(hist_df.columns):
        return pd.DataFrame()

    d = hist_df.copy()
    season_year = pd.to_numeric(d["SeasonYear"], errors="coerce")
    total_toi = pd.to_numeric(d["TotalTOIMins"], errors="coerce").fillna(0.0)
    gp = pd.to_numeric(d["GP"], errors="coerce").fillna(0.0)
    return d[
        season_year.ge(TOI_PROJECTION_COVERAGE_START_YEAR)
        & total_toi.gt(0)
        & gp.ge(MIN_TOI_HIST_GP)
    ].copy()


def process_players(
    players: dict,
    metric: str,
    hist_df: pd.DataFrame,
    id_to_name_map: dict,
    clone_details_map: dict,
    season_type: str,
    stat_category: str,
    do_era: bool,
    do_predict: bool,
    do_smooth: bool,
    do_cumul: bool,
    games_mode: bool,
    selected_season: str | int = "All",
    league_filter: list | None = None,
) -> tuple:
    """Run the player pipeline and return chart data, raw caches, clones, and peaks."""
    processed_dfs  = []
    raw_dfs_cache  = []
    ml_clones_dict = {}
    peak_info      = {}

    _league_filter = [] if league_filter is None else league_filter
    season_mode = str(selected_season) != "All"

    # Era-adjust historical data once per category before the player loop.
    # This avoids per-player era adjustment inside the projection path and keeps
    # skater/goalie adjustments isolated from each other.
    if do_predict:
        hist_df_skater = apply_era_to_hist(hist_df, do_era, is_goalie=False)
        hist_df_goalie = apply_era_to_hist(hist_df, do_era, is_goalie=True)
    else:
        hist_df_skater = hist_df
        hist_df_goalie = hist_df

    for pid, name in players.items():
        if season_mode:
            try:
                season_year = int(selected_season)
            except Exception:
                continue
            raw_df, base_name, pos_code = get_player_season_game_log(int(pid), name, season_year)
        else:
            raw_df, base_name, pos_code = get_player_raw_stats(pid, name)
        if raw_df.empty:
            continue

        # --- Skater / goalie gatekeeper ---
        is_goalie = raw_df['Saves'].sum() > 0 or raw_df['Wins'].sum() > 0
        if stat_category == "Skater" and is_goalie:
            continue
        if stat_category == "Goalie" and not is_goalie:
            continue

        if 'PlayerID' not in raw_df.columns:
            raw_df['PlayerID'] = int(pid)
        raw_df['PlayerID'] = pd.to_numeric(raw_df['PlayerID'], errors='coerce').fillna(int(pid)).astype(int)
        raw_df['PositionCode'] = str(pos_code or ('G' if is_goalie else 'S'))
        raw_df['BaseName'] = base_name
        raw_dfs_cache.append(raw_df.copy())

        # --- Step 3: League filter + optional NHLe conversion ---
        if season_mode:
            raw_df = raw_df[
                raw_df['League'].apply(normalize_league_abbrev) == 'NHL'
            ].copy()
        else:
            _selected_norm = {
                normalize_league_abbrev(_lg)
                for _lg in _league_filter
                if normalize_league_abbrev(_lg)
            }
            raw_df = raw_df[
                raw_df['League'].apply(normalize_league_abbrev).isin(_selected_norm)
            ].copy()
        if raw_df.empty:
            continue
        # When Era is on, translate skater scoring from other leagues into a rough
        # NHL-equivalent level before the NHL-only era adjustment runs. With Era off,
        # keep the selected leagues raw.
        if 'NHLeMultiplier' not in raw_df.columns:
            raw_df['NHLeMultiplier'] = raw_df['League'].apply(
                lambda _lg: NHLE_MULTIPLIERS.get(
                    normalize_league_abbrev(_lg), NHLE_DEFAULT_MULTIPLIER
                )
            )
        if do_era and stat_category == "Skater":
            _mult = raw_df['NHLeMultiplier'].fillna(NHLE_DEFAULT_MULTIPLIER)
            raw_df['Points'] *= _mult
            raw_df['Goals'] *= _mult
            raw_df['Assists'] *= _mult

        # --- Step 4: Season type filter ---
        if season_type != "Both":
            raw_df = raw_df[raw_df['GameType'] == season_type]
        if raw_df.empty:
            continue

        # --- Step 5: Era adjustment (NHL rows only) ---
        if do_era and stat_category == "Skater":
            # FIX #4: Adjust Goals and Assists independently, not just Points.
            # Era multipliers derived from NHL GF/GP data; apply to NHL rows only
            # so non-NHL rows are not double-adjusted after the league-normalisation step.
            _nhl_mask = raw_df['League'].apply(normalize_league_abbrev) == 'NHL'
            if _nhl_mask.any():
                _era_mults = raw_df.loc[_nhl_mask, 'SeasonYear'].apply(get_era_multiplier)
                raw_df.loc[_nhl_mask, 'Points']  *= _era_mults.values
                raw_df.loc[_nhl_mask, 'Goals']   *= _era_mults.values
                raw_df.loc[_nhl_mask, 'Assists']  *= _era_mults.values

        if do_era and is_goalie:
            # Era-normalize goalie stats to the 2018+ baseline (NHL rows only).
            # Must target WeightedGAA and WeightedSV — the GP-weighted pre-groupby sums —
            # because Save % and GAA don't exist in raw_df until post-groupby.
            # Shutouts adjusted inversely: harder to record in high-scoring eras.
            _nhl_mask = raw_df['League'].apply(normalize_league_abbrev) == 'NHL'
            if _nhl_mask.any():
                _era_mults  = raw_df.loc[_nhl_mask, 'SeasonYear'].apply(get_era_multiplier)
                _sv_offsets = raw_df.loc[_nhl_mask, 'SeasonYear'].apply(get_goalie_era_sv_offset)
                _gp_nhl     = raw_df.loc[_nhl_mask, 'GP']
                raw_df.loc[_nhl_mask, 'WeightedGAA'] = (
                    raw_df.loc[_nhl_mask, 'WeightedGAA'] * _era_mults.values
                ).clip(lower=0)
                raw_df.loc[_nhl_mask, 'WeightedSV'] = (
                    raw_df.loc[_nhl_mask, 'WeightedSV'] + _sv_offsets.values * 100 * _gp_nhl.values
                ).clip(lower=0)
                raw_df.loc[_nhl_mask, 'Shutouts'] = (
                    raw_df.loc[_nhl_mask, 'Shutouts'] / _era_mults.values
                )

        # --- Step 6a: Games Played mode branch ---
        if season_mode:
            df = raw_df.sort_values(['GameDate', 'GameId']).reset_index(drop=True)
            df['CumGP'] = range(1, len(df) + 1)

            if do_cumul:
                cum_gp = df['CumGP'].astype(float)
                cum_points = df['Points'].cumsum()
                cum_goals = df['Goals'].cumsum()
                cum_assists = df['Assists'].cumsum()
                cum_wins = df['Wins'].cumsum()
                cum_shutouts = df['Shutouts'].cumsum()
                cum_pim = df['PIM'].cumsum()
                cum_saves = df['Saves'].cumsum()
                cum_pm = df['+/-'].cumsum()
                cum_toi = df['TotalTOIMins'].cumsum()
                cum_wsv = df['WeightedSV'].cumsum()
                cum_wgaa = df['WeightedGAA'].cumsum()
                cum_shots = df['Shots'].cumsum()

                df['Points'] = cum_points
                df['Goals'] = cum_goals
                df['Assists'] = cum_assists
                df['Wins'] = cum_wins
                df['Shutouts'] = cum_shutouts
                df['PIM'] = cum_pim
                df['Saves'] = cum_saves
                df['+/-'] = cum_pm
                df['GP'] = cum_gp

                df['PPG'] = cum_points / cum_gp
                df['TOI'] = cum_toi / cum_gp
                df['SH%'] = (cum_goals / cum_shots.replace(0, float('nan')) * 100).fillna(0)
                df['Save %'] = cum_wsv / cum_gp
                df['GAA'] = cum_wgaa / cum_gp
            else:
                game_gp = df['GP'].replace(0, 1)
                df['PPG'] = df['Points'] / game_gp
                df['TOI'] = df['TotalTOIMins'] / game_gp
                df['SH%'] = (df['Goals'] / df['Shots'].replace(0, float('nan')) * 100).fillna(0)
                df['Save %'] = df['WeightedSV'] / game_gp
                df['GAA'] = df['WeightedGAA'] / game_gp

        elif games_mode:
            # Group by SeasonYear first to collapse Regular+Playoffs into one row
            age_per_season = raw_df.groupby('SeasonYear')['Age'].max()
            df = raw_df.groupby('SeasonYear').sum(numeric_only=True).reset_index()
            df['Age'] = df['SeasonYear'].map(age_per_season)
            df = df.sort_values('SeasonYear').reset_index(drop=True)

            # CumGP is always computed — it is the x-axis column in both sub-modes
            cum_gp = df['GP'].cumsum()
            df['CumGP'] = cum_gp

            if do_cumul:
                # Cumulative sub-branch: overwrite counting stats with career totals
                cum_points   = df['Points'].cumsum()
                cum_goals    = df['Goals'].cumsum()
                cum_assists  = df['Assists'].cumsum()
                cum_wins     = df['Wins'].cumsum()
                cum_shutouts = df['Shutouts'].cumsum()
                cum_pim      = df['PIM'].cumsum()
                cum_saves    = df['Saves'].cumsum()
                cum_pm       = df['+/-'].cumsum()
                cum_toi      = df['TotalTOIMins'].cumsum()
                cum_wsv      = df['WeightedSV'].cumsum()
                cum_wgaa     = df['WeightedGAA'].cumsum()
                cum_shots    = df['Shots'].cumsum()

                df['Points']   = cum_points
                df['Goals']    = cum_goals
                df['Assists']  = cum_assists
                df['Wins']     = cum_wins
                df['Shutouts'] = cum_shutouts
                df['PIM']      = cum_pim
                df['Saves']    = cum_saves
                df['+/-']      = cum_pm
                df['GP']       = cum_gp   # career GP to date

                df['PPG']    = cum_points / cum_gp
                df['TOI']    = cum_toi / cum_gp
                df['SH%']    = (cum_goals / cum_shots.replace(0, float('nan')) * 100).fillna(0)
                df['Save %'] = cum_wsv / cum_gp
                df['GAA']    = cum_wgaa / cum_gp

            else:
                # Per-season sub-branch: keep raw season stats; derive rate stats per season
                season_gp    = df['GP'].copy()   # season GP before any override
                df['PPG']    = df['Points'] / season_gp
                df['TOI']    = df['TotalTOIMins'] / season_gp
                df['SH%']    = (df['Goals'] / df['Shots'].replace(0, float('nan')) * 100).fillna(0)
                df['Save %'] = df['WeightedSV'] / season_gp
                df['GAA']    = df['WeightedGAA'] / season_gp
                # GP stays as season_gp; CumGP is the x-axis

        else:
            # --- Step 6b: Age mode branch ---
            # FIX #2: Preserve SeasonYear as max per age (not sum).
            season_year_max = raw_df.groupby('Age')['SeasonYear'].max()
            df = raw_df.groupby('Age').sum(numeric_only=True).reset_index()
            df['SeasonYear'] = df['Age'].map(season_year_max)

            df['PPG']    = df['Points'] / df['GP']
            df['TOI']    = df['TotalTOIMins'] / df['GP']
            df['SH%']    = (df['Goals'] / df['Shots'] * 100).fillna(0)
            df['Save %'] = df['WeightedSV'] / df['GP']
            df['GAA']    = df['WeightedGAA'] / df['GP']

        df['BaseName'] = base_name
        df['Player']   = base_name

        # --- Step 7: Origin-anchor zero row (Games Played cumulative mode) ---
        if games_mode and do_cumul:
            # Anchor every player's line at career game 0 so all share the same
            # x=0 origin.  Without this, each line starts at end of first season.
            # Not applied in per-season mode — lines naturally start at first season.
            _zero = {col: 0 for col in df.columns}
            _zero.update({
                'CumGP': 0, 'GP': 0,
                'Age':   int(df['Age'].iloc[0]) if not df.empty else 18,
                'SeasonYear': int(df['SeasonYear'].iloc[0]) if 'SeasonYear' in df.columns and not df.empty else 0,
                'Player': base_name, 'BaseName': base_name,
            })
            # Rate stats have no meaningful value at game 0 — leave as NaN
            for _rs in ('PPG', 'TOI', 'SH%', 'Save %', 'GAA'):
                if _rs in _zero:
                    _zero[_rs] = float('nan')
            if 'GameDate' in df.columns:
                _zero['GameDate'] = None
            if 'GameId' in df.columns:
                _zero['GameId'] = 0
            df = pd.concat([pd.DataFrame([_zero]), df], ignore_index=True)

        # --- Step 8: Peak detection (pre-smoothing, pre-cumsum) ---
        _peak_x = _peak_age = _peak_sy = _peak_game_number = None
        _peak_game_date = None
        _peak_raw_val = None
        try:
            if metric in df.columns and not df[metric].dropna().empty:
                if games_mode and do_cumul and metric not in RATE_STATS:
                    # Cumulative data: extract per-season increments for peak detection
                    _incremental = df[metric].diff().fillna(df[metric])
                    _pidx = (
                        _incremental.replace(0, float('nan')).idxmin()
                        if metric == 'GAA'
                        else _incremental.idxmax()
                    )
                else:
                    _series = df[metric].replace(0, float('nan')) if metric == 'GAA' else df[metric]
                    _pidx   = _series.idxmin() if metric == 'GAA' else _series.idxmax()
                if pd.notna(_pidx):
                    _pr      = df.loc[_pidx]
                    _peak_age = int(_pr['Age'])
                    _peak_sy  = int(_pr['SeasonYear']) if 'SeasonYear' in df.columns else None
                    _peak_x   = float(_pr['CumGP']) if games_mode else float(_peak_age)
                    _peak_game_number = int(_pr['CumGP']) if 'CumGP' in df.columns else None
                    _peak_game_date = _pr.get('GameDate') if hasattr(_pr, 'get') else None
                    if games_mode and do_cumul and metric not in RATE_STATS:
                        peak_raw = _incremental.loc[_pidx]
                        _peak_raw_val = float(peak_raw) if pd.notna(peak_raw) else None
                    else:
                        peak_raw = _pr.get(metric) if hasattr(_pr, 'get') else None
                        _peak_raw_val = float(peak_raw) if pd.notna(peak_raw) else None
        except Exception:
            pass

        # --- Step 9: KNN projection or linear fallback ---
        max_age = df['Age'].max()
        can_project = (
            not games_mode
            and do_predict
            and max_age < 40
            and metric not in NO_PROJECTION_METRICS
        )
        if can_project and metric == "TOI":
            can_project = _can_project_toi(df, stat_category)

        # Thin‑data guard: no projection if career is too short
        if can_project and metric != "TOI":
            seasons  = int(df['Age'].nunique()) if 'Age' in df.columns else 0
            total_gp = float(df['GP'].sum()) if 'GP' in df.columns else 0
            min_gp   = MIN_CAREER_GP_FOR_PROJ_GOALIE if is_goalie else MIN_CAREER_GP_FOR_PROJ_SKATER
            if seasons < MIN_SEASONS_FOR_PROJ or total_gp < min_gp:
                can_project = False

        if can_project:
            career_df = _filter_toi_projection_rows(df) if metric == "TOI" else df.copy()
            player_knn_hist = hist_df_goalie if is_goalie else hist_df_skater
            if metric == "TOI":
                player_knn_hist = _filter_toi_knn_hist(player_knn_hist)
            player_pos_code = 'G' if is_goalie else pos_code
            use_ml    = not player_knn_hist.empty and metric in ML_SUPPORTED_METRICS
            proj_rows = []

            if use_ml:
                proj_rows, clone_names = run_knn_projection(
                    career_df      = career_df,
                    metric         = metric,
                    hist_df        = player_knn_hist,
                    is_goalie      = is_goalie,
                    pos_code       = player_pos_code,
                    do_era         = False,
                    season_type    = season_type,
                    stat_category  = stat_category,
                    id_to_name_map = id_to_name_map,
                    clone_details_map = clone_details_map,
                )
                if not proj_rows:
                    use_ml = False
                else:
                    ml_clones_dict[base_name] = clone_names
            if not use_ml and metric not in KNN_ONLY_PROJECTION_METRICS:
                proj_rows = run_linear_fallback(
                    career_df  = career_df,
                    metric     = metric,
                    max_age    = int(max_age),
                    stat_category = stat_category,
                )
                ml_clones_dict[base_name] = []
            elif not use_ml:
                ml_clones_dict[base_name] = []

            if proj_rows:
                df = pd.concat([df, pd.DataFrame(proj_rows)], ignore_index=True)

        # --- Step 10: Cumulative toggle (age mode only) ---
        if do_cumul and not games_mode:
            # games_mode handles cumulation in Step 6a to avoid double-application
            df[metric] = df[metric].cumsum()

        # --- Step 11: 3-season rolling average smoothing ---
        if do_smooth:
            df[metric] = df[metric].rolling(window=3, min_periods=1).mean()

        # --- Step 12: Real / projection split ---
        if not games_mode and do_predict and df['Age'].max() > max_age:
            real_part = df[df['Age'] <= max_age].copy()
            proj_part = df[df['Age'] >= max_age].copy()
            if not proj_part.empty:
                proj_part['Player'] = f"{base_name} (Proj)"
                final_player_df = pd.concat([real_part, proj_part], ignore_index=True)
            else:
                final_player_df = real_part.copy()
        else:
            final_player_df = df.copy()

        # Look up the star's chart y-value at the peak position (post-smoothing/cumsum)
        if _peak_x is not None and _peak_sy is not None:
            x_col_lk  = 'CumGP' if games_mode else 'Age'
            real_only = final_player_df[
                ~final_player_df['Player'].str.contains(r'\(Proj\)', na=False)
            ]
            match = real_only[real_only[x_col_lk] == _peak_x]
            if not match.empty and metric in match.columns:
                peak_info[base_name] = {
                    'x':            _peak_x,
                    'y':            float(match[metric].iloc[0]),
                    'raw_peak_val': _peak_raw_val if _peak_raw_val is not None else float(match[metric].iloc[0]),
                    'age':          _peak_age,
                    'season_year':  _peak_sy,
                    'game_number':  _peak_game_number,
                    'game_date':    _peak_game_date,
                    'pid':          pid,
                }

        processed_dfs.append(final_player_df)

    return processed_dfs, raw_dfs_cache, ml_clones_dict, peak_info
