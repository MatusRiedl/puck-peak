"""Expected goals (xG) from NHL play-by-play: shot features, scoring, summaries and goalies.

OFFLINE RESEARCH MODULE. The Streamlit app does not import it. ``train_win_prob.py
--phase2-report`` uses it to test whether xG shares and projected-starter goalie ratings
improve the win-probability model. At v1.01.9 they did not: the mean log-loss change
over 2021-22..2025-26 was -0.0001 against a required +0.002. So the runtime model stays
on Elo, goal differential, shot shares and rest. The code is kept so the check can be
re-run each offseason and reused for goal-total markets.

Pure pandas/numpy apart from the shot model's coefficients, which the trainer fits.

Conventions that the tests pin:

- Only unblocked attempts (shots on goal, misses and goals) are modelled. A blocked shot
  is recorded where it was blocked, not where it was taken.
- A blocked-shot event is owned by the blocking team, so its attempting team is the
  other side.
- Shootouts and penalty shots are excluded. Empty-net attempts are flagged and left
  out of team xG and goalie workloads.
- The starting goalie is the first goalie in net for a shot the team faced. That
  avoids one boxscore request per game.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nhl.team_ratings import canonical_team_abbrev

UNBLOCKED_TYPES = frozenset({"shot-on-goal", "missed-shot", "goal"})
ATTEMPT_TYPES = UNBLOCKED_TYPES | {"blocked-shot"}
NET_X = 89.0
REBOUND_SECONDS = 3.0
RUSH_SECONDS = 4.0

_SHOT_TYPE_FEATURES = {
    "snap": "type_snap",
    "slap": "type_slap",
    "backhand": "type_backhand",
    "tip-in": "type_tip_in",
    "deflected": "type_deflected",
    "wrap-around": "type_wrap_around",
}
"""Shot types with their own indicator. Wrist shots are the baseline; anything else is ``type_other``."""

XG_FEATURES: tuple[str, ...] = (
    "distance",
    "log_distance",
    "distance_sq",
    "angle",
    "angle_sq",
    "distance_x_angle",
    "behind_net",
    "type_snap",
    "type_slap",
    "type_backhand",
    "type_tip_in",
    "type_deflected",
    "type_wrap_around",
    "type_other",
    "rebound",
    "rush",
    "power_play",
    "short_handed",
    "empty_net",
    "log_seconds_since_previous",
)
"""Design-matrix columns of the xG model, in artifact order."""

SHOT_COLUMNS = [
    "GameId",
    "SeasonYear",
    "GameTypeId",
    "GameDate",
    "Period",
    "ShootingTeam",
    "DefendingTeam",
    "IsHomeShot",
    "IsGoal",
    "ShotType",
    "X",
    "Y",
    "Distance",
    "Angle",
    "Rebound",
    "Rush",
    "SecondsSincePrevious",
    "ShooterSkaters",
    "DefenderSkaters",
    "EmptyNet",
    "FiveOnFive",
    "GoalieId",
]
"""One row per unblocked, non-shootout, non-penalty-shot attempt."""


def _clock_seconds(value: object) -> float:
    """Convert an ``MM:SS`` period clock into seconds, or NaN."""
    try:
        minutes, seconds = str(value).split(":")
        return float(minutes) * 60.0 + float(seconds)
    except (TypeError, ValueError):
        return float("nan")


def _parse_situation(code: object) -> tuple[int, int, int, int] | None:
    """Split a situation code into (away goalie, away skaters, home skaters, home goalie)."""
    text = str(code or "").strip()
    if len(text) != 4 or not text.isdigit():
        return None
    return int(text[0]), int(text[1]), int(text[2]), int(text[3])


def _attempting_team_id(kind: str, owner_id: int | None, home_id: int, away_id: int) -> int | None:
    """Return the team that took a shot attempt (blocked shots are owned by the blocker)."""
    if owner_id not in (home_id, away_id):
        return None
    if kind == "blocked-shot":
        return away_id if owner_id == home_id else home_id
    return owner_id


def _zone_for_team(zone: object, owner_id: int | None, team_id: int) -> str:
    """Express an event's zone code from ``team_id``'s point of view."""
    code = str(zone or "").upper()
    if code not in ("O", "D", "N") or owner_id is None:
        return ""
    if owner_id == team_id or code == "N":
        return code
    return "D" if code == "O" else "O"


def _attack_signs(plays: list[dict], home_id: int, away_id: int) -> dict[tuple[int, int], float]:
    """Infer each team's attacking direction per period from its offensive-zone shots.

    Only needed for seasons without ``homeTeamDefendingSide`` (2017-18 and earlier).

    Returns:
        ``(period, team_id) -> +1.0`` when the team attacks the net at x = +89, else ``-1.0``.
    """
    votes: dict[tuple[int, int], list[float]] = {}
    for play in plays:
        if play.get("typeDescKey") not in UNBLOCKED_TYPES:
            continue
        details = play.get("details") or {}
        owner = details.get("eventOwnerTeamId")
        x_coord = details.get("xCoord")
        if owner not in (home_id, away_id) or x_coord is None or str(details.get("zoneCode") or "").upper() != "O":
            continue
        period = int((play.get("periodDescriptor") or {}).get("number", 0) or 0)
        votes.setdefault((period, owner), []).append(1.0 if float(x_coord) >= 0 else -1.0)
    return {key: (1.0 if np.median(values) >= 0 else -1.0) for key, values in votes.items()}


def parse_play_by_play(payload: dict) -> dict:
    """Extract shot rows, starting goalies and goalie names from one play-by-play payload.

    Args:
        payload: ``/v1/gamecenter/{id}/play-by-play`` JSON.

    Returns:
        ``shots`` (``SHOT_COLUMNS`` frame) and ``meta`` (game identity, franchise-canonical
        teams, starter ids per side, goalie id -> name).
    """
    game_id = int(payload.get("id", 0) or 0)
    home = payload.get("homeTeam") or {}
    away = payload.get("awayTeam") or {}
    home_id = int(home.get("id", 0) or 0)
    away_id = int(away.get("id", 0) or 0)
    home_abbr = canonical_team_abbrev(home.get("abbrev"))
    away_abbr = canonical_team_abbrev(away.get("abbrev"))
    season_raw = str(payload.get("season") or "")
    season_year = int(season_raw[:4]) if season_raw[:4].isdigit() else game_id // 1_000_000
    game_type = int(payload.get("gameType", 0) or 0) or (game_id // 10_000) % 100
    game_date = str(payload.get("gameDate") or "")[:10]

    goalie_names = {}
    for spot in payload.get("rosterSpots", []) or []:
        if str(spot.get("positionCode") or "").upper() != "G":
            continue
        first = str((spot.get("firstName") or {}).get("default", "") or "").strip()
        last = str((spot.get("lastName") or {}).get("default", "") or "").strip()
        goalie_names[int(spot.get("playerId", 0) or 0)] = f"{first} {last}".strip()

    plays = sorted((play for play in payload.get("plays", []) or [] if isinstance(play, dict)), key=lambda play: play.get("sortOrder", 0) or 0)
    inferred_signs = _attack_signs(plays, home_id, away_id)
    team_abbr = {home_id: home_abbr, away_id: away_abbr}

    rows: list[dict] = []
    starters: dict[str, int] = {}
    previous: dict | None = None
    for play in plays:
        kind = str(play.get("typeDescKey") or "")
        descriptor = play.get("periodDescriptor") or {}
        period = int(descriptor.get("number", 0) or 0)
        period_type = str(descriptor.get("periodType") or "").upper()
        seconds = _clock_seconds(play.get("timeInPeriod"))
        details = play.get("details") or {}
        owner = details.get("eventOwnerTeamId")
        owner = int(owner) if isinstance(owner, (int, float)) else None

        if kind in ATTEMPT_TYPES and period_type != "SO":
            shooter_id = _attempting_team_id(kind, owner, home_id, away_id)
            defender_id = away_id if shooter_id == home_id else home_id
            goalie_in_net = details.get("goalieInNetId")
            if shooter_id is not None and goalie_in_net and team_abbr[defender_id] not in starters:
                starters[team_abbr[defender_id]] = int(goalie_in_net)

        if kind in UNBLOCKED_TYPES and period_type != "SO" and owner in (home_id, away_id):
            situation = _parse_situation(play.get("situationCode")) or (1, 5, 5, 1)
            away_goalie, away_skaters, home_skaters, home_goalie = situation
            is_home = owner == home_id
            shooter_skaters = home_skaters if is_home else away_skaters
            defender_skaters = away_skaters if is_home else home_skaters
            defending_goalie = away_goalie if is_home else home_goalie
            x_coord, y_coord = details.get("xCoord"), details.get("yCoord")
            penalty_shot = shooter_skaters <= 1 or defender_skaters == 0
            if x_coord is not None and y_coord is not None and not penalty_shot:
                side = str(play.get("homeTeamDefendingSide") or "").lower()
                if side in ("left", "right"):
                    home_sign = 1.0 if side == "left" else -1.0
                    sign = home_sign if is_home else -home_sign
                else:
                    sign = inferred_signs.get((period, owner), 1.0 if float(x_coord) >= 0 else -1.0)
                x_norm = float(x_coord) * sign
                y_norm = float(y_coord) * sign
                depth = NET_X - x_norm
                distance = math.hypot(depth, y_norm)
                angle = math.atan2(abs(y_norm), depth)

                same_period = previous is not None and previous["period"] == period
                since_previous = seconds - previous["seconds"] if same_period and not math.isnan(seconds) else 99.0
                since_previous = min(max(since_previous, 0.0), 99.0) if not math.isnan(since_previous) else 99.0
                rebound = bool(
                    same_period and previous["attempt_team"] == owner and since_previous <= REBOUND_SECONDS
                )
                previous_zone = _zone_for_team(previous["zone"], previous["owner"], owner) if same_period else ""
                rush = bool(same_period and previous_zone in ("N", "D") and since_previous <= RUSH_SECONDS)

                rows.append(
                    {
                        "GameId": game_id,
                        "SeasonYear": season_year,
                        "GameTypeId": game_type,
                        "GameDate": game_date,
                        "Period": period,
                        "ShootingTeam": team_abbr[owner],
                        "DefendingTeam": team_abbr[away_id if is_home else home_id],
                        "IsHomeShot": bool(is_home),
                        "IsGoal": kind == "goal",
                        "ShotType": str(details.get("shotType") or "").lower(),
                        "X": x_norm,
                        "Y": y_norm,
                        "Distance": distance,
                        "Angle": angle,
                        "Rebound": rebound,
                        "Rush": rush,
                        "SecondsSincePrevious": since_previous,
                        "ShooterSkaters": shooter_skaters,
                        "DefenderSkaters": defender_skaters,
                        "EmptyNet": defending_goalie == 0,
                        "FiveOnFive": shooter_skaters == 5 and defender_skaters == 5 and home_goalie == 1 and away_goalie == 1,
                        "GoalieId": int(details["goalieInNetId"]) if details.get("goalieInNetId") else np.nan,
                    }
                )

        if not math.isnan(seconds):
            previous = {
                "period": period,
                "seconds": seconds,
                "owner": owner,
                "zone": details.get("zoneCode"),
                "attempt_team": _attempting_team_id(kind, owner, home_id, away_id) if kind in ATTEMPT_TYPES else None,
            }

    shots = pd.DataFrame(rows, columns=SHOT_COLUMNS)
    meta = {
        "game_id": game_id,
        "season_year": season_year,
        "game_type_id": game_type,
        "game_date": game_date,
        "home_team": home_abbr,
        "away_team": away_abbr,
        "home_starter": starters.get(home_abbr),
        "away_starter": starters.get(away_abbr),
        "goalie_names": goalie_names,
    }
    return {"shots": shots, "meta": meta}


def xg_design_matrix(shots: pd.DataFrame) -> np.ndarray:
    """Build the xG model's design matrix in ``XG_FEATURES`` order.

    Args:
        shots: ``SHOT_COLUMNS`` frame.

    Returns:
        Float matrix with one row per shot.
    """
    distance = shots["Distance"].to_numpy(dtype=float)
    angle = shots["Angle"].to_numpy(dtype=float)
    shot_type = shots["ShotType"].fillna("").astype(str).to_numpy()
    # A pulled goalie's sixth skater is an extra attacker, which the NHL scores as even
    # strength, so skaters are capped at five before comparing.
    shooter_skaters = np.minimum(shots["ShooterSkaters"].to_numpy(dtype=float), 5.0)
    defender_skaters = np.minimum(shots["DefenderSkaters"].to_numpy(dtype=float), 5.0)
    columns = {
        "distance": distance,
        "log_distance": np.log1p(distance),
        "distance_sq": distance ** 2 / 1000.0,
        "angle": angle,
        "angle_sq": angle ** 2,
        "distance_x_angle": distance * angle / 10.0,
        "behind_net": (shots["X"].to_numpy(dtype=float) > NET_X).astype(float),
        "rebound": shots["Rebound"].to_numpy(dtype=float),
        "rush": shots["Rush"].to_numpy(dtype=float),
        "power_play": (shooter_skaters > defender_skaters).astype(float),
        "short_handed": (shooter_skaters < defender_skaters).astype(float),
        "empty_net": shots["EmptyNet"].to_numpy(dtype=float),
        "log_seconds_since_previous": np.log1p(shots["SecondsSincePrevious"].to_numpy(dtype=float)),
    }
    known_types = set(_SHOT_TYPE_FEATURES)
    for raw_type, feature in _SHOT_TYPE_FEATURES.items():
        columns[feature] = (shot_type == raw_type).astype(float)
    columns["type_other"] = np.array([value not in known_types and value != "wrist" for value in shot_type], dtype=float)
    return np.column_stack([columns[name] for name in XG_FEATURES]) if len(shots) else np.zeros((0, len(XG_FEATURES)))


def validate_xg_model(payload: object) -> dict | None:
    """Return a normalized xG model block, or ``None`` when it is absent or malformed."""
    if not isinstance(payload, dict):
        return None
    features = [str(name) for name in payload.get("feature_order", [])]
    if features != list(XG_FEATURES):
        return None
    try:
        coefficients = [float(value) for value in payload["coefficients"]]
        scaler_mean = [float(value) for value in payload["scaler_mean"]]
        scaler_scale = [float(value) if float(value) != 0 else 1.0 for value in payload["scaler_scale"]]
        intercept = float(payload.get("intercept", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not (len(coefficients) == len(scaler_mean) == len(scaler_scale) == len(features)):
        return None
    return {
        "feature_order": features,
        "coefficients": coefficients,
        "intercept": intercept,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "version": str(payload.get("version", "") or ""),
    }


def score_expected_goals(shots: pd.DataFrame, xg_model: dict) -> np.ndarray:
    """Return the goal probability of every shot.

    Args:
        shots: ``SHOT_COLUMNS`` frame.
        xg_model: Output of ``validate_xg_model``.

    Returns:
        Probabilities in shot order.
    """
    if shots is None or shots.empty:
        return np.zeros(0)
    matrix = xg_design_matrix(shots)
    standardized = (matrix - np.array(xg_model["scaler_mean"])) / np.array(xg_model["scaler_scale"])
    logits = xg_model["intercept"] + standardized @ np.array(xg_model["coefficients"])
    return 1.0 / (1.0 + np.exp(-logits))


XG_GAME_COLUMNS = [
    "GameId", "SeasonYear", "GameTypeId", "GameDate", "HomeTeam", "AwayTeam",
    "HomeXgf5v5", "AwayXgf5v5", "HomeXgfAll", "AwayXgfAll", "HomeStarter", "AwayStarter",
]
"""One row per game: 5v5 xG and all-situations xG (empty nets excluded) for each side."""

GOALIE_GAME_COLUMNS = [
    "GameId", "SeasonYear", "GameTypeId", "GameDate", "Team", "GoalieId", "GoalieName",
    "Shots", "XgAgainst", "GoalsAgainst", "Started",
]
"""One row per goalie appearance: unblocked non-empty-net shots faced, xG against, goals against."""


def summarize_games(shots: pd.DataFrame, expected_goals: np.ndarray, metas: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate scored shots into per-game team xG and per-goalie workloads.

    The trainer (thousands of games) and the runtime (one game at a time) both use this
    function, so the numbers they produce cannot drift apart.

    Args:
        shots: ``SHOT_COLUMNS`` rows for any number of games.
        expected_goals: xG per shot, aligned with ``shots``.
        metas: ``parse_play_by_play`` metas for the same games.

    Returns:
        ``(games, goalie_games)`` with ``XG_GAME_COLUMNS`` and ``GOALIE_GAME_COLUMNS``.
    """
    meta_frame = pd.DataFrame(
        [
            {
                "GameId": int(meta["game_id"]),
                "SeasonYear": int(meta["season_year"]),
                "GameTypeId": int(meta["game_type_id"]),
                "GameDate": meta["game_date"],
                "HomeTeam": meta["home_team"],
                "AwayTeam": meta["away_team"],
                "HomeStarter": meta.get("home_starter"),
                "AwayStarter": meta.get("away_starter"),
            }
            for meta in metas or []
            if not meta.get("missing")
        ],
        columns=["GameId", "SeasonYear", "GameTypeId", "GameDate", "HomeTeam", "AwayTeam", "HomeStarter", "AwayStarter"],
    )
    if meta_frame.empty:
        return pd.DataFrame(columns=XG_GAME_COLUMNS), pd.DataFrame(columns=GOALIE_GAME_COLUMNS)

    frame = shots.copy() if shots is not None else pd.DataFrame(columns=SHOT_COLUMNS)
    frame["xG"] = np.asarray(expected_goals, dtype=float) if len(frame) else np.zeros(0)
    frame["EmptyNet"] = frame["EmptyNet"].astype(bool)
    frame["FiveOnFive"] = frame["FiveOnFive"].astype(bool)
    frame = frame[frame["GameId"].isin(meta_frame["GameId"])]

    def _team_totals(subset: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """Pivot xG taken per game into home and away columns."""
        totals = subset.groupby(["GameId", "IsHomeShot"])["xG"].sum().unstack("IsHomeShot")
        return pd.DataFrame(
            {
                "GameId": totals.index,
                f"HomeXgf{suffix}": totals[True].to_numpy() if True in totals.columns else 0.0,
                f"AwayXgf{suffix}": totals[False].to_numpy() if False in totals.columns else 0.0,
            }
        )

    games = meta_frame.merge(_team_totals(frame[frame["FiveOnFive"]], "5v5"), on="GameId", how="left")
    games = games.merge(_team_totals(frame[~frame["EmptyNet"]], "All"), on="GameId", how="left")
    for column in ("HomeXgf5v5", "AwayXgf5v5", "HomeXgfAll", "AwayXgfAll"):
        games[column] = pd.to_numeric(games[column], errors="coerce").fillna(0.0)
    for column in ("HomeStarter", "AwayStarter"):
        games[column] = pd.to_numeric(games[column], errors="coerce").astype("Int64")

    faced = frame[~frame["EmptyNet"]].dropna(subset=["GoalieId"])
    if faced.empty:
        return games[XG_GAME_COLUMNS], pd.DataFrame(columns=GOALIE_GAME_COLUMNS)
    goalie_games = (
        faced.groupby(["GameId", "DefendingTeam", "GoalieId"])
        .agg(Shots=("xG", "size"), XgAgainst=("xG", "sum"), GoalsAgainst=("IsGoal", "sum"))
        .reset_index()
        .rename(columns={"DefendingTeam": "Team"})
    )
    goalie_games["GoalieId"] = goalie_games["GoalieId"].astype("int64")
    goalie_games["GoalsAgainst"] = goalie_games["GoalsAgainst"].astype(int)
    goalie_games = goalie_games.merge(meta_frame, on="GameId", how="left")
    home_side = goalie_games["Team"] == goalie_games["HomeTeam"]
    starter = np.where(home_side, goalie_games["HomeStarter"].astype("float"), goalie_games["AwayStarter"].astype("float"))
    goalie_games["Started"] = goalie_games["GoalieId"].astype(float).to_numpy() == starter
    names = {
        (int(meta["game_id"]), int(goalie_id)): name
        for meta in metas or []
        for goalie_id, name in (meta.get("goalie_names") or {}).items()
    }
    goalie_games["GoalieName"] = [names.get((int(game_id), int(goalie_id)), "") for game_id, goalie_id in zip(goalie_games["GameId"], goalie_games["GoalieId"])]
    return games[XG_GAME_COLUMNS], goalie_games[GOALIE_GAME_COLUMNS]


# ---------------------------------------------------------------------------
# Goalies: shrunk goals saved above expected and projected starters
# ---------------------------------------------------------------------------

DEFAULT_GOALIE_PARAMS: dict[str, float] = {
    "goalie_prior_shots": 1500.0,
    "goalie_season_decay": 0.5,
    "starter_window": 10.0,
}
"""Shots of league-average play every rating starts with, per-season decay of old workload, starts window."""

GSAX_SHOT_SCALE = 30.0
"""Ratings are expressed as goals saved above expected per 30 unblocked shots (about one game)."""


class GoalieHistory:
    """Point-in-time goalie ratings and team start histories built from goalie-game rows.

    Every query takes a date and only uses games before that date, so the trainer can ask
    for pregame values without leaking the game's own result.
    """

    def __init__(self, goalie_games: pd.DataFrame, params: dict | None = None):
        """Index goalie workloads and team starts for fast point-in-time lookups.

        Args:
            goalie_games: ``GOALIE_GAME_COLUMNS`` rows.
            params: Overrides for ``DEFAULT_GOALIE_PARAMS``.
        """
        self.params = dict(DEFAULT_GOALIE_PARAMS)
        self.params.update({key: float(value) for key, value in (params or {}).items() if key in DEFAULT_GOALIE_PARAMS})
        self.names: dict[int, str] = {}
        self._goalie_dates: dict[int, list[str]] = {}
        self._goalie_states: dict[int, list[tuple[int, float, float]]] = {}
        self._team_starts: dict[str, list[tuple[str, int, int]]] = {}
        self._team_start_dates: dict[str, list[str]] = {}

        if goalie_games is None or goalie_games.empty:
            return
        ordered = goalie_games.sort_values(["GameDate", "GameId"], kind="stable")
        decay = self.params["goalie_season_decay"]
        running: dict[int, list[float]] = {}
        for row in ordered.itertuples(index=False):
            goalie_id = int(row.GoalieId)
            season = int(row.SeasonYear)
            if str(row.GoalieName or ""):
                self.names[goalie_id] = str(row.GoalieName)
            state = running.setdefault(goalie_id, [season, 0.0, 0.0])
            if season > state[0]:
                factor = decay ** (season - state[0])
                state[1] *= factor
                state[2] *= factor
                state[0] = season
            state[1] += float(row.XgAgainst) - float(row.GoalsAgainst)
            state[2] += float(row.Shots)
            self._goalie_dates.setdefault(goalie_id, []).append(str(row.GameDate))
            self._goalie_states.setdefault(goalie_id, []).append((state[0], state[1], state[2]))
            if bool(row.Started):
                self._team_starts.setdefault(str(row.Team), []).append((str(row.GameDate), int(row.GameId), goalie_id))
        self._team_start_dates = {team: [start[0] for start in starts] for team, starts in self._team_starts.items()}

    def rating(self, goalie_id: int | None, as_of_date: str, season_year: int) -> float:
        """Return a goalie's shrunk GSAx per 30 unblocked shots before ``as_of_date``.

        Args:
            goalie_id: NHL player id, or ``None``.
            as_of_date: ``YYYY-MM-DD``; games on or after it are ignored.
            season_year: Season being predicted, used to decay older workload.

        Returns:
            Rating in goals per 30 shots. Zero means league average, or no data.
        """
        if goalie_id is None or goalie_id not in self._goalie_dates:
            return 0.0
        dates = self._goalie_dates[goalie_id]
        position = _bisect_left(dates, str(as_of_date)) - 1
        if position < 0:
            return 0.0
        state_season, gsax, shots = self._goalie_states[goalie_id][position]
        factor = self.params["goalie_season_decay"] ** max(int(season_year) - int(state_season), 0)
        return GSAX_SHOT_SCALE * (gsax * factor) / (shots * factor + self.params["goalie_prior_shots"])

    def recent_starts(self, team: str, as_of_date: str) -> list[tuple[str, int, int]]:
        """Return the team's last ``starter_window`` starts before ``as_of_date``, oldest first."""
        starts = self._team_starts.get(team, [])
        position = _bisect_left(self._team_start_dates.get(team, []), str(as_of_date))
        window = int(self.params["starter_window"])
        return starts[max(position - window, 0):position]

    def projected_starter(self, team: str, as_of_date: str, back_to_back: bool = False) -> int | None:
        """Project a game's starter from recent starts, before any official confirmation.

        Args:
            team: Franchise abbreviation.
            as_of_date: Game date; only earlier starts are used.
            back_to_back: Whether the team plays the day before.

        Returns:
            Goalie id, or ``None`` with no start history.
        """
        return choose_starter(self.recent_starts(team, as_of_date), as_of_date, back_to_back)


def choose_starter(starts: list[tuple[str, int, int]], game_date: str, back_to_back: bool = False) -> int | None:
    """Pick a projected starter from a team's recent starts.

    The goalie with the most starts in the window gets the net; ties go to the more
    recent start. On the second night of a back-to-back the first-choice goalie rests:
    if last night's start is already on record and he made it, the next most used
    goalie starts. If last night's game has not been played yet (a card published a
    day ahead), the first choice is assumed to play it and rest tonight.

    Args:
        starts: ``(date, game_id, goalie_id)`` starts, oldest first, all before the game.
        game_date: ``YYYY-MM-DD`` of the game being projected.
        back_to_back: Whether the team plays the day before.

    Returns:
        Goalie id, or ``None`` with no starts.
    """
    if not starts:
        return None
    counts: dict[int, list] = {}
    for order, (_, _, goalie_id) in enumerate(starts):
        entry = counts.setdefault(goalie_id, [0, -1])
        entry[0] += 1
        entry[1] = order
    ranked = sorted(counts, key=lambda goalie_id: (counts[goalie_id][0], counts[goalie_id][1]), reverse=True)
    choice = ranked[0]
    if not back_to_back or len(ranked) < 2:
        return choice
    previous_day = (pd.Timestamp(game_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    last_date, _, last_goalie = starts[-1]
    if last_date == previous_day:
        return ranked[1] if last_goalie == choice else choice
    return ranked[1]


def _bisect_left(values: list[str], target: str) -> int:
    """Return the first index whose value is not less than ``target`` (sorted list)."""
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low


RESEARCH_FEATURES: tuple[str, ...] = ("xg_share_shrunk_diff", "xg_all_share_shrunk_diff", "goalie_gsax_diff")
"""Play-by-play features evaluated offline. The runtime model never uses them (Phase 2 gate)."""


def research_game_features(
    games: pd.DataFrame,
    xg_games: pd.DataFrame,
    goalie_features: pd.DataFrame | None = None,
    prior_weight: float = 8.0,
    share_prior_keep: float = 0.6,
) -> pd.DataFrame:
    """Build pregame xG-share and projected-goalie features for offline evaluation.

    Uses the same shrinkage rule as ``team_ratings.compute_team_form``: a season-to-date
    average blended with a prior worth ``prior_weight`` games, taken from last season
    and pulled toward 0.5. Playoff games use each team's end-of-regular-season value.
    Games without play-by-play are skipped rather than counted as 0.5.

    Args:
        games: Game table with ``GameId``, ``SeasonYear``, ``GameTypeId``, ``GameDate``,
            ``HomeTeam`` and ``AwayTeam``.
        xg_games: ``XG_GAME_COLUMNS`` rows.
        goalie_features: ``build_goalie_features`` output, or ``None`` for zeros.
        prior_weight: Prior strength in games.
        share_prior_keep: Share of last season's distance from 0.5 kept in the prior.

    Returns:
        ``GameId`` plus every feature in ``RESEARCH_FEATURES``.
    """
    base = games[["GameId", "SeasonYear", "GameTypeId", "GameDate", "HomeTeam", "AwayTeam"]].merge(
        xg_games[["GameId", "HomeXgf5v5", "AwayXgf5v5", "HomeXgfAll", "AwayXgfAll"]], on="GameId", how="left"
    )
    regular = base[base["GameTypeId"].eq(2)]

    def _share(own: pd.Series, other: pd.Series) -> pd.Series:
        """Own share of the two totals, NaN when both are zero or missing."""
        total = own + other
        return (own / total).where(total > 0)

    rows = pd.concat(
        [
            pd.DataFrame({"GameId": regular["GameId"], "SeasonYear": regular["SeasonYear"], "GameDate": regular["GameDate"], "Team": regular["HomeTeam"],
                          "xg_share": _share(regular["HomeXgf5v5"], regular["AwayXgf5v5"]), "xg_all_share": _share(regular["HomeXgfAll"], regular["AwayXgfAll"])}),
            pd.DataFrame({"GameId": regular["GameId"], "SeasonYear": regular["SeasonYear"], "GameDate": regular["GameDate"], "Team": regular["AwayTeam"],
                          "xg_share": _share(regular["AwayXgf5v5"], regular["HomeXgf5v5"]), "xg_all_share": _share(regular["AwayXgfAll"], regular["HomeXgfAll"])}),
        ],
        ignore_index=True,
    ).sort_values(["Team", "SeasonYear", "GameDate", "GameId"], kind="stable").reset_index(drop=True)

    metrics = ("xg_share", "xg_all_share")
    priors = rows.groupby(["Team", "SeasonYear"])[list(metrics)].mean().reset_index()
    priors["SeasonYear"] = priors["SeasonYear"] + 1
    for metric in metrics:
        priors[f"{metric}_prior"] = 0.5 + (priors[metric] - 0.5) * share_prior_keep
    rows = rows.merge(priors[["Team", "SeasonYear"] + [f"{metric}_prior" for metric in metrics]], on=["Team", "SeasonYear"], how="left")

    keys = [rows["Team"], rows["SeasonYear"]]
    season_end = rows[["Team", "SeasonYear"]].drop_duplicates().copy()
    for metric in metrics:
        prior = rows[f"{metric}_prior"].fillna(0.5)
        values = rows[metric].fillna(0.0)
        valid = rows[metric].notna().astype(float)
        sums = values.groupby(keys).cumsum()
        counts = valid.groupby(keys).cumsum()
        rows[f"{metric}_pre"] = (prior_weight * prior + sums - values) / (prior_weight + counts - valid)
        totals = pd.DataFrame({"Team": rows["Team"], "SeasonYear": rows["SeasonYear"], "sum": sums, "count": counts, "prior": prior}).groupby(["Team", "SeasonYear"]).last().reset_index()
        totals[f"{metric}_end"] = (prior_weight * totals["prior"] + totals["sum"]) / (prior_weight + totals["count"])
        season_end = season_end.merge(totals[["Team", "SeasonYear", f"{metric}_end"]], on=["Team", "SeasonYear"], how="left")

    features = base[["GameId", "SeasonYear", "HomeTeam", "AwayTeam"]].copy()
    for side, team_column in (("home", "HomeTeam"), ("away", "AwayTeam")):
        pregame = rows[["GameId", "Team"] + [f"{metric}_pre" for metric in metrics]].rename(
            columns={"Team": team_column, **{f"{metric}_pre": f"{side}_{metric}" for metric in metrics}}
        )
        ending = season_end.rename(columns={"Team": team_column, **{f"{metric}_end": f"{side}_{metric}_end" for metric in metrics}})
        features = features.merge(pregame, on=["GameId", team_column], how="left").merge(ending, on=[team_column, "SeasonYear"], how="left")
        for metric in metrics:
            features[f"{side}_{metric}"] = features[f"{side}_{metric}"].fillna(features[f"{side}_{metric}_end"]).fillna(0.5)

    features["xg_share_shrunk_diff"] = features["home_xg_share"] - features["away_xg_share"]
    features["xg_all_share_shrunk_diff"] = features["home_xg_all_share"] - features["away_xg_all_share"]
    if goalie_features is not None and not goalie_features.empty:
        features = features.merge(goalie_features[["GameId", "HomeGoalieGsax", "AwayGoalieGsax"]], on="GameId", how="left")
        features["goalie_gsax_diff"] = features["HomeGoalieGsax"].fillna(0.0) - features["AwayGoalieGsax"].fillna(0.0)
    else:
        features["goalie_gsax_diff"] = 0.0
    return features[["GameId"] + list(RESEARCH_FEATURES)]


def build_goalie_features(games: pd.DataFrame, history: GoalieHistory) -> pd.DataFrame:
    """Return the projected starters' pregame ratings for every game.

    Args:
        games: Game table rows with ``GameId``, ``GameDate``, ``SeasonYear``, ``HomeTeam``,
            ``AwayTeam`` and, when available, ``HomeBackToBack`` / ``AwayBackToBack``.
        history: ``GoalieHistory`` covering games before these dates.

    Returns:
        ``GameId``, ``HomeProjectedStarter``, ``AwayProjectedStarter``, ``HomeGoalieGsax``
        and ``AwayGoalieGsax``.
    """
    rows = []
    home_b2b = games["HomeBackToBack"] if "HomeBackToBack" in games.columns else pd.Series(0, index=games.index)
    away_b2b = games["AwayBackToBack"] if "AwayBackToBack" in games.columns else pd.Series(0, index=games.index)
    for row, home_flag, away_flag in zip(games.itertuples(index=False), home_b2b, away_b2b):
        date = str(row.GameDate)
        season = int(row.SeasonYear)
        home_starter = history.projected_starter(row.HomeTeam, date, bool(home_flag))
        away_starter = history.projected_starter(row.AwayTeam, date, bool(away_flag))
        rows.append(
            {
                "GameId": int(row.GameId),
                "HomeProjectedStarter": home_starter,
                "AwayProjectedStarter": away_starter,
                "HomeGoalieGsax": history.rating(home_starter, date, season),
                "AwayGoalieGsax": history.rating(away_starter, date, season),
            }
        )
    return pd.DataFrame(rows, columns=["GameId", "HomeProjectedStarter", "AwayProjectedStarter", "HomeGoalieGsax", "AwayGoalieGsax"])
