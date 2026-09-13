import itertools
import unittest

import pandas as pd

from nhl import season_sim
from nhl.win_prob import validate_model_artifact

DIVISIONS = {
    ("Eastern", "Atlantic"): ["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"],
    ("Eastern", "Metropolitan"): ["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"],
    ("Western", "Central"): ["CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG"],
    ("Western", "Pacific"): ["ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK"],
}


def _artifact(sd_preseason=0.0, sd_late=0.0, coefficient=0.01, intercept=0.0):
    """Return a one-feature artifact where the logit is ``intercept + coefficient * elo_diff``."""
    return validate_model_artifact(
        {
            "model_version": 2,
            "feature_order": ["elo_diff"],
            "coefficients": [coefficient],
            "intercept": intercept,
            "scaler_mean": [0.0],
            "scaler_scale": [1.0],
            "simulation": {"strength_sd_preseason": sd_preseason, "strength_sd_late": sd_late},
        }
    )


def _teams(points_by_team=None):
    """Return simulator team dicts for a 32-team league."""
    points_by_team = points_by_team or {}
    return [
        {
            "team_abbr": team,
            "conference": conference,
            "division": division,
            "points": float(points_by_team.get(team, 0.0)),
            "regulation_wins": 0.0,
            "regulation_plus_ot_wins": 0.0,
            "wins": 0.0,
        }
        for (conference, division), members in DIVISIONS.items()
        for team in members
    ]


def _inputs(elo_by_team=None):
    """Return team rating inputs, 1505 unless overridden."""
    elo_by_team = elo_by_team or {}
    return {team: {"elo": float(elo_by_team.get(team, 1505.0))} for members in DIVISIONS.values() for team in members}


def _final_points():
    """Final standings with an unambiguous order inside every division and wild-card race."""
    points = {}
    bases = {"Atlantic": 110, "Metropolitan": 108, "Central": 112, "Pacific": 104}
    for (_, division), members in DIVISIONS.items():
        for position, team in enumerate(members):
            points[team] = bases[division] - 6 * position
    return points


class SeriesProbabilityTests(unittest.TestCase):
    """Cover the exact best-of-7 calculation."""

    def test_even_teams_split_the_series(self):
        """Evenly matched teams are a coin flip whatever the venue pattern."""
        self.assertAlmostEqual(float(season_sim.series_win_probability(0.5, 0.5)), 0.5)

    def test_existing_series_score_is_respected(self):
        """A 3-0 lead needs one win in four tries; 0-3 needs four straight."""
        self.assertAlmostEqual(float(season_sim.series_win_probability(0.5, 0.5, 3, 0)), 0.9375)
        self.assertAlmostEqual(float(season_sim.series_win_probability(0.5, 0.5, 0, 3)), 0.0625)
        self.assertEqual(float(season_sim.series_win_probability(0.2, 0.2, 4, 2)), 1.0)
        self.assertEqual(float(season_sim.series_win_probability(0.9, 0.9, 1, 4)), 0.0)

    def test_home_ice_lifts_the_higher_seed(self):
        """Winning more home games than road games is worth more than a coin flip."""
        self.assertGreater(float(season_sim.series_win_probability(0.6, 0.5)), 0.5)


class SimulateSeasonTests(unittest.TestCase):
    """Cover standings, seeding and bracket logic of the season simulator."""

    def test_final_standings_seed_top_three_per_division_plus_two_wild_cards(self):
        """With the season over, qualification and division titles are certain."""
        points = _final_points()
        result = season_sim.simulate_season(_teams(points), _inputs(), None, _artifact(), n_sims=400, seed=3)
        teams = result["teams"]

        qualified = {team for team, values in teams.items() if values["make_playoffs"] == 1.0}
        expected = set()
        for conference in ("Eastern", "Western"):
            conference_divisions = [members for (conf, _), members in DIVISIONS.items() if conf == conference]
            division_top = {team for members in conference_divisions for team in members[:3]}
            others = sorted(
                (team for members in conference_divisions for team in members[3:]),
                key=lambda team: -points[team],
            )
            expected |= division_top | set(others[:2])
        self.assertEqual(qualified, expected)
        self.assertEqual({team for team, values in teams.items() if values["make_playoffs"] == 0.0}, set(points) - expected)
        self.assertEqual(
            {team for team, values in teams.items() if values["win_division"] == 1.0},
            {"BOS", "CAR", "CHI", "ANA"},
        )

    def test_best_division_winner_draws_the_second_wild_card(self):
        """BOS (110) outranks CAR (108), so BOS meets the second wild card (NYI, 90)."""
        points = _final_points()
        east_others = sorted(
            [team for team in DIVISIONS[("Eastern", "Atlantic")][3:] + DIVISIONS[("Eastern", "Metropolitan")][3:]],
            key=lambda team: -points[team],
        )
        first_card, second_card = east_others[0], east_others[1]
        # Make the second wild card unbeatable so the first-round pairing is observable.
        result = season_sim.simulate_season(
            _teams(points), _inputs({second_card: 5000.0}), None, _artifact(), n_sims=400, seed=5,
        )
        teams = result["teams"]

        self.assertEqual(teams["BOS"]["win_round_1"], 0.0)
        self.assertEqual(teams[second_card]["win_cup"], 1.0)
        self.assertGreater(teams["CAR"]["win_round_1"], 0.3)
        self.assertGreater(teams[first_card]["win_round_1"], 0.3)

    def test_probabilities_are_conserved_and_the_strongest_team_leads(self):
        """One champion, sixteen playoff teams and four division winners per simulation."""
        teams = _teams()
        games = []
        conference_members = {
            conference: [team for (conf, _), members in DIVISIONS.items() if conf == conference for team in members]
            for conference in ("Eastern", "Western")
        }
        for members in conference_members.values():
            for home, away in itertools.combinations(members, 2):
                games.append({"HomeTeam": home, "AwayTeam": away, "HomeBackToBack": 0, "AwayBackToBack": 0})
        remaining = pd.DataFrame(games)
        inputs = _inputs({"COL": 1700.0})

        result = season_sim.simulate_season(teams, inputs, remaining, _artifact(sd_preseason=0.3), n_sims=3000, seed=11)
        values = result["teams"]

        self.assertAlmostEqual(sum(team["win_cup"] for team in values.values()), 1.0, places=9)
        self.assertAlmostEqual(sum(team["make_playoffs"] for team in values.values()), 16.0, places=9)
        self.assertAlmostEqual(sum(team["win_division"] for team in values.values()), 4.0, places=9)
        self.assertEqual(max(values, key=lambda team: values[team]["win_cup"]), "COL")
        self.assertGreater(values["COL"]["projected_points"], values["DAL"]["projected_points"])

    def test_same_seed_gives_identical_output(self):
        """Reruns with unchanged inputs must not jitter the board."""
        remaining = pd.DataFrame(
            [{"HomeTeam": "BOS", "AwayTeam": "TOR", "HomeBackToBack": 0, "AwayBackToBack": 1}] * 20
        )
        first = season_sim.simulate_season(_teams(), _inputs(), remaining, _artifact(sd_preseason=0.2), n_sims=500, seed=42)
        second = season_sim.simulate_season(_teams(), _inputs(), remaining, _artifact(sd_preseason=0.2), n_sims=500, seed=42)

        self.assertEqual(first, second)

    def test_live_bracket_keeps_decided_series_and_conditions_on_series_score(self):
        """An eliminated team has no Cup chance; a 3-0 lead converts about 94% of the time."""
        teams_by_division = {division: members for (_, division), members in DIVISIONS.items()}
        series = {}
        for letter, division in zip(("A", "C", "E", "G"), ("Atlantic", "Metropolitan", "Central", "Pacific")):
            members = teams_by_division[division]
            series[letter] = {"top": members[0], "bottom": members[3], "top_wins": 0, "bottom_wins": 0, "winner": "", "round": 1}
            series[chr(ord(letter) + 1)] = {"top": members[1], "bottom": members[2], "top_wins": 0, "bottom_wins": 0, "winner": "", "round": 1}
        series["A"].update({"top_wins": 4, "bottom_wins": 1, "winner": "BOS"})
        series["B"].update({"top_wins": 3, "bottom_wins": 0})
        bracket = {"series": series, "round_one_complete": True, "champion": ""}

        result = season_sim.simulate_season(_teams(_final_points()), _inputs(), None, _artifact(), bracket=bracket, n_sims=6000, seed=9)
        values = result["teams"]

        self.assertEqual(values["FLA"]["make_playoffs"], 1.0)
        self.assertEqual(values["FLA"]["win_round_1"], 0.0)
        self.assertEqual(values["FLA"]["win_cup"], 0.0)
        self.assertEqual(values["BOS"]["win_round_1"], 1.0)
        self.assertAlmostEqual(values["BUF"]["win_round_1"], 0.9375, delta=0.02)
        self.assertEqual(values["TOR"]["make_playoffs"], 0.0)
        self.assertAlmostEqual(sum(team["win_cup"] for team in values.values()), 1.0, places=9)

    def test_rejects_leagues_outside_the_wild_card_format(self):
        """A table without two conferences of two divisions cannot be seeded."""
        teams = [team for team in _teams() if team["conference"] == "Eastern"]
        with self.assertRaises(ValueError):
            season_sim.simulate_season(teams, _inputs(), None, _artifact(), n_sims=10)


class InputAdapterTests(unittest.TestCase):
    """Cover the payload adapters feeding the simulator."""

    def test_parse_playoff_bracket_finds_champion_and_round_one_state(self):
        """The Final's winning team id names the champion."""
        series = []
        for number, letter in enumerate("ABCDEFGH"):
            series.append({
                "seriesLetter": letter,
                "playoffRound": 1,
                "topSeedTeam": {"id": 100 + number, "abbrev": f"T{number}A"},
                "bottomSeedTeam": {"id": 200 + number, "abbrev": f"T{number}B"},
                "topSeedWins": 4,
                "bottomSeedWins": 2,
                "winningTeamId": 100 + number,
            })
        series.append({
            "seriesLetter": "O",
            "playoffRound": 4,
            "topSeedTeam": {"id": 12, "abbrev": "CAR"},
            "bottomSeedTeam": {"id": 54, "abbrev": "VGK"},
            "topSeedWins": 4,
            "bottomSeedWins": 2,
            "winningTeamId": 12,
        })
        parsed = season_sim.parse_playoff_bracket({"series": series})

        self.assertTrue(parsed["round_one_complete"])
        self.assertEqual(parsed["champion"], "CAR")
        self.assertEqual(parsed["series"]["A"]["winner"], "T0A")

    def test_empty_bracket_means_no_playoffs_yet(self):
        """Before the playoffs the endpoint returns an empty series list."""
        parsed = season_sim.parse_playoff_bracket({"series": []})

        self.assertFalse(parsed["round_one_complete"])
        self.assertEqual(parsed["champion"], "")

    def test_remaining_games_skip_completed_and_final_games_but_keep_back_to_backs(self):
        """A finished game still counts as the night before for the next one."""
        schedule = pd.DataFrame(
            {
                "SeasonYear": [2026] * 4,
                "GameTypeId": [2, 2, 2, 2],
                "GameId": [1, 2, 3, 4],
                "GameDate": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-05"],
                "GameStateId": [7, 1, 1, 1],
                "HomeTeam": ["TOR", "BOS", "TOR", "MTL"],
                "AwayTeam": ["MTL", "TOR", "BOS", "TOR"],
            }
        )
        remaining = season_sim.remaining_regular_season_games(schedule, completed_game_ids={2})

        self.assertEqual(list(remaining["GameId"]), [3, 4])
        self.assertEqual(int(remaining.set_index("GameId").loc[3, "HomeBackToBack"]), 1)
        self.assertEqual(int(remaining.set_index("GameId").loc[4, "AwayBackToBack"]), 0)

    def test_teams_from_standings_can_drop_last_seasons_record(self):
        """Before opening night only division membership carries over."""
        standings = pd.DataFrame(
            [{"teamAbbrev": "ARI", "conferenceName": "Western", "divisionName": "Central", "points": 90, "regulationWins": 30, "regulationPlusOtWins": 35, "wins": 40, "gamesPlayed": 82}]
        )
        with_record = season_sim.teams_from_standings(standings)
        without_record = season_sim.teams_from_standings(standings, include_record=False)

        self.assertEqual(with_record[0]["team_abbr"], "UTA")
        self.assertEqual(with_record[0]["points"], 90.0)
        self.assertEqual(without_record[0]["points"], 0.0)
        self.assertEqual(without_record[0]["division"], "Central")


if __name__ == "__main__":
    unittest.main()
