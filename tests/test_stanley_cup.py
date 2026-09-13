import unittest

import pandas as pd

from nhl.stanley_cup import (
    BOARD_MODE_CHAMPION,
    BOARD_MODE_ODDS,
    BOARD_MODE_PRESEASON,
    BOARD_MODE_STANDINGS,
    build_stanley_cup_board,
)


def _standings(season_id=20252026):
    """Return a four-team standings frame split across two divisions."""
    rows = [
        ("COL", "Colorado Avalanche", "Avalanche", "Western", "Central", 1, 1, 63, 43, 11, 9, 95),
        ("DAL", "Dallas Stars", "Stars", "Western", "Central", 2, 3, 64, 40, 14, 10, 90),
        ("EDM", "Edmonton Oilers", "Oilers", "Western", "Pacific", 1, 9, 65, 32, 25, 8, 72),
        ("CGY", "Calgary Flames", "Flames", "Western", "Pacific", 2, 27, 64, 25, 32, 7, 57),
    ]
    return pd.DataFrame(
        [
            {
                "teamAbbrev": abbr,
                "teamName": name,
                "teamCommonName": common,
                "teamLogo": f"https://assets.nhle.com/logos/nhl/svg/{abbr}_light.svg",
                "conferenceName": conference,
                "divisionName": division,
                "divisionSequence": division_sequence,
                "leagueSequence": league_sequence,
                "gamesPlayed": games,
                "wins": wins,
                "losses": losses,
                "otLosses": ot_losses,
                "points": points,
                "standingsDateTimeUtc": "2026-03-12T20:28:00Z",
                "seasonId": season_id,
            }
            for abbr, name, common, conference, division, division_sequence, league_sequence, games, wins, losses, ot_losses, points in rows
        ]
    )


def _projection(phase="regular_season"):
    """Return a season projection where Dallas, not points leader Colorado, is favoured."""
    odds = {
        "COL": (0.21, 0.97, 112.0),
        "DAL": (0.26, 0.95, 108.0),
        "EDM": (0.06, 0.61, 91.0),
        "CGY": (0.004, 0.08, 75.0),
    }
    return {
        "state": "projection",
        "phase": phase,
        "season_year": 2025,
        "n_sims": 10000,
        "teams": {
            abbr: {
                "win_cup": cup,
                "make_playoffs": playoffs,
                "projected_points": projected,
                "points_p10": projected - 8,
                "points_p90": projected + 8,
                "win_division": 0.4,
                "win_conference": cup * 2,
            }
            for abbr, (cup, playoffs, projected) in odds.items()
        },
    }


class StanleyCupBoardTests(unittest.TestCase):
    """Cover board assembly from standings plus the season projection."""

    def test_favorite_is_the_highest_simulated_cup_probability(self):
        """Rank by simulated Cup odds, not by points."""
        board = build_stanley_cup_board(_standings(), _projection())

        self.assertEqual(board["mode"], BOARD_MODE_ODDS)
        self.assertEqual(board["favorite_team_abbr"], "DAL")
        self.assertTrue(board["favorite_team"]["is_favorite"])
        self.assertAlmostEqual(board["favorite_team"]["cup_pct"], 0.26)
        self.assertIn("26.0% to win the 2025-26 Stanley Cup", board["summary_text"])
        self.assertIn("10,000 simulations", board["summary_text"])
        self.assertEqual([team["team_abbr"] for team in board["contenders"]], ["DAL", "COL", "EDM", "CGY"])
        self.assertEqual([division["division_name"] for division in board["divisions"]], ["Central", "Pacific"])
        central = board["divisions"][0]["teams"]
        self.assertEqual([team["team_abbr"] for team in central], ["COL", "DAL"])
        self.assertEqual(central[0]["points"], 95)
        self.assertAlmostEqual(central[0]["playoff_pct"], 0.97)

    def test_preseason_board_hides_last_seasons_record_and_sorts_by_projection(self):
        """Before opening night the standings only supply division membership."""
        board = build_stanley_cup_board(_standings(), _projection(phase="preseason"))

        self.assertEqual(board["mode"], BOARD_MODE_PRESEASON)
        self.assertIn("preseason projection", board["generated_at_label"])
        central = board["divisions"][0]["teams"]
        self.assertEqual([team["team_abbr"] for team in central], ["COL", "DAL"])
        self.assertEqual(central[0]["points"], 0)
        self.assertEqual(central[0]["games_played"], 0)
        self.assertAlmostEqual(central[0]["projected_points"], 112.0)

    def test_champion_board_names_the_winner_and_has_no_favorite(self):
        """Once the Final is decided the board reports the champion instead of odds."""
        board = build_stanley_cup_board(
            _standings(),
            {"state": "champion", "season_year": 2025, "champion": "EDM", "teams": {}},
        )

        self.assertEqual(board["mode"], BOARD_MODE_CHAMPION)
        self.assertEqual(board["favorite_team_abbr"], "")
        self.assertEqual(board["champion_team"]["team_abbr"], "EDM")
        self.assertTrue(board["champion_team"]["is_champion"])
        self.assertEqual(board["summary_text"], "Edmonton Oilers won the 2025-26 Stanley Cup.")
        self.assertIn("Final 2025-26 standings", board["generated_at_label"])

    def test_unavailable_projection_falls_back_to_plain_standings(self):
        """Without a projection the board still renders, with no pick and no odds."""
        board = build_stanley_cup_board(_standings(), {"state": "unavailable", "teams": {}})

        self.assertEqual(board["mode"], BOARD_MODE_STANDINGS)
        self.assertEqual(board["favorite_team"], {})
        self.assertIsNone(board["teams"][0]["cup_pct"])
        self.assertEqual(board["teams"][0]["team_abbr"], "COL")

    def test_empty_standings_return_an_empty_board(self):
        """No standings, no board."""
        board = build_stanley_cup_board(pd.DataFrame(), _projection())

        self.assertEqual(board["divisions"], [])
        self.assertEqual(board["teams"], [])


if __name__ == "__main__":
    unittest.main()
