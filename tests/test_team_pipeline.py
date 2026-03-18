import unittest
from unittest.mock import patch

import pandas as pd

import nhl.team_pipeline as team_pipeline


class TeamPipelineTests(unittest.TestCase):
    """Cover the selected-season team season-progress branch."""

    def test_process_teams_selected_season_builds_running_team_metrics(self):
        """Use one row per game and compute season-to-date values after each game."""
        game_log_df = pd.DataFrame(
            {
                "GameDate": ["2023-10-10", "2023-10-12", "2023-10-15"],
                "GameId": [2001, 2002, 2003],
                "gameTypeId": [2, 2, 2],
                "GameType": ["Regular", "Regular", "Regular"],
                "GP": [1.0, 1.0, 1.0],
                "Wins": [1.0, 0.0, 1.0],
                "Losses": [0.0, 1.0, 0.0],
                "OTLosses": [0.0, 0.0, 0.0],
                "Ties": [0.0, 0.0, 0.0],
                "Points": [2.0, 0.0, 2.0],
                "Goals": [4.0, 2.0, 3.0],
                "GoalsAgainst": [2.0, 3.0, 1.0],
                "PP%": [20.0, 0.0, 50.0],
                "OpponentAbbrev": ["MTL", "BOS", "BUF"],
                "OpponentName": ["Montreal Canadiens", "Boston Bruins", "Buffalo Sabres"],
                "HomeRoadFlag": ["H", "R", "H"],
                "ResultCode": ["W", "L", "W"],
                "ResultLabel": ["W", "L", "W"],
                "TeamAbbrev": ["TOR"] * 3,
                "TeamName": ["Toronto Maple Leafs"] * 3,
            }
        )

        with patch.object(team_pipeline, "get_team_season_game_log", return_value=game_log_df):
            processed = team_pipeline.process_teams(
                teams={"TOR": "Toronto Maple Leafs"},
                all_team_df=pd.DataFrame({"teamAbbrev": ["TOR"]}),
                metric="GF/G",
                season_type="Regular",
                do_cumul=False,
                do_smooth=False,
                games_mode=True,
                selected_season=2023,
            )

        self.assertEqual(len(processed), 1)
        df = processed[0]
        self.assertEqual(df["CumGP"].tolist(), [1, 2, 3])
        self.assertEqual(df["Points"].tolist(), [2.0, 2.0, 4.0])
        self.assertEqual(df["Wins"].tolist(), [1.0, 1.0, 2.0])
        self.assertEqual(df["Goals"].tolist(), [4.0, 6.0, 9.0])
        self.assertEqual(df["GoalsAgainst"].tolist(), [2.0, 5.0, 6.0])
        self.assertEqual(df["RecordLabel"].tolist(), ["1-0", "1-1", "2-1"])
        self.assertEqual(df["GF/G"].round(3).tolist(), [4.0, 3.0, 3.0])
        self.assertEqual(df["GA/G"].round(3).tolist(), [2.0, 2.5, 2.0])
        self.assertEqual(df["Win%"].round(1).tolist(), [100.0, 50.0, 66.7])
        self.assertEqual(df["PP%"].round(1).tolist(), [20.0, 10.0, 23.3])
        self.assertEqual(df["PPG"].round(3).tolist(), [10.8, 8.1, 8.1])


if __name__ == "__main__":
    unittest.main()
