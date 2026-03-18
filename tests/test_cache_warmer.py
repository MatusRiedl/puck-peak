import os
import unittest
from unittest.mock import call, patch

import nhl.cache_warmer as cache_warmer


class CacheWarmerTests(unittest.TestCase):
    """Cover optional background cache warmer startup and task wiring."""

    def setUp(self):
        self._env_patch = patch.dict(os.environ, {}, clear=False)
        self._env_patch.start()
        cache_warmer._warmer_started = False

    def tearDown(self):
        cache_warmer._warmer_started = False
        self._env_patch.stop()

    def test_start_background_warmer_disabled_by_default(self):
        """Missing enable flag should leave the warmer dormant."""
        with patch("nhl.cache_warmer.threading.Thread") as mock_thread:
            started = cache_warmer.start_background_warmer()

        self.assertFalse(started)
        mock_thread.assert_not_called()

    def test_start_background_warmer_enabled_starts_exactly_one_thread_per_tier(self):
        """Enabled flag should start 3 daemon tier threads once."""
        os.environ["PUCKPEAK_CACHE_WARMER_ENABLED"] = "1"

        class FakeThread:
            created = []

            def __init__(self, target=None, name=None, args=(), daemon=None):
                self.target = target
                self.name = name
                self.args = args
                self.daemon = daemon
                self.started = False
                FakeThread.created.append(self)

            def start(self):
                self.started = True

        with patch("nhl.cache_warmer.threading.Thread", FakeThread):
            started = cache_warmer.start_background_warmer()

        self.assertTrue(started)
        self.assertEqual(len(FakeThread.created), 3)
        self.assertEqual(
            [thread.name for thread in FakeThread.created],
            ["cache-warmer-live", "cache-warmer-seasonal", "cache-warmer-historical"],
        )
        self.assertTrue(all(thread.daemon for thread in FakeThread.created))
        self.assertTrue(all(thread.started for thread in FakeThread.created))

    def test_start_background_warmer_second_call_does_not_spawn_duplicates(self):
        """Repeated startup attempts in the same process should be ignored."""
        os.environ["PUCKPEAK_CACHE_WARMER_ENABLED"] = "1"

        class FakeThread:
            created = []

            def __init__(self, target=None, name=None, args=(), daemon=None):
                self.target = target
                self.name = name
                self.args = args
                self.daemon = daemon
                FakeThread.created.append(self)

            def start(self):
                return None

        with patch("nhl.cache_warmer.threading.Thread", FakeThread):
            first_started = cache_warmer.start_background_warmer()
            second_started = cache_warmer.start_background_warmer()

        self.assertTrue(first_started)
        self.assertFalse(second_started)
        self.assertEqual(len(FakeThread.created), 3)

    def test_run_live_cycle_warms_matchup_and_upcoming_games(self):
        """Live cycle should warm the home screen matchup and predictions rail."""
        with patch("nhl.cache_warmer.get_live_or_recent_game", return_value=("EDM", "DAL")) as mock_game, patch(
            "nhl.cache_warmer.get_featured_players",
            return_value={"players": {}, "teams": {}},
        ) as mock_featured, patch(
            "nhl.cache_warmer.get_upcoming_games",
            return_value=[],
        ) as mock_upcoming:
            cache_warmer._run_live_cycle()

        mock_game.assert_called_once_with()
        mock_featured.assert_called_once_with("EDM", "DAL")
        mock_upcoming.assert_called_once_with(limit=8, days_ahead=14)

    def test_run_live_cycle_skips_featured_players_when_no_matchup_is_available(self):
        """No featured-player warm-up should run when the live game lookup fails."""
        with patch("nhl.cache_warmer.get_live_or_recent_game", return_value=None), patch(
            "nhl.cache_warmer.get_featured_players",
        ) as mock_featured, patch(
            "nhl.cache_warmer.get_upcoming_games",
            return_value=[],
        ):
            cache_warmer._run_live_cycle()

        mock_featured.assert_not_called()

    def test_run_seasonal_cycle_warms_seeded_teams_and_players(self):
        """Seasonal cycle should touch the exact seeded team and player list."""
        with patch("nhl.cache_warmer.load_all_team_seasons") as mock_team_seasons, patch(
            "nhl.cache_warmer.get_current_nhl_standings",
        ) as mock_standings, patch(
            "nhl.cache_warmer.get_team_roster",
            return_value={},
        ) as mock_team_roster, patch(
            "nhl.cache_warmer.get_player_landing",
            return_value={},
        ) as mock_player_landing:
            cache_warmer._run_seasonal_cycle()

        mock_team_seasons.assert_called_once_with()
        mock_standings.assert_called_once_with()
        self.assertEqual(
            mock_team_roster.call_args_list,
            [call("EDM"), call("PIT"), call("WSH"), call("COL"), call("TOR"), call("NYR")],
        )
        self.assertEqual(
            mock_player_landing.call_args_list,
            [
                call(8478402),
                call(8471675),
                call(8471214),
                call(8477492),
                call(8479318),
                call(8478048),
            ],
        )

    def test_run_historical_cycle_warms_maps_and_top_lists(self):
        """Historical cycle should warm shared lookup maps and all configured top lists."""
        with patch("nhl.cache_warmer.get_id_to_name_map", return_value={}) as mock_id_map, patch(
            "nhl.cache_warmer.get_clone_details_map",
            return_value={},
        ) as mock_clone_map, patch(
            "nhl.cache_warmer.get_top_50",
            return_value={},
        ) as mock_top_50, patch(
            "nhl.cache_warmer.get_top_50_goalies",
            return_value={},
        ) as mock_goalies:
            cache_warmer._run_historical_cycle()

        self.assertEqual(mock_id_map.call_args_list, [call("Skater"), call("Goalie")])
        self.assertEqual(mock_clone_map.call_args_list, [call("Skater"), call("Goalie")])
        self.assertEqual(
            mock_top_50.call_args_list,
            [call("Points"), call("Goals"), call("Assists")],
        )
        mock_goalies.assert_called_once_with()

    def test_run_safe_task_logs_and_swallows_exceptions(self):
        """One failing task must not crash the warmer thread."""

        def _boom():
            raise RuntimeError("boom")

        with patch.object(cache_warmer.log, "exception") as mock_log:
            result = cache_warmer._run_safe_task("exploding-task", _boom)

        self.assertIsNone(result)
        mock_log.assert_called_once()


if __name__ == "__main__":
    unittest.main()
