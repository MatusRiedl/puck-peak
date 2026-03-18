import unittest

import pandas as pd

from nhl.baselines import _shape_goalie_save_tail, _shape_skater_late_tail
from nhl.knn_engine import _stabilize_late_target


class TailShapingTests(unittest.TestCase):
    def test_skater_tail_breaks_synthetic_092_chain(self):
        base = pd.DataFrame(
            {
                'Points': {
                    33: 44.0,
                    34: 42.0,
                    35: 40.0,
                    36: 36.8,
                    37: 33.856,
                    38: 31.1475,
                    39: 28.6557,
                    40: 26.3632,
                    41: 24.2541,
                }
            }
        )
        age_counts = pd.Series({36: 393, 37: 266, 38: 146, 39: 89, 40: 45, 41: 21})

        shaped = _shape_skater_late_tail(base, age_counts)['Points']
        tail = [float(shaped.loc[age]) for age in range(36, 42)]
        ratios = [tail[i] / tail[i - 1] for i in range(1, len(tail))]

        self.assertTrue(all(tail[i] < tail[i - 1] for i in range(1, len(tail))))
        self.assertTrue(any(abs(r - 0.92) > 0.02 for r in ratios[2:]))

    def test_goalie_save_tail_is_curved_not_straight(self):
        goalie_base = pd.DataFrame(
            {
                'Save %': {
                    34: 91.3,
                    35: 91.5,
                    36: 91.6,
                    37: 91.8,
                    38: 91.7,
                    39: 91.9,
                    40: 92.0,
                    41: 92.1,
                }
            }
        )
        age_counts = pd.Series({35: 84, 36: 58, 37: 42, 38: 26, 39: 19, 40: 10, 41: 11})

        shaped = _shape_goalie_save_tail(goalie_base, age_counts)['Save %']
        tail = [float(shaped.loc[age]) for age in range(34, 42)]
        diffs = [round(tail[i] - tail[i - 1], 4) for i in range(1, len(tail))]

        self.assertTrue(all(d < 0 for d in diffs))
        self.assertTrue(all(-0.45 <= d <= -0.05 for d in diffs))
        self.assertGreater(len(set(diffs)), 2)

    def test_sparse_skater_target_blocks_late_rebound(self):
        stabilized = _stabilize_late_target(
            next_avg=54.0,
            last_avg=20.0,
            metric='Points',
            age=39,
            stat_category='Skater',
            clone_count=1,
        )

        self.assertAlmostEqual(stabilized, 18.8)
        self.assertLess(stabilized, 20.0)

    def test_sparse_goalie_save_target_caps_upward_spike(self):
        stabilized = _stabilize_late_target(
            next_avg=94.0,
            last_avg=90.5,
            metric='Save %',
            age=39,
            stat_category='Goalie',
            clone_count=2,
        )

        self.assertAlmostEqual(stabilized, 90.3)
        self.assertLess(stabilized, 90.5)


if __name__ == '__main__':
    unittest.main()