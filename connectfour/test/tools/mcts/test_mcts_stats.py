import unittest

from connectfour.tools.mcts import stats


class TestMCTSStats(unittest.TestCase):
    def test_empty(self):
        empty = stats.empty(42)

        self.assertEqual(empty.probability, 42)
        self.assertEqual(empty.visits, 0)
        self.assertEqual(empty.rewards, 0)

    def test_update(self):
        initial = stats.MCTSNodeStats(42, 5, 5)
        updated = stats.update(initial, -1)

        self.assertEqual(updated.probability, 42)
        self.assertEqual(updated.visits, 6)
        self.assertEqual(updated.rewards, 4)


if __name__ == '__main__':
    unittest.main()
