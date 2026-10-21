import unittest

from soniclake.pareto import pareto_frontier


class ParetoTests(unittest.TestCase):
    def test_frontier(self):
        rows = [
            {"name": "a", "storage": 1, "quality": 0.8},
            {"name": "b", "storage": 2, "quality": 0.7},
            {"name": "c", "storage": 3, "quality": 0.9},
        ]
        frontier = pareto_frontier(rows, minimize=("storage",), maximize=("quality",))
        self.assertEqual([row["name"] for row in frontier], ["a", "c"])
