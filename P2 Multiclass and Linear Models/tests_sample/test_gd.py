import unittest

import gd


class TestGD(unittest.TestCase):
    def setUp(self):
        pass

    def test_gd_basic(self):
        """Evaluate basic GD"""

        weight, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 10, 0.2)
        target = 1.0034641051795872
        self.assertTrue(abs(weight - target) < 1e-4)