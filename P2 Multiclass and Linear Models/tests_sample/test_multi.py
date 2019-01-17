import unittest

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import multiclass, datasets, runClassifier, util


class TestAVA(unittest.TestCase):
    def setUp(self):
        self.dataset = datasets.WineDataSmall

    def test_ava_5(self):
        """Evaluate AVA on 5 classes"""

        h = multiclass.AVA(5, lambda: DecisionTreeClassifier(max_depth=3))
        h.train(self.dataset.X, self.dataset.Y)
        P = h.predictAll(self.dataset.Xte)

        targetAccRange = (0.33, 0.36)
        acc = np.mean(P == self.dataset.Yte)
        self.assertTrue(acc > targetAccRange[0] and acc < targetAccRange[1])

    