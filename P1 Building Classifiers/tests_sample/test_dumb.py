import unittest

import dumbClassifiers, datasets
import numpy as np


class TestDumb(unittest.TestCase):
    def setUp(self):
        pass

    def test_most_freq_train(self):
        """Evaluate training accuracy of most frequent classifier"""
        h = dumbClassifiers.AlwaysPredictMostFrequent({})
        h.train(datasets.SentimentData.X, datasets.SentimentData.Y)
        m = np.mean((datasets.SentimentData.Y > 0) == (h.predictAll(datasets.SentimentData.X) > 0))
        self.assertTrue(abs(0.5041666666666667 - m) <= 1e-3)

    def test_most_freq_test(self):
        """Evaluate testing accuracy of most frequent classifier"""
        h = dumbClassifiers.AlwaysPredictMostFrequent({})
        h.train(datasets.SentimentData.Xte, datasets.SentimentData.Yte)
        m = np.mean((datasets.SentimentData.Yte > 0) == (h.predictAll(datasets.SentimentData.Xte) > 0))
        self.assertTrue(abs(0.5025 - m) <= 1e-3)
