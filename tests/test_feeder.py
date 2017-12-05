import unittest
from sequence_modeling import BatchFeeder
import numpy as np


class Test(unittest.TestCase):

    def setUp(self):
        # procedure before each test
        pass

    def tearDown(self):
        # procedure after each test
        pass

    def test_1d(self):
        inputs = np.arange(100)
        outputs = np.array([0] * 35 + [1] * 65)
        feeder = BatchFeeder(inputs, outputs, 10, 0.2, fix_validation=True)
        self.assertEqual(np.sum(feeder.y_valid == 0), np.sum(feeder.y_valid == 1))
        x, y = feeder.next()
        self.assertEqual(len(x), 10)
        self.assertEqual(len(y), 10)

    def test_2d(self):
        inputs = np.array([np.arange(100), np.arange(100)]).T
        outputs = np.array([0] * 35 + [1] * 65)
        feeder = BatchFeeder(inputs, outputs, 10, 0.2, fix_validation=True)
        self.assertEqual(np.sum(feeder.y_valid == 0), np.sum(feeder.y_valid == 1))
        x, y = feeder.next()
        self.assertEqual(len(x), 10)
        self.assertEqual(len(y), 10)

    def test_validation(self):
        inputs = np.array([np.arange(100), np.arange(100)]).T
        outputs = np.array([0] * 35 + [1] * 65)
        feeder_0 = BatchFeeder(inputs, outputs, 10, 0.2, fix_validation=True)
        feeder_1 = BatchFeeder(inputs, outputs, 10, 0.2, fix_validation=True)
        for i in range(20):
            self.assertEqual(feeder_0.y_valid[i], feeder_1.y_valid[i])
            self.assertEqual(list(feeder_0.x_valid[i]), list(feeder_1.x_valid[i]))
        self.assertEqual(np.sum(feeder_0.y_valid == 0), np.sum(feeder_0.y_valid == 1))


if __name__ == '__main__':
    unittest.main()
