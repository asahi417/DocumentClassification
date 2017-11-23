import unittest
import numpy as np
from sequence_modeling.vectorize.util import padding


class Test(unittest.TestCase):

    def setUp(self):
        # procedure before each test
        pass

    def tearDown(self):
        # procedure after each test
        pass

    def test_padding(self):
        x = np.ones((8, 10))
        _x = padding(x, 8)
        self.assertEqual(len(x), len(_x))

        _x = padding(x, 4)
        self.assertEqual(4, len(_x))
        # self.assertEqual(x[0: 4], _x)

        _x = padding(x, 12)
        self.assertEqual(12, len(_x))
        # self.assertEqual(x, _x[0:12])

        x = np.ones(8)
        _x = padding(x, 8)
        self.assertEqual(len(x), len(_x))

        _x = padding(x, 6)
        self.assertEqual(6, len(_x))

        _x = padding(x, 10)
        self.assertEqual(10, len(_x))


if __name__ == '__main__':
    unittest.main()
