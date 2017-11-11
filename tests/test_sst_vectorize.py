import unittest
import numpy as np
from sequence_modeling.sst_vectorize import padding


class Test(unittest.TestCase):

    def setUp(self):
        # procedure before each test
        pass

    def tearDown(self):
        # procedure after each test
        pass

    def test_padding(self):
        x = np.ones((8, 10))
        _x = padding(x, 10)
        self.assertEqual(x, _x)

        _x = padding(x, 8)
        self.assertEqual(8, len(_x))
        self.assertEqual(x[0: 8], _x)

        _x = padding(x, 12)
        self.assertEqual(12, len(_x))
        self.assertEqual(x, _x[0:10])


if __name__ == '__main__':
    unittest.main()
