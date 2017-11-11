import unittest
import numpy as np
from sequence_modeling import sst


class Test(unittest.TestCase):

    def setUp(self):
        # procedure before each test
        pass

    def tearDown(self):
        # procedure after each test
        pass

    def test_sst(self):
        path = "./data/stanfordSentimentTreebank"
        re = sst(path, drop_neutral=False, binary=False, cut_off=None)
        self.assertEqual(len(re["label"]), re["original_size"])
        self.assertEqual(len(re["label"]), len(re["sentence"]))
        self.assertEqual(len(re["balance"]), 5)
        self.assertEqual(np.sum(re["balance"]), len(re["label"]))

        re = sst(path, drop_neutral=True, binary=False, cut_off=None)
        self.assertEqual(len(re["label"]), len(re["sentence"]))
        self.assertEqual(len(re["balance"]), 4)
        self.assertEqual(np.sum(re["balance"]), len(re["label"]))

        re = sst(path, drop_neutral=True, binary=True, cut_off=None)
        self.assertEqual(len(re["label"]), len(re["sentence"]))
        self.assertEqual(len(re["balance"]), 2)
        self.assertEqual(np.sum(re["balance"]), len(re["label"]))

    def test_sst_cutoff(self):
        path = "./data/stanfordSentimentTreebank"

        re = sst(path, drop_neutral=True, binary=False, cut_off=3)
        self.assertEqual(len(re["label"]), len(re["sentence"]))
        self.assertEqual(len(re["balance"]), 4)
        self.assertEqual(np.sum(re["balance"]), len(re["label"]))

        re = sst(path, drop_neutral=True, binary=True, cut_off=3)
        self.assertEqual(len(re["label"]), len(re["sentence"]))
        self.assertEqual(len(re["balance"]), 2)
        self.assertEqual(np.sum(re["balance"]), len(re["label"]))


if __name__ == '__main__':
    unittest.main()
