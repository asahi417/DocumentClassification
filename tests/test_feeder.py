import unittest
from sequence_modeling import sst
from sequence_modeling import BatchFeeder


class Test(unittest.TestCase):

    def setUp(self):
        # procedure before each test
        pass

    def tearDown(self):
        # procedure after each test
        pass

    def test_feeder(self):
        path = "./data/stanfordSentimentTreebank"
        re = sst(path, drop_neutral=True, binary=False, cut_off=3)
        fed = BatchFeeder(re["sentence"], re["label"], 100)
        tmp = fed.next()
        self.assertEqual(len(tmp[0]), len(tmp[1]))
        self.assertEqual(len(tmp[0]), 100)


if __name__ == '__main__':
    unittest.main()
