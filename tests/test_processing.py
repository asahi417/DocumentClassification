import unittest
from sequence_modeling import Process


class Test(unittest.TestCase):

    def setUp(self):
        # procedure before each test
        pass

    def tearDown(self):
        # procedure after each test
        pass

    def test_none(self):
        _data = [""]

        p = Process(mode="onehot", parameter={"length_word": 40, "length_char": 24})
        _tmp = p(_data)
        __shape = _tmp.shape
        print("char_none", __shape)
        self.assertEqual(__shape[0], 1)
        self.assertEqual(__shape[1], 40)
        self.assertEqual(__shape[2], 24)
        # self.assertEqual(__shape[3], p.__char_dict_l)

        p = Process(mode="embed", parameter={"length_word": 40, "dim": 30})
        _tmp = p(_data)
        __shape = _tmp.shape
        print("word_none", __shape)
        self.assertEqual(__shape[0], 1)
        self.assertEqual(__shape[1], 40)
        self.assertEqual(__shape[2], 30)

    def test(self):
        _data = ["I have a pen.", "I have an apple."]
        p = Process(mode="onehot", parameter={"length_word": 40, "length_char": 24})
        _tmp = p(_data)
        __shape = _tmp.shape
        print("char", __shape)
        self.assertEqual(__shape[0], 2)
        self.assertEqual(__shape[1], 40)
        self.assertEqual(__shape[2], 24)
        # self.assertEqual(__shape[3], p.__char_dict_l)

        p = Process(mode="embed", parameter={"length_word": 40, "dim": 30})
        _tmp = p(_data)
        __shape = _tmp.shape
        print("word", __shape)
        self.assertEqual(__shape[0], 2)
        self.assertEqual(__shape[1], 40)
        self.assertEqual(__shape[2], 30)


if __name__ == '__main__':
    unittest.main()
