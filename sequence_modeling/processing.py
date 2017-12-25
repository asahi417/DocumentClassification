import numpy as np
import re
from string import ascii_lowercase
import json
import os


def padding(x, length):
    """
    :param x: data number x feature
    :param length: padding threshold to data number
    :return:
    """
    if len(x) == length:
        return x
    elif len(x) > length:
        return x[0:length]
    elif len(x) < length:
        __shape = [length - len(x)] + list(x.shape[1:])
        pad = np.zeros(tuple(__shape))
        return np.vstack([x, pad]) if x.ndim > 1 else np.hstack([x, pad])


def clean_word(word):
    """ Remove non alphabet and lowercase
    :param word:
    :return: if len(word) == 1 return None, else cleaned word
    """
    word = re.sub("[^%s]" % ascii_lowercase, r"", word.lower())
    return None if len(word) == 1 else word


class Process:
    """ Preprcessing functions
    Usage
    ```
    import Process
    p = Process(mode="onehot", parameter={"length_word": 40, "length_char": 33}
    p(data)
    """
    __random_dict = dict()
    __char_dict = dict()

    def __init__(self, mode, parameter):
        """
        :param mode: onehot for character, embed for word (average among the word or not)
        :param parameter:
        """
        self.mode = mode
        self.seed = 0  # seed for random
        if mode == "onehot":
            self.__char_dict = dict()
            for ind, c in enumerate(ascii_lowercase):
                self.__char_dict[c] = ind + 1
            self.char_dict_size = len(self.__char_dict)
            self.__length_word = parameter["length_word"]
            self.__length_char = parameter["length_char"]
            self.__process = self.__onehot_char
        elif mode == "embed":
            self.__dict_save_path = parameter["path"] if "path" in parameter.keys() else "./random_dict.json"
            self.__model = parameter["model"] if "model" in parameter.keys() else None
            self.__model_dim = parameter["dim"]
            self.__random_dict = dict()
            # if dictionary used in pre-trained
            if os.path.exists(self.__dict_save_path):
                with open(self.__dict_save_path, "r") as f:
                    self.__random_dict = json.load(f)
            if mode == "embed":
                self.__sentence_length = parameter["length_word"]
                self.__process = self.__embed
            elif mode == "embed_avg":
                self.__process = self.__embed_avg

    def __call__(self, data):
        return self.__process(data)

    def finalize(self):
        if len(self.__random_dict) != 0 and self.mode == "embed":
            with open(self.__dict_save_path, "w") as f:
                json.dump(self.__random_dict, f)

    def __onehot_char(self, data):
        """ Convert one hot vector for character
        :param data: numpy array [sentence, word, char]
        :return: one hot vector
        """
        vector = []
        for ind, _d in enumerate(data):  # loop for sentence
            _vec = []
            for __d in _d.split(' '):  # loop for word
                __d = clean_word(__d)
                if __d is not None:
                    __vec = []
                    for ___d in __d:  # loop for char
                        __vec.append(self.__char_dict[___d])
                    _oh = padding(np.vstack([np.zeros(self.char_dict_size),
                                             np.eye(self.char_dict_size)])[__vec], self.__length_char)
                    _vec.append(_oh)
            # if no word is contained in sentence, give zero vector with shape (1, word, char)
            _vec = np.zeros((1, self.__length_char, self.char_dict_size)) if len(_vec) == 0 else np.array(_vec)
            vector.append(padding(_vec, self.__length_word))
        return np.array(vector)

    def __embed(self, data):
        """ Word embedding by given model
        the model's embedded value must be in [-1, 1]
        :param data: numpy array, [sentence, word]
        :return:
        """
        vector = []
        for ind, _d in enumerate(data):  # loop for sentence
            _vec = []
            for __d in _d.split(' '):  # loop for word
                __d = clean_word(__d)
                if __d is not None:
                    if self.__model is None:
                        # full random embedding
                        if __d not in self.__random_dict.keys():
                            np.random.seed(self.seed)
                            self.seed += 1
                            self.__random_dict[__d] = list(np.random.rand(self.__model_dim) * 2 - 1)
                        _vec.append(self.__random_dict[__d])
                    else:
                        # partially random embedding
                        try:
                            _vec.append(self.__model[__d])
                        except KeyError:
                            if __d not in self.__random_dict.keys():
                                np.random.seed(self.seed)
                                self.seed += 1
                                self.__random_dict[__d] = list(np.random.rand(self.__model_dim) * 2 - 1)
                            _vec.append(self.__random_dict[__d])
            _vec = np.zeros((1, self.__model_dim)) if len(_vec) == 0 else np.array(_vec)
            vector.append(padding(_vec, self.__sentence_length))
        return np.array(vector)

    def __embed_avg(self, data):
        """ Word embedding by given model
        the model's embedded value must be in [-1, 1]
        if not in model, randomly embed
        :param data: numpy array, [sentence, word]
        :return:
        """
        vector = []
        for ind, _d in enumerate(data):  # loop for sentence
            _vec = []
            for __d in _d.split(' '):  # loop for word
                __d = clean_word(__d)
                if __d is not None:
                    if self.__model is None:
                        # full random embedding
                        if __d not in self.__random_dict.keys():
                            np.random.seed(self.seed)
                            self.seed += 1
                            self.__random_dict[__d] = list(np.random.rand(self.__model_dim) * 2 - 1)
                        _vec.append(self.__random_dict[__d])
                    else:
                        # partially random embedding
                        try:
                            _vec.append(self.__model[__d])
                        except KeyError:
                            if __d not in self.__random_dict.keys():
                                np.random.seed(self.seed)
                                self.seed += 1
                                self.__random_dict[__d] = list(np.random.rand(self.__model_dim) * 2 - 1)
                            _vec.append(self.__random_dict[__d])
            _vec = np.zeros((1, self.__model_dim)) if len(_vec) == 0 else np.array(_vec)
            vector.append(_vec.mean(0))
        return np.array(vector)
