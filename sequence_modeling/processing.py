import numpy as np
import re
from string import ascii_lowercase


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

    def __init__(self, mode, parameter):
        """
        :param mode: onehot, embed, random
        :param parameter:
        """
        if mode == "onehot":
            self.__char_dict = dict()
            for ind, c in enumerate(ascii_lowercase):
                self.__char_dict[c] = ind + 1
            self.__char_dict_l = len(self.__char_dict)
            self.__length_word = parameter["length_word"]
            self.__length_char = parameter["length_char"]
            self.__process = self.__onehot_char
        elif mode == "embed":
            self.__model = parameter["model"] if "model" in parameter.keys() else None
            # self.__model = parameter["model"]
            self.__model_dim = parameter["dim"]
            self.__sentence_length = parameter["length_word"]
            self.__random_dict = dict()
            self.__process = self.__embed
        # elif mode == "random":
        #     self.__dim = parameter["dim"]
        #     self.__sentence_length = parameter["length_word"]
        #     self.__random_dict = dict()
        #     self.__process = self.__random

    def __call__(self, data):
        return self.__process(data)

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
                    _oh = padding(np.vstack([np.zeros(self.__char_dict_l),
                                             np.eye(self.__char_dict_l)])[__vec], self.__length_char)
                    _vec.append(_oh)
            # if no word is contained in sentence, give zero vector with shape (1, word, char)
            _vec = np.zeros((1, self.__length_char, self.__char_dict_l)) if len(_vec) == 0 else np.array(_vec)
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
                            self.__random_dict[__d] = np.random.rand(self.__model_dim) * 2 - 1
                        _vec.append(self.__random_dict[__d])
                    else:
                        # partially random embedding
                        try:
                            _vec.append(self.__model[__d])
                        except KeyError:
                            if __d not in self.__random_dict.keys():
                                self.__random_dict[__d] = np.random.rand(self.__model_dim) * 2 - 1
                            _vec.append(self.__random_dict[__d])
            _vec = np.zeros((1, self.__model_dim)) if len(_vec) == 0 else np.array(_vec)
            vector.append(padding(_vec, self.__sentence_length))
        return np.array(vector)

    # def __random(self, data):
    #     """ Random embedding, range in [-1, 1]
    #     :param data: numpy array, [sentence, word]
    #     :return:
    #     """
    #     vector = []
    #     for ind, _d in enumerate(data):  # loop for sentence
    #         _vec = []
    #         for __d in _d.split(' '):  # loop for word
    #             __d = clean_word(__d)
    #             if __d is not None:
    #                 if __d not in self.__random_dict.keys():
    #                     self.__random_dict[__d] = np.random.rand(self.__model_dim) * 2 - 1
    #                 _vec.append(self.__random_dict[__d])
    #         _vec = np.zeros((1, self.__model_dim)) if len(_vec) == 0 else np.array(_vec)
    #         vector.append(padding(_vec, self.__sentence_length))
    #     return np.array(vector)
