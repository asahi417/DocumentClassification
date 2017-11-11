import gensim
import numpy as np
from . import sst


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
        pad = np.zeros((length - len(x), x.shape[1]))
        return np.vstack([x, pad])


def sst_vectorize(_length, sst_path, embed_path, save_path):
    data = sst(sst_path, binary=True)
    model = gensim.models.KeyedVectors.load_word2vec_format(embed_path, binary=True)
    cnt = 0
    nan_vec = []
    vector = []
    for ind, _d in enumerate(data["sentence"]):
        _vec = []
        for __d in _d.split(' '):
            try:
                _vec.append(model[__d])
            except KeyError:
                pass
        if len(_vec) == 0:
            nan_vec.append(ind)
            cnt += 1
        else:
            _vec = padding(np.vstack(_vec), _length)
            vector.append(np.expand_dims(_vec, 0))
    vector = np.vstack(vector)
    label = data["label"][nan_vec]
    np.savez("%s/vectorized_data_%i.npz" % (save_path, _length),
             senctence=vector, label=label, pad_length=_length, nan_cnt=cnt)


if __name__ == '__main__':
    # padding length
    _length = 30
    _sst_path = "./stanfordSentimentTreebank"
    _embed_path = "./GoogleNews-vectors-negative300.bin"
    sst_vectorize(_length, _sst_path, _embed_path, "./data")
