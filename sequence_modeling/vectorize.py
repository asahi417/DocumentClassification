import gensim
import numpy as np
from .train import create_log
import os


def padding(x, _len):
    """
    :param x: data number x feature
    :param _len: padding threshold to data number
    :return:
    """
    if len(x) == _len:
        return x
    elif len(x) > _len:
        return x[0:_len]
    elif len(x) < _len:
        pad = np.zeros((_len - len(x), x.shape[1]))
        return np.vstack([x, pad])


def vectorize(sentences, label, length, embed_path, save_path):
    """
    Vectorize sentences based on given embedding model for supervised learning

    :param sentences: target sentence
    :param label: label related to the sentence
    :param length: padding threshold
    :param embed_path: path to the embedding model
    :param save_path: path to save the embedded vectors
    :return:
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    logger = create_log("%s/embedding_%i.log" % (save_path, length))
    logger.info("loading model .....")
    model = gensim.models.KeyedVectors.load_word2vec_format(embed_path, binary=True)
    cnt = 0
    proper_index = []
    vector = []

    cnt_for_log, full_size = 10, len(sentences) - 1

    for ind, _d in enumerate(sentences):
        _vec = []
        for __d in _d.split(' '):
            try:
                _vec.append(model[__d])
            except KeyError:
                pass
        if len(_vec) == 0:
            cnt += 1
        else:
            proper_index.append(ind)
            _vec = padding(np.vstack(_vec), length)
            vector.append(np.expand_dims(_vec, 0))
        if np.floor(ind / full_size * 100).astype(int) == cnt_for_log:
            logger.info("%i %s" % (cnt_for_log, "%"))
            cnt_for_log += 10
    logger.info("finally shaped %i x %i x %i ....." % (len(vector), length, model.vector_size))
    logger.info("saving vectors .....")
    vector = np.vstack(vector)
    label = label[proper_index]
    np.savez("%s/vectorized_data_%i.npz" % (save_path, length),
             sentence=vector, label=label, pad_length=length, nan_cnt=cnt)


# if __name__ == '__main__':
#     # padding length
#     _length = 30
#     _sst_path = "./stanfordSentimentTreebank"
#     _embed_path = "./GoogleNews-vectors-negative300.bin"
#     # sst_vectorize(_length, _sst_path, _embed_path, "./data")
