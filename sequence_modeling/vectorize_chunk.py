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


def randomize(target):
    index = np.arange(len(target))
    np.random.shuffle(index)
    return target[index]


def vectorize_chunk(sentences, label, length, embed_path, save_path, chunk_size=1000):
    """
    Vectorize sentences based on given embedding model for supervised learning

    :param sentences: target sentence
    :param label: label related to the sentence
    :param length: padding threshold
    :param int chunk_size: save chunk size
    :param embed_path: path to the embedding model
    :param save_path: path to save the embedded vectors
    :return:
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    logger = create_log("%s/embedding.log" % save_path)
    logger.info("loading model .....")
    model = gensim.models.KeyedVectors.load_word2vec_format(embed_path, binary=True)
    save_index, nan_cnt = 0, 0
    vector, proper_index = [], []

    sentences = randomize(sentences)

    for ind, _d in enumerate(sentences):
        _vec = []
        for __d in _d.split(' '):
            try:
                _vec.append(model[__d])
            except KeyError:
                pass
        if len(_vec) == 0:
            nan_cnt += 1
        else:
            proper_index.append(ind)
            _vec = padding(np.vstack(_vec), length)
            vector.append(np.expand_dims(_vec, 0))
        if len(proper_index) % chunk_size == 0:
            save_index += 1
            logger.info("saving %i " % save_index)
            vector = np.vstack(vector)
            np.savez("%s/vectorized_pad%i_model%i_chunk%i" % (save_path, length, model.vector_size, save_index),
                     x=vector, y=label[proper_index])
            proper_index = []
            vector = []
        if ind % 10000 == 0:
            logger.info("processed %i/%i" % (ind, len(sentences)))
    if len(proper_index) != 0:
        save_index += 1
        logger.info("saving residual length: %i " % len(proper_index))
        vector = np.vstack(vector)
        np.savez("%s/vectorized_pad%i_model%i_chunk%i" % (save_path, length, model.vector_size, save_index),
                 x=vector, y=label[proper_index])

    # logger.info("finally shaped %i x %i x %i ....." % (len(vector), length, model.vector_size))
    # logger.info("finally shaped (%i, %i)" % vector.shape)
    logger.info("nan: %i" % nan_cnt)


# if __name__ == '__main__':
#     # padding length
#     _length = 30
#     _sst_path = "./stanfordSentimentTreebank"
#     _embed_path = "./GoogleNews-vectors-negative300.bin"
#     # sst_vectorize(_length, _sst_path, _embed_path, "./data")
