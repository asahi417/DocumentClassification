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
    logger = create_log("%s/embedding.log" % save_path)
    logger.info("loading model .....")
    model = gensim.models.KeyedVectors.load_word2vec_format(embed_path, binary=True)
    cnt = 0
    proper_index = []
    vector = []
    # vector = sparse.csr_matrix(np.empty((0, int(length * model.vector_size))))

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

            # _vec = padding(np.vstack(_vec), length).flatten()
            # _vec = sparse.csr_matrix(np.expand_dims(_vec, 0))
            # vector = sparse.vstack([vector, _vec])

        if np.floor(ind / full_size * 100).astype(int) == cnt_for_log:
            logger.info("%i %s" % (cnt_for_log, "%"))
            cnt_for_log += 10
    # logger.info("finally shaped %i x %i x %i ....." % (len(vector), length, model.vector_size))
    # logger.info("finally shaped (%i, %i)" % vector.shape)
    logger.info("nan: %i" % cnt)
    logger.info("saving vectors .....")
    # sparse.save_npz("%s/vectorized_pad%i_model%i_sentence.npz" % (save_path, length, model.vector_size), vector)
    vector = np.vstack(vector)
    np.save("%s/vectorized_pad%i_model%i_sentence.npy" % (save_path, length, model.vector_size), vector)
    np.save("%s/vectorized_pad%i_model%i_label.npy" % (save_path, length, model.vector_size), label[proper_index])
    # np.save("%s/vectorized_%i_label.npy" % (save_path, length), label[proper_index])


# if __name__ == '__main__':
#     # padding length
#     _length = 30
#     _sst_path = "./stanfordSentimentTreebank"
#     _embed_path = "./GoogleNews-vectors-negative300.bin"
#     # sst_vectorize(_length, _sst_path, _embed_path, "./data")
