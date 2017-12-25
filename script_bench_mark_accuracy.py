import numpy as np
import os
import logging
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import gensim

from tweet_classification import BatchFeeder, Process
from data.util import data_set
from sklearn.linear_model import LogisticRegression

"""
Script for bench mark check
- Model: Logistic Regression, SVM
- Evaluation: Accuracy for 5-Fold stratified cross validation 
"""


def create_log(name):
    """Logging."""
    if os.path.exists(name):
        os.remove(name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # handler for logger file
    handler1 = logging.FileHandler(name)
    handler1.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    # handler for standard output
    handler2 = logging.StreamHandler()
    handler2.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


if __name__ == '__main__':
    _log = create_log("./log/bench_mark.log")
    # load data
    data = data_set()
    x, y = data["sentence"], data["label"]
    ratio = float(np.sum(y == 0) / len(y))
    _log.info("Balance of positive label: %0.3f" % ratio)
    w2v = gensim.models.KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin", binary=True)
    pre_process = Process("embed_avg", {"dim": w2v.vector_size, "model": w2v, "path": "./data/random_dict.json",
                                        "conversion_dict": data["dictionary"]})

    # preprocess
    _log.info("Cross validation (precision)")
    feeder = BatchFeeder(x, y, len(y), process=pre_process)
    _x, _y = feeder.next()
    # train
    clf = svm.SVC(verbose=False, shrinking=False)
    cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
    scores = cross_val_score(clf, _x, _y, cv=cv, scoring="average_precision")
    _log.info("SVM full score: %s" % scores)
    _log.info("SVM mean: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    clf = LogisticRegression(verbose=False)
    cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
    scores = cross_val_score(clf, _x, _y, cv=cv, scoring="average_precision")
    _log.info("LR full score: %s" % scores)
    _log.info("LR mean: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    # preprocess
    _log.info("Fixed validation (accuracy for balanced data)")
    feeder = BatchFeeder(x, y, 6089, validation=0.05, process=pre_process, fix_validation=False)
    x, y = feeder.next()
    x_v, y_v = feeder.next_valid()
    print(x.shape, y.shape)
    print(x_v.shape, y_v.shape)
    # train
    clf = svm.SVC(verbose=False, shrinking=False)
    clf.fit(x, y)
    acc = np.sum(y_v == clf.predict(x_v))/len(y_v)
    _log.info("SVM-accuracy: %0.3f" % acc)

    clf = LogisticRegression(verbose=False)
    clf.fit(x, y)
    acc = np.sum(y_v == clf.predict(x_v))/len(y_v)

    _log.info("LR-accuracy: %0.3f" % acc)

    feeder.finalize()
