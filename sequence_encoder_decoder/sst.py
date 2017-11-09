import numpy as np
import pandas as pd


def sst(path="./", drop_neutral=True, cut_off=3):
    df = sst_formatting(path)
    label = quantize_label(df["label"].values)
    df["label"] = label
    if drop_neutral:
        df = df[df.label != 3]
    if cut_off:
        df["count"] = [len(i.split(' ')) for i in df["data"].values]






def sst_formatting(path):
    with open("%s/sentiment_labels.txt" % path) as f:
        _tmp = [i.split('|') for i in f.read().split('\n')]
        _tmp.pop(-1)
        _tmp.pop(0)
        _tmp = np.array(_tmp)
        _df1 = pd.DataFrame(_tmp[:, 1].astype(float), columns=["label"], index=_tmp[:, 0].astype(int))

    with open("%s/dictionary.txt" % path) as f:
        _tmp = [i.split('|') for i in f.read().split('\n')]
        _tmp.pop(-1)
        _tmp = np.array(_tmp)
        _df2 = pd.DataFrame(_tmp[:, 0], columns=["data"], index=_tmp[:, 1].astype(int))

    return _df1.join(_df2, how="inner")


def quantize_label(score):
    """
    [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
    for very negative, negative, neutral, positive, very positive, respectively.

    :param score:
    :return: very negative: 1, negative: 2, neutral: 3, positive: 4, very positive: 5
    """
    label = np.zeros(len(score))
    label[score <= 0.2] += 1
    label[score <= 0.4] += 1
    label[score <= 0.6] += 1
    label[score <= 0.8] += 1
    label[score <= 1] += 1
    return label.astype(int)


