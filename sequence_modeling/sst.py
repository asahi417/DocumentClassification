import numpy as np
import pandas as pd


def sst(path="./stanfordSentimentTreebank", drop_neutral=True, cut_off=None, binary=False):
    """

    :param str path: path to the `stanfordSentimentTreebank`
    :param bool drop_neutral: if ignore neutral label or not
    :param cut_off: cut off the term based on frequency. If None, no cut off
    :param binary: binarize label or not
    :return dict: Stanford Sentiment Treebank data
    """
    df = sst_formatting(path)
    label = quantize_label(df["label"].values)
    df["label"] = label
    original_size = len(df)
    if cut_off:
        df["cnt"] = [len(i.split(' ')) for i in df["data"].values]
        df = df[df.cnt >= cut_off]
        label = df["label"].values
    if drop_neutral:
        df = df[df.label != 3]
        if binary:
            label = df["label"].values
            label[label > 3] = 1
            label[label != 1] = 0
            df["label"] = label
            bal = [np.sum(label == 1), np.sum(label == 0)]
        else:
            bal = [np.sum(label == i) for i in [1, 2, 4, 5]]
    else:
        bal = [np.sum(label == i) for i in [1, 2, 3, 4, 5]]
    return {"label": df.label.values, "sentence": df.data.values, "original_size": original_size, "balance": bal}


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


if __name__ == '__main__':
    data = sst("./data/stanfordSentimentTreebank", binary=True)
    length = []
    for _d in data["sentence"]:
        length.append(len(_d.split(' ')))
    print("Word distribution")
    print("max", np.sort(length)[-100:])
    print("min", np.sort(length)[:100])
    print("mean %0.2f" % np.mean(length))
    print("median %0.2f" % np.median(length))
    print("balance", data["balance"])
    print("size: %i -> %i" % (data["original_size"], len(data["label"])))
