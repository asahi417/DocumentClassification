import os
import sys
import argparse
import tweet_classification
import gensim
from data.util import data_set
from glob import glob
import numpy as np


def get_options(parser):
    parser.add_argument('model', action='store', nargs='?', const=None, default='char_cnn', type=str, choices=None,
                        metavar=None, help='Name of model to use. (default: char_cnn)')
    return parser.parse_args()


def controller(model_name, embed_model):
    # load data
    data = data_set()
    model, architecture, pre_process, model_inputs = \
        tweet_classification.get_model_instance(model_name, embed_model, usr_dict=data["dictionary"])
    _list = glob("./log/%s/*" % model_name)
    print("Sentence classifier ('q' to quite)")
    print("- saved list (model: %s )" % model_name)
    for __ind, __l in enumerate(_list):
        print("  id: %i, name: %s" % (__ind, __l))
    while True:
        __in = input("select log (id) >>> ")
        if __in == "q":
            break
        elif __in == "":
            continue
        if __in.isdigit():
            if len(_list) <= int(__in):
                print("Index out of range !")
                continue
            model = model(architecture, load_model="%s/model.ckpt" % _list[int(__in)])
            while True:
                __in = input("sentence ('/v' to see the validation) >>> ")
                if __in == "q":
                    sys.exit()
                elif __in == "":
                    continue
                elif __in == "/v":  # check the validation
                    target, label = balanced_validation_split(data["sentence"], data["label"], 0.05)
                    while True:
                        __cnt = input("size of validation (max %i) >>> " % len(label))
                        if __cnt == "":
                            continue
                        elif not __cnt.isdigit():
                            print("non digit value !")
                        elif int(__cnt) > len(label):
                            print("over size.")
                        else:
                            target, label = target[:int(__cnt)], label[:int(__cnt)]
                            break
                    _data = processing(target, pre_process)
                    feed_dict = model_inputs(model, _data)
                    feed_dict[model.is_training] = False
                    prediction = model.sess.run([model.prediction], feed_dict=feed_dict)
                    acc = 0
                    for _p, _l, _s in zip(prediction[0], label, target):
                        _e = int(_p > 0.5)
                        acc += (_e == _l)
                        print("est: %i (%0.3f), true: %i, sentence: %s" % (_e, _p, _l, _s))
                    acc = acc / len(label)
                    print("accuracy:%0.3f" % acc)
                else:
                    _data = processing([__in], pre_process)
                    feed_dict = model_inputs(model, _data)
                    feed_dict[model.is_training] = False
                    prediction = model.sess.run([model.prediction], feed_dict=feed_dict)
                    print(prediction[0] > 0.5, "prob (%0.3f)" % prediction[0])
        else:
            print("Invalid command !")


def processing(x, process):
    if process is not None:
        if type(process) == list:
            return [_process(x) for _process in process]
        else:
            return process(x)
    else:
        return x


def balanced_validation_split(x, y, ratio):
    """y should be 1 dimension array"""
    _ind = [i for i in range(len(x))]
    np.random.seed(0)
    np.random.shuffle(_ind)
    y, x = y[_ind], x[_ind]
    size = int(np.floor(len(x) * ratio) / 2)
    # binary label index
    _y0 = y[y == 0]
    _y1 = y[y == 1]
    _x0 = x[y == 0]
    _x1 = x[y == 1]
    _ind = int(np.min([np.min([len(_y0), len(_y1)]), size]))
    y_valid = np.hstack([_y0[:_ind], _y1[:_ind]])
    if x.ndim == 1:
        x_valid = np.hstack([_x0[:_ind], _x1[:_ind]])
    else:
        x_valid = np.vstack([_x0[:_ind], _x1[:_ind]])
    return x_valid, y_valid


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    _parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    args = get_options(_parser)
    # w2v
    w2v = gensim.models.KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin", binary=True)
    # load model
    controller(args.model, w2v)






