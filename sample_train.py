import os
import argparse
import sequence_modeling
import gensim
import numpy as np


def get_options(parser):
    parser.add_argument('model', action='store', nargs='?', const=None, default='char_cnn', type=str, choices=None,
                        metavar=None, help='Name of model to use. (default: char_cnn)')
    parser.add_argument('-e', '--epoch', action='store', nargs='?', const=None, default=300, type=int,
                        choices=None, help='Epoch number. (default: 300)', metavar=None)
    parser.add_argument('-b', '--batch', action='store', nargs='?', const=None, default=100, type=int,
                        choices=None, help='Batch size. (default: 100)', metavar=None)
    parser.add_argument('-l', '--lr', action='store', nargs='?', const=None, default=0.00005, type=float,
                        choices=None, help='Learning rate. (default: 0.00005)', metavar=None)
    parser.add_argument('-c', '--clip', action='store', nargs='?', const=None, default=None, type=float,
                        choices=None, help='Gradient clipping. (default: None)', metavar=None)
    parser.add_argument('-k', '--keep', action='store', nargs='?', const=None, default=1, type=float,
                        choices=None, help='Keep rate for Dropout. (default: 1)', metavar=None)
    return parser.parse_args()


def get_model_instance(model_name, label_size=2):
    # set up pre processing
    w2v = gensim.models.KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin", binary=True)
    _pre_process = sequence_modeling.Process("embed", {"length_word": 40, "dim": w2v.vector_size, "model": w2v})

    def _model_inputs(model, x): return {model.x: np.expand_dims(x, 3)}

    if model_name == "char_cnn":
        _net = {"label_size": label_size, "input_char": [40, 33, 26], "input_word": [40, 300],
                "char_embed_dim": 5, "char_cnn_unit": 10, "char_cnn_kernel": 3, "word_embed_dim": 30,
                "cnn_unit": 300, "cnn_kernel": 5, "hidden_unit": 300}
        __model = sequence_modeling.model.CharCNN
        _pre_process = [sequence_modeling.Process("onehot", {"length_word": 40, "length_char": 33}), _pre_process]

        def _model_inputs(model, x): return {model.x_char: x[0], model.x_word: x[1]}
    elif model_name == "gap_cnn":
        _net = {"label_size": label_size, "n_input": [40, 300, 1]}
        __model = sequence_modeling.model.GapCNN
    elif model_name == "deep_cnn":
        _net = {"label_size": label_size, "n_input": [40, 300, 1]}
        __model = sequence_modeling.model.DeepCNN
    elif model_name == "shallow_cnn":
        _net = {"label_size": label_size, "n_input": [40, 300, 1]}
        __model = sequence_modeling.model.ShallowCNN
    elif model_name == "lstm":
        _net = {"label_size": label_size, "n_input": [40, 300], "n_hidden_1": 64, "n_hidden_2": 128, "n_hidden_3": 256}
        __model = sequence_modeling.model.LSTM
        _model_inputs = None
    else:
        raise ValueError("unknown model!")
    return __model, _net, _pre_process, _model_inputs


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    _parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    args = get_options(_parser)

    # path
    path = "./log/%s/l%0.4f_e%i" % (args.model, args.lr, args.epoch)
    if args.clip:
        path += "_c%0.2f" % args.clip
    if args.keep != 1:
        path += "_k%0.2f" % args.keep

    # load model
    _model, net, pre_process, model_inputs = get_model_instance(args.model)
    _model = _model(network_architecture=net, learning_rate=args.lr, max_grad_norm=args.clip, keep_prob=args.keep)

    # load data
    data = sequence_modeling.sst("./data/stanfordSentimentTreebank", binary=True, cut_off=2)
    _x, _y = data["sentence"], data["label"]

    # train
    feeder = sequence_modeling.BatchFeeder(_x, _y, batch_size=args.batch, validation=0.05, process=pre_process)
    sequence_modeling.train(epoch=args.epoch, model=_model, feeder=feeder, save_path=path, model_inputs=model_inputs)





