import os
import argparse
import sequence_modeling
import gensim
from data.util import data_set


def get_options(parser):
    parser.add_argument('model', action='store', nargs='?', const=None, default='char_cnn', type=str, choices=None,
                        metavar=None, help='Name of model to use. (default: char_cnn)')
    parser.add_argument('-e', '--epoch', action='store', nargs='?', const=None, default=100, type=int,
                        choices=None, help='Epoch number. (default: 100)', metavar=None)
    parser.add_argument('-b', '--batch', action='store', nargs='?', const=None, default=100, type=int,
                        choices=None, help='Batch size. (default: 100)', metavar=None)
    parser.add_argument('-l', '--lr', action='store', nargs='?', const=None, default=0.0001, type=float,
                        choices=None, help='Learning rate. (default: 0.0001)', metavar=None)
    parser.add_argument('-c', '--clip', action='store', nargs='?', const=None, default=None, type=float,
                        choices=None, help='Gradient clipping. (default: None)', metavar=None)
    parser.add_argument('-k', '--keep', action='store', nargs='?', const=None, default=1.0, type=float,
                        choices=None, help='Keep rate for Dropout. (default: 1)', metavar=None)
    parser.add_argument('-n', '--norm', action='store', nargs='?', const=None, default=False, type=bool,
                        choices=None, help='Batch normalization (default: False)', metavar=None)
    parser.add_argument('-d', '--decay_learning_rate', action='store', nargs='?', const=None, default=None, type=float,
                        choices=None, help='Decay learning rate (default: None)', metavar=None)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    _parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    args = get_options(_parser)

    path = "./log/%s/l%0.6f_e%i_b%i" % (args.model, args.lr, args.epoch, args.batch)
    if args.clip:
        path += "_c%0.2f" % args.clip
    if args.keep != 1.0:
        path += "_k%0.2f" % args.keep
    if args.norm:
        path += "_norm"
    if args.decay_learning_rate is not None:
        path += "_d%0.2f" % args.decay_learning_rate

    # word2vec
    embedding_model = \
        gensim.models.KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin", binary=True)

    # load data
    data = data_set()
    _x, _y = data["sentence"], data["label"]

    # load model
    model, net, pre_process, model_inputs = sequence_modeling.get_model_instance(args.model, embedding_model)
    if "lstm" in args.model:
        model = model(network_architecture=net, learning_rate=args.lr, max_grad_norm=args.clip, keep_prob=args.keep,
                      lr_schedule=args.decay_learning_rate, layer_norm=args.norm)
    else:
        model = model(network_architecture=net, learning_rate=args.lr, max_grad_norm=args.clip, keep_prob=args.keep,
                      lr_schedule=args.decay_learning_rate, batch_norm=args.norm)

    # train
    feeder = sequence_modeling.BatchFeeder(_x, _y, batch_size=args.batch, validation=0.05, process=pre_process)
    sequence_modeling.train(epoch=args.epoch, model=model, feeder=feeder, save_path=path, model_inputs=model_inputs)





