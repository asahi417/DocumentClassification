import os
import argparse
import numpy as np
import sequence_modeling as sed


def get_options(parser):
    parser.add_argument('model', action='store', nargs='?', const=None, default='cnn', type=str, choices=None,
                        metavar=None, help='Name of model to use. (default: cnn) \n'
                                           '- cnn: \n'
                                           '- lstm: ')
    parser.add_argument('-e', '--epoch', action='store', nargs='?', const=None, default=150, type=int,
                        choices=None, help='Epoch number. (default: 150)', metavar=None)
    parser.add_argument('-l', '--lr', action='store', nargs='?', const=None, default=0.005, type=float,
                        choices=None, help='Learning rate. (default: 0.005)', metavar=None)
    parser.add_argument('-c', '--clip', action='store', nargs='?', const=None, default=None, type=float,
                        choices=None, help='Gradient clipping. (default: None)', metavar=None)
    return parser.parse_args()


if __name__ == '__main__':
    _length = 30
    _sst_path = "./data/stanfordSentimentTreebank"
    _embed_path = "./data/GoogleNews-vectors-negative300.bin"

    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    _parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    args = get_options(_parser)

    sed.sst_vectorize(_length, _sst_path, _embed_path, "./data")
    data = np.load("./data/vectorized_data_%i.npz" % _length)
    feeder = sed.train(epoch=args.epoch, clip=args.clip, lr=args.lr, model=args.model,
                       x=data["sentence"], y=data["label"], valid=0.3, save_path="./log")





