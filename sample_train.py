import os
import argparse
import json
import numpy as np
import sequence_modeling


def get_options(parser):
    parser.add_argument('model', action='store', nargs='?', const=None, default='cnn', type=str, choices=None,
                        metavar=None, help='Name of model to use. (default: cnn) \n'
                                           '- cnn: \n'
                                           '- lstm: ')
    parser.add_argument('-e', '--epoch', action='store', nargs='?', const=None, default=150, type=int,
                        choices=None, help='Epoch number. (default: 150)', metavar=None)
    parser.add_argument('-b', '--batch', action='store', nargs='?', const=None, default=500, type=int,
                        choices=None, help='Batch size. (default: 500)', metavar=None)
    parser.add_argument('-l', '--lr', action='store', nargs='?', const=None, default=0.001, type=float,
                        choices=None, help='Learning rate. (default: 0.001)', metavar=None)
    parser.add_argument('-c', '--clip', action='store', nargs='?', const=None, default=None, type=float,
                        choices=None, help='Gradient clipping. (default: None)', metavar=None)
    parser.add_argument('-p', '--pad', action='store', nargs='?', const=None, default=40, type=int,
                        choices=None, help='Padding value for word embedding. (default: 40)', metavar=None)
    parser.add_argument('-co', '--cut', action='store', nargs='?', const=None, default=2, type=int,
                        choices=None, help='Cut off for word embedding. (default: 2)', metavar=None)
    parser.add_argument('-k', '--keep', action='store', nargs='?', const=None, default=1, type=float,
                        choices=None, help='Keep rate for Dropout. (default: 1)', metavar=None)
    parser.add_argument('-v', '--valid', action='store', nargs='?', const=None, default=2, type=int,
                        choices=None, help='Validation chunk size. (default: 2)', metavar=None)

    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    _parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    args = get_options(_parser)

    with open("./network_architectures/%s.json" % args.model) as f:
        net = json.load(f)
    # auto encoder model need batch size
    net["batch_size"] = args.batch
    net["n_input"] = [args.pad] + net["n_input"][1:]

    feeder = sequence_modeling.ChunkBatchFeeder(data_path="./data/embed_p%i_c%i" % (args.pad, args.cut), batch_size=args.batch,
                                                chunk_for_validation=args.valid, balance_validation=True)
    sequence_modeling.train_chunk(epoch=args.epoch, clip=args.clip, lr=args.lr, model=args.model, feeder=feeder,
                                  save_path="./log", network_architecture=net, keep_prob=args.keep,
                                  r_keep_prob=args.keep)





