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
    parser.add_argument('-l', '--lr', action='store', nargs='?', const=None, default=0.005, type=float,
                        choices=None, help='Learning rate. (default: 0.005)', metavar=None)
    parser.add_argument('-c', '--clip', action='store', nargs='?', const=None, default=None, type=float,
                        choices=None, help='Gradient clipping. (default: None)', metavar=None)
    parser.add_argument('-p', '--pad', action='store', nargs='?', const=None, default=30, type=int,
                        choices=None, help='Gradient clipping. (default: None)', metavar=None)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    _parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    args = get_options(_parser)

    with open("./network_architectures/%s.json" % args.model) as f:
        net = json.load(f)

    feeder = sequence_modeling.ChunkBatchFeeder(data_path="./data/embed_%i" % args.pad, batch_size=net["batch_size"],
                                                chunk_for_validation=2)
    sequence_modeling.train_chunk(epoch=args.epoch, clip=args.clip, lr=args.lr, model=args.model,
                                  feeder=feeder, save_path="./log", network_architecture=net)





