import os
import argparse
import json
import numpy as np
import sequence_modeling


if __name__ == '__main__':
    feeder = sequence_modeling.ChunkBatchFeeder(data_path="./data/embed_30", batch_size=100, chunk_for_validation=2,
                                                balance_validation=True)
    for i in range(feeder.iteration_in_epoch_valid):
        x, y = feeder.next_valid()
        print(i, np.sum(y == 0), np.sum(y == 1))
    # feeder = sequence_modeling.train_chunk(epoch=args.epoch, clip=args.clip, lr=args.lr, model=args.model,
    #                                        feeder=feeder, save_path="./log", network_architecture=net)





