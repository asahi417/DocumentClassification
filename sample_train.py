import os
import logging
import argparse
import tensorflow as tf
import numpy as np
import sequence_modeling
import gensim
from data.util import data_set


def train(epoch, model, feeder, input_format, save_path="./", lr_decay=1.0, test=False):
    """ Train model based on mini-batch of input data.

    :param model: model instance
    :param str save_path: Path to save
    :param int epoch:
    :param feeder: Feeding data.
    :param input_format: (optional) Input data format for `model`. For instance
            def input_format(model, x):
                return {model.x_char: x[0], model.x_word: x[1]}
        This example is used when char and word vector is fed through the `feeder`. This function has to return dict.
        By default, def input_format(model, x): return {model.x: x}
    :param float lr_decay: learning rate will be divided by lr_decay each epoch
    """

    # logger
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    logger = create_log("%s/log" % save_path)
    logger.info(model.__doc__)
    logger.info("train: epoch (%i), size (%i), iteration in an epoch (%i), batch size(%i)"
                % (epoch, feeder.n, feeder.iterator_length, feeder.batch_size))
    logger.info("validation: size (%i)" % feeder.n_valid)
    result = []

    # Initializing the tensor flow variables
    tf_writer = tf.summary.FileWriter(save_path, model.sess.graph)
    model.sess.run(tf.global_variables_initializer())
    for _e in range(epoch):

        _result = []
        for _b in range(feeder.iterator_length):  # Train
            _x, _y = feeder.next()
            feed_dict = input_format(model, _x)
            feed_dict[model.y] = _y
            feed_dict[model.is_train] = True
            if lr_decay != 1.0:
                feed_dict[model.lr_decay] = lr_decay ** (np.ceil(_e / 100) - 1)  # every 100 epoch, (decay) ** lr_index
            loss, acc, _ = model.sess.run([model.loss, model.accuracy, model.train], feed_dict=feed_dict)
            _result.append([loss, acc])
            if test:
                logger.info("iter %i: acc %0.3f, loss %0.3f" % (_b, loss, acc))

        _result_valid = []
        for _b in range(feeder.iterator_length_valid):  # Test
            _x, _y = feeder.next_valid()
            feed_dict = input_format(model, _x)
            feed_dict[model.y] = _y
            summary, loss, acc = model.sess.run([model.summary, model.loss, model.accuracy], feed_dict=feed_dict)
            _result_valid.append([loss, acc])
            tf_writer.add_summary(summary, int(_e))

        _result_full = np.append(np.mean(_result, 0), np.mean(_result_valid, 0))
        logger.info("epoch %i: acc %0.3f, loss %0.3f, train acc %0.3f, train loss %0.3f"
                    % (_e, _result_full[3], _result_full[2], _result_full[1], _result_full[0]))
        result.append(_result_full)
        if _e % 50 == 0:
            model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, _e))
            np.savez("%s/progress-%i-acc.npz" % (save_path, _e), loss=np.array(result))
    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    feeder.finalize()  # destructor of feeder
    np.savez("%s/statistics.npz" % save_path, loss=np.array(result))


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


def get_options(parser):
    share_param = {'nargs': '?', 'action': 'store', 'const': None, 'choices': None, 'metavar': None}
    parser.add_argument('model', help='Name of model to use. (default: cnn_char)',
                        default='cnn_char', type=str, **share_param)
    parser.add_argument('-e', '--epoch', help='Epoch number. (default: 500)',
                        default=500, type=int, **share_param)
    parser.add_argument('-b', '--batch', help='Batch size. (default: 100)',
                        default=100, type=int, **share_param)
    parser.add_argument('-l', '--lr', help='Learning rate. (default: 0.0001)',
                        default=0.0001, type=float, **share_param)
    parser.add_argument('-c', '--clip', help='Gradient clipping. (default: None)',
                        default=None, type=float, **share_param)
    parser.add_argument('-k', '--keep', help='Keep rate for Dropout. (default: 1.0)',
                        default=1.0, type=float, **share_param)
    parser.add_argument('-n', '--norm', help='Decay for batch normalization. if batch is 100, 0.95 (default: None)',
                        default=None, type=float, **share_param)
    parser.add_argument('-d', '--decay_lr', help='Decay index for learning rate (default: 1.0)',
                        default=1.0, type=float, **share_param)
    return parser.parse_args()


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    _parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    args = get_options(_parser)

    # save path
    path = "./log/%s/l%0.6f_e%i_b%i" % (args.model, args.lr, args.epoch, args.batch)
    if args.decay_lr != 1.0:  # learning rate decaying
        path += "_d%0.2f" % args.decay_lr
    if args.clip is not None:  # gradient clipping
        path += "_c%0.2f" % args.clip
    if args.norm is not None:  # batch normalization
        path += "_n%0.3f" % args.norm
    if args.keep != 1.0:  # dropout
        path += "_k%0.2f" % args.keep

    # word2vec
    embedding_model = \
        gensim.models.KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin", binary=True)

    # load data
    data = data_set()

    # load model
    model_instance = sequence_modeling.get_model_instance(model_name=args.model,
                                                          embedding_model=embedding_model,
                                                          learning_rate=args.lr,
                                                          gradient_clip=args.clip,
                                                          batch_norm=args.norm,
                                                          keep_prob=args.keep)

    # train
    data_feeder = sequence_modeling.BatchFeeder(data["sentence"], data["label"], batch_size=args.batch,
                                                validation=0.05, process=model_instance["processing"])
    train(model=model_instance["model"], input_format=model_instance["input_format"],
          epoch=args.epoch, feeder=data_feeder, save_path=path, lr_decay=args.decay_lr,
          test=False)
