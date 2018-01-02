import os
import logging
import argparse
import sequence_modeling
import gensim
from data.util import data_set


def train(epoch, model, feeder, model_inputs, save_path="./", batch_norm=False, keep_prob=1.0, lr_decay=1.0):
    """ Train model based on mini-batch of input data.

    :param model: model instance
    :param str save_path: Path to save
    :param int epoch:
    :param feeder: Feeding data.
    :param model_inputs: (optional) Input data format for `model`. For instance
            def model_inputs(model, x):
                return {model.x_char: x[0], model.x_word: x[1]}
        This example is used when char and word vector is fed through the `feeder`. This function has to return dict.
        By default, def model_inputs(model, x): return {model.x: x}
    :param bool batch_norm: "True" to use batch norm
    :param float keep_prob: dropout keep prob
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
            feed_dict = model_inputs(model, _x)
            feed_dict[model.y] = _y
            if batch_norm:
                feed_dict[model.batch_norm] = batch_norm
            if keep_prob != 1.0:
                feed_dict[model.keep_prob] = keep_prob
            if lr_decay != 1.0:
                feed_dict[model.lr_decay] = lr_decay
            if model.lr_schedule is not None:
                feed_dict[model.lr_index] = np.ceil(_e / 100) - 1  # every 100 epoch, (decay)**lr_index
            loss, acc, _ = model.sess.run([model.loss, model.accuracy, model.train], feed_dict=feed_dict)
            _result.append([loss, acc])

        _result_valid = []
        for _b in range(feeder.iterator_length_valid):  # Test
            _x, _y = feeder.next_valid()
            feed_dict = model_inputs(model, _x)
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
            np.savez("%s/progress-%i-acc.npz" % (save_path, _e), loss=np.array(result), clip=model.max_grad_norm,
                     learning_rate=model.learning_rate, epoch=epoch)
    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    feeder.finalize()  # destructor of feeder
    np.savez("%s/statistics.npz" % save_path, loss=np.array(result), learning_rate=model.learning_rate, epoch=epoch,
             clip=model.max_grad_norm)


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
    model_information = sequence_modeling.get_model_instance(args.model, embedding_model, args.lr, args.clip)
    # train
    data_feeder = sequence_modeling.BatchFeeder(_x, _y, batch_size=args.batch, validation=0.05,
                                                process=model_information["processing"])
    train(model=model_information["model_instance"], model_inputs=model_information["input_format"],
          epoch=args.epoch, feeder=data_feeder, save_path=path)





