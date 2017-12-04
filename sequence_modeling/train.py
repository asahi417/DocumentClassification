import os
import logging
import numpy as np
import tensorflow as tf


def model_inputs_default(model, x): return {model.x: x}


def train(epoch, model, feeder, model_inputs=None, save_path="./"):
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
    """

    model_inputs = model_inputs_default if model_inputs is None else model_inputs
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
            feed_dict[model.is_training] = True
            loss, acc, _ = model.sess.run([model.loss, model.accuracy, model.train], feed_dict=feed_dict)
            _result.append([loss, acc])

        _result_valid = []
        for _b in range(feeder.iterator_length_valid):  # Test
            _x, _y = feeder.next_valid()
            feed_dict = model_inputs(model, _x)
            feed_dict[model.y] = _y
            feed_dict[model.is_training] = False
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

