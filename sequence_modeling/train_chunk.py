import os
import logging
import json
import sys
import numpy as np
import tensorflow as tf
from os.path import abspath as abp, dirname as drn


def train_chunk(epoch, lr, clip, model, feeder, save_path="./", network_architecture=None):
    """ Train model based on mini-batch of input data.

    :param str model: name of model (cnn, lstm)
    :param int epoch:
    :param str network_architecture:
    :param float lr: learning rate
    :param float clip: value of gradient clipping
    :param feeder:
    :param str save_path: path to save
    :return:
    """

    # load model
    path = drn(abp(__file__))
    if not network_architecture:
        with open("%s/model/%s.json" % (path, model)) as f:
            network_architecture = json.load(f)
    if model == "cnn":
        from .model import CNN
        _model = CNN(network_architecture=network_architecture, learning_rate=lr, max_grad_norm=clip, save_path=path)
        volumed_input = True
    elif model == "lstm":
        from .model import LSTM
        _model = LSTM(network_architecture=network_architecture, learning_rate=lr, max_grad_norm=clip, save_path=path)
        volumed_input = False
    else:
        sys.exit("unknown model %s " % model)

    path = "%s/%s/" % (save_path, model)

    # logger
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    logger = create_log(path+"log")
    logger.info(_model.__doc__)
    logger.info("train: epoch (%i), iteration in batch (%i), batch size(%i)"
                % (epoch, feeder.iteration_in_epoch, feeder.batch_size))

    result = []
    # Initializing the tensor flow variables
    _model.sess.run(tf.global_variables_initializer())
    for _e in range(epoch):
        _result = []
        for _b in range(feeder.iteration_in_epoch):
            # train
            _x, _y = feeder.next()
            _x = np.expand_dims(_x, 3) if volumed_input else _x
            feed_val = [_model.summary, _model.loss, _model.accuracy, _model.train]
            feed_dict = {_model.x: _x, _model.y: _y, _model.is_training: True}
            summary, loss, acc, _ = _model.sess.run(feed_val, feed_dict=feed_dict)
            # print(pred.shape, np.sum(pred == 0), np.sum(pred == 1), np.sum(_y == 0), np.sum(_y == 1))
            # print(pred[0:10])
            _result.append([loss, acc])
            _model.writer.add_summary(summary, int(_b + _e * _model.network_architecture["batch_size"]))

        # validation
        _result_valid = []
        for _b in range(feeder.iteration_in_epoch_valid):
            _x_valid, _y_valid = feeder.next_valid()
            _x_valid = np.expand_dims(_x_valid, 3) if volumed_input else _x_valid
            feed_dict = {_model.x: _x_valid, _model.y: _y_valid, _model.is_training: False}
            loss, acc = _model.sess.run([_model.loss, _model.accuracy], feed_dict=feed_dict)
            _result_valid.append([loss, acc])
        # _result_valid = np.mean(_result_valid, 0)
        _result_full = np.append(np.mean(_result, 0), np.mean(_result_valid, 0))
        logger.info("epoch %i: acc %0.3f, loss %0.3f, train acc %0.3f, train loss %0.3f"
                    % (_e, _result_full[3], _result_full[2], _result_full[1], _result_full[0]))
        result.append(_result_full)
        if _e % 50 == 0:
            _model.saver.save(_model.sess, "%s/progress-%i-model.ckpt" % (path, _e))
            np.savez("%s/progress-%i-acc.npz" % (path, _e), loss=np.array(result), clip=_model.max_grad_norm,
                     learning_rate=_model.learning_rate, epoch=epoch)
    _model.saver.save(_model.sess, "%s/model.ckpt" % path)
    np.savez("%s/statistics.npz" % path, loss=np.array(result), learning_rate=_model.learning_rate, epoch=epoch,
             clip=_model.max_grad_norm)


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

