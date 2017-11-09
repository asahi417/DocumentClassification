import os
import logging
import json
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from os.path import abspath as abp, dirname as drn


def mnist_train(epoch, lr, clip):
    """ Train model based on mini-batch of input data.

    :param str model: name of model (cvae_cnn3, cvae_fc3, vae)
    :param int epoch:
    :param float lr: learning rate
    :param float clip: value of gradient clipping
    :return:
    """

    from . import model_info

    # load model
    path = drn(abp(__file__))
    with open("%s/model/%s.json" % (path, model_info["model"])) as f:
        if model_info["model"] == "cvae_cnn3":
            from .model import CvaeCnn3
            _model = CvaeCnn3(network_architecture=json.load(f), learning_rate=lr, max_grad_norm=clip, save_path=path)
            inp_img = True
        elif model_info["model"] == "cvae_fc3":
            from .model import CvaeFc3
            _model = CvaeFc3(network_architecture=json.load(f), learning_rate=lr, max_grad_norm=clip, save_path=path)
            inp_img = False
        else:
            sys.exit("unknown model %s " % model_info["model"])

    path = "%s/%s/" % (model_info["model_path"], model_info["model"])

    # load mnist
    data = read_data_sets('MNIST_data', one_hot=True)
    n = data.train.num_examples
    n_iter = int(n / _model.network_architecture["batch_size"])
    # logger
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    logger = create_log(path+"log")
    logger.info(_model.__doc__)
    logger.info("train: data size(%i), batch num(%i), batch size(%i)"
                % (n, n_iter, _model.network_architecture["batch_size"]))
    result = []
    # Initializing the tensor flow variables
    _model.sess.run(tf.global_variables_initializer())
    for _e in range(epoch):
        _result = []
        for _b in range(n_iter):
            # train
            _x, _y = data.train.next_batch(_model.network_architecture["batch_size"])
            if inp_img:
                _x = np.expand_dims(_x.reshape(_model.network_architecture["batch_size"], 28, 28), 3)
            feed_val = [_model.summary, _model.loss, _model.re_loss, _model.latent_loss, _model.train]
            feed_dict = {_model.x: _x, _model.y: _y}
            summary, loss, re_loss, latent_loss, _ = _model.sess.run(feed_val, feed_dict=feed_dict)
            _result.append([loss, re_loss, latent_loss])
            _model.writer.add_summary(summary, int(_b + _e * _model.network_architecture["batch_size"]))
        _result = np.mean(_result, 0)
        logger.info("epoch %i: loss %0.3f, re loss %0.3f, latent loss %0.3f" % (_e, _result[0], _result[1], _result[2]))
        result.append(_result)
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
