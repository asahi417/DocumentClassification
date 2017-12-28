import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer
import tensorflow.contrib.slim as slim


def convolution(x, weight_shape, stride, padding="SAME", bias=True, batch_norm=False,
                initializer=xavier_initializer_conv2d(seed=0)):
    """ 2d convolution layer
    - weight_shape: width, height, input channel, output channel
    - stride: batch, w, h, c
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    x = tf.nn.conv2d(x, weight, strides=stride, padding=padding)
    if batch_norm:
        return batch_normalization(x)
    elif bias:
        return tf.add(x, tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32))
    else:
        return x


def full_connected(x, weight_shape, initializer=xavier_initializer(seed=0), bias=True, batch_norm=False):
    """ fully connected layer
    - weight_shape: input size, output size
    - priority: batch norm (remove bias) > dropout and bias term
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    x = tf.matmul(x, weight)
    if batch_norm:
        return batch_normalization(x)
    else:
        if bias:
            return tf.add(x, tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32))
        else:
            return x


def batch_normalization(x, epsilon=1e-4):
    batch_size = x.shape.as_list()[0]
    batch_mean, batch_var = tf.nn.moments(x, [0])  # return mu and var for the batch axis
    scale = tf.Variable(tf.ones([batch_size]))
    beta = tf.Variable(tf.zeros([batch_size]))
    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)


class GapCNN(object):
    """ single spatial CNN with temporal max pooling (global max pooling)
    `Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).`

    convolution over all feature (kernel: 12) -> max pooling over all temporal
    - if input [40, 300]: [40, 300, 1] -> [20, 1, 16] -> [1, 1, 16]
    - each convolution layer has [convolution -> activation -> dropout]
    - output: one hot vector of label (multi class, 2 dim), 0 or 1 (binary class, 1 dim)
    """

    def __init__(self, network_architecture, activation=tf.nn.relu, learning_rate=0.001,
                 load_model=None, max_grad_norm=None, keep_prob=1.0, batch_norm=False, lr_schedule=None):
        """
        :param dict network_architecture: dictionary with following elements
            n_input: shape of input (list: sequence, feature, channel)
            label_size: unique number of label
            batch_size: size of mini-batch
        :param activation: activation function (tensor flow function)
        :param float learning_rate:
        :param float keep_prob: (option) keep rate for dropout for last FC.
        :param bool batch_norm: (option) if True, apply BN for training
        :param float max_grad_norm: (option) clipping gradient value
        :param str load_model: (option) load saved model
        :param float lr_schedule: (option) decay rate
        """
        self.network_architecture = network_architecture
        self.binary_class = True if self.network_architecture["label_size"] == 2 else False
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.keep_prob = keep_prob
        self.batch_norm = batch_norm
        self.lr_schedule = lr_schedule

        # Initializer
        if "relu" in self.activation.__name__:
            self.ini_c, self.ini = variance_scaling_initializer(seed=0), variance_scaling_initializer(seed=0)
        else:
            self.ini_c, self.ini = xavier_initializer_conv2d(seed=0), xavier_initializer(seed=0)

        # Create network
        self._create_network()
        # Summary
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        # Launch the session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        # Summary writer for tensor board
        self.summary = tf.summary.merge_all()
        # Load model
        if load_model:
            tf.reset_default_graph()
            self.saver.restore(self.sess, load_model)

    def _create_network(self):
        """ Create Network, Define Loss Function and Optimizer """
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None] + self.network_architecture["n_input"], name="input")
        if self.binary_class:
            self.y = tf.placeholder(tf.float32, [None], name="output")
        else:
            self.y = tf.placeholder(tf.float32, [None, self.network_architecture["label_size"]], name="output")

        self.is_training = tf.placeholder(tf.bool)
        _keep_prob = self.keep_prob if self.is_training is True else 1.0
        _batch_norm = self.batch_norm if self.is_training is True else False

        # CNN over feature
        # -print(self.x.shape)
        _kernel = [12, self.network_architecture["n_input"][1], 1, 16]
        _stride = [2, self.network_architecture["n_input"][1]]
        _layer = convolution(self.x, _kernel, _stride, self.ini_c, batch_norm=_batch_norm)
        _layer = self.activation(_layer)
        _layer = tf.nn.dropout(_layer, _keep_prob)

        # Pooling over all temporal
        # -print(_layer.shape)
        _kernel = [1, _layer.shape.as_list()[1], 1, 1]
        _layer = tf.nn.max_pool(_layer, ksize=_kernel, strides=[1, 1, 1, 1], padding='VALID')
        _layer = tf.nn.dropout(_layer, _keep_prob)

        # Prediction, Loss and Accuracy
        # -print(_layer.shape)
        _layer = slim.flatten(_layer)
        _shape = _layer.shape.as_list()
        if self.binary_class:
            # last layer to get logit and prediction
            _logit = tf.squeeze(full_connected(_layer, [_shape[-1], 1], self.ini, batch_norm=_batch_norm))
            self.prediction = tf.sigmoid(_logit)
            # logistic loss
            _loss = self.y * tf.log(self.prediction + 1e-8) + (1 - self.y) * tf.log(1 - self.prediction + 1e-8)
            self.loss = - tf.reduce_mean(_loss)
            # accuracy
            _prediction = tf.cast((self.prediction > 0.5), tf.float32)
            self.accuracy = 1 - tf.reduce_mean(tf.abs(self.y - _prediction))
        else:
            # last layer to get logit
            # _logit = full_connected(_layer, [_shape[-1], self.network_architecture["label_size"]], self.ini)
            _logit = full_connected(_layer, [_shape[-1], self.network_architecture["label_size"]], self.ini,
                                    batch_norm=_batch_norm)
            self.prediction = tf.nn.softmax(_logit)
            # cross entropy
            self.loss = - tf.reduce_sum(self.y * tf.log(self.prediction + 1e-8))
            # accuracy
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.prediction, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        if self.max_grad_norm:
            _var = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, _var), self.max_grad_norm)
            self.train = optimizer.apply_gradients(zip(grads, _var))
        else:
            self.train = optimizer.minimize(self.loss)
        # saver
        self.saver = tf.train.Saver()


if __name__ == '__main__':
    import os
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    net = {"label_size": 2, "n_input": [40, 300, 1]}
    CNN(net)
