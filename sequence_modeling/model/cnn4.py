import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer
import tensorflow.contrib.slim as slim


def convolution(x, weight_shape, stride, initializer, padding="SAME"):
    """ 2d convolution layer
    - weight_shape: width, height, input channel, output channel
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
    return tf.add(tf.nn.conv2d(x, weight, strides=[1, stride[0], stride[1], 1], padding=padding), bias)


def full_connected(x, weight_shape, initializer):
    """ fully connected layer
    - weight_shape: input size, output size
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
    return tf.add(tf.matmul(x, weight), bias)


class CNN(object):
    """ CNN classifier ver 4
    CNN over feature  -> CNN over sequence -> FC x2
    - output: one hot vector of label (multi class, 2 dim), 0 or 1 (binary class, 1 dim)
    """

    def __init__(self, network_architecture, activation=tf.nn.relu, learning_rate=0.001,
                 save_path=None, load_model=None, max_grad_norm=None, keep_prob=0.9):
        """
        :param dict network_architecture: dictionary with following elements
            n_input: shape of input (list: sequence, feature, channel)
            label_size: unique number of label
            batch_size: size of mini-batch
        :param activation: activation function (tensor flow function)
        :param float learning_rate:
        :param str save_path: path to save
        :param str load_model: load saved model
        """
        self.network_architecture = network_architecture
        self.binary_class = True if self.network_architecture["label_size"] == 2 else False
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.keep_prob = keep_prob

        # Initializer
        if "relu" in self.activation.__name__:
            self.ini_c, self.ini = variance_scaling_initializer(), variance_scaling_initializer()
        else:
            self.ini_c, self.ini = xavier_initializer_conv2d(), xavier_initializer()

        # Create network
        self._create_network()

        # Summary
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        # Launch the session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        # Summary writer for tensor board
        self.summary = tf.summary.merge_all()
        if save_path:
            self.writer = tf.summary.FileWriter(save_path, self.sess.graph)
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
        _keep_prob = self.keep_prob if self.is_training is True else 1

        # CNN over feature
        print(self.x.shape)
        _kernel = [12, self.network_architecture["n_input"][1], 1, 16]
        _stride = [6, 1]
        _layer = convolution(self.x, _kernel, _stride, self.ini_c, padding="VALID")
        _layer = self.activation(_layer)
        print(_layer.shape)

        # CNN over sequential direction
        _kernel = [_layer.shape.as_list()[1], 1, 16, 32]
        _layer = convolution(_layer, _kernel, [1, 1], self.ini_c, padding="VALID")

        print(_layer.shape)
        _layer = slim.flatten(_layer)
        _layer = tf.nn.dropout(_layer, _keep_prob)
        print(_layer.shape)
        # Prediction, Loss and Accuracy
        _shape = _layer.shape.as_list()
        _layer = tf.squeeze(full_connected(_layer, [_shape[-1], 8], self.ini))
        _layer = tf.nn.dropout(_layer, _keep_prob)
        if self.binary_class:
            # last layer to get logit and prediction
            _logit = tf.squeeze(full_connected(_layer, [8, 1], self.ini))
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
            _logit = full_connected(_layer, [8, self.network_architecture["label_size"]], self.ini)
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
    net = {
        "label_size": 2,
        "n_input": [40, 300, 1]
    }
    CNN(net)
