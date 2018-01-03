import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer


def convolution(x, weight_shape, stride, padding="SAME", bias=True, initializer=xavier_initializer_conv2d(seed=0)):
    """ 2d convolution layer
    - weight_shape: width, height, input channel, output channel
    - stride: batch, w, h, c
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    x = tf.nn.conv2d(x, weight, strides=stride, padding=padding)
    if bias:
        return tf.add(x, tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32))
    else:
        return x


def full_connected(x, weight_shape, bias=True, initializer=xavier_initializer(seed=0)):
    """ fully connected layer
    - weight_shape: input size, output size
    - priority: batch norm (remove bias) > dropout and bias term
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    x = tf.matmul(x, weight)
    if bias:
        return tf.add(x, tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32))
    else:
        return x


class GapCNN(object):
    """ single spatial CNN with temporal max pooling (global max pooling)
    `Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).`

    convolution over all feature (kernel: 12) -> max pooling over all temporal
    - if input [40, 300]: [40, 300, 1] -> [20, 1, 16] -> [1, 1, 16]
    - each convolution layer has [convolution -> activation -> dropout]
    - output: one hot vector of label (multi class, 2 dim), 0 or 1 (binary class, 1 dim)
    """

    def __init__(self, network_architecture, learning_rate=0.0001, load_model=None, gradient_clip=None,
                 batch_norm=None, keep_prob=1.0):
        """
        :param dict network_architecture: dictionary with following elements
            n_input: shape of input (list: sequence, feature, channel)
            label_size: unique number of label
            batch_size: size of mini-batch
        :param float learning_rate:
        :param float gradient_clip: (option) clipping gradient value
        :param float keep_prob: (option) keep rate for dropout for last FC.
        :param str load_model: (option) load saved model
        :param float batch_norm: (option) decay for batch norm. ex) default for 0.999
                                 https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        """
        self.network_architecture = network_architecture
        self.binary_class = True if self.network_architecture["label_size"] == 2 else False
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.keep_prob = keep_prob
        self.batch_norm = batch_norm

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

        self.is_train = tf.placeholder_with_default(False, [])
        _keep_prob = tf.where(self.is_train, self.keep_prob, 1.0)

        # CNN over feature
        _kernel = [12, self.network_architecture["n_input"][1], 1, 16]
        _stride = [2, self.network_architecture["n_input"][1]]
        _layer = convolution(self.x, _kernel, _stride)
        if self.batch_norm is not None:
            _layer = tf.contrib.layers.batch_norm(_layer, decay=self.batch_norm, is_training=self.is_train)
        _layer = tf.nn.relu(_layer)
        _layer = tf.nn.dropout(_layer, _keep_prob)

        # Pooling over all temporal
        _kernel = [1, _layer.shape.as_list()[1], 1, 1]
        _layer = tf.nn.max_pool(_layer, ksize=_kernel, strides=[1, 1, 1, 1], padding='VALID')

        # Prediction, Loss and Accuracy
        _shape = _layer.shape.as_list()
        if self.binary_class:
            _weight = [_shape[-1], 1]

            _layer = tf.nn.dropout(_layer, _keep_prob)
            _layer = full_connected(_layer, _weight)

            if self.batch_norm is not None:
                _layer = tf.contrib.layers.batch_norm(_layer, decay=self.batch_norm, is_training=self.is_train)

            self.prediction = tf.sigmoid(tf.squeeze(_layer, axis=1))
            # logistic loss
            _loss = self.y * tf.log(self.prediction + 1e-8) + (1 - self.y) * tf.log(1 - self.prediction + 1e-8)
            self.loss = - tf.reduce_mean(_loss)
            # accuracy
            _prediction = tf.cast((self.prediction > 0.5), tf.float32)
            self.accuracy = 1 - tf.reduce_mean(tf.abs(self.y - _prediction))
        else:
            _weight = [_shape[-1], self.network_architecture["label_size"]]

            _layer = tf.nn.dropout(_layer, _keep_prob)
            _layer = full_connected(_layer, _weight)

            if self.batch_norm is not None:
                _layer = tf.contrib.layers.batch_norm(_layer, decay=self.batch_norm, is_training=self.is_train)

            self.prediction = tf.nn.softmax(_layer)
            # cross entropy
            self.loss = - tf.reduce_sum(self.y * tf.log(self.prediction + 1e-8))
            # accuracy
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.prediction, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Define optimizer and learning rate: lr = lr/lr_decay
        # need for BN -> https://www.tensorflow.org/versions/r1.1/api_docs/python/tf/contrib/layers/batch_norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.lr_decay = tf.placeholder_with_default(1.0, [])
            optimizer = tf.train.AdamOptimizer(self.learning_rate / self.lr_decay)
            if self.gradient_clip is not None:
                _var = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, _var), self.gradient_clip)
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
