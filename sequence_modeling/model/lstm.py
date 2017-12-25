import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer


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


class LSTM(object):
    """ LSTM classifier
        * recurrent dropout and layer norm for LSTM
        * batch norm for full connect
    - input -> bi LSTM x 3 -> last hidden unit -> FC -> output
    - output: one hot vector of label (multi class, 2 dim), 0 or 1 (binary class, 1 dim)
    """

    def __init__(self, network_architecture, learning_rate=0.0001,
                 load_model=None, max_grad_norm=None, keep_prob=1.0, lr_schedule=False, layer_norm=False):
        """
        :param dict network_architecture: dictionary with following elements
            n_input: shape of input (list: sequence, feature, channel)
            label_size: unique number of label
            batch_size: size of mini-batch
        :param float learning_rate:
        :param str load_model: load saved model
        :param bool lr_schedule: if True, decay learning rate x 10^-1 by each 100 epoch
        """
        self.network_architecture = network_architecture
        self.binary_class = True if self.network_architecture["label_size"] == 2 else False
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.keep_prob = keep_prob
        self.lr_schedule = lr_schedule
        self.layer_norm = layer_norm

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
        # input: length, channel
        self.x = tf.placeholder(tf.float32, [None] + self.network_architecture["n_input"], name="input")
        if self.binary_class:
            self.y = tf.placeholder(tf.float32, [None], name="output")
        else:
            self.y = tf.placeholder(tf.float32, [None, self.network_architecture["label_size"]], name="output")
        self.is_training = tf.placeholder(tf.bool)

        _keep_prob = self.keep_prob if self.is_training is True else 1.0
        _layer_norm = self.layer_norm if self.is_training is True else False

        cell_bw, cell_fw = [], []
        for i in range(1, 4):
            _cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.network_architecture["n_hidden_%i" % i],
                                                          dropout_keep_prob=_keep_prob, layer_norm=_layer_norm)
            cell_fw.append(_cell)

            _cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.network_architecture["n_hidden_%i" % i],
                                                          dropout_keep_prob=_keep_prob, layer_norm=_layer_norm)
            cell_bw.append(_cell)
        cell_bw, cell_fw = tf.contrib.rnn.MultiRNNCell(cell_bw), tf.contrib.rnn.MultiRNNCell(cell_fw)

        _layer = tf.nn.dropout(self.x, _keep_prob)
        (output_fw, output_bw), (states_fw, states_bw) = \
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=_layer, dtype=tf.float32)
        cell = tf.concat([states_fw[-1][-1], states_bw[-1][-1]], axis=1)
        _shape = cell.shape.as_list()

        # Prediction, Loss and Accuracy
        if self.binary_class:
            # last layer to get logit and prediction
            _logit = tf.squeeze(full_connected(cell, [_shape[-1], 1], batch_norm=_layer_norm))
            self.prediction = tf.sigmoid(_logit)
            # logistic loss
            _loss = self.y * tf.log(self.prediction + 1e-8) + (1 - self.y) * tf.log(1 - self.prediction + 1e-8)
            self.loss = - tf.reduce_mean(_loss)
            # accuracy
            _prediction = tf.cast((self.prediction > 0.5), tf.float32)
            self.accuracy = 1 - tf.reduce_mean(tf.abs(self.y - _prediction))
        else:
            # last layer to get logit
            _logit = full_connected(cell, [_shape[-1], self.network_architecture["label_size"]], batch_norm=_layer_norm)
            self.prediction = tf.nn.softmax(_logit)
            # cross entropy
            self.loss = - tf.reduce_sum(self.y * tf.log(self.prediction + 1e-8))
            # accuracy
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.prediction, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Define optimizer
        if self.lr_schedule is not None:
            self.lr_index = tf.placeholder(tf.float32)
            lr = self.learning_rate * self.lr_schedule ** self.lr_index
        else:
            lr = self.learning_rate
        optimizer = tf.train.AdamOptimizer(lr)
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
        "n_input": [10, 300],
        "n_hidden_1": 64,
        "n_hidden_2": 128,
        "n_hidden_3": 256,
        "label_size": 2,
        "batch_size": 100
        }
    LSTM(net)
