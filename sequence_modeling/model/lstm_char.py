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


class CharLSTM(object):
    """ LSTM classifier with character level feature
        * recurrent dropout and layer norm for LSTM
        * batch norm for full connect
    - inputs:
        - onehot char vector (max word num in sentence, max char num in word, character size)
        - embedded vector of word (max word num in sentence, embedded dimension)
    - input -> bi LSTM x 3 -> last hidden unit -> FC -> output
    - output: one hot vector of label (multi class, 2 dim), 0 or 1 (binary class, 1 dim)
    """

    def __init__(self, network_architecture, learning_rate=0.0001, load_model=None, gradient_clip=None,
                 batch_norm=None, keep_prob=1.0):
        """
        :param dict network_architecture: dictionary with following elements
            n_input: shape of input (list: sequence, feature, channel)
            label_size: unique number of label
            batch_size: size of mini-batch
        :param float learning_rate: default 0.001
        :param float gradient_clip: (option) clipping gradient value
        :param float keep_prob: (option) keep probability of dropout
        :param str load_model: load saved model
        :param float batch_norm: (option) decay for batch norm. ex) default for 0.999
                                 https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        """
        self.network_architecture = network_architecture
        self.binary_class = True if self.network_architecture["label_size"] == 2 else False
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.batch_norm = batch_norm
        self.keep_prob = keep_prob

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
        # tf Graph input (batch, word size, character size, 26)
        self.x_char = tf.placeholder(tf.float32, [None] + self.network_architecture["input_char"], name="input_char")
        self.x_word = tf.placeholder(tf.float32, [None] + self.network_architecture["input_word"], name="input_word")
        if self.binary_class:
            self.y = tf.placeholder(tf.float32, [None], name="output")
        else:
            self.y = tf.placeholder(tf.float32, [None, self.network_architecture["label_size"]], name="output")
        # Regularization
        self.is_train = tf.placeholder_with_default(False, [])
        _keep_prob = tf.where(self.is_train, self.keep_prob, 1.0)
        _layer_norm = self.batch_norm is not None

        with tf.variable_scope("word_level"):
            cell_bw, cell_fw = [], []
            for i in range(1, 4):
                _cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.network_architecture["n_hidden_%i" % i],
                                                              dropout_keep_prob=_keep_prob, layer_norm=_layer_norm)
                cell_fw.append(_cell)
                _cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.network_architecture["n_hidden_%i" % i],
                                                              dropout_keep_prob=_keep_prob, layer_norm=_layer_norm)
                cell_bw.append(_cell)
            cell_bw, cell_fw = tf.contrib.rnn.MultiRNNCell(cell_bw), tf.contrib.rnn.MultiRNNCell(cell_fw)

            _layer = tf.nn.dropout(self.x_word, _keep_prob)
            (output_fw, output_bw), (states_fw, states_bw) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, dtype=tf.float32, inputs=_layer)
            cell_word = tf.concat([states_fw[-1][-1], states_bw[-1][-1]], axis=1)

        with tf.variable_scope("character_level"):
            cell_bw, cell_fw = [], []
            for i in range(1, 4):
                _cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.network_architecture["n_hidden_%i" % i],
                                                              dropout_keep_prob=_keep_prob, layer_norm=_layer_norm)
                cell_fw.append(_cell)
                _cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.network_architecture["n_hidden_%i" % i],
                                                              dropout_keep_prob=_keep_prob, layer_norm=_layer_norm)
                cell_bw.append(_cell)
            cell_bw, cell_fw = tf.contrib.rnn.MultiRNNCell(cell_bw), tf.contrib.rnn.MultiRNNCell(cell_fw)

            _layer = tf.nn.dropout(self.x_char, _keep_prob)
            _layer = self.embedding_char(_layer,
                                         embed_dim_c=self.network_architecture["char_embed_dim"],
                                         embed_dim_w=self.network_architecture["char_cnn_kernel"],
                                         wk=self.network_architecture["char_cnn_unit"])
            (output_fw, output_bw), (states_fw, states_bw) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=_layer, dtype=tf.float32)
            cell_char = tf.concat([states_fw[-1][-1], states_bw[-1][-1]], axis=1)

        _layer = tf.concat([cell_word, cell_char], axis=1)
        _layer = tf.nn.dropout(_layer, _keep_prob)
        _shape = _layer.shape.as_list()

        # Prediction, Loss and Accuracy
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
        
    def embedding_char(self, x, embed_dim_c=5, embed_dim_w=10, wk=5):
        """ Word representation by character level embedding.

        :param x: input (batch, max word num in sentence, max char num in word, vocabulary size)
        :param int embed_dim_c: embedding dimension of word
        :param int embed_dim_w: embedding dimension of character
        :param int wk: word context window
        :return embedding: embedded vector (batch, max word num, embed_dim_w)
        """
        __shape = x.shape.as_list()
        # character level embedding (word size x character size x embed_dim_c)
        embedding = convolution(x, [1, 1, __shape[3], embed_dim_c], [1, 1, 1, 1], bias=False, padding='VALID')
        if self.batch_norm is not None:
            embedding = tf.contrib.layers.batch_norm(embedding, decay=self.batch_norm, is_training=self.is_train)

        # mutual character representation by convolution (word size x character size x embed_dim_w)
        embedding = convolution(embedding, [1, wk, embed_dim_c, embed_dim_w], [1, 1, 1, 1], bias=False, padding='SAME')
        if self.batch_norm is not None:
            embedding = tf.contrib.layers.batch_norm(embedding, decay=self.batch_norm, is_training=self.is_train)

        # word representation by max pool over character (word size x 1 x embed_dim_c)
        embedding = tf.nn.max_pool(embedding, ksize=[1, 1, __shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
        return tf.squeeze(embedding, axis=2)


if __name__ == '__main__':
    import os
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    net = {
        "input_char": [40, 33, 26], "input_word": [40, 300],
        "n_hidden_1": 64,
        "n_hidden_2": 128,
        "n_hidden_3": 256,
        "label_size": 2,
        "batch_size": 100
        }
    CharLSTM(net)
