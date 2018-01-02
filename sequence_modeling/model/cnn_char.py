import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer


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


def batch_normalization(x):
    batch_size = x.shape.as_list()[0]
    batch_mean, batch_var = tf.nn.moments(x, [0])  # return mu and var for the batch axis
    scale = tf.Variable(tf.ones([batch_size]))
    beta = tf.Variable(tf.zeros([batch_size]))
    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon=1e-4)


def embedding_char(x, embed_dim_c=5, embed_dim_w=10, wk=5, batch_norm=False):
    """ Word representation by character level embedding.

    :param x: input (batch, max word num in sentence, max char num in word, vocabulary size)
    :param int embed_dim_c: embedding dimension of word
    :param int embed_dim_w: embedding dimension of character
    :param int wk: word context window
    :param bool batch_norm: if False, no batch normalization
    :return embedding: embedded vector (batch, max word num, embed_dim_w)
    """
    __shape = x.shape.as_list()
    # character level embedding (word size x character size x embed_dim_c)
    embedding = convolution(x, [1, 1, __shape[3], embed_dim_c], [1, 1, 1, 1], bias=False, padding='VALID',
                            batch_norm=batch_norm)
    # mutual character representation by convolution (word size x character size x embed_dim_w)
    embedding = convolution(embedding, [1, wk, embed_dim_c, embed_dim_w], [1, 1, 1, 1], padding='SAME',
                            batch_norm=batch_norm)
    # word representation by max pool over character (word size x 1 x embed_dim_c)
    embedding = tf.nn.max_pool(embedding, ksize=[1, 1, __shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
    return tf.squeeze(embedding, axis=2)


def embedding_word(x, embed_dim=30, batch_norm=False):
    """ Word representation by character level embedding.

    :param x: input (batch, max word num in sentence, embedded dim)
    :param int embed_dim: embedding dimension of word
    :param bool batch_norm: if False, no batch normalization
    :return embedding: embedded vector (batch, max word num, embed_dim)
    """
    # (batch, max word num in sentence, vocabulary size) -> (batch, max word num in sentence, embedded dim, 1)
    x = tf.expand_dims(x, 3)
    v_size = x.shape.as_list()[2]
    embedding = convolution(x, [1, v_size, 1, embed_dim], stride=[1, 1, 1, 1], bias=False, padding="VALID",
                            batch_norm=batch_norm)
    return tf.squeeze(embedding, axis=2)


class CharCNN(object):
    """ character-level CNN with BN and scheduled learning rate
    `Dos Santos, Cícero Nogueira, and Maira Gatti. "Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts." COLING. 2014.`

        - inputs:
            - onehot char vector (max word num in sentence, max char num in word, character size)
            - embedded vector of word (max word num in sentence, embedded dimension)
        - network
            - word -(CNN)-> feature by word (w)
            - char -(CNN)-> char embedded vector for each word -(CNN)-> feature by char (c)
            [c, w] -> CNN + max pool over feature -> FC -> output
    """

    def __init__(self, network_architecture, learning_rate=0.0001, load_model=None, max_grad_norm=None):
        """
        :param dict network_architecture: dictionary with following elements
            input_char: shape of char input (list: max word num in sentence, max char num in word, vocabulary size)
            input_word: shape of word input (list: max word num in sentence, vocabulary size)
            label_size: unique number of label
            char_embed_dim: character-level embedding dimension
            char_cnn_unit: channel size of cnn for character to word representation
            char_cnn_kernel: kernel size of cnn for character to word representation
            word_embed_dim: word embedding dimension
            cnn_unit: channel size of cnn for word and character feature
            cnn_kernel: kernel size of cnn for word and character feature
            hidden_unit: hidden unit number for
        :param float learning_rate: default 0.001
        :param float max_grad_norm: (option) clipping gradient value
        :param str load_model: (option) load saved model
        """
        self.network_architecture = network_architecture
        self.binary_class = True if self.network_architecture["label_size"] == 2 else False
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

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
        """ Create Network, Define Loss Function and Optimizer
        input: x_char, x_word, batch_norm, keep_prob, y (training)
        """
        # tf Graph input (batch, word size, character size, 26)
        self.x_char = tf.placeholder(tf.float32, [None] + self.network_architecture["input_char"], name="input_char")
        self.x_word = tf.placeholder(tf.float32, [None] + self.network_architecture["input_word"], name="input_word")
        if self.binary_class:
            self.y = tf.placeholder(tf.float32, [None], name="output")
        else:
            self.y = tf.placeholder(tf.float32, [None, self.network_architecture["label_size"]], name="output")
        # Regularization
        self.batch_norm = tf.placeholder_with_default(False, tf.bool)
        self.keep_prob = tf.placeholder_with_default(1.0, tf.float32)

        # embedding
        embed_char = embedding_char(tf.nn.dropout(self.x_char, self.keep_prob),
                                    embed_dim_c=self.network_architecture["char_embed_dim"],
                                    embed_dim_w=self.network_architecture["char_cnn_kernel"],
                                    wk=self.network_architecture["char_cnn_unit"], batch_norm=self.batch_norm)
        embed_word = embedding_word(tf.nn.dropout(self.x_word, self.keep_prob),
                                    embed_dim=self.network_architecture["word_embed_dim"], batch_norm=self.batch_norm)
        embed = tf.expand_dims(tf.concat([embed_char, embed_word], 2), 3)

        # CNN
        _, __word_size, __feature_size = embed.shape.as_list()[0:3]
        _kernel = [self.network_architecture["cnn_kernel"], __feature_size, 1, self.network_architecture["cnn_unit"]]
        # convolution over feature
        embed = tf.nn.dropout(embed, self.keep_prob)
        _layer = convolution(embed, _kernel, [1, 1, __feature_size, 1], batch_norm=self.batch_norm)

        # max pooling over word
        _layer = tf.nn.max_pool(_layer, ksize=[1, __word_size, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

        # Activation
        _layer = tf.squeeze(_layer, axis=[1, 2])
        _weight = [self.network_architecture["cnn_unit"], self.network_architecture["hidden_unit"]]
        _layer = tf.nn.dropout(_layer, self.keep_prob)
        _layer = full_connected(_layer, _weight, batch_norm=self.batch_norm)
        _layer = tf.nn.tanh(_layer)

        # Prediction, Loss and Accuracy
        if self.binary_class:
            _weight = [self.network_architecture["hidden_unit"], 1]
            _layer = full_connected(_layer, _weight, batch_norm=self.batch_norm)
            _layer = tf.nn.dropout(_layer, self.keep_prob)
            self.prediction = tf.sigmoid(tf.squeeze(_layer, axis=1))
            # logistic loss
            _loss = self.y * tf.log(self.prediction + 1e-8) + (1 - self.y) * tf.log(1 - self.prediction + 1e-8)
            self.loss = - tf.reduce_mean(_loss)
            # accuracy
            _prediction = tf.cast((self.prediction > 0.5), tf.float32)
            self.accuracy = 1 - tf.reduce_mean(tf.abs(self.y - _prediction))
        else:
            _weight = [self.network_architecture["hidden_unit"], self.network_architecture["label_size"]]
            _layer = full_connected(_layer, _weight, batch_norm=self.batch_norm)
            _layer = tf.nn.dropout(_layer, self.keep_prob)
            self.prediction = tf.nn.softmax(_layer)
            # cross entropy
            self.loss = - tf.reduce_sum(self.y * tf.log(self.prediction + 1e-8))
            # accuracy
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.prediction, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Define optimizer and learning rate: lr = lr/lr_decay
        self.lr_decay = tf.placeholder_with_default(1.0, tf.float32)
        optimizer = tf.train.AdamOptimizer(self.learning_rate / self.lr_decay)
        if self.max_grad_norm is not None:
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
    net = {"label_size": 2, "input_char": [40, 33, 26], "input_word": [40, 300],
           "char_embed_dim": 5, "char_cnn_unit": 10, "char_cnn_kernel": 3, "word_embed_dim": 30,
           "cnn_unit": 300, "cnn_kernel": 5,
           "hidden_unit": 300}
    _model = CharCNN(net)

