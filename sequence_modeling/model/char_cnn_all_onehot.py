import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer
import tensorflow.contrib.slim as slim


def convolution(x, weight_shape, stride, initializer=xavier_initializer_conv2d(), padding="SAME", bias=True):
    """ 2d convolution layer
    - weight_shape: width, height, input channel, output channel
    - stride: batch, w, h, c
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    if bias:
        bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
        return tf.add(tf.nn.conv2d(x, weight, strides=stride, padding=padding), bias)
    else:
        return tf.nn.conv2d(x, weight, strides=stride, padding=padding)


def full_connected(x, weight_shape, initializer=xavier_initializer()):
    """ fully connected layer
    - weight_shape: input size, output size
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
    return tf.add(tf.matmul(x, weight), bias)


def embedding_char(x, embed_dim_c=5, embed_dim_w=10, wk=5):
    """ Word representation by character level embedding.

    :param x: input (batch, max word num in sentence, max char num in word, vocabulary size)
    :param int embed_dim_c: embedding dimension of word
    :param int embed_dim_w: embedding dimension of character
    :param int wk: word context window
    :return embedding: embedded vector (batch, max word num, embed_dim_w)
    """
    __shape = x.shape.as_list()
    # character level embedding (word size x character size x embed_dim_c)
    embedding = convolution(x, [1, 1, __shape[3], embed_dim_c], stride=[1, 1, 1, 1], bias=False, padding='VALID')
    # mutual character representation by convolution (word size x character size x embed_dim_w)
    embedding = convolution(embedding, [1, wk, embed_dim_c, embed_dim_w], [1, 1, 1, 1], padding='SAME')
    # word representation by max pool over character (word size x 1 x embed_dim_c)
    embedding = tf.nn.max_pool(embedding, ksize=[1, 1, __shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')
    return tf.squeeze(embedding, axis=2)


def embedding_word(x, embed_dim=30):
    """ Word representation by character level embedding.

    :param x: input (batch, max word num in sentence, vocabulary size)
    :param int embed_dim: embedding dimension of word
    :return embedding: embedded vector (batch, max word num, embed_dim)
    """
    # (batch, max word num in sentence, vocabulary size) -> (batch, max word num in sentence, vocabulary size, 1)
    x = tf.expand_dims(x, 3)
    v_size = x.shape.as_list()[2]
    embedding = convolution(x, [1, v_size, 1, embed_dim], stride=[1, 1, 1, 1], bias=False, padding="VALID")
    return tf.squeeze(embedding, axis=2)


class CharCNN(object):
    """ character-level CNN
        - inputs: onehot vector of word and character
            word -(CNN)-> word embedded vector (w)
            char -(CNN)-> char embedded vector for each word -(CNN)-> feature by char (c)
            [c, w] -> CNN + max pool over feature -> FC -> output
    """

    def __init__(self, network_architecture, learning_rate=0.001, load_model=None, max_grad_norm=None,
                 activation=tf.nn.tanh, keep_prob=1):
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
        :param float keep_prob: keep rate for dropout for last FC. default 1, which means no dropout
        :param float max_grad_norm: if not None, clipping gradient value
        :param activation: (option) activation function (tensor flow function). default is tanh
        :param str load_model: (option) load saved model
        """
        self.network_architecture = network_architecture
        self.binary_class = True if self.network_architecture["label_size"] == 2 else False
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.keep_prob = keep_prob
        self.activation = activation

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
        self.is_training = tf.placeholder(tf.bool)
        _keep_prob = self.keep_prob if self.is_training is True else 1

        # embedding
        embed_char = embedding_char(self.x_char, embed_dim_c=self.network_architecture["char_embed_dim"],
                                    embed_dim_w=self.network_architecture["char_cnn_kernel"],
                                    wk=self.network_architecture["char_cnn_unit"])
        embed_word = embedding_word(self.x_word, embed_dim=self.network_architecture["word_embed_dim"])
        embed = tf.expand_dims(tf.concat([embed_char, embed_word], 2), 3)
        # print("embedding:", embed.shape)

        # CNN
        _, __word_size, __feature_size = embed.shape.as_list()[0:3]
        _kernel = [self.network_architecture["cnn_kernel"], __feature_size, 1,
                   self.network_architecture["cnn_unit"]]
        # convolution over feature
        _layer = convolution(embed, _kernel, [1, 1, __feature_size, 1])
        # max pooling over word
        _layer = tf.nn.max_pool(_layer, ksize=[1, __word_size, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        # print("cnn:", _layer.shape)

        # Activation
        _layer = tf.squeeze(_layer, axis=[1, 2])
        _weight = [self.network_architecture["cnn_unit"], self.network_architecture["hidden_unit"]]
        _layer = full_connected(_layer, _weight, initializer=self.ini)
        _layer = tf.nn.dropout(_layer, _keep_prob)
        _layer = self.activation(_layer)

        # Prediction, Loss and Accuracy
        if self.binary_class:
            _weight = [self.network_architecture["hidden_unit"], 1]
            logit = tf.squeeze(full_connected(_layer, _weight), axis=1)
            # print("logit:", logit.shape)
            self.prediction = tf.sigmoid(logit)
            # logistic loss
            _loss = self.y * tf.log(self.prediction + 1e-8) + (1 - self.y) * tf.log(1 - self.prediction + 1e-8)
            self.loss = - tf.reduce_mean(_loss)
            # accuracy
            _prediction = tf.cast((self.prediction > 0.5), tf.float32)
            self.accuracy = 1 - tf.reduce_mean(tf.abs(self.y - _prediction))
        else:
            _weight = [self.network_architecture["hidden_unit"], self.network_architecture["label_size"]]
            logit = full_connected(_layer, _weight)
            # print("logit:", logit.shape)
            self.prediction = tf.nn.softmax(logit)
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
    net = {"label_size": 2, "input_char": [40, 33, 26], "input_word": [40, 19105],
           "char_embed_dim": 5, "char_cnn_unit": 10, "char_cnn_kernel": 3, "word_embed_dim": 30,
           "cnn_unit": 300, "cnn_kernel": 5,
           "hidden_unit": 300}
    CharCNN(net)
