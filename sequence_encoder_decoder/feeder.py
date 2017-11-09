import numpy as np


class BatchFeeder:
    """ Simple feeder of mini-batch for feeding a subset of numpy matrix into tf network. """

    def __init__(self, x, y, batch_size, ini_random=True):
        """
        Parameter
        ----------------
        X: input data, numpy array, 1st dimension should be data length
        y: output data, numpy array, 1st dimension should be data length
        batch_size: int mini batch size
        ini_random: (optional, default True) initialize with random
        """

        # assert len(x) == len(y)  # check whether X and Y have the matching sample size.
        self.x, self.y, self.n, self.index = x, y, len(x), 0
        self.batch_size = batch_size
        self.base_index = np.arange(self.n)
        if ini_random:
            _ = self.randomize(np.arange(len(x)))
        # self.val = None

    def next(self):
        if self.index + self.batch_size > self.n:
            self.index = 0
            self.base_index = self.randomize(self.base_index)
        _x = self.x[self.index:self.index + self.batch_size]
        _y = self.y[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return _x, _y

    def randomize(self, index):
        np.random.shuffle(index)
        self.y = self.y[index]
        self.x = self.x[index]
        return index

    # def create_validation(self, batch_size):
    #     self.val = (self.x[-1*int(batch_size):], self.y[-1*int(batch_size):])
    #     self.x = self.x[:-1*int(batch_size)]
    #     self.y = self.y[:-1*int(batch_size)]
    #     self.n = len(self.x)-int(batch_size)

