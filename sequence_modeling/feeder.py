import numpy as np


class BatchFeeder:
    """ Simple feeder of mini-batch for feeding a subset of numpy matrix into tf network. """

    def __init__(self, x, y, batch_size, ini_random=True, validation=0.7):
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
        if validation:
            self.randomize()
            self.create_validation(validation)
        else:
            self.valid_x = self.valid_y = None
            if ini_random:
                self.randomize()

    def next(self):
        if self.index + self.batch_size > self.n:
            self.index = 0
            self.randomize()
        _x = self.x[self.index:self.index + self.batch_size]
        _y = self.y[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return _x, _y

    def randomize(self):
        index = np.arange(self.n)
        np.random.shuffle(index)
        self.x, self.y = self.x[index], self.y[index]

    def create_validation(self, ratio):
        ratio = np.floor(len(self.x)*ratio).astype(int)
        self.valid_x, self.valid_y = self.x[-1 * ratio:], self.y[-1 * ratio:]
        self.x, self.y = self.x[:-1 * ratio], self.y[:-1 * ratio]
        self.n = len(self.x)

