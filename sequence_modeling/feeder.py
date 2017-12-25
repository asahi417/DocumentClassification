import numpy as np


def randomize(x, y=None, seed=0):
    """ Randomize numpy array."""
    index = [i for i in range(len(x))]
    np.random.seed(seed)
    np.random.shuffle(index)
    if y is not None:
        return x[index], y[index]
    else:
        return x[index]


class BatchFeeder:
    """ Batch feeder for train model."""
    _index = 0
    _index_valid = 0
    n_valid, iterator_length_valid = 0, 0  # number of validation data, iterate number for each batch
    y_valid, x_valid = None, None

    def __init__(self, inputs, outputs, batch_size, validation=None, process=None, fix_validation=False):
        """
        :param outputs: outputs data, first dimension is iterator
        :param inputs: input data, first dimension is iterator (should be 1 dimension array)
        :param int batch_size: batch size
        :param float validation: proportion of validation data
        :param process: (optional) list of pre-processing function or single function
        :param fix_validation: (optional) fix validation set or not
        """
        self.batch_size = batch_size
        self.process = process
        # if validation, split chunk into validation and training, get validation chunk
        if validation is not None:
            if not fix_validation:
                inputs, outputs = randomize(inputs, outputs)
            self.balanced_validation_split(inputs, outputs, validation)
            self.iterator_length_valid = int(np.ceil(len(self.y_valid) / self.batch_size))
            self.n_valid = len(self.y_valid)
        else:
            self.x, self.y = inputs, outputs
        self.n = len(self.y)
        self.iterator_length = int(np.floor(len(self.y) / self.batch_size))

    def finalize(self):
        if self.process is not None:
            if type(self.process) == list:
                for __process in self.process:
                    __process.finalize()
            else:
                self.process.finalize()

    def balanced_validation_split(self, x, y, ratio):
        """y should be 1 dimension array"""
        size = int(np.floor(len(x) * ratio) / 2)
        # binary label index
        _y0 = y[y == 0]
        _y1 = y[y == 1]
        _x0 = x[y == 0]
        _x1 = x[y == 1]
        _ind = int(np.min([np.min([len(_y0), len(_y1)]), size]))
        self.y_valid = np.hstack([_y0[:_ind], _y1[:_ind]])
        __y = np.hstack([_y0[_ind:], _y1[_ind:]])
        if x.ndim == 1:
            self.x_valid = np.hstack([_x0[:_ind], _x1[:_ind]])
            __x = np.hstack([_x0[_ind:], _x1[_ind:]])
        else:
            self.x_valid = np.vstack([_x0[:_ind], _x1[:_ind]])
            __x = np.vstack([_x0[_ind:], _x1[_ind:]])
        self.x, self.y = randomize(__x, __y)

    def next(self):
        """ next batch (size is `self.batch_size`) """
        if self._index + self.batch_size >= len(self.y):
            self._index = 0
            self.x, self.y = randomize(self.x, self.y)
        _x = self.x[self._index:self._index + self.batch_size]
        _y = self.y[self._index:self._index + self.batch_size]
        self._index += self.batch_size
        if self.process is not None:
            if type(self.process) == list:
                return [_process(_x) for _process in self.process], _y
            else:
                return self.process(_x), _y
        else:
            return _x, _y

    def next_valid(self):
        """ next balanced validation batch (size is `self.batch_size`) """
        if self._index_valid + self.batch_size >= len(self.y_valid):
            _x = self.x_valid[self._index_valid:]
            _y = self.y_valid[self._index_valid:]
            self._index_valid = 0
        else:
            _x = self.x_valid[self._index_valid:self._index_valid + self.batch_size]
            _y = self.y_valid[self._index_valid:self._index_valid + self.batch_size]
            self._index_valid += self.batch_size

        if self.process is not None:
            if type(self.process) == list:
                return [_process(_x) for _process in self.process], _y
            else:
                return self.process(_x), _y
        else:
            return _x, _y


