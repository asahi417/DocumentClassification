import numpy as np


def randomize(x, y=None):
    """ Randomize numpy array."""
    index = [i for i in range(len(x))]
    np.random.shuffle(index)
    if y is not None:
        return x[index], y[index]
    else:
        return x[index]


class BatchFeeder:
    """ Batch feeder for train model."""
    _index = 0
    _index_valid = 0

    def __init__(self, inputs, outputs, batch_size, validation=0.2, process=None):
        """
        :param outputs: outputs data, first dimension is iterator
        :param inputs: input data, first dimension is iterator
        :param int batch_size: batch size
        :param float validation: proportion of validation data
        :param process: (optional) list of pre-processing function or single function
        """
        self.batch_size = batch_size
        self.process = process
        # if validation, split chunk into validation and training, get validation chunk
        if validation:
            inputs, outputs = randomize(inputs, outputs)
            ind = int(np.floor(len(inputs) * (1 - validation)))
            self.x, self.x_valid = inputs[:ind], inputs[ind:]
            self.y, self.y_valid = outputs[:ind], outputs[ind:]
            self.iterator_length_valid = int(np.floor(len(self.y_valid) / self.batch_size))
        else:
            self.x, self.y = inputs, outputs
        self.iterator_length = int(np.floor(len(self.y) / self.batch_size))

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
            self._index_valid = 0
            # self.x_valid, self.y_valid = randomize(self.x_valid, self.y_valid)
        _x = self.x_valid[self._index_valid:self._index_valid + self.batch_size]
        _y = self.y_valid[self._index_valid:self._index_valid + self.batch_size]
        self._index_valid += self.batch_size

        # binary label index
        __y0 = _y[_y == 0]
        __y1 = _y[_y == 1]
        __x0 = _x[_y == 0]
        __x1 = _x[_y == 1]

        # Reshaped for minimum label data
        _ind = int(np.min([len(__y0), len(__y1)]))
        _y = np.hstack([__y0[:_ind], __y1[:_ind]])
        _x = np.hstack([__x0[:_ind], __x1[:_ind]])
        if self.process is not None:
            if type(self.process) == list:
                return [_process(_x) for _process in self.process], _y
            else:
                return self.process(_x), _y
        else:
            return _x, _y


