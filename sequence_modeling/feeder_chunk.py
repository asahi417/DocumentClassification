import numpy as np
from glob import glob


def randomize(target):
    """ Randomize numpy array."""
    index = [i for i in range(len(target))]
    np.random.shuffle(index)
    return target[index]


class ChunkBatchFeeder:
    """ Chunk feeder of mini-batch for feeding a subset of numpy matrix into tf network. """

    x, y = None, None
    chunk_size = 0
    index_data_in_chunk = 0  # index for data in chunk (reset for each chunk)
    index_data_in_epoch = 0  # index for data in epoch (reset for each epoch)
    index_chunk = 0  # index for chunk  (reset for each epoch)
    index_chunk_valid = 0  # index for validation chunk  (reset for each epoch)

    def __init__(self, data_path, batch_size, chunk_for_validation=1):
        """
        Parameter
        ----------------
        X: input data, numpy array, 1st dimension should be data length
        y: output data, numpy array, 1st dimension should be data length
        batch_size: int mini batch size
        ini_random: (optional, default True) initialize with random
        """

        self.batch_size = batch_size
        # if validation, split chunk into validation and training
        if chunk_for_validation:
            self.validation = True
            _tmp = glob("%s/*.npz" % data_path)
            self.chunk_list_valid = [_tmp.pop(0) for _ in range(chunk_for_validation)]
            self.chunk_list = np.array(_tmp)
        else:
            self.validation = False
            self.chunk_list = np.array(glob("%s/*.npz" % data_path))
        # load first chunk data
        self.load_chunk()
        # iteration number in one epoch
        self.iteration_in_epoch = int((len(self.chunk_list) - 1) * self.chunk_size / self.batch_size)
        if self.validation:
            self.iteration_in_epoch_valid = len(self.chunk_list_valid)

    def load_chunk(self):
        data = np.load(self.chunk_list[self.index_chunk])
        self.chunk_size = len(data["y"])
        self.x, self.y = data["x"], data["y"]
        # index for chunk
        self.index_chunk += 1
        # index for data in chunk
        self.index_data_in_chunk = 0

    def next(self):
        """ next batch (size is `self.batch_size`) """
        # if iteration in epoch is finished, reset the indices
        if self.index_data_in_epoch == self.iteration_in_epoch:
            self.index_data_in_epoch = 0
            self.index_chunk = 0
            self.chunk_list = randomize(self.chunk_list)
            self.load_chunk()

        if self.index_data_in_chunk + self.batch_size <= self.chunk_size:
            _x = self.x[self.index_data_in_chunk:self.index_data_in_chunk + self.batch_size]
            _y = self.y[self.index_data_in_chunk:self.index_data_in_chunk + self.batch_size]
            self.index_data_in_chunk += self.batch_size
        else:
            res = self.chunk_size - self.index_data_in_chunk
            _x = self.x[self.index_data_in_chunk:self.chunk_size]
            _y = self.y[self.index_data_in_chunk:self.chunk_size]
            self.load_chunk()
            res = self.batch_size - res
            _x = np.vstack([_x, self.x[self.index_data_in_chunk:self.index_data_in_chunk + res]])
            _y = np.hstack([_y, self.y[self.index_data_in_chunk:self.index_data_in_chunk + res]])
            self.index_data_in_chunk += res
        self.index_data_in_epoch += 1
        return _x, _y

    def next_valid(self):
        """ next data chunk for validation """
        if not self.validation:
            raise ValueError("No validation setting.")
        if self.index_chunk_valid == self.iteration_in_epoch_valid:
            self.index_chunk_valid = 0
        data = np.load(self.chunk_list_valid[self.index_chunk_valid])
        self.index_chunk_valid += 1
        return data["x"], data["y"]
