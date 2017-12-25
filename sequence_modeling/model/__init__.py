from .lstm import LSTM
from .gap_cnn import GapCNN
from .char_cnn import CharCNN


class InputFormat:

    @staticmethod
    def char_word(model, input_data):
        return {model.x_char: input_data[0], model.x_word: input_data[1]}

    @staticmethod
    def word_3d(model, input_data):
        return {model.x: np.expand_dims(input_data, 3)}

    @staticmethod
    def basic(model, x):
        return {model.x: x}

