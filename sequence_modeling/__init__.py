from . import model
from .feeder import BatchFeeder
from .train import train
from .processing import Process


def get_model_instance(model_name, embedding_model, label_size=2, n_word=40, n_char=33):
    # set up pre processing
    _pre_process = Process("embed", {"length_word": n_word, "dim": embedding_model.vector_size,
                                     "model": embedding_model, "path": "./data/random_dict.json"})
    if model_name == "char_cnn":
        _model = model.CharCNN
        _pre_process = [Process("onehot", {"length_word": n_word, "length_char": n_char}),
                        _pre_process]
        _net = {"input_char": [n_word, n_char, _pre_process[0].char_dict_size],
                "input_word": [n_word, embedding_model.vector_size],
                "char_embed_dim": 5, "char_cnn_unit": 10, "char_cnn_kernel": 3, "word_embed_dim": 30,
                "label_size": label_size, "cnn_unit": 300, "cnn_kernel": 5, "hidden_unit": 300}
        _model_inputs = model.InputFormat.char_word
    elif model_name == "gap_cnn":
        _net = {"label_size": label_size, "n_input": [n_word, embedding_model.vector_size, 1]}
        _model = model.GapCNN
        _model_inputs = model.InputFormat.word_3d
    elif model_name == "lstm":
        _net = {"label_size": label_size, "n_input": [n_word, embedding_model.vector_size],
                "n_hidden_1": 64, "n_hidden_2": 128, "n_hidden_3": 256}
        _model = model.LSTM
        _model_inputs = model.InputFormat.basic
    else:
        raise ValueError("unknown model!")
    return _model, _net, _pre_process, _model_inputs
