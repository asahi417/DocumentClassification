# Short Sentence Modelling
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/SequenceModeling/blob/master/LICENSE)

Implement classification model for [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html).
Input data are sentences, which is review of movie and output label is binarized as *Good* or *Bad*.
Here, neutral evaluation is removed.

- This code is supported python 3 and tensorflow 1.3.0.

## Models
Currently following models are available:
- [`Gap CNN`](sequence_modeling/model/cnn_gap.py)
    - [1] Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
- [`CNN with character level feature`](sequence_modeling/model/cnn_char.py)
    - [2] Dos Santos, Cícero Nogueira, and Maira Gatti. "Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts." COLING. 2014.
- [`LSTM`](sequence_modeling/model/lstm.py):
    - bi LSTM x 3 -> hidden unit of last bi LSTM -> Full connect -> output (good, bad)
- [`LSTM with character level feature`](sequence_modeling/model/lstm_char.py)
    - concatenate(word feature, character feature) -> Full connect -> output (good, bad)
    - word feature:  
    word -> embed -> bi LSTM x 3 -> hidden unit of last bi LSTM -> feature  
    - character feature:  
    word -> *character embedding* (same as [2]) -> bi LSTM x 3 -> hidden unit of last bi LSTM -> feature   
 
Dropout (recurrent dropout for LSTM) and batch normalization for CNN and fully connected layer have been implemented in all models.

Todo:
* Layer normalization: when I use [`LayerNormBasicLSTMCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LayerNormBasicLSTMCell) with layer_norm, accuracies becomes NaN and it have not solved yet.


## How to use.
### Setup
First, clone the repository and run setup.

```
git clone https://github.com/asahi417/SequenceModeling 
cd SequenceModeling
pip install -r requirements.txt
python setup.py test
```

Then, you need to install [pretrained word2vector by Google news](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)
to get `GoogleNews-vectors-negative300.bin.gz`.
Also, you have to access [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html)
to get `stanfordSentimentTreebank.zip`. Finally, move them to `SequenceModeling/data/`
and unzip those files. 
Thus, the directory would be looks like
```
SequenceModeling/
    ├ data/
    : ├ __init__.py
      ├ sst.py
      ├ GoogleNews-vectors-negative300.bin
      └ stanfordSentimentTreebank/
``` 


### Train model
You can use sample script by 
```
python sample_train.py cnn_char -e 100
```
then you will get following log.
<p align="center">
  <img src="./img/learning.png" width="900">
  <br><i> training progress log</i>
</p>


### Base line check
To see how efficient the model is, you can see the base line accuracy by SVM and logistic regression.
```
>>> python bench_mark_accuracy.py 
```


