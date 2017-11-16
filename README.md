# Short Sentence Modelling
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/SequenceModeling/blob/master/LICENSE)

Implement classification model for [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html).
Input data are sentences, which is review of movie and output label is binarized as *Good* or *Bad*.
Here, neutral evaluation is removed.

- This code is supported python 3 and tensorflow 1.3.0.

## How to use.
### Setup
First, clone the repository and run setup.

```
git clone https://github.com/asahi417/SequenceModeling 
cd SequenceModeling
pip install -r requirements.txt
python setup.py test
mkdir data
```

Then, you need to install [pretrained word2vector by Google news](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download)
to get `GoogleNews-vectors-negative300.bin.gz`.
Also, you have to access [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html)
to get `stanfordSentimentTreebank.zip`. Finally, move them to `SequenceModeling/data/`
and unzip those files. 

### Embedding word

To train the model, you need to get vector representation of the corpus. 
In python, 

```python
>>> import sequence_modeling
>>> data = sequence_modeling.sst("./data/stanfordSentimentTreebank", binary=True, cut_off=2)
>>> sentences = data["sentence"]
>>> label = data["label"]
>>> sequence_modeling.vectorize_chunk(sentences=sentences, label=label, length=40, chunk_size=5000,
                                      embed_path="./data/GoogleNews-vectors-negative300.bin",
                                      save_path="./data/embed_p40_c2")
```

or you can use sample script by 
```
python sample_embedding.py -p 40 -c 2
```

### Train model
In python
```python
>>> import sequence_modeling
>>> feeder = sequence_modeling.ChunkBatchFeeder(data_path="./data/embed_p40_c2", batch_size=500,
                                                chunk_for_validation=2, balance_validation=True)
>>> net = {"label_size": 2, "n_input": [45, 300, 1]}
>>> sequence_modeling.train_chunk(epoch=150, clip=None, lr=0.001, model="cnn3", feeder=feeder,
                                  save_path="./log", network_architecture=net, keep_prob=0.9)

```

or you can use sample script by 
```
python sample_train.py cnn3
```

### References
- Socher, Richard, et al. "Recursive deep models for semantic compositionality over a sentiment treebank." Proceedings of the 2013 conference on empirical methods in natural language processing. 2013.