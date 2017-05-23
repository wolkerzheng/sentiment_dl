#!/bin/bash/env python
# -*- coding: utf-8 -*-
__author__ = 'wolker'
"""
precision：0.8716
score: 0.94293
"""
import pandas as pd
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, Input,Conv1D,GlobalMaxPooling1D
import numpy as np
import os
def loadW2v():

    return Word2Vec.load("300features_40minwords_10context")

def loadWordindex():
    word_index = {}
    with open('wordidx.txt ', 'r') as f:
        lines = f.readlines()
        for l in lines:
            t = l.strip().split(',')
            word_index[t[0]] = int(t[1])
    return word_index

def loadData():
    x_train_sequences = []
    x_test_sequences = []
    with open('x_train_sequences.txt','r') as f:
        lines = f.readlines()
        for l in lines:
                t = l.strip().split(',')
                x_train_sequences.append([int(x) for x in t if x!=''])


    with open('x_test_sequences.txt','r') as f:
        lines = f.readlines()
        for l in lines:
                t = l.strip().split(',')
                x_test_sequences.append([int(x) for x in t if x != ''])


    return x_train_sequences,x_test_sequences

if __name__ == '__main__':

    MAX_SEQUENCE_LENGTH = 100
    MAX_NB_WORDS = 200000
    EMBDDING_DIM = 300
    nfilters = 250
    kernel_size = 3
    batch_size = 64
    epoch = 10


    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)
    word_index = loadWordindex()

    x_train_sequences, x_test_sequences = loadData()


    data_1 = pad_sequences(x_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = train['sentiment']
    print data_1.shape,labels.shape     #(25000, 100) (25000,)

    data_2 = pad_sequences(x_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # test_ids = test["id"]

    print '构造词向量矩阵'
    w2vModel = Word2Vec.load("300features_40minwords_10context")
    # 取词向量或分词的最小，之所以加1,因为有一维代表改词没有在训练的词向量中的表示
    nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((nb_words, EMBDDING_DIM))
    for word, i in word_index.items():
        if word in w2vModel.vocab:
            embedding_matrix[i] = w2vModel[word]
    print embedding_matrix.shape  # (101226, 300)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedding_layer = Embedding(nb_words,
                                EMBDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(input_layer)

    cnn_layer = Conv1D(filters=nfilters,kernel_size=4,
                       activation='relu',strides=1)(embedding_layer)
    maxPool_layer = GlobalMaxPooling1D()(cnn_layer)

    hidden_layer = Dense(256,activation='tanh')(maxPool_layer)
    # hidden_layer = Dropout(0.3)(hidden_layer)
    ouput_layer = Dense(1,activation='sigmoid')(hidden_layer)

    model = Model(inputs=input_layer,outputs=ouput_layer)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(data_1, labels,
              batch_size=batch_size,
              epochs=epoch,
              validation_split=0.1)

    res = model.predict(data_2, batch_size=batch_size)
    print 'res is', res.shape
    # print res[0:10, 0]
    output = pd.DataFrame(data={"id": test["id"], "sentiment": res[:, 0]})
    output.to_csv("simple_cnn.csv", index=False, quoting=3)
    print "Wrote simple_cnn.csv"

