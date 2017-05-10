#!/bin/bash/env python
# -*- coding: utf-8 -*-
__author__ = 'wolker'

import os
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import numpy as np
from gensim.models import Word2Vec

import KaggleWord2VecUtility
from keras.models import Model
from keras.layers import Dense,Dropout,Activation,Embedding,Input,LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
def getCleanReviews(reviews):
    """

    :param reviews:
    :return:
    """
    clean_reviews = []
    for review in reviews["review"]:
        #一定要加encode('utf-8')，要不然报错！！！
        #python2对于字符编码问题要慎重
        review_text = BeautifulSoup(review, "lxml").get_text().encode('utf-8')
        review_text = re.sub(r"[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        text = " ".join(words)

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        clean_reviews.append(text)
        # clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews

if __name__ == '__main__':

    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)

    MAX_SEQUENCE_LENGTH = 100  # 每句话最大的单词数
    MAX_NB_WORDS = 200000
    EMBDDING_DIM = 300
    batch_size = 5000
    epoch = 10


    train_text = getCleanReviews(train)
    test_text = getCleanReviews(test)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train_text + test_text)
    x_train_sequences = tokenizer.texts_to_sequences(train_text)
    x_test_sequences = tokenizer.texts_to_sequences(test_text)
    print 'x_train_sequences', len(x_train_sequences)
    word_index = tokenizer.word_index
    print 'Found %s unique tokens' % len(word_index), type(word_index)

    data_1 = pad_sequences(x_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = train['sentiment']
    # print data_1.shape,labels.shape     #(25000, 100) (25000,)

    data_2 = pad_sequences(x_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # test_ids = test["id"]

    print '构造词向量矩阵'
    w2vModel =  Word2Vec.load("300features_40minwords_10context")
    # 取词向量或分词的最小，之所以加1,因为有一维代表改词没有在训练的词向量中的表示
    nb_words = min(MAX_NB_WORDS,len(word_index))+1
    embedding_matrix = np.zeros((nb_words,EMBDDING_DIM))
    for word,i in word_index.items():
        if word in w2vModel.vocab:
            embedding_matrix[i] = w2vModel[word]
    print embedding_matrix.shape    #(101226, 300)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    num_classes = 1
    #定义普通LSTM结构


    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(nb_words,
                                EMBDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    # embedding输出：(None, MAX_SEQUENCE_LENGTH, EMBDDING_DIM)
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    hidden_layer1 = LSTM(128,activation='relu')(embedded_sequences_1)
    hidden_layer1 = Dropout(0.25)(hidden_layer1)
    pred = Dense(1,activation='sigmoid')(embedded_sequences_1)
    model = Model(inputs=sequence_1_input,outputs=pred)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(data_1, labels,
                        epochs=epoch, batch_size=batch_size,
                        validation_split=0.1)

    res = model.predict(data_2, batch_size=batch_size)
    print 'res is', res.shape
    print res[0:10, 0]
    output = pd.DataFrame(data={"id": test["id"], "sentiment": res[:, 0]})
    output.to_csv("Word2Vec_lstm.csv", index=False, quoting=3)
    print "Wrote Word2Vec_lstm.csv"
