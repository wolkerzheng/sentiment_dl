#!/bin/bash/env python
# -*- coding: utf-8 -*-
__author__ = 'wolker'

"""
普通lstm precision: 0.79
score:0.92377

"""
import os
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import numpy as np
from gensim.models import Word2Vec
from keras.models import Model
from keras.layers import Dense,Dropout,Activation,Embedding,Input,LSTM
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import activations, Wrapper
import attention_lstm as atten
class AttentionLSTM(LSTM):
    def __init__(self, output_dim, attention_vec, attn_activation='tanh', single_attention_param=False, **kwargs):
        self.attention_vec = attention_vec
        self.attn_activation = activations.get(attn_activation)
        self.single_attention_param = single_attention_param

        super(AttentionLSTM, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        super(AttentionLSTM, self).build(input_shape)

        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception('Layer could not be build: No information about expected input shape.')

        self.U_a = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.output_dim,), name='{}_b_a'.format(self.name))

        self.U_m = self.inner_init((attention_dim, self.output_dim),
                                   name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.output_dim,), name='{}_b_m'.format(self.name))

        if self.single_attention_param:
            self.U_s = self.inner_init((self.output_dim, 1),
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights += [self.U_a, self.U_m, self.U_s, self.b_a, self.b_m, self.b_s]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        h, [h, c] = super(AttentionLSTM, self).step(x, states)
        attention = states[4]

        m = self.attn_activation(K.dot(h, self.U_a) * attention + self.b_a)
        # Intuitively it makes more sense to use a sigmoid (was getting some NaN problems
        # which I think might have been caused by the exponential function -> gradients blow up)
        s = K.sigmoid(K.dot(m, self.U_s) + self.b_s)

        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.output_dim, axis=1)
        else:
            h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = super(AttentionLSTM, self).get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants


class AttentionLSTMWrapper(Wrapper):
    def __init__(self, layer, attention_vec, attn_activation='tanh', single_attention_param=False, **kwargs):
        assert isinstance(layer, LSTM)
        self.supports_masking = True
        self.attention_vec = attention_vec
        self.attn_activation = activations.get(attn_activation)
        self.single_attention_param = single_attention_param
        super(AttentionLSTMWrapper, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

        super(AttentionLSTMWrapper, self).build()

        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception('Layer could not be build: No information about expected input shape.')

        self.U_a = self.layer.inner_init((self.layer.output_dim, self.layer.output_dim), name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.layer.output_dim,), name='{}_b_a'.format(self.name))

        self.U_m = self.layer.inner_init((attention_dim, self.layer.output_dim), name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.layer.output_dim,), name='{}_b_m'.format(self.name))

        if self.single_attention_param:
            self.U_s = self.layer.inner_init((self.layer.output_dim, 1), name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = self.layer.inner_init((self.layer.output_dim, self.layer.output_dim), name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.layer.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights = [self.U_a, self.U_m, self.U_s, self.b_a, self.b_m, self.b_s]

    def get_output_shape_for(self, input_shape):
        return self.layer.get_output_shape_for(input_shape)

    def step(self, x, states):
        h, [h, c] = self.layer.step(x, states)
        attention = states[4]

        m = self.attn_activation(K.dot(h, self.U_a) * attention + self.b_a)
        s = K.sigmoid(K.dot(m, self.U_s) + self.b_s)

        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.layer.output_dim, axis=1)
        else:
            h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = self.layer.get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.layer.stateful:
            initial_states = self.layer.states
        else:
            initial_states = self.layer.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.layer.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.layer.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.layer.unroll,
                                             input_length=input_shape[1])
        if self.layer.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.layer.states[i], states[i]))

        if self.layer.return_sequences:
            return outputs
        else:
            return last_output
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

    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)

    MAX_SEQUENCE_LENGTH = 100  # 每句话最大的单词数
    MAX_NB_WORDS = 200000
    EMBDDING_DIM = 300
    batch_size = 32
    epoch = 10

    word_index = loadWordindex()

    x_train_sequences, x_test_sequences = loadData()

    data_1 = pad_sequences(x_train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = train['sentiment']
    print data_1.shape, labels.shape  # (25000, 100) (25000,)

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
    hidden_layer1 = LSTM(EMBDDING_DIM,return_sequences=True,activation='relu')(embedded_sequences_1)
    hidden_layer1 = Dropout(0.3)(hidden_layer1)
    hidden_layer1 = atten.Attention()(hidden_layer1)
    # hidden_layer1 = LSTM(EMBDDING_DIM, activation='relu')(embedded_sequences_1)
    # hidden_layer1 = Dropout(0.3)(hidden_layer1)
    # hidden_layer1 =
    pred = Dense(1,activation='sigmoid')(hidden_layer1)
    model = Model(inputs=sequence_1_input,outputs=pred)
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    # model = Sequential()
    # model.add(Embedding(nb_words, EMBDDING_DIM,
    #                     input_length=MAX_SEQUENCE_LENGTH))

    # model.add(LSTM(EMBDDING_DIM,
    #                dropout=0.2,
    #                recurrent_dropout=0.2))
    #
    # model.add(Dense(1, activation='sigmoid'))
    # model.add(Embedding(nb_words, EMBDDING_DIM))
    # model.add(LSTM(EMBDDING_DIM, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(data_1, labels,
                        epochs=epoch, batch_size=batch_size,
                        validation_split=0.1)

    res = model.predict(data_2, batch_size=batch_size)
    print 'res is', res.shape
    # print res[0:10, 0]
    output = pd.DataFrame(data={"id": test["id"], "sentiment": res[:, 0]})
    output.to_csv("Word2Vec_lstm.csv", index=False, quoting=3)
    print "Wrote Word2Vec_lstm.csv"
