import string
import re
import os
import nltk
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
SEED = 1013
np.random.seed(SEED)
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples 
from stance_utils import *
#from parameters import *
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dropout,Concatenate,Dense, Embedding, SpatialDropout1D, Flatten, GRU, Bidirectional, Conv1D,MaxPooling1D

from tensorflow.keras.layers import RNN, Dropout,Concatenate,Dense, Embedding,LSTMCell, LSTM, SpatialDropout1D, Flatten, GRU, Bidirectional, Conv1D, Input,MaxPooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from sklearn.model_selection import StratifiedKFold
stemmer = PorterStemmer()
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
stopwords_english = stopwords.words('english')
from sklearn.preprocessing import LabelEncoder
import keras.backend as K
from keras.layers import Lambda
import random
import matplotlib.pyplot as plt

























def bicond(units,opt, embedding_matrix, x_t, batch_size, sentence_maxlen,num_classes): # Check this model again....
    embedded_inputs = tf.nn.embedding_lookup(embedding_matrix, x_t)
    print(embedded_inputs.shape)
    inputs = embedded_inputs[:batch_size]
    h_0 = tf.convert_to_tensor(np.zeros([batch_size, units]).astype(np.float32))
    c_0 = tf.convert_to_tensor(np.zeros([batch_size, units]).astype(np.float32))
    start_state = [h_0, c_0]
    lstm = LSTM(units, return_sequences=True, return_state=True)
    fw_output, fw_h_0, fw_c_0 = lstm(inputs,initial_state = [h_0, c_0])
    bw_output, bw_h_0, bw_c_0 = lstm(inputs[::-1],initial_state = [h_0, c_0]) # feeding data backwords
    
    inputs2 = Input(shape=(sentence_maxlen), name = 'Input')
    embedded_inputs = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], name = 'Embedding')(inputs2)
    lstm = LSTM(units,activation='tanh',dropout=0.1,name = 'lstm')(embedded_inputs, initial_state = [h_0, fw_c_0])
    b_lstm = LSTM(units,activation='tanh',dropout=0.1, go_backwards = True,name = 'back_lstm')(embedded_inputs, initial_state = [h_0, bw_c_0])
    cond_out = []
    cond_out.append(lstm)
    cond_out.append(b_lstm)
    concat_output = Concatenate()(cond_out)
    flat = Flatten(name = 'Flatten')(concat_output)
    output = (Dense(num_classes,activation='softmax',name = 'Dense'))(flat)
    model = Model(inputs=inputs2, outputs=output, name = 'bicond')
    model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])    
    model.summary()
    
    return model

def biLSTM(embedding_matrix, num_classes):
    model = Sequential(name = 'biLSTM')
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix]))
    model.add(Dropout(0.2))
    model.add(LSTM(64,return_sequences=True,dropout=0.3))
    model.add(Bidirectional(LSTM(64,dropout=0.3)))
    #model.add(Flatten())
    #add a dropout here
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

def biLSTMCNN(embedding_matrix, num_classes,sentence_maxlen ):
    inputs = Input(shape=(sentence_maxlen,))
    embedded_inputs = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix])(inputs)
    embedded_inputs = Dropout(0.2)(embedded_inputs)
    lstm = Bidirectional(LSTM(64,return_sequences=True,dropout=0.3))(embedded_inputs)
    convs = []
    for each_filter_size in [3,4,5]:
        #print(rnn.shape)
        each_conv = Conv1D(100, each_filter_size, activation='relu')(lstm)
        each_conv = MaxPooling1D(sentence_maxlen-each_filter_size+1)(each_conv)
        each_conv = Flatten()(each_conv)
        #print(each_conv.shape)
        convs.append(each_conv)
        
    output = Concatenate()(convs)
    output = Dropout(0.5)(output)
    output = (Dense(num_classes,activation='softmax'))(output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy']) 
    return model

def biGRU(embedding_matrix, num_classes):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix]))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(64,return_sequences=True,dropout=0.3)))
    model.add(Bidirectional(GRU(64,dropout=0.3)))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model

def biGRUCNN(embedding_matrix, num_classes):
    inputs = Input(shape=(sentence_maxlen,))
    embedded_inputs = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix])(inputs)
    embedded_inputs = Dropout(0.2)(embedded_inputs)
    rnn = Bidirectional(GRU(64,return_sequences=True,dropout=0.3))(embedded_inputs)
    convs = []
    for each_filter_size in [3,4,5]:
        #print(rnn.shape)
        each_conv = Conv1D(100, each_filter_size, activation='relu')(rnn)
        each_conv = MaxPooling1D(sentence_maxlen-each_filter_size+1)(each_conv)
        each_conv = Flatten()(each_conv)
        #print(each_conv.shape)
        convs.append(each_conv)
        
    output = Concatenate()(convs)
    output = Dropout(0.5)(output)
    output = (Dense(3,activation='softmax'))(output)
    model = Model(inputs=inputs, outputs=output, name = 'biGRUCNN')
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])    
    
    return model
    
    
    
