import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import nltk
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from itertools import chain

stopwords = nltk.corpus.stopwords

test = pd.read_csv('test.csv', header=0, encoding='utf-8')


f = open('glove.6B/glove.6B.50d.txt','r')
glove = {}
for line in f:
    split_line = line.split()
    word = split_line[0]
    vector = [float(val) for val in split_line[1:]]
    glove[word] = vector
gv = pd.DataFrame.from_dict(glove)

def tokenize_sentence(sentence):
    return nltk.word_tokenize(str(sentence).lower())

def glove_sentence(sentence):
    return [gv[word] for word in sentence if word in gv]

def get_sentence_features(sentence):
    return pd.Series([np.sum(sentence), np.mean(sentence)]).fillna(0)

def sent_col_to_features(panda_col):
    # Take sum and mean of each sentence
    return panda_col.apply(lambda sentence: get_sentence_features(glove_sentence(tokenize_sentence(sentence)))) 

import pickle
filename = 'svc_model.p'
clf_svc = pickle.load(open(filename, 'rb'))    

test_array = test
test_test_size = len(test)
q1_test = sent_col_to_features(test_array.question1[:test_test_size])
q2_test = sent_col_to_features(test_array.question2[:test_test_size])
test_features = pd.concat([q1_test , q2_test],axis=1)

sent_col_to_features(test_features).to_pickle('test_features.p')

submission_predictions = clf_svc.predict(test_features)

submission_predictions = pd.DataFrame(submission_predictions, columns=['is_duplicate'])
submission_predictions.to_csv("submission.csv")
