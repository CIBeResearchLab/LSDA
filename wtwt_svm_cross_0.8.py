#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json
from nltk.tokenize import TweetTokenizer
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.util import ngrams
#from google.colab import drive
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy import sparse
import os
from stance_utils import *
import warnings
warnings.filterwarnings('ignore')
import datetime
now = datetime.datetime.now()
from scipy.sparse import csr_matrix


# In[ ]:





# In[2]:


df_hlt_train = pd.read_csv("/data/parush/wtwt/healthcare_train.txt", sep='\t')
df_hlt_test = pd.read_csv("/data/parush/wtwt/healthcare_test.txt", sep='\t')
df_ent_train = pd.read_csv("/data/parush/wtwt/entertainment_train.txt", sep='\t')
df_ent_test = pd.read_csv("/data/parush/wtwt/entertainment_test.txt", sep='\t')
print("Length of Health_train", len(df_hlt_train))
print("Length of Health_test", len(df_hlt_test))
print("Length of ent_train", len(df_ent_train))
print("Length of ent_test", len(df_ent_test))


# In[3]:


aug = True
classes = {'support':0, 'refute': 1, 'comment': 2, 'unrelated': 3}
file = 'rich_data/ent_pol_hlt/ent_pol_hlt0.8.json'


# In[ ]:





# In[4]:


def process_tweet(tweet):
    '''
    Input: 
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
    
    '''
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    ### START CODE HERE ###
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and # remove stopwords
            word not in string.punctuation): # remove punctuation
            #tweets_clean.append(word)
            stem_word = stemmer.stem(word) # stemming word
            tweets_clean.append(stem_word)
    ### END CODE HERE ###
    tweets_clean = " ".join(tweets_clean)
    return tweets_clean


# In[ ]:





# In[5]:


vectorizer = 'tfidf'   # set 'count' or 'tfidf'
analyzer = 'both'  # set 'word' or 'both' ( word and char)


# In[6]:


if vectorizer == 'count':
    if analyzer == 'word':
        vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,1))
    else:
        vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,3))
        char_vectorizer = CountVectorizer(analyzer='char',ngram_range=(2,5))
else:
    if analyzer == 'word':
        vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,1))
    else:
        vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,3))
        char_vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(2,5))
        
        
        
        


# In[7]:


def cross_splitter(df_train, df_test, classes, file):
    print("Started splitting 0.8 threshold pre-processing")
    

    
    train_corpus = [process_tweet(i) for i in df_train['tweet'].tolist()]
    train_labels = [classes[i] for i in df_train['stance'].tolist()]
    test_corpus = [process_tweet(i) for i in df_test['tweet'].tolist()]
    test_labels = [classes[i] for i in df_test['stance'].tolist()]
    c_len = len(train_corpus)
    print("Before augmenting length ", c_len)
    if aug:
        with open(file,'r') as new_file:
            data = json.load(new_file)
            for line in data:
                tweet = line['tweet'].strip()
                stance = line['stance'].strip()
                train_corpus.append(process_tweet(tweet))
                train_labels.append(classes[stance])
    
    print("Added {} more examples".format(len(train_corpus)-c_len))
    print("Total tweet {} and labels {}".format(len(train_corpus), len(train_labels)))
    
    if analyzer == 'word':
        ngram_vectorized_data = vectorizer.fit_transform(train_corpus)
        test_ngram_vectorized_data = vectorizer.transform(test_corpus)
        #ngram_vectorized_data = sparse.csr_matrix(ngram_vectorized_data)
        #test_ngram_vectorized_data = sparse.csr_matrix(test_ngram_vectorized_data)
        return ngram_vectorized_data, train_labels, test_ngram_vectorized_data, test_labels
    else:
        ngram_vectorized_data = vectorizer.fit_transform(train_corpus)
        char_vectorized_data = char_vectorizer.fit_transform(train_corpus)
        l = np.hstack((ngram_vectorized_data.toarray(), char_vectorized_data.toarray()))
        train_vectorized_data = sparse.csr_matrix(l)
     
        test_ngram_vectorized_data = vectorizer.transform(test_corpus)
        test_char_vectorized_data = char_vectorizer.transform(test_corpus)
        l2 = np.hstack((test_ngram_vectorized_data.toarray(), test_char_vectorized_data.toarray()))
        test_vectorized_data = sparse.csr_matrix(l2)
        
        return train_vectorized_data, train_labels, test_vectorized_data,test_labels


# In[8]:



#X_train, y_train, X_test, y_test =  cross_splitter(df_hlt_train, df_ent_test,classes, file)
X_train, y_train, X_test, y_test =  cross_splitter(df_ent_train, df_hlt_test,classes, file)

# In[9]:


unique,count = np.unique(y_train,return_counts=True)
print(dict(zip(unique, count)))


# In[ ]:





# In[ ]:





# In[10]:


unique,count = np.unique(y_test,return_counts=True)
print(dict(zip(unique, count)))


# In[11]:


from sklearn.model_selection import StratifiedKFold


# In[ ]:


# Set the parameters by cross-validation
print("Started at ", now)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']


for score in scores:
    
    print("# Tuning hyper-parameters for %s" % score)
    print()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state = 2 )
    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % score, cv = cv
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred, digits = 4,))
    print()
    print("Finishes at ", now)


# In[ ]:


now = datetime.datetime.now()
print("Finishes at ", now)
#print(classification_report(y_true, y_pred, digits = 4,))


# # Test on other Target
# False

# In[9]:


# X_test_, y_test_ = get_test_data_and_labels(test_data_file_m,TARGETS_m[2])


# In[10]:


# y_true_, y_pred_ = y_test_, clf.predict(X_test_)
# print(classification_report(y_true_, y_pred_, digits = 4, labels = [0,1]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




