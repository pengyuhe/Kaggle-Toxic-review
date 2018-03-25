import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from Model.Useful_layers import *
from Model.Get_model import get_model
from Model.ROC_callback import RocAucMetricCallback
#from nltk.corpus import stopwords
#from nltk.stem import SnowballStemmer
from string import punctuation
from keras.layers import Embedding
#from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import sys

import json


def get_embed_matrix(EMBEDDING_FILE,EMBEDDING_DIM,MAX_NB_WORDS,word_index):
    embeddings_index = {}
    f = open(EMBEDDING_FILE)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs
        try:
            word_stem=stemmer.stem(word)
            embeddings_index_stem[word_stem] = coefs
        except:
            pass
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    print('Preparing embedding matrix')
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print("Nb_words ",nb_words)
    return embedding_matrix,nb_words


def Text_to_Tokens(train_df,test_df,list_classes,MAX_SEQUENCE_LENGTH,MAX_NB_WORDS):

    list_sentences_train = train_df["comment_text"].fillna("NA").values
    train_label = train_df[list_classes].values
    list_sentences_test = test_df["comment_text"].fillna("NA").values


    comments = []
    for text in list_sentences_train:
        comments.append(text)
        
    test_comments=[]
    for text in list_sentences_test:
        test_comments.append(text)

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters="")
    tokenizer.fit_on_texts(comments + test_comments)

    sequences = tokenizer.texts_to_sequences(comments)
    test_sequences = tokenizer.texts_to_sequences(test_comments)

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', train_label.shape)

    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of test_data tensor:', test_data.shape)


    return train_data,train_label,test_data,word_index

if __name__=="__main__":

        
    with open('config.json') as json_data_file:
        config_dic = json.load(json_data_file)
        print(config_dic)

    feature_dic=config_dic['feature_dic']
    para_dic=config_dic['para_dic']
    data_dic=config_dic['data_dic']

    data_path=data_dic["data_path"]
    embed_path=data_dic["embed_path"]

    TRAIN_DATA_FILE=data_path+data_dic['train_file']
    TEST_DATA_FILE=data_path+ data_dic['test_file']
    SUB_FILE=data_path+data_dic['submission_file']


    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)
    sample_submission = pd.read_csv(SUB_FILE)

    MAX_SEQUENCE_LENGTH = para_dic['MAX_SEQUENCE_LENGTH']
    MAX_NB_WORDS = para_dic['MAX_NB_WORDS']
    VALIDATION_SPLIT = para_dic["VALIDATION_SPLIT"]


    train_data, train_label, test_data, word_index=Text_to_Tokens(train_df,test_df,\
                                                                  list_classes,\
                                                                  MAX_SEQUENCE_LENGTH,MAX_NB_WORDS)

    ### Each embedding file should also be associated with its dimension
    Embed_names=data_dic['embed_file']
    DIM=data_dic['embed_dim']

    ### initialize output
    sample_submission[list_classes]=0.0
    
    N_bag=0
    for Ename,EMBEDDING_DIM in zip(Embed_names,DIM):

        ########################################
        ## bagging_data
        ## To save time, instead of doing k fold,
        ## we do bagging + average of different embeddings
        ########################################
        # np.random.seed(1234)
        perm = np.random.permutation(len(train_data))
        idx_train = perm[:int(len(train_data)*(1-VALIDATION_SPLIT))]
        idx_val = perm[int(len(train_data)*(1-VALIDATION_SPLIT)):]

        temp_train_data=train_data[idx_train]
        labels_train=train_label[idx_train]
        print(temp_train_data.shape,labels_train.shape)

        temp_val_data=train_data[idx_val]
        labels_val=train_label[idx_val]

        print(temp_val_data.shape,labels_val.shape)



        #### Generate embedding
        EMBEDDING_FILE=embed_path+Ename

        embedding_matrix,nb_words=get_embed_matrix(EMBEDDING_FILE,EMBEDDING_DIM,MAX_NB_WORDS,word_index)

        embedding_layer = Embedding(nb_words,
                EMBEDDING_DIM,
                weights=[embedding_matrix],
                input_length=MAX_SEQUENCE_LENGTH,
                trainable=False)    

        for RNN in ['LSTM','GRU']:
            model=get_model(embedding_layer,RNN=RNN,embed_size=EMBEDDING_DIM,Feature_dic=feature_dic,Para_dic=para_dic)

            
            ### Model name
            #STAMP = Ename+RNN
            STAMP=Ename.split('/')[-1]+RNN
            print(STAMP)

            early_stopping =EarlyStopping(monitor='roc_auc_val', patience=2,mode='max')
            bst_model_path = STAMP + '.h5'
            model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
            
            ### Fit model, with ROC early stopping
            hist = model.fit(temp_train_data, labels_train,         validation_data=(temp_val_data, labels_val),         epochs=50, batch_size=128, shuffle=True,          callbacks=[RocAucMetricCallback(),early_stopping, model_checkpoint])
                     
            ### Do prediction
            model.load_weights(bst_model_path)
            y_test = model.predict([test_data], batch_size=512, verbose=1)

            ### update prediction 
            sample_submission[list_classes] +=y_test
            N_bag+=1.0
    
    sample_submission[list_classes]/=N_bag
    sample_submission.to_csv('Submission.csv', index=False)
    print "Success!"
