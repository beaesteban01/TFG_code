import pandas as pd
import io
import requests
import numpy as np
import os

import tensorflow.contrib.learn as skflow
import string

from scipy.stats import zscore

from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from sklearn import preprocessing

#####################################################
#os.system('python 22_05_Transform_columns.py')
#No puedo ponerlo asi por que entonces dice que no esta definido df
#Seguro que hay alguna manera
#####################################################

path = "../july_reduced copia.csv"
# This file is a CSV, just no CSV extension or headers
df_not_chunk = pd.read_csv(path, header=None, chunksize=200000)

##########################CHUNKSIZE###################################################
chunk_list = [] #append echa chunk df here

#Each chunk is in a df format
for df in df_not_chunk:
    #si no imprime no funciona, da error en el print
    #print(df)

#########################################################################################


    print("Read {} rows.".format(len(df)))
    # df = df.sample(frac=0.1, replace=False) # Uncomment this line to sample only 10% of the dataset
    df.dropna(inplace=True,axis=1) # For now, just drop NA's (rows with missing values)


    # The CSV file has no column heads, so add them
    df.columns = [
        'time',
        'duration',
        'sip',
        'dip',
        'source_port',
        'dest_port',
        'protocol',
        'flags',
        'forward_status',
        'type_service',
        'pack_exch',
        'bytes',
        'attack_tag'
    ]

    #print(df[0:3])

    ENCODING = 'utf-8'

    
 
    # Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
    def encode_text_dummy(df, name):
        dummies = pd.get_dummies(df[name])
        for x in dummies.columns:
            dummy_name = f"{name}-{x}"
            df[dummy_name] = dummies[x]
        df.drop(name, axis=1, inplace=True)

    #Clean 'date' column and convert to Int type
    def clean_date(s):
        s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
        s_removed = s.replace(" ", "")
        s_int = int(s_removed)
        return s_int

    ########## CLEAN IP #######################
    def clean_ip(s):
        s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
        s_int = int(s)
        return s_int

    # Encode a numeric column as zscores
    def encode_numeric_zscore(df, name, mean=None, sd=None):
        if mean is None:
            mean = df[name].mean()

        if sd is None:
            sd = df[name].std()

        df[name] = (df[name] - mean) / sd


    # Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
    def encode_text_index(df, name):
        le = preprocessing.LabelEncoder()
        df[name] = le.fit_transform(df[name])
        return le.classes_

    def minmax(df, name):
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(df[name])
    #LAS QUE YA SON NUMEROS --> no los normalizo de momento
    # encode_numeric_zscore(df, 'duration')
    # encode_numeric_zscore(df, 'source_port')
    # encode_numeric_zscore(df, 'dest_port')
    # encode_numeric_zscore(df, 'forward_status')
    # encode_numeric_zscore(df, 'type_service')
    # encode_numeric_zscore(df, 'pack_exch')
    # encode_numeric_zscore(df, 'bytes')

    df.drop('time', 1, inplace=True)

    encode_text_dummy(df, 'protocol')
    encode_text_dummy(df, 'flags')
    #encode_text_dummy(df, 'attack_tag')

    outcomes = encode_text_index(df, 'attack_tag')
    num_classes = len(outcomes)

    #Me crea una columna AL FINAL nueva con los valores transformdos asi 20160318105240
    #df['time'] = df['time'].apply(clean_date)

    df['sip'] = df['sip'].apply(clean_ip)
    df['dip'] = df['dip'].apply(clean_ip)

    #encode_numeric_zscore(df, 'time')
    minmax(df, 'sip')
    minmax(df, 'dip')


    print(df.shape)
    print(df[0:3])

   