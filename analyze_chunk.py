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

path = "agosto.csv"
# This file is a CSV, just no CSV extension or headers
df_not_chunk = pd.read_csv(path, header=None, chunksize=200000)

##########################CHUNKSIZE###################################################
chunk_list = [] #append echa chunk df here

#Each chunk is in a df format
for df in df_not_chunk:

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

    def expand_categories(values):
        result = []
        s = values.value_counts()
        t = float(len(values))
        for v in s.index:
            result.append("{}:{}%".format(v,round(100*(s[v]/t),2)))
        return "[{}]".format(",".join(result))
            
    def analyze(filename):
        print()
        print("Analyzing: {}".format(filename))
        df = pd.read_csv(filename,encoding=ENCODING)
        cols = df.columns.values
        total = float(len(df))

        print("{} rows".format(int(total)))
        for col in cols:
            uniques = df[col].unique()
            unique_count = len(uniques)
            if unique_count>100:
                print("** {}:{} ({}%)".format(col,unique_count,int(((unique_count)/total)*100)))
            else:
                print("** {}:{}".format(col,expand_categories(df[col])))
                expand_categories(df[col])

    analyze(path)