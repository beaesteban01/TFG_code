import tensorflow.contrib.learn as skflow
import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
import re
import string

path = "../966MB_UGR16.csv"
# This file is a CSV, just no CSV extension or headers
df = pd.read_csv(path, header=None)

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

col_date = df['time']
# print(type(col_date))
# print(type(col_date[3]))

# def remove_punctuation(df, name, keep_apostrophe=False):
#     for i in col_date:
#         #sentence = df.at[i, 'time']
#         removed = re.sub(r'[^a-zA-Z0-9]', r' ', col_date[i])
#         test3_1 = removed.replace(" ", "")
#         return test3_1



def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    s_removed = s.replace(" ", "")
    s_int = int(s_removed)
    return s_int


#Me crea una columna nueva con los. valores transformdos asi 20160318 105240
df['cleaned'] = df['time'].apply(remove_punctuation)
col_cleaned = df['cleaned']
print(type(col_cleaned[3]))



#col_2 = remove_punctuation(df, 'time')
print(df[0:3])

def prueba(df, name):
    columna = pd.column(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)