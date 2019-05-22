import tensorflow.contrib.learn as skflow
import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore

import string

path = "../966MB_UGR16.csv"
# This file is a CSV, just no CSV extension or headers
df = pd.read_csv(path, header=None)

print("Read {} rows.".format(len(df)))

#MENTIRA QUE NO FUNCIONA
df = df.sample(frac=0.1, replace=False) # Uncomment this line to sample only 10% of the dataset
#print(df.shape)
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


# ############ CLEAN DATE ##############
# def clean_date(s):
#     s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
#     s_removed = s.replace(" ", "")
#     s_int = int(s_removed)
#     return s_int

# #Me crea una columna AL FINAL nueva con los valores transformdos asi 20160318105240
# df['cleaned_time'] = df['time'].apply(clean_date)
# df.drop('time', axis=1, inplace=True)
# # col_cleaned = df['cleaned']
# # print(type(col_cleaned[3]))
# ############################################


########## CLEAN IP #######################
def clean_ip(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    s_int = int(s)
    return s_int

df['cleaned_sip'] = df['sip'].apply(clean_ip)
df.drop('sip', axis=1, inplace=True)

#############################################

# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


encode_numeric_zscore(df, 'cleaned_sip') #----> FUNCIONA!


#df['dip'].apply(clean_ip)


print(df[0:3])

