import tensorflow.contrib.learn as skflow
import pandas as pd
import os
import numpy as np
import string
from sklearn import metrics
from scipy.stats import zscore

# import re
# import string

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

#analyze(path)

######RESULT######
#####################################################
# 9999998 rows
# ** 2016-03-18 10:52:40:10719 (0%)
# ** 0.000:35412 (0%)
# ** 127.204.60.89:892565 (8%)
# ** 42.219.153.89:4098 (0%)
# ** 123:65536 (0%)
# ** 425:64147 (0%)
# ** UDP:[TCP:62.27%,UDP:36.56%,ICMP:1.05%,GRE:0.06%,ESP:0.05%,IPIP:0.01%,IPv6:0.0%]
# ** .A....:[.A....:40.27%,.AP.SF:26.33%,.AP.S.:7.04%,.AP...:6.6%,.APRSF:4.37%,.A...F:4.04%,....S.:2.5%,.APRS.:2.18%,.AP..F:1.85%,.A..SF:1.66%,.A.R..:0.87%,.A..S.:0.64%,.A.R.F:0.45%,...R..:0.45%,.APR..:0.44%,.APR.F:0.18%,.A.RS.:0.06%,...RS.:0.04%,.A.RSF:0.02%,......:0.01%,UAP..F:0.0%,UAP.S.:0.0%,..P.S.:0.0%]
# ** 0:[0:100.0%]
# ** 0.1:[0:66.46%,40:20.13%,72:8.62%,8:1.48%,64:1.3%,42:0.76%,24:0.4%,2:0.39%,26:0.17%,104:0.07%,75:0.05%,16:0.04%,20:0.02%,192:0.02%,74:0.02%,96:0.01%,28:0.01%,4:0.01%,224:0.01%,43:0.01%,6:0.0%,18:0.0%,12:0.0%,184:0.0%,32:0.0%,73:0.0%,10:0.0%,152:0.0%,23:0.0%,88:0.0%,13:0.0%,3:0.0%,48:0.0%,66:0.0%,160:0.0%,56:0.0%,194:0.0%,80:0.0%,15:0.0%,25:0.0%,9:0.0%,1:0.0%,17:0.0%,14:0.0%,92:0.0%,200:0.0%,19:0.0%,21:0.0%,22:0.0%,30:0.0%,41:0.0%,5:0.0%,240:0.0%,11:0.0%,27:0.0%,68:0.0%,7:0.0%,136:0.0%,29:0.0%]
# ** 1:8245 (0%)
# ** 76:116716 (1%)
# ** background:[background:99.7%,blacklist:0.28%,anomaly-spam:0.02%]
########################################################

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



#LAS QUE YA SON NUMEROS
#encode_numeric_zscore(df, 'duration')
# encode_numeric_zscore(df, 'source_port')
# encode_numeric_zscore(df, 'dest_port')
# encode_numeric_zscore(df, 'forward_status')
# encode_numeric_zscore(df, 'type_service')
# encode_numeric_zscore(df, 'pack_exch')
# encode_numeric_zscore(df, 'bytes')

#DE MOMENTO SOLO TEXT-DUMMY
encode_text_dummy(df, 'protocol')
encode_text_dummy(df, 'flags')
encode_text_dummy(df, 'attack_tag')

#Me crea una columna AL FINAL nueva con los valores transformdos asi 20160318105240
df['cleaned_time'] = df['time'].apply(clean_date)
df.drop('time', axis=1, inplace=True)
df['cleaned_sip'] = df['sip'].apply(clean_ip)
df.drop('sip', axis=1, inplace=True)
df['cleaned_dip'] = df['dip'].apply(clean_ip)
df.drop('dip', axis=1, inplace=True)

encode_numeric_zscore(df, 'cleaned_time')
encode_numeric_zscore(df, 'cleaned_sip')
encode_numeric_zscore(df, 'cleaned_dip')


print(df.shape)
print(df[0:3])







