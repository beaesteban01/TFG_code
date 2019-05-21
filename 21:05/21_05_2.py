import tensorflow.contrib.learn as skflow
import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore


#SI NO ESTA EN LA CARPETA NO LO LEE
#dataset = pd.read_csv("966MB_UGR16.csv")
path = "../966MB_UGR16.csv"
names = ["Time", "Duration", "SIP", "DIP", "SPort", "DPort", "Protocol", "Flags", "FwStat", "TypOFServ", "PackEx", "NumBytes", "Ataque"]
df = pd.read_csv(path, sep=',', names = names, na_values=['NA','?'] )

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
# # Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
# def encode_text_dummy(df, name):
#     dummies = pd.get_dummies(df[name])
#     for x in dummies.columns:
#         dummy_name = f"{name}-{x}"
#         df[dummy_name] = dummies[x]
#     df.drop(name, axis=1, inplace=True)

# encode_numeric_zscore(df, 'duration')
# encode_text_dummy(df, 'protocol_type')