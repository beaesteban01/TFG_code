import os
import pandas as pd
import numpy as np

#SI NO ESTA EN LA CARPETA NO LO LEE
#dataset = pd.read_csv("966MB_UGR16.csv")
path = "../966MB_UGR16.csv"
df = pd.read_csv(path, sep=',', names = ["Time", "Duration", "SIP", "DIP", "SPort", "DPort", "Protocol", "Flags", "FwStat", "TypOFServ", "PackEx", "NumBytes", "Ataque"])

#print(df.head()) #Comprobar que se añade la cabecera
#print (dataset.shape) #(9999998, 13)

# Strip non-numerics
df = df.select_dtypes(include=['int', 'float'])

#print(df[0:5])

#creamos esta lista de los nombres de la cabecera para que esten separados por comas y se pueda iterar sobre ellos
headers = list(df.columns.values)  
fields = []

for field in headers:
    fields.append({
        'name' : field,
        'mean': df[field].mean(),
        'var': df[field].var(),
        'sdev': df[field].std()
    })

#for field in fields:
  #  print(field)
print(fields[0:3])
#Esto solo nos da la media, la varianza y la desviacion estandar, lo quiero para algo?

#print(df[0:5])
#DE MOMENTO NO HEMOS AÑADIDO LAS COLUMNAS NO NUMERICAS





# #SHUFFLING, GROUPING AND SHORTING
# np.random.seed(42) # Uncomment this line to get the same shuffle each time
# df = df.reindex(np.random.permutation(df.index))
# df.reset_index(inplace=True, drop=True)
# print(df[0:3])

