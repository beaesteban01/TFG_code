import os
import pandas as pd
import numpy as np

#SI NO ESTA EN LA CARPETA NO LO LEE
#dataset = pd.read_csv("966MB_UGR16.csv")
path = "../966MB_UGR16.csv"
names = ["Time", "Duration", "SIP", "DIP", "SPort", "DPort", "Protocol", "Flags", "FwStat", "TypOFServ", "PackEx", "NumBytes", "Ataque"]
df = pd.read_csv(path, sep=',', names = names, na_values=['NA','?'] )

#print(df.head()) #Comprobar que se añade la cabecera
#print (dataset.shape) #(9999998, 13)

#Miro cuantos tipos de cada columna (no numerica) tengo
time = list(df['Time'].unique())
print(f'Number of times: {len(time)}')

sip = list(df['SIP'].unique())
print(f'Number of Source IP: {len(sip)}')

dip = list(df['DIP'].unique())
print(f'Number of dest IP: {len(dip)}')

protocol = list(df['Protocol'].unique())
print(f'Number of protocols: {len(protocol)}')
print(f'Protocols: {protocol}')

flags = list(df['Flags'].unique())
print(f'Number of flags: {len(flags)}')

attack = list(df['Ataque'].unique())
print(f'Number of attacks tags: {len(attack)}')
print(f'Attack: {attack}')