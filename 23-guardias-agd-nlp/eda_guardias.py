import pandas as pd
import matplotlib.pyplot as plt

# Guardias 2021
df = pd.read_csv('guardias_res.csv',sep=';')
inicial = df['Inicial'].value_counts()
realizo = df['Realizo'].value_counts()
final = pd.concat([inicial,realizo],axis=1)
final.plot(kind='bar')
plt.xticks(rotation=45)
plt.show()

#Guardias 2022 primer semestre
df = pd.read_csv('guardias_1_2022.csv',sep=';')
inicial = df['Inicial'].value_counts()
realizo = df['Realizo'].value_counts()
final = pd.concat([inicial,realizo],axis=1)
final.plot(kind='bar')
plt.xticks(rotation=45)
plt.show()