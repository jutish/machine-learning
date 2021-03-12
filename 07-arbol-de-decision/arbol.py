# A partir de atributos de cantantes y de un histórico de canciones que
# alcanzaron entrar al Billboard 100 (U.S.) en 2013 y 2014 crearemos un árbol
# que nos permita intentar predecir si un nuevo cantante podrá llegar a
# número uno.
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage  # Shell python con esteroides,
from subprocess import check_call            # lo instale pip install ipython
from PIL import Image, ImageDraw, ImageFont  # Permite manejar archivos
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Análisis Exploratorio Inicial
df = pd.read_csv('artists_billboard_fix3.csv')
print(df.shape)
print(df.head())
print(df.columns)
# Agrupamos para saber cuantos artistas llegaron al puesto nro 1
print(df.groupby(['top']).size())
sb.catplot(x='top', data=df, kind='count')  # .catplot reemplazo a .factorplot
plt.show()
# Contamos cuantos artistas hay por sexo
print(df.groupby(['artist_type']).size())
sb.catplot(x='artist_type', data=df, kind='count')
plt.show()
# Contamos cuantas canciones por mood
print(df.groupby(['mood']).size())
sb.catplot(x='mood', data=df, kind='count', aspect=3)
plt.show()
# Contamos por tempo
print(df.groupby(['tempo']).size())
print(df.groupby(['tempo', 'top']).size())
sb.catplot(x='tempo', data=df, hue='top', kind='count')
plt.show()
# Por genero musical
print(df.groupby(['genre']).size())
sb.catplot(x='genre', data=df, kind='count', aspect=3)
plt.show()
# Por año de nacimiento de los artistas
print(df.groupby(['anioNacimiento']).size())
sb.catplot(x='anioNacimiento', data=df, kind='count', aspect=3)
plt.show()

# Como dijimos antes, no tenemos “equilibrio” en la cantidad de etiquetas
# top (1->141) y “no-top”(0->494) de las canciones.
# Para balancear en lugar de agregar mas Top's usaremos el parametro
# clas_weight

# Preparamos los datos
# Vamos a arreglar el problema de los años de nacimiento que están en cero.
# Realmente el “feature” o característica que queremos obtener es :
# “sabiendo el año de nacimiento del cantante, calcular qué edad tenía al
# momento de aparecer en el Billboard”. Por ejemplo un artista que nació en
# 1982 y apareció en los charts en 2012, tenía 30 años.


# Primero vamos a sustituir los ceros de la columna “anioNacimiento”por
# el valor None -que es es nulo en Python-.
def edad_fix(anio):
    if anio == 0:
        return None
    return anio


df['anioNacimiento'] = df.apply(lambda x:
                                edad_fix(x['anioNacimiento']), axis=1)
