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


# Luego vamos a calcular las edades en una nueva columna “edad_en_billboard”
# restando el año de aparición (los 4 primeros caracteres de chart_date)
# al año de nacimiento. En las filas que estaba el año en None, tendremos como
# resultado edad None.
def calculaEdad(anio, cuando):
    cad = str(cuando)
    momento = cad[:4]
    if anio == 0.0:
        return None
    return int(momento) - anio


df['edad_en_billiboard'] = df.apply(lambda x: calculaEdad(x['anioNacimiento'],
                                    x['chart_date']), axis=1)

# Y finalmente asignaremos edades aleatorias a los registros faltantes:
# para ello, obtenemos el promedio de edad de nuestro conjunto (avg) y
# su desvío estándar (std) -por eso necesitábamos las edades en None- y
# pedimos valores random a la función que van desde avg – std hasta avg +
# std. En nuestro caso son edades de entre 21 a 37 años.

age_avg = df['edad_en_billiboard'].mean()
age_std = df['edad_en_billiboard'].std()
age_null_count = df['edad_en_billiboard'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std,
                                         size=age_null_count)
# Retorna un array [True, False, True, ...] indicando si hay nulos o no
conValoresNulos = np.isnan(df['edad_en_billiboard'])
# Donde hay valores nulos los reemplazo por la lista de edades aleatorias
df.loc[conValoresNulos, 'edad_en_billiboard'] = age_null_random_list
df['edad_en_billboard'] = df['edad_en_billiboard'].astype(int)
print('Edad promedio: ', age_avg)
print('Desvio Standard Edad: ', age_std)
print('Intervalo para asignar edad aleatoria: ', str(age_avg - age_std), ' : ',
      str(age_avg + age_std))

# Podemos visualizar los valores que agregamos (en color verde)
# En color naranja las edades que no fueron top
# Y en azul las edades que si fueron top
# El eje no es mas que el index, es decir el nro de registro.
f1 = df['edad_en_billiboard'].values
f2 = df.index
colores = ['orange', 'blue', 'green']
asignar = []
for index, row in df.iterrows():
    if(conValoresNulos[index]):
        asignar.append(colores[2])  # Verde
    else:
        asignar.append(colores[row['top']])  # 0, no top, orange - 1, top, blue

plt.scatter(f1, f2, c=asignar, s=30)
plt.axis([15, 50, 0, 650])
plt.show()

# Mapeo de datos
# Vamos a transformar varios de los datos de entrada en valores categóricos.
# Las edades, las separamos en: menor de 21 años, entre 21 y 26, etc.
# las duraciones de canciones también, por ej. entre 150 y 180 segundos,
# etc. Para los estados de ánimo (mood) agrupé los que eran similares.
# El Tempo que puede ser lento, medio o rápido queda mapeado: 0-Rapido,
# 1-Lento, 2-Medio (por cantidad de canciones en cada tempo:
# el Medio es el que más tiene)

# Mood maping
df['moodEncoded'] = df['mood'].map({
    'Energizing': 6,
    'Empowering': 6,
    'Cool': 5,
    'Yearning': 4,  # anhelo, deseo, ansia
    'Excited': 5,  # emocionado
    'Defiant': 3, 
    'Sensual': 2, 
    'Gritty': 3,  # coraje 
    'Sophisticated': 4,
    'Aggressive': 4,  # provocativo
    'Fiery': 4,  # caracter fuerte
    'Urgent': 3, 
    'Rowdy': 4,  # ruidoso alboroto
    'Sentimental': 4,
    'Easygoing': 1,  # sencillo
    'Melancholy': 4,
    'Romantic': 2,
    'Peaceful': 1,
    'Brooding': 4,  # melancolico
    'Upbeat': 5,  # optimista alegre
    'Stirring': 5,  # emocionante
    'Lively': 5,  # animado
    'Other': 0,
    '':0
    }).astype(int)

# Tempo mapping
df['tempoEncoded'] = df['tempo'].map({'Fast Tempo': 0,
    'Medium Tempo': 2,
    'Slow Tempo': 1,
    '': 0}).astype(int)

# Genre Mapping
df['genreEncoded'] = df['genre'].map({'Urban': 4,
    'Pop': 3,
    'Traditional': 2,
    'Alternative & Punk': 1,
    'Electronica': 1,
    'Rock': 1,
    'Soundtrack': 0,
    'Jazz': 0,
    'Other':0,
    '': 0}).astype(int)

# Artist_type Mapping
df['artist_typeEncoded'] = df['artist_type'].map({'Female': 2, 'Male': 3,
    'Mixed': 1, '': 0}).astype(int)

# Mapping edad en la que llegaron al billboard
df.loc[df['edad_en_billboard'] <= 21, 'edadEncoded'] = 0
df.loc[(df['edad_en_billboard'] > 21) & (df['edad_en_billboard'] <= 26),
    'edadEncoded'] = 1
df.loc[(df['edad_en_billboard'] > 26) & (df['edad_en_billboard'] <= 30),
    'edadEncoded'] = 2
df.loc[(df['edad_en_billboard'] > 30) & (df['edad_en_billboard'] <= 40),
    'edadEncoded'] = 3
df.loc[df['edad_en_billboard'] > 40, 'edadEncoded'] = 4

# Mapping Song Duration
df.loc[df['durationSeg'] <= 150, 'durationEncoded'] = 0
df.loc[(df['durationSeg'] > 150) & (df['durationSeg'] <= 180),
    'durationEncoded'] = 1
df.loc[(df['durationSeg'] > 180) & (df['durationSeg'] <= 210),
    'durationEncoded'] = 2
df.loc[(df['durationSeg'] > 210) & (df['durationSeg'] <= 240),
    'durationEncoded'] = 3
df.loc[(df['durationSeg'] > 240) & (df['durationSeg'] <= 270),
    'durationEncoded'] = 4
df.loc[(df['durationSeg'] > 270) & (df['durationSeg'] <= 300),
    'durationEncoded'] = 5
df.loc[ df['durationSeg'] > 300, 'durationEncoded'] = 6

# Finalmente obtenemos un nuevo conjunto de datos llamado artists_encoded
# con el que tenemos los atributos definitivos para crear nuestro árbol.
# Para ello, quitamos todas las columnas que no necesitamos con “drop”:
drop_elements = ['id', 'title', 'artist', 'mood', 'tempo', 'genre',
                'artist_type', 'chart_date', 'edad_en_billiboard',
                'durationSeg', 'anioNacimiento']
artists_encoded = df.drop(drop_elements, axis = 1)
print(artists_encoded.columns)

# Como quedan los top en relación a los datos mapeados
# Revisemos en tablas cómo se reparten los top=1 en los diversos atributos
# mapeados. Sobre la columna sum, estarán los top, pues al ser valor 0 o 1,
# sólo se sumarán los que sí llegaron al número 1.

print(artists_encoded[['moodEncoded', 'top']].groupby(['moodEncoded'],
    as_index=False).agg(['mean', 'count', 'sum']))
