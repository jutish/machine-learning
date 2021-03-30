# https://www.aprendemachinelearning.com/k-means-en-python-paso-a-paso/

# K-Means es un algoritmo no supervisado de Clustering. Se utiliza cuando
# tenemos un montón de datos sin etiquetar. El objetivo de este algoritmo
# es el de encontrar “K” grupos (clusters) entre los datos crudos.
# En este artículo repasaremos sus conceptos básicos y veremos un
# ejemplo paso a paso en python que podemos descargar.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D
# Seteamos el plot
plt.rcParams['figure.figsize']
plt.style.use('ggplot')
# Cargamos el csv
df = pd.read_csv('analisis.csv')
print(df.describe())

# El archivo contiene diferenciadas 9  actividades laborales-
# 1 Actor # 2 Cantante # 3 Modelo # 4 Tv/series # 5 Radio # 6 Tecnología
# 7 Deportes # 8 Politica # 9 Escritor
print(df.groupby(['categoria']).size())

# Veremos graficamente nuestros datos para tener una idea de la
# dispersión de los mismos:
df.drop(['categoria'],axis=1).hist()
plt.show()

# En este caso seleccionamos 3 dimensiones: op, ex y ag y las cruzamospara ver
# si nos dan alguna pista de su agrupación y la relación con sus categorías.
sb.pairplot(df.dropna(), hue='categoria', height=4, vars=['op', 'ex', 'ag'], 
            kind='scatter',
            diag_kind='hist')
plt.show()

# Concretamos la estructura de datos que utilizaremos para alimentar el
# algoritmo. Como se ve, sólo cargamos las columnas op, ex y ag en
# nuestra variable X.
X = np.array(df[['op', 'ex', 'ag']])
y = np.array(df['categoria'])

# Ploteamos
fig = plt.figure()
ax = Axes3D(fig)
colores = ['blue', 'red', 'green', 'blue', 'cyan', 'yellow', 'orange',
           'black', 'pink', 'brown', 'purple']
asignar = []
for row in y:
    asignar.append(colores[row])

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar, s=60)
plt.show()

# Vamos a hallar el valor de K haciendo una gráfica e intentando hallar el
# “punto de codo” que comentábamos antes. Este es nuestro resultado:
Nc = range(1, 20)
# Creo un array de objetos KMeans con distintos tamaños de Clusters uso la
# sintaxis NestedList Compresion que es propia de Python
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
# Realmente la curva es bastante “suave”. Considero a 5 como un buen número
# para K. Según vuestro criterio podría ser otro.

# Ejecutamos el algoritmo para 5 clusters y obtenemos las etiquetas y 
# los centroids.
kmeans = KMeans(n_clusters=5).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

# Ahora veremos esto en una gráfica 3D con colores para los grupos y
# veremos si se diferencian: (las estrellas marcan el centro de cada cluster)
# Predicting the clusters
labels = kmeans.predict(X)
print(labels.shape)
# Obtengo los centroides
C = kmeans.cluster_centers_
colores = ['red', 'green', 'blue', 'cyan', 'yellow']
asignar = []
for row in labels:
    asignar.append([row])
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar, s=60, alpha=1)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000, alpha=1)
# Aqui podemos ver que el Algoritmo de K-Means con K=5 ha agrupado a los
# 140 usuarios Twitter por su personalidad, teniendo en cuenta las 3
# dimensiones que utilizamos: Openess, Extraversion y Agreeablenes.
# Pareciera que no hay necesariamente una relación en los grupos con
# sus actividades de Celebrity.
plt.show()


