# https://www.aprendemachinelearning.com/comprende-principal-component-analysis/
# Comprende Principal Component Analysis
# En este artículo veremos una herramienta muy importante para nuestro kit de
# Machine Learning y Data Science: PCA para Reducción de dimensiones.
# Como bonus-track veremos un ejemplo rápido-sencillo en Python usando
# Scikit-learn.

# Utilizaré un archivo csv de entrada de un ejercicio anterior,
# ./03-naive-bayes/comprar_alquilar.csv
# en el cual decidíamos si convenía alquilar o comprar casa dadas
# 9 dimensiones.
# En este ejemplo:

# normalizamos los datos de entrada,
# aplicamos PCA y veremos que con 5 de las nuevas dimensiones (y descartando 4)
# obtendremos hasta un 85% de variación explicada y buenas predicciones.
# Realizaremos 2 gráficas:
# una con el acumulado de variabilidad explicada y
# una gráfica 2D, en donde el eje X e Y serán los 2 primero 
# componentes principales obtenidos por PCA.
# Y veremos cómo los resultados “comprar ó alquilar” tienen bastante
# buena separación en 2 dimensiones.

# Importamos las librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Cargamos los datos
dataFrame = pd.read_csv('../03-naive-bayes/comprar_alquilar.csv')

# Normalizamos los datos de entrada
scaler = StandardScaler()
df = dataFrame.drop(['comprar'], axis=1)  #Me quedo solo con los atributos
scaler.fit(df)  # Calculo la media para poder hacer la transformacion
X_scaled = scaler.transform(df)  # Escalo los datos y los normalizo

# Instanciamos objeto PCA y lo aplicamos
pca = PCA(n_components=9)  # Otra opción es instanciar pca sólo con dimensiones nuevas hasta obtener un mínimo "explicado" ej.: pca=PCA(.85)
pca.fit(X_scaled)  # Obtener los componentes principales
X_pca = pca.transform(X_scaled)  # Convertimos nuestros datos con las nuevas dimensiones de PCA
print('Shape of X_scaled: ', X_scaled.shape)
print('Shape of X_pca: ',X_pca.shape)
expl = pca.explained_variance_ratio_
print('Cuanto explica el modelo cada atributo: ',expl)
print('Suma: ', sum(expl[0:5]))  #Vemos que con 5 componentes tenemos algo mas del 85% de varianza explicada

# Graficamos el acumulado de varianza explicada en las nuevas dimensiones
plt.plot(np.cumsum(expl))
plt.xlabel('Number of componets')
plt.ylabel('Cumulative explained variance')
plt.show()

# Graficamos en 2 Dimensiones, tomando los 2 primeros componentes principales
Xax = X_pca[:,0]  # Todas las filas 1ra columna/atrr.
Yax = X_pca[:,1]  # Todas las filas 2da columna/attr.
labels = dataFrame['comprar'].values
cdict = {0:'red', 1:'green'}
labl = {0:'Alquilar', 1:'Comprar'}
marker = {0:'*',1:'o'}
alpha = {0:.3, 1:.5}
fig, ax = plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
    ix=np.where(labels==l)
    ax.scatter(Xax[ix],Yax[ix],c=cdict[l],label=labl[l],s=40,marker=marker[l],
    	       alpha=alpha[l])
plt.xlabel("First Principal Component", fontsize=14)
plt.ylabel("Second Principal Component", fontsize=14)
plt.legend()
plt.show()

# Conclusiones Finales
# Con PCA obtenemos:

# Una medida de como cada variable se asocia con las otras (matriz de covarianza)
# La dirección en las que nuestros datos están dispersos (autovectores)
# La relativa importancia de esas distintas direcciones (autovalores)
# PCA combina nuestros predictores y nos permite deshacernos de los autovectores
# de menor importancia relativa.


