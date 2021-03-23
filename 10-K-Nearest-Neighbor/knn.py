import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sb
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Leemos el .csv y describimos el mismo
df = pd.read_csv('reviews_sentiment.csv', sep=';')
print(df.head(), df.shape)
print(df.describe())

# Visualizamos
# Vemos que la distribución de “estrellas” no está balanceada.
# esto no es bueno. Convendría tener las mismas cantidades en las
# salidas, para no tener resultados “tendenciosos”. Para este ejercicio
# lo dejaremos así, pero en la vida real, debemos equilibrarlos.
# La gráfica de Valores de Sentimientos parece bastante una campana movida
# levemente hacia la derecha del cero y la cantidad de palabras se centra
# sobre todo de 0 a 10.
df.hist()
plt.show()

# Veamos realmente cuantas Valoraciones de Estrellas tenemos:
# Nota Esteban: Las estrellas son en este ejemplo nuestras clases o categorias
# Es bueno que existan una uniformidad en la cantidad de ejemplos por clase
# Cuando esto no sucede hablamos de clases desbalanceadas. En el ejercicio
# 07 y sobre todo en el 13 se trabaja sobre como solucionar esto.
print(df.groupby(['Star Rating']).size())

# Una grafica mas bonita
sb.catplot(x='Star Rating', data=df, kind='count', aspect=3)
plt.show()

# Graficamos mejor la cantidad de palabras y confirmamos que la
# mayoría están entre 1 y 10 palabras.
sb.catplot(x='wordcount', data=df, kind='count')
plt.show()

# Preparamos nuestro "X" e "y" de entrada y los set de train y test.
print(df.columns)
X = df[['wordcount', 'sentimentValue']].values
y = df['Star Rating'].values #  Values devuelve un numpy.ndarray
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)  # Escala los valores de X entre 0 y 1
X_test = scaler.fit_transform(X_test)

# Definimos el valor de k en 7 (esto realmente lo sabemos más adelante,
# ya veréis) y creamos nuestro clasificador.
k = 7
knn = KNeighborsClassifier(k)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
# NOTA: como verán utilizamos la clase KNeighborsClassifier de SciKit Learn
# puesto que nuestras etiquetas son valores discretos (estrellas del 1 al 5).
# Pero deben saber que también existe la clase KneighborsRegressor para
# etiquetas con valores continuos.
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Y ahora, la gráfica que queríamos ver!
h = .02  # step size in the mesh
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99', '#ffffb3','#b3ffff',
                             '#c2f0c2'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933', '#FFFF00', '#00ffff',
                            '#00FF00'])
# We create an instance of Neighbours Classifier and fit the data.
clf = KNeighborsClassifier(k, weights='distance')
clf.fit(X, y)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
    
patch0 = mpatches.Patch(color='#FF0000', label='1')
patch1 = mpatches.Patch(color='#ff9933', label='2')
patch2 = mpatches.Patch(color='#FFFF00', label='3')
patch3 = mpatches.Patch(color='#00ffff', label='4')
patch4 = mpatches.Patch(color='#00FF00', label='5')
plt.legend(handles=[patch0, patch1, patch2, patch3,patch4])

    
plt.title("5-Class classification (k = %i, weights = '%s')"
              % (k, 'distance'))
plt.show()

# Antes vimos que asignamos el valor n_neighbors=7 como valor de “k” y
# obtuvimos buenos resultados. ¿Pero de donde salió ese valor?.
# Pues realmente tuve que ejecutar este código que viene a continuación,
# donde vemos distintos valores k y la precisión obtenida.
# En la gráfica vemos que con valores k=7 a k=14 es donde mayor precisión
# se logra
k_range = range(1,20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])
plt.show()

# Clasificar y/o Predecir nuevas muestras
# Ya tenemos nuestro modelo y nuestro valor de k. Ahora,
# lo lógico será usarlo! Pues supongamos que nos llegan nuevas
# reviews! veamos como predecir sus estrellas de 2 maneras. La primera:
# Review de 5 palabras y 1 en Sentiment, predice que nos daran 5 Estrellas
print(clf.predict([[5, 1.0]]))
# Pero también podríamos obtener las probabilidades que de nos den 1, 2,3,4 o 5
# estrellas con predict_proba():
print(clf.predict_proba([[5, 1.0]]))