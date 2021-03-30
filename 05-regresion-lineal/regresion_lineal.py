# https://www.aprendemachinelearning.com/regresion-lineal-en-espanol-con-python/

# Predecir cuántas veces será compartido un artículo de Machine Learning.
# Regresión lineal simple en Python (con 1 variable)
import numpy as np # soporte para crear vectores y matrices grandes multidimensionales, junto con una gran colección de funciones matemáticas de alto nivel 
import pandas as pd #  manipulación y análisis de datos
import seaborn as sb #  data visualization library based on matplotlib
import matplotlib.pyplot as plt # generación de gráficos a partir de datos contenidos en listas o array
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Cargamos datos de entrada
data = pd.read_csv('articulos_ml.csv')
print(data.shape)
print(data.columns)
print(data.head())
print(data.info())
print(data.describe())

# Intentaremos ver con nuestra relación lineal, si hay una correlación 
# entre la cantidad de palabras del texto y la cantidad de Shares obtenidos.

#Visualizamos las caracteristicas de los datos de entrada
data.drop(['Title', 'url', 'Elapsed days'], 1).hist()
plt.show()

# Vamos a filtrar los datos de cantidad de palabras para quedarnos con los registros 
# con menos de 3500 palabras y también con los que tengan Cantidad de compartidos 
# menos a 80.000. Lo gratificaremos pintando en azul los puntos con menos de 1808 
# palabras (la media) y en naranja los que tengan más.

# Vamos a RECORTAR los datos en la zona donde se concentran más los puntos
# esto es en el eje X: entre 0 y 3.500
# y en el eje Y: entre 0 y 80.000
filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares'] <= 80000)]
colores = ['orange', 'blue']
tamanios = [30, 60]
f1 = filtered_data['Word count'].values #usando .values devuelve un numpy.ndarray sino un pandas.series
f2 = filtered_data['# Shares'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de Cantidad de Palabras (1808)
asignar = []
words_mean = data['Word count'].mean()

for index, row in filtered_data.iterrows():
	if(row['Word count'] > words_mean):
		asignar.append(colores[0])
	else:
		asignar.append(colores[1])

plt.scatter(f1, f2, c = asignar, s = tamanios[0])
plt.show()	

# Vamos a crear nuestros datos de entrada por el momento sólo Word Count y 
# como etiquetas los # Shares. Creamos el objeto LinearRegression y lo hacemos 
# “encajar” (entrenar) con el método fit(). Finalmente imprimimos los coeficientes y puntajes obtenidos.	

# dataX = filtered_data['Word count'] #Asi devuelve un series.Series
# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX = filtered_data[['Word count']] #Asi devuelve un frame.DataFrame
X_train = np.array(dataX) # Devuelve numpy.array
y_train = filtered_data['# Shares'].values # Devuelve un numpy.array
# Creamos el objeto de regresion lineal
regr = linear_model.LinearRegression()

#Entrenamos el modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coeficientes: ', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: ', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))

# De la ecuación de la recta y = mX + b nuestra pendiente “m” es el coeficiente 5,69 
# y el término independiente “b” es 11200. Tenemos un Error Cuadrático medio enorme… 
# por lo que en realidad este modelo no será muy bueno 😉 Pero estamos aprendiendo a usarlo, 
# que es lo que nos importa ahora 🙂 Esto también se ve reflejado en el puntaje de Varianza 
# que debería ser cercano a 1.0.

# Predicción en regresión lineal simple
# Vamos a intentar probar nuestro algoritmo, suponiendo que quisiéramos 
# predecir cuántos “compartir” obtendrá un articulo sobre ML de 2000 palabras
y_dosMil = regr.predict([[2000]])
print(int(y_dosMil))

# Regresión Lineal Múltiple en Python
# Vamos a extender el ejercicio utilizando más de una variable de entrada para el modelo.
# Esto le da mayor poder al algoritmo de Machine Learning, pues de esta manera podremos 
# obtener predicciones más complejas.
# Nuestra “ecuación de la Recta”, ahora pasa a ser:
# Y = b + m1 X1 + m2 X2 + … + m(n) X(n) y deja de ser una recta)

# En nuestro caso, utilizaremos 2 “variables predictivas” para poder graficar en 3D, 
# pero recordar que para mejores predicciones podemos utilizar más de 2 entradas y 
# prescindir del grafico. Nuestra primer variable seguirá siendo la cantidad de palabras 
# y la segunda variable será la suma de 3 columnas de entrada: la cantidad de enlaces, 
# comentarios y cantidad de imágenes. Vamos a programar!

# Para poder graficar en 3D, haremos una variable nueva que será la suma de los enlaces, 
# comentarios e imágenes
suma = filtered_data['# of Links'] + filtered_data['# of comments'].fillna(0) + filtered_data['# Images video']
dataX2 = pd.DataFrame()
dataX2['Word count'] = filtered_data['Word count']
dataX2['Suma'] = suma
XY_train = np.array(dataX2)
z_train = filtered_data['# Shares'].values

# Ya tenemos nuestras 2 variables de entrada en XY_train y 
# nuestra variable de salida pasa de ser “Y” a ser el eje “Z”.

#Creamos un nuevo objeto de regresion Lineal
regr2 = linear_model.LinearRegression()

# Entrenamos el modelo, esta vez, con 2 dimensiones
# obtendremos 2 coeficientes, para graficar un plano
regr2.fit(XY_train, z_train)

# Hacemos la predicción con la que tendremos puntos sobre el plano hallado
z_pred = regr2.predict(XY_train)

# Los coeficientes son 2 c/uno afecta a una variable predictiva (X1, X2) -> Y = b + m1 X1 + m2 X2
print(f'Coefecientes {regr2.coef_}')

# Error cuadrático medio
print("Mean squared error: %.2f" % mean_squared_error(z_train, z_pred))

# Evaluamos el puntaje de varianza (siendo 1.0 el mejor posible)
print('Variance score: %.2f' % r2_score(z_train, z_pred))

# Como vemos, obtenemos 2 coeficientes (cada uno correspondiente a nuestras 2 variables predictivas), 
# pues ahora lo que graficamos no será una linea si no, un plano en 3 Dimensiones.
# El error obtenido sigue siendo grande, aunque algo mejor que el anterior y 
# el puntaje de Varianza mejora casi el doble del anterior 
# (aunque sigue siendo muy malo, muy lejos del 1).

fig = plt.figure()
ax = Axes3D(fig)

# Creamos una malla, sobre la cual graficaremos el plano
xx, yy = np.meshgrid(np.linspace(0, 3500, num = 10), np.linspace(0, 60, num = 10))

fig = plt.figure()
ax = Axes3D(fig)
 
# Creamos una malla, sobre la cual graficaremos el plano
xx, yy = np.meshgrid(np.linspace(0, 3500, num=10), np.linspace(0, 60, num=10))
 
# calculamos los valores del plano para los puntos x e y
nuevoX = (regr2.coef_[0] * xx)
nuevoY = (regr2.coef_[1] * yy) 
 
# calculamos los correspondientes valores para z. Debemos sumar el punto de intercepción
z = (nuevoX + nuevoY + regr2.intercept_)
 
# Graficamos el plano
ax.plot_surface(xx, yy, z, alpha=0.2, cmap='hot')
 
# Graficamos en azul los puntos en 3D
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_train, c='blue',s=30)
 
# Graficamos en rojo, los puntos que 
ax.scatter(XY_train[:, 0], XY_train[:, 1], z_pred, c='red',s=40)
 
# con esto situamos la "camara" con la que visualizamos
ax.view_init(elev=30., azim=65)
        
ax.set_xlabel('Cantidad de Palabras')
ax.set_ylabel('Cantidad de Enlaces,Comentarios e Imagenes')
ax.set_zlabel('Compartido en Redes')
ax.set_title('Regresión Lineal con Múltiples Variables')	
plt.show()


# Si quiero predecir cuántos "Shares" voy a obtener por un artículo con: 
# 2000 palabras y con enlaces: 10, comentarios: 4, imagenes: 6
# según nuestro modelo, hacemos:
z_Dosmil = regr2.predict([[2000, 10+4+6]])
print(int(z_Dosmil))


