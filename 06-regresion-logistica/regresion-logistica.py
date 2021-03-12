# Para nuestro ejercicio he creado un archivo csv con datos de entrada a modo de 
# ejemplo para clasificar si el usuario que visita un sitio web usa como 
# sistema operativo Windows, Macintosh o Linux.

# Nuestra información de entrada son 4 características que tomé de una web que utiliza 
# Google Analytics y son:

# Duración de la visita en Segundos
# Cantidad de Páginas Vistas durante la Sesión
# Cantidad de Acciones del usuario (click, scroll, uso de checkbox, sliders,etc)
# Suma del Valor de las acciones (cada acción lleva asociada una valoración de importancia)
# Clase 0 (Windows) 1 (Macintosh) 2 (Linux)
# [duracion, paginas, acciones, valor, clase]

import pandas as pd 
import numpy as np 
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sb

#Leemos el .csv
df = pd.read_csv('usuarios_win_mac_lin.csv')
print(df.head())
print(df.describe())
print(df.info())

# Luego analizaremos cuantos resultados tenemos de cada tipo usando la función groupby 
# y vemos que tenemos 86 usuarios “Clase 0”, es decir Windows, 40 usuarios Mac y 44 de Linux.
print(df.groupby('clase').size())
df.drop(['clase'], 1).hist()
plt.show()
sb.pairplot(df.dropna(), hue='clase',height=4,vars=["duracion", "paginas","acciones","valor"],kind='reg')
# plt.show()

# Ahora cargamos las variables de las 4 columnas de entrada en X excluyendo la columna “clase” 
# con el método drop(). En cambio agregamos la columna “clase” en la variable y. 
# Ejecutamos X.shape para comprobar la dimensión de nuestra matriz con datos de entrada 
# de 170 registros por 4 columnas.
X = np.array(df.drop(['clase'],1))
y = np.array(df['clase'])
print(X.shape)
print(y.shape)

# Y creamos nuestro modelo y hacemos que se ajuste (fit) a nuestro conjunto de entradas X y salidas ‘y’.
model = linear_model.LogisticRegression()
model.fit(X, y)

# Una vez compilado nuestro modelo, le hacemos clasificar todo nuestro conjunto de 
# entradas X utilizando el método “predict(X)” y revisamos algunas de sus salidas 
# y vemos que coincide con las salidas reales de nuestro archivo csv.
predictions = model.predict(X)
print(predictions)

# Y confirmamos cuan bueno fue nuestro modelo utilizando model.score() 
# que nos devuelve la precisión media de las predicciones, en nuestro caso del 77%.
print(model.score(X,y))

# Validación de nuestro modelo
# Una buena práctica en Machine Learning es la de subdividir nuestro conjunto de datos 
# de entrada en un set de entrenamiento y otro para validar el modelo 
# (que no se utiliza durante el entrenamiento y por lo tanto la máquina desconoce). 
# Esto evitará problemas en los que nuestro algoritmo pueda fallar por “sobregeneralizar”
# el conocimiento.
# Para ello, subdividimos nuestros datos de entrada en forma aleatoria (mezclados) 
# utilizando 80% de registros para entrenamiento y 20% para validar.
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size = validation_size, random_state = seed)

# Volvemos a compilar nuestro modelo de Regresión Logística pero esta vez sólo con 80% 
# de los datos de entrada y calculamos el nuevo scoring que ahora nos da 74%.
# Esteban:
# Kfold y cross_val_score basicamente toman X_train e Y_train lo dividen en 10 grupos`(folds)
# con cada grupo usan 9 para entrenar el modelo (model.fit()) y uno para testear
# guardan el score de cada grupo y al final sumarizan todos los scores.
# muy bien explicado aca: https://machinelearningmastery.com/k-fold-cross-validation/
name = 'Logistic Regression'
kfold = model_selection.KFold(n_splits = 10, shuffle = True, random_state = seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

# Y ahora hacemos las predicciones -en realidad clasificación- utilizando nuestro 
# “cross validation set”, es decir del subconjunto que habíamos apartado. 
# En este caso vemos que los aciertos fueron del 85% pero hay que tener en cuenta 
# que el tamaño de datos era pequeño.
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))

# Finalmente vemos en pantalla la “matriz de confusión” donde muestra 
# cuantos resultados equivocados tuvo de cada clase (los que no están en la diagonal), 
# por ejemplo predijo 3 usuarios que eran Mac como usuarios de Windows y predijo a 2 
# usuarios Linux que realmente eran de Windows.
print(confusion_matrix(Y_validation, predictions))
#      W  M  L
# W [[16  0  2]
# M  [ 3  3  0]
# L  [ 0  0 10]]

# También podemos ver el reporte de clasificación con nuestro conjunto de Validación. 
# En nuestro caso vemos que se utilizaron como “soporte” 18 registros windows, 
# 6 de mac y 10 de Linux (total de 34 registros). Podemos ver la precisión con que se
# acertaron cada una de las clases y vemos que por ejemplo de Macintosh tuvo 3 aciertos y 
# 3 fallos (0.5 recall). La valoración que de aqui nos conviene tener en cuenta es la de 
# F1-score, que tiene en cuenta la precisión y recall. El promedio de F1 es de 84% lo cual 
# no está nada mal.
print(classification_report(Y_validation, predictions))

# Clasificación (o predicción) de nuevos valores

# Como último ejercicio, vamos a inventar los datos de entrada de  
# navegación de un usuario ficticio que tiene estos valores:

# Tiempo Duración: 10
# Paginas visitadas: 3
# Acciones al navegar: 5
# Valoración: 9
# Lo probamos en nuestro modelo y vemos que lo clasifica como un usuario tipo 2, 
# es decir, de Linux.
print(model.predict([[10, 3, 5, 9]]))
