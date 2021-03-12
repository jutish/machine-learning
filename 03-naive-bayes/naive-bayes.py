"""
Hoy veremos un nuevo ejercicio práctico,
intentando llevar los algoritmos de Machine Learning a
ejemplos claros y de la vida real,
repasaremos la teoría del Teorema de Bayes (video) de estadística
para poder tomar una decisión muy importante: ¿me conviene comprar casa ó
alquilar?

https://www.aprendemachinelearning.com/
        comprar-casa-o-alquilar-naive-bayes-usando-python/
"""

# Importemos las librerías que usaremos y visualicemos la información que
# tenemos de entrada:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Y carguemos la info del archivo csv:
dataFrame = pd.read_csv(r'comprar_alquilar.csv')
print(dataFrame.head(10))

# Veamos qué cantidad de muestras de comprar o alquilar tenemos:
print(dataFrame.groupby('comprar').size())

# Hagamos un histograma de las características quitando la columna de
# resultados (comprar):
dataFrame.drop(['comprar'], axis=1).hist()
plt.show()

# Vamos a hacer algo: procesemos algunas de estas columnas.
# Por ejemplo, podríamos agrupar los diversos gastos.
# También crearemos una columna llamada financiar que será la resta
# del precio de la vivienda con los ahorros de la famili
dataFrame['gastos'] = dataFrame['gastos_comunes'] + dataFrame['pago_coche']
+dataFrame['gastos_otros']
dataFrame['financiar'] = dataFrame['vivienda'] - dataFrame['ahorros']
print(dataFrame.drop(['gastos_comunes', 'pago_coche', 'gastos_otros'], axis=1)
               .head(10))

# Y ahora veamos un resumen estadístico que nos brinda la librería Pandas
# con describe():
reduced = dataFrame.drop(['gastos_comunes', 'pago_coche', 'gastos_otros'],
                         axis=1).describe()
print(reduced)

# Feature Selection ó Selección de Características
# En este ejercicio haremos Feature Selection para mejorar nuestros resultados
# con este algoritmo. En vez de utilizar las 11 columnas de datos de  entrada
# que tenemos, vamos a  utilizar una Clase de SkLearn llamada SelectKBest con
# la que  seleccionaremos las 5 mejores # características y usaremos sólo esas
X = dataFrame.drop(['comprar'], axis=1)
y = dataFrame['comprar']

best = SelectKBest(k=5)
X_new = best.fit_transform(X, y)
X_new.shape
selected = best.get_support(indices=True)
print(X.columns[selected])

# Bien, entonces usaremos 5 de las 11 características que teníamos.
# Las que “más aportan” al momento de clasificar. Veamos qué grado de
# correlación tienen: ['ingresos', 'ahorros', 'hijos', 'trabajo', 'financiar']
used_features = X.columns[selected]
colormap = plt.cm.viridis
plt.figure(figsize=(12, 12))
plt.title('Pearsons correlation of Features', y=1.05, size=15)
sb.heatmap(dataFrame[used_features].astype(float).corr(), linewidths=0.1,
           vmax=1.0, square=True, cmap=colormap, linecolor='white',
           annot=True)
plt.show()

# Crear el modelo Gaussian Naive Bayes con SKLearn
# Primero vamos a dividir nuestros datos de entrada en entrenamiento y test.
X_train, X_test = train_test_split(dataFrame, test_size=0.2, random_state=6)
y_train = X_train['comprar']
y_test = X_test['comprar']

# Y creamos el modelo, lo ponemos a aprender con fit() y
# obtenemos predicciones sobre nuestro conjunto de test

# Instantiate the classifier
gnb = GaussianNB()

# Train classifier
gnb.fit(
    X_train[used_features].values,
    y_train
)

y_pred = gnb.predict(X_test[used_features])

print('Precisión en el set de Entrenamiento: {:.2f}'
      .format(gnb.score(X_train[used_features], y_train)))
print('Precisión en el set de Test: {:.2f}'
      .format(gnb.score(X_test[used_features], y_test)))

# Probemos el modelo: ¿Comprar o Alquilar?
# Ahora, hagamos 2 predicciones para probar nuestra máquina:
# Será una familia sin hijos con 2.000€ de ingresos que quiere comprar
# una casa de 200.000€ y tiene sólo 5.000€ ahorrados.
# El otro será una familia con 2 hijos con ingresos por 6.000€ al mes, 34.000
# en ahorros y consultan si comprar una casa de 320.000€.

#                 ['ingresos', 'ahorros', 'hijos', 'trabajo', 'financiar']
print(gnb.predict([[2000,        5000,     0,       5,         200000],
                   [6000,        34000,    2,       5,         320000]]))
# Resultado esperado 0-Alquilar, 1-Comprar casa
