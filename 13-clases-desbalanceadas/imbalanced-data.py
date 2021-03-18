# https://www.aprendemachinelearning.com/clasificacion-con-datos-desbalanceados/
# Instala la librería de Imbalanced Learn desde linea de comando con:
# (toda la documentación en la web oficial imblearn)
# pip install -U imbalanced-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
from collections import Counter

LABELS = ['Normal', 'Fraud']

df = pd.read_csv('creditcard.csv')
print(df.head())

# Veamos cuantas filas tenemos y cuantas hay de cada clase:
print(df.shape)
print(pd.value_counts(df['Class'], sort=True))
count_classes = pd.value_counts(df['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.xticks(range(2), LABELS)
plt.title("Frequency by observation number")
plt.xlabel("Class")
plt.ylabel("Number of Observations")
plt.show()

# Estrategias para el manejo de Datos Desbalanceados:

# Ajuste de Parámetros del modelo: Ejemplos son ajuste de peso en árboles,
# también en logisticregression`tenemos el parámetro class_weight= “balanced”
# que utilizaremos en este ejemplo.

# Modificar el Dataset: Podemos eliminar muestras de la clase mayoritaria
# para reducirlo e intentar equilibrar la situación. Tiene como “peligroso”
# que podemos prescindir de muestras importantes

# Muestras artificiales: Podemos intentar crear muestras sintéticas
# (no idénticas) utilizando diversos algoritmos que intentan seguir
# la tendencia del grupo

# Balanced Ensemble Methods: Utiliza las ventajas de hacer ensamble de
# métodos, es decir, entrenar diversos modelos y entre todos obtener el
# resultado final (por ejemplo “votando”) pero se asegura de tomar muestras
# de entrenamiento equilibradas.

# Probando el Modelo “a secas” -sin estrategias- con LogisticRegression
# Definimos nuestras muestras (X) y etiquetas (y)
X = df.drop(['Class'], axis=1)
y = df['Class']
# Dividimos en sets de entrenamiento (70%) y prueba(30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)


# Creamos una funcion que crea el modelo que usaremos cada vez
def run_model(X_train, X_test, y_train, y_test):
    clf_base = LogisticRegression(C=1.0, penalty='l2', random_state=1,
                                  solver='newton-cg')
    clf_base.fit(X_train, y_train)
    return clf_base


# Ejecutamos el modelo tal cual
model = run_model(X_train, X_test, y_train, y_test)


# Definimos una funcion para mostrar los resultados
def mostrar_resultados(y_test, y_pred, estrategia=''):
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'******** {estrategia} **********')
    print(classification_report(y_test, y_pred))
    plt.figure(figsize=(12, 12))
    sb.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS,
               annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()


# Aqui vemos la confusion matrix y en la clase 2 (es lo que nos interesa
# detectar) vemos 51 fallos y 97 aciertos dando un recall de 0.66 y es el
# valor que queremos mejorar.
y_pred = model.predict(X_test)
mostrar_resultados(y_test, y_pred, 'Regresion Logistica')


# Estrategia: Penalización para compensar
def run_model_balanced(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(C=1.0, penalty='l2', random_state=1,
                             solver='newton-cg', class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf


model = run_model_balanced(X_train, X_test, y_train, y_test)
y_pred = model.predict(X_test)
mostrar_resultados(y_test, y_pred, 'Penalizacion')


# Estrategia: Undersampling en la clase mayoritaria
# Lo que haremos es utilizar un algoritmo para reducir la clase mayoritaria.
# Lo haremos usando un algoritmo que hace similar al k-nearest neighbor para
# ir seleccionando cuales eliminar. Fijemonos que reducimos bestialmente de
# 199.020 muestras de clase cero (la mayoría) y pasan a ser 688. y Con esas
# muestras entrenamos el modelo.

us = NearMiss(sampling_strategy=0.5, n_neighbors=3, version=2)

X_train_res, y_train_res = us.fit_resample(X_train, y_train)
print(f'Distribution before resampling {Counter(y_train)}')
print(f'Distribution after resampling {Counter(y_train_res)}')

model = run_model(X_train_res, X_test, y_train_res, y_test)
y_pred = model.predict(X_test)
mostrar_resultados(y_test, y_pred, 'Undersampling')

# Estrategia: Oversampling de la clase minoritaria
# En este caso, crearemos muestras nuevas “sintéticas” de la clase minoritaria.
# Usando RandomOverSampler. Y vemos que pasamos de 344 muestras de
# fraudes a 99.510.
os = RandomOverSampler(sampling_strategy=0.5)
X_train_res, y_train_res = os.fit_resample(X_train, y_train)
print(f'Distribution before resampling {Counter(y_train)}')
print(f'Distribution after resampling {Counter(y_train_res)}')
model = run_model(X_train_res, X_test, y_train_res, y_test)
y_pred = model.predict(X_test)
mostrar_resultados(y_test, y_pred, 'Oversampling')

# Estrategia: Combinamos resampling con Smote-Tomek
# Ahora probaremos una técnica muy usada que consiste en aplicar
# en simultáneo un algoritmo de undersampling y otro de oversampling
# a la vez al dataset. En este caso usaremos SMOTE para oversampling:
# busca puntos vecinos cercanos y agrega puntos “en linea recta” entre ellos.
# Y usaremos Tomek para undersampling que quita los de distinta clase que sean
# nearest neighbor y deja ver mejor el decisión boundary
# (la zona limítrofe de nuestras clases).
os_us = SMOTETomek(sampling_strategy=0.5)
X_train_res, y_train_res = os_us.fit_resample(X_train, y_train)
print(f'Distribution before resampling {Counter(y_train)}')
print(f'Distribution after resampling {Counter(y_train_res)}')
model = run_model(X_train_res, X_test, y_train_res, y_test)
y_pred = model.predict(X_test)
mostrar_resultados(y_test, y_pred, 'Smote-Tomek')

# Estrategia: Ensamble de Modelos con Balanceo
# Para esta estrategia usaremos un Clasificador de Ensamble
# que usa Bagging y el modelo será un DecisionTree. Veamos como se comporta:
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)
# Train the classifier
bbc.fit(X_train, y_train)
y_pred = bbc.predict(X_test)
mostrar_resultados(y_test, y_pred, 'Ensamble BBC')
