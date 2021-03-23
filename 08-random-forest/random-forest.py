import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Cómo funciona Random Forest?
# Random Forest funciona así:

# Seleccionamos k features (columnas) de las m totales (siendo k menor a m) y
# creamos un árbol de decisión con esas k características. Creamos n árboles
# variando siempre la cantidad de k features y también podríamos variar la
# cantidad de muestras que pasamos a esos árboles
# (esto es conocido como “bootstrap sample”) Tomamos cada uno de los n árboles
# y le pedimos que hagan una misma clasificación. Guardamos el resultado de
# cada árbol obteniendo n salidas. Calculamos los votos obtenidos para cada
# “clase” seleccionada y consideraremos a la más votada como la
# clasificación final de nuestro “bosque”.

# Load csv
df = pd.read_csv('../13-clases-desbalanceadas/creditcard.csv')

# Split into train and test
X = df.drop(['Class'], axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
print('X_Train: ', X_train.shape, ' y_train: ', y_train.shape)
print('X_Test: ', X_test.shape, ' y_test: ', y_test.shape)

# Crear el modelo con 100 arboles

# n_estimators: será la cantidad de árboles que generaremos.
# max_features: la manera de seleccionar la cantidad máxima de features
# para cada árbol.
# min_sample_leaf: número mínimo de elementos en las hojas para permitir un
# nuevo split (división) del nodo. oob_score: es un método que emula el
# cross-validation en árboles y permite mejorar la precisión y
# evitar overfitting.
# boostrap: para utilizar diversos tamaños de muestras para entrenar.
# Si se pone en falso, utilizará siempre el dataset completo.
# n_jobs: si tienes multiples cores en tu CPU, puedes indicar cuantos
# puede usar el modelo al entrenar para acelerar el entrenamiento.

model = RandomForestClassifier(n_estimators=100,
                               bootstrap=True,
                               verbose=2,
                               max_features='sqrt')

# A entrenar
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Mostramos la confusion matrix
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sb.heatmap(cm, xticklabels=['Normal', 'Fraud'],
           yticklabels=['Normal', 'Fraud'],
           annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

# Los algoritmos Tree-Based -en inglés- son muchos, todos parten de la idea principal
# de árbol de decisión y la mejoran con diferentes tipos de ensambles y técnicas.Tenemos
# que destacar a 2 modelos que según el caso logran superar a las mismísimas redes neuronales!
# son XGboost y LightGBM. Si te parecen interesantes puede que en el futuro escribamos
# sobre ellos.
