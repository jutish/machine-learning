# https://www.aprendemachinelearning.com/sets-de-entrenamiento-test-validacion-cruzada/

# Ejemplo K-Folds en Python
# Veamos en código python usando la librería de data science scikit-learn 
# como podemos hacer el cross-validation con K-Folds:

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

print(iris.feature_names)  # The names of the dataset columns.
print(iris.target_names)  # The names of target classes.
print(iris.data[0:2,])  # The data matrix
print(iris.target[0:2,])  # The classification target
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.2, 
                                                    random_state=0)
kf = KFold(n_splits=5)
clf = LogisticRegression()
clf.fit(X_train, y_train)
score = clf.score(X_train, y_train)
print('Metrica del modelo', score)
scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring='accuracy')
print('Metricas cross validation', scores)
print('Media de cross_validation', scores.mean())
y_pred = clf.predict(X_test)
score_pred = metrics.accuracy_score(y_test, y_pred)
print('Metrica en test', score_pred)

# Recomendaciones finales:
# En principio separar Train y Test en una proporción de 80/20
# Hacer Cross Validation siempre que podamos:
# No usar K-folds. Usar Stratified-K-folds en su lugar.
# La cantidad de “folds” dependerá del tamaño del dataset que tengamos,
# pero la cantidad usual es 5 (pues es similar al 80-20 que hacemos con
# train/test).
# Para problemas de tipo time-series usar TimeSeriesSplit
# Si el Accuracy (ó métrica que usamos) es similar en los conjuntos de Train
# (donde hicimos Cross Validation) y Test, podemos dar por bueno al modelo.