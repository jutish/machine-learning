# https://www.aprendemachinelearning.com/sistemas-de-recomendacion/

# User-based collobaritive filtering
# Agrupa a los usuarios en base a tener gustos similares, es decir como han
# clasificado n-productos. Luego ofrecen a otro usuario con gusto similar
# un producto que este ultimo no haya clasificado.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import sklearn
# Leemos los datos
df_users = pd.read_csv('users.csv')
df_repos = pd.read_csv('repos.csv')
df_ratings = pd.read_csv('ratings.csv')
# Analizamos
print(df_users.head(), df_users.shape)
print(df_repos.head(), df_repos.shape)
print(df_ratings.head(), df_ratings.shape)
print('Users: ', df_ratings.userId.unique().shape[0])
print('Repos: ', df_ratings.repoId.unique().shape[0])
plt.hist(df_ratings.rating, bins=8)
plt.show()
print(df_ratings.groupby(['rating']).size())  # Dos formas de hacer lo mismo
# Equivale a SELECT rating, COUNT(userId).. group by rating
print(df_ratings.groupby(['rating'])['userId'].count())  # Esta se usa mas
# Creamos la matriz usuarios/ratings
df_matrix = pd.pivot_table(df_ratings, values='rating', index='userId',
                           columns='repoId').fillna(0)


# Sparsity
# Veamos el porcentaje de sparcity que tenemos:
# Es decir que porcentaje del total de
# datos son 0, es decir que hay que rellenar. 
ratings_matrix = df_matrix.values
sparsity = float(len(ratings_matrix.nonzero()[0]))
sparsity /= (ratings_matrix.shape[0] * ratings_matrix.shape[1])
sparsity *= 100
print('Sparsity: {:4.2f}%'.format(100 - sparsity))

#Dividimos en train y test
ratings_train, ratings_test = train_test_split(ratings_matrix, test_size=0.2,
                                               random_state=42)

# Matriz de Similitud: Distancias por Coseno
# Representa que tan similares son los gustos de un usuario con otro
# Basado en lo que han opinado sobre un determinado producto.
sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(ratings_matrix)

plt.imshow(sim_matrix)
plt.colorbar()
# Cuanto más cercano a 1, mayor similitud entre esos usuarios.
plt.show()

# Predicciones -ó llamémosle “Sugeridos para ti”-
sim_matrix_train = sim_matrix[0:24,0:24]  #24 x 24
sim_matrix_test = sim_matrix[24:,24:]

# sim_matrix_train --> 24 x 24 Contine la similitud de gustos 
# ratings_train -->  24 x 167 Contine la puntuacion a cada producto
# En el producto punto o multiplicacion matricial el nro de columnas
# de la primera matriz debe ser igual al numero de filas de las segunda.
# la matriz obtenida es de 24 x 167 y la divide por 167 x 1 (la transpuso .T)
users_predictions = sim_matrix_train.dot(ratings_train) / np.array(
                                     [np.abs(sim_matrix_train).sum(axis=1)]).T

# print(users_predictions.shape)

plt.rcParams['figure.figsize'] = (20.0, 5.0)
plt.imshow(users_predictions);
plt.colorbar()
plt.show()

# Vamos a tomar de ejemplo mi usuario de Github que es jbagnato.
USUARIO_EJEMPLO = 'jbagnato'
data = df_users[df_users['username'] == USUARIO_EJEMPLO]
usuario_ver = data.iloc[0]['userId'] - 1  # resta 1 para obtener el index de pandas.

# print('Data:\n', data)

# Retorna un array ordenado para el usuario de menor a mayor puntuaciones
# las ultimas tres del array son las mejores predicciones. Ver argsort()
user0 = users_predictions.argsort()[usuario_ver] 

# Veamos los tres recomendados con mayor puntaje en la predic para este usuario
for i, aRepo in enumerate(user0[-3:]):
    selRepo = df_repos[df_repos['repoId'] == (aRepo+1)]
    print(selRepo.iloc[0]['title'], 'puntaje:', users_predictions[usuario_ver]
                                                                      [aRepo])

# Sobre el test set comparemos el mean squared error 
# con el conjunto de entrenamiento:

def get_mse(preds, actuals):
    if preds.shape[1] != actuals.shape[1]:
        actuals = actuals.T
    preds = preds[actuals.nonzero()].flatten()
    actuals = actuals[actuals.nonzero()].flatten()
    return mean_squared_error(preds, actuals)
 
get_mse(users_predictions, ratings_train)
 
# Realizo las predicciones para el test set
users_predictions_test = sim_matrix.dot(ratings_matrix) / np.array([np.abs(sim_matrix).sum(axis=1)]).T
users_predictions_test = users_predictions_test[24:30,:]
 
get_mse(users_predictions_test, ratings_test)


