8# User-based collobaritive filtering
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
print(df_matrix)