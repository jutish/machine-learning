import pandas as pd
import spacy
import numpy as np
import os

# Read data
data = pd.read_csv('guardias.csv',sep=';')

print(data['Solución'])

# Load the large model to get the vectors
nlp = spacy.load('es_core_news_lg')

# Load or make a vector of the problems
if os.path.isfile('problems_vectors.npy'):
    # Load data_vector if exists
    vectors = np.load('problems_vectors.npy')
    print('Problems Vector loaded! Shape: ',vectors.shape)
else:
    # We just want the vectors so we can turn off other models in the pipeline
    # The result is a matrix of 1204 rows and 300 columns.
    with nlp.disable_pipes():
        vectors = np.array([nlp(problem).vector for problem in data['Problema']])
    # Save the problem vector to avoid the upper step next time.
    np.save('problems_vectors',vectors)
    print('Problems Vector saved! Shape: ',vectors.shape)

# For a new problem we find similar problems and check their solutions
# We use the cosine similarity.
def cosine_similarity(a, b):
    return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))

# Problem
problem = "En chazon no funciona el pasaje de datos de playa a acopios llamo\
llamo david artez"

# Vectorize the problem
problem_vec = nlp(problem).vector

# Center the document vectors
# Calculate the mean for the document vectors, should have shape (300,)
vec_mean = vectors.mean(axis=0)

# Substract the mean for the vectors
centered_vec = vectors - vec_mean

# Calculate similarities for each document in the dataset
# Make sure to subtract the mean from the problem vector
sims = np.array([cosine_similarity((problem_vec - vec_mean), centered) for centered in centered_vec])

# Order indexes from sims. It's sorted asc. 
sorted_indexes = sims.argsort()

# 
sol_qty = 10
solutions_indexes = sorted_indexes[: -(sol_qty+1) : -1]
solutions = data.iloc[solutions_indexes][['Problema','Solución']]
solutions.to_excel('Solucion.xlsx')

