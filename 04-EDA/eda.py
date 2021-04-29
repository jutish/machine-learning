# https://www.aprendemachinelearning.com/analisis-exploratorio-de-datos-pandas-python/

# Un EDA de pocos minutos con Pandas (Python)
# Vamos a hacer un ejemplo en pandas de un EDA bastante sencillo pero con
# fines educativos.

# Vamos a leer un csv directamente desde una URL de GitHub que
# contiene información geográfica básica de los países del mundo
# y vamos a jugar un poco con esos datos.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sb

# url = 'https://raw.githubusercontent.com/lorey/list-of-countries/master/csv/
#                countries.csv'
url = 'countries.csv'
df = pd.read_csv(url, sep=';')
print(df.head(5))

# Nombre de columnas
print('Cantidad de filas y columnas: ', df.shape)
print('Nombre de columnas: ', df.columns)

# Columnas, nulos y tipo de datos
print(df.info())

# descripción estadística de los datos numéricos
print(df.describe())


# Verifiquemos si hay correlación entre los datos
corr = df.set_index('alpha_3').corr()
# Ploteamos usando statsmodel.api
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()
# Ploteamos usando seaborn (Para mas opciones ver el ejemplo usado
# en 03-naives-bayes)
sb.heatmap(corr)
plt.show()

# Cargamos un segundo archivo csv para ahondar en el crecimiento de
# la población en los últimos años, filtramos a España y visualizamos
# url = 'https://raw.githubusercontent.com/DrueStaples/Population_Growth/
#                master/countries.csv'
url = 'countries_pop.csv'
df_pop = pd.read_csv(url)
print(df_pop.head(5))
df_pop_es = df_pop[df_pop['country'] == 'Spain']
print(df_pop_es.head())
df_pop_es.drop(['country'], axis=1)['population'].plot(kind='bar')
plt.show()

# Hagamos la comparativa con otro país, por ejemplo con el crecimiento
# poblacional en Argentina
df_pop_ar = df_pop[(df_pop['country'] == 'Argentina')]
print(df_pop_ar.head(5))

anios = df_pop_es['year'].unique()
pop_ar = df_pop_ar['population'].values
pop_es = df_pop_es['population'].values
df_plot = pd.DataFrame({
    'Argentina': pop_ar,
    'Spain': pop_es
    }, index=anios)
df_plot.plot(kind='bar')
plt.show()

# Ahora filtremos todos los paises hispano-hablantes
df_espanol = df.replace(np.nan, '', regex=True)
df_espanol = df_espanol[(df_espanol['languages'].str.contains('es'))]
print(df_espanol)
df_espanol.set_index('alpha_3')[['population', 'area']].plot(kind='bar',
                                                             rot=65,
                                                             figsize=(20, 10))
plt.show()

# Vamos a hacer detección de Outliers, (con fines educativos)
# en este caso definimos como limite superior (e inferior) la media más (menos)
# “2 veces la desviación estándar” que muchas veces es tomada como máximos de
# tolerancia.
anomalies = []


# Funcion ejemplo para la deteccion de outliers
def find_anomalies(data):
    # Set upper and lower limit to 2 standar deviation
    data_std = data.std()
    data_mean = data.mean()
    anomaly_cut_off = data_std * 2
    lower_limit = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off
    print(lower_limit.iloc[0])
    print(upper_limit.iloc[0])
    # Generate outliers
    for index, row in data.iterrows():
        outlier = row  # Obtener primer columna
        if ((outlier.iloc[0] > upper_limit.iloc[0]) or
                (outlier.iloc[0] < lower_limit.iloc[0])):
            anomalies.append(index)
    return anomalies


print(find_anomalies(df_espanol.set_index('alpha_3')[['population']]))

# Quitemos BRA y USA por ser outlies y volvamos a graficar:
df_espanol.drop([30, 233], inplace=True)
df_espanol.set_index('alpha_3')[['population', 'area']].sort_values(
    ["population"]).plot(kind='bar', rot=65,
                         figsize=(20, 10))
plt.show()
