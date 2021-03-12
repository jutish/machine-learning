import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Outliers en una dimension	- Edades
edades = np.array([22, 22, 23, 23, 23, 23, 26, 27, 27, 28, 30, 30, 30, 30, 31,
                   32, 33, 34, 80])
edad_unique, counts = np.unique(edades, return_counts=True)

sizes = counts * 100
colors = ['blue'] * len(edad_unique)
colors[-1] = 'red'

plt.axhline(1, color='k', linestyle='--')
plt.scatter(edad_unique, np.ones(len(edad_unique)), s=sizes, color=colors)
plt.yticks([])
plt.show()

# Outliers 2D variables - Edad y Salario
""" Ahora supongamos que tenemos 2 variables: edad e ingresos
Hagamos una gráfica en 2D. Además, usaremos una fórmula para
trazar un círculo que delimitará los valores outliers:
Los valores que superen el valor de la “media más 2 desvíos estándar”
(el área del círculo) quedarán en rojo."""

salario_anual_miles = np.array([16, 20, 15, 21, 19, 17, 33, 22, 31, 32, 56, 30,
                                22, 31, 30, 16, 2, 22, 23])
media_x = salario_anual_miles.mean()
std_2x = salario_anual_miles.std() * 2
media_y = edades.mean()
std_2y = edades.std() * 2

colors = ['blue'] * len(salario_anual_miles)

for index, x in enumerate(salario_anual_miles):
    if abs(x - media_x) > std_2x:
        colors[index] = 'red'

for index, x in enumerate(edades):
    if abs(x - media_y) > std_2y:
        colors[index] = 'red'

# Dentro del circulo azul, los valores que están en la media
# y en rojo los outliers: 3 valores que superan en más de
# 2 veces el desvío estándar.

plt.scatter(edades, salario_anual_miles, s=100, color=colors)
plt.axhline(media_x, color='k', linestyle='--')
plt.axvline(media_y, color='k', linestyle='--')

v = media_x  # y-position of the center
u = media_y  # x-position of the center
b = std_2x   # radius on the y-axis
a = std_2y   # radius on the x-axis
t = np.linspace(0, 2 * pi, 100)
plt.plot(u+a*np.cos(t), v+b*np.sin(t))

plt.xlabel('Edad')
plt.ylabel('Salario Anual (miles)')
plt.show()

"""
Outliers 3D variables - Edad, Salario y Compras
¿Qué pasa si añadimos una 3ra dimensión nuestro dataset?
Por ejemplo, la dimensión de “compras por mes” de cada usuario
"""

fig = plt.figure(figsize=(7, 7))
ax = fig.gca(projection='3d')

compras_mes = np.array([1, 2, 1, 20, 1, 0, 3, 2, 3, 0, 5, 3, 2, 1, 0, 1, 2, 2,
                        2])
media_z = compras_mes.mean()
std_z = compras_mes.std() * 2

for index, x in enumerate(compras_mes):
    if abs(x - media_z) > std_z:
        colors[index] = 'red'

ax.scatter(edades, salario_anual_miles, compras_mes, s=20, c=colors)
plt.xlabel('Edad')
plt.ylabel('Salario Anual (miles)')
ax.set_zlabel('Compras mensuales')
plt.show()
