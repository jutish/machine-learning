# https://www.aprendemachinelearning.com/ejemplo-web-scraping-python-ibex35-bolsa-valores/
import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime

# indicar la ruta
url_page = 'https://www.bolsamadrid.es/esp/aspx/Indices/Resumen.aspx'
page = requests.get(url_page).text
soup = BeautifulSoup(page, 'lxml')
tabla = soup.find('table', attrs={'id': 'ctl00_Contenido_tblÍndices'})
# Iteramos la tabla
name = ''
price = ''
nroFila = 0
for fila in tabla.find_all('tr'):
    if nroFila == 1:
        nroCelda = 0
        for celda in fila.find_all('td'):
            if nroCelda == 0:
                name = celda.text
                print('Indice: ', name)
            if nroCelda == 2:
                price = celda.text
                print('Precio: ', price)
        nroCelda += 1
    nroFila += 1
# Grabamos el CSV. Lo abrimos con Append para ir agregando datos.
with open('bolsa_ibex35.csv', 'a') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([name, price, datetime.now()])
