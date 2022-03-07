# Get games results from Argentina Football League
# Only retrieve results from first division games
# https://www.promiedos.com.ar/

import pandas as pd
import requests as rq
import numpy as np
import time
from bs4 import BeautifulSoup
from datetime import datetime

# Pandas data frame to save results
data = pd.DataFrame({'date_name':[],
                     'local_team_id':[],
                     'local_team':[],
                     'local_result':[],
                     'visitor_result':[],
                     'visitor_team':[],
                     'visitor_team_id':[]})

# Get all teams id's
teams_url = 'https://www.promiedos.com.ar/historiales'
page = rq.get(teams_url).text
soup = BeautifulSoup(page, 'lxml')
teams_list = list()
for div in soup.find_all(id='clubhist'):
    teams_list.append(int(div.a['href'][12:]))

# Download game results from every team on teams_list
evaluated = list()
for team1 in teams_list:
    for team2 in teams_list:
        if team1 != team2:
            if (team1,team2) not in evaluated and \
               (team2,team1)  not in (evaluated):
                evaluated.append((team1,team2))
                # Web URL
                url = 'https://www.promiedos.com.ar/historial.php?'
                url += f'equipo1={team1}&equipo2={team2}&modo=todos'
                try:
                    print(url)
                    page = rq.get(url).text
                    soup = BeautifulSoup(page, 'lxml')
                    flag_game_date = False
                    for tr in soup.find_all('tr'):
                        try:
                            if tr['class'][0] == 'diapart':  # Game's date
                                date_name = tr.td.string
                                flag_game_date = True
                        except:
                            if flag_game_date:
                                flag_game_date = False
                                local_team = tr.contents[1].span.string
                                local_result = tr.contents[2].span.string
                                visitor_result = tr.contents[3].span.string
                                visitor_team = tr.contents[4].span.string
                                row = {'date_name':date_name,
                                       'local_team_id':team1,
                                       'local_team':local_team,
                                       'local_result':local_result,
                                       'visitor_result':visitor_result,
                                       'visitor_team':visitor_team,
                                       'visitor_team_id':team2}
                                data = data.append(row, ignore_index=True)
                                time.sleep(1)
                except Exception as e:
                    print("Ha ocurrido un error: ",type(e).__name__)

data.to_csv('results.csv')
        