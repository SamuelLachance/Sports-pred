from optparse import OptionContainer
from textblob.en.np_extractors import filter_insignificant
from http.cookiejar import LoadError
import os
import requests
from textblob import TextBlob
import datetime
import re
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import statsmodels.api as sm
import pickle
import time
import random
import json
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#remove accents

team_abbr_to_name = {
    'ANA': 'Anaheim Ducks',
    'BOS': 'Boston Bruins',
    'BUF': 'Buffalo Sabres',
    'CGY': 'Calgary Flames',
    'CAL': 'Calgary Flames',
    'CAR': 'Carolina Hurricanes',
    'CHI': 'Chicago Blackhawks',
    'COL': 'Colorado Avalanche',
    'CBJ': 'Columbus Blue Jackets',
    'CLB':'Columbus Blue Jackets',
    'DAL': 'Dallas Stars',
    'DET': 'Detroit Red Wings',
    'EDM': 'Edmonton Oilers',
    'FLA': 'Florida Panthers',
    'LAK': 'Los Angeles Kings',
    'LA': 'Los Angeles Kings',
    'MIN': 'Minnesota Wild',
    'MTL': 'Montreal Canadiens',
    'MON': 'Montreal Canadiens',
    'NSH': 'Nashville Predators',
    'NAS': 'Nashville Predators',
    'NJD': 'New Jersey Devils',
    'NJ': 'New Jersey Devils',
    'NYI': 'New York Islanders',
    'NYR': 'New York Rangers',
    'OTT': 'Ottawa Senators',
    'PHI': 'Philadelphia Flyers',
    'PIT': 'Pittsburgh Penguins',
    'SEA': 'Seattle Kraken',
    'SJS': 'San Jose Sharks',
    'SJ': 'San Jose Sharks',
    'STL': 'St. Louis Blues',
    'TBL': 'Tampa Bay Lightning',
    'TB': 'Tampa Bay Lightning',
    'TOR': 'Toronto Maple Leafs',
    'UTA': 'Utah Utah HC',
    'UTAH': 'Utah Utah HC',
    'VAN': 'Vancouver Canucks',
    'VGK': 'Vegas Golden Knights',
    'VEG': 'Vegas Golden Knights',
    'WSH': 'Washington Capitals',
    'WAS': 'Washington Capitals',
    'WPG': 'Winnipeg Jets',
    'WIN': 'Winnipeg Jets'
}

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii

def weird_division(x,y):
    if y == 0:
        return 0
    return x / y

#get the time
yesterday = (datetime.date.today()- datetime.timedelta(days=1)).strftime('%Y-%m-%d')
today = (datetime.date.today() + datetime.timedelta(days=0)).strftime('%Y-%m-%d')
year = 20242025

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

urlNHL = 'https://www.naturalstattrick.com/teamtable.php?fromseason=20242025&thruseason=20242025&stype=2&sit=sva&score=all&rate=y&team=all&loc=B&gpf=c&gp=10&fd=&td='

# Make the request with headers
req = requests.get(urlNHL, headers=headers)

print(req.text) 

# Parse the content
soup = BeautifulSoup(req.content, 'html5lib')

print(soup)

top = list(soup.children)

body = soup.find('body')

print(body)

body.find_all('th')

columns = [item.text for item in body.find_all('th')]
columns

body.find_all('td')

data = [e.text for e in body.find_all('td')]

start = 0
table= []

#loop through entire data

while start+len(columns) <= len(data):
    player = []
    #use length of columns as iteration stop point to get list of info for 1 player
    for i in range(start,start+len(columns)):
        player.append(data[i])
    #add player row to list
    table.append(player)
    #start at next player
    start += len(columns)

df = pd.DataFrame(table, columns = columns, dtype = 'float').set_index('')

df.head()

urlNHL2 = 'https://www.naturalstattrick.com/teamtable.php?fromseason=20242025&thruseason=20242025&stype=2&sit=pp&score=all&rate=y&team=all&loc=B&gpf=c&gp=25&fd=&td='
req2 = requests.get(urlNHL2, headers=headers)

soup2 = BeautifulSoup(req2.content, 'html5lib')

top2 = list(soup2.children)

body2 = soup2.find('body')

body2.find_all('th')

columns2 = [item.text for item in body2.find_all('th')]
columns2

body2.find_all('td')

data2 = [e.text for e in body2.find_all('td')]

start2 = 0
table2 = []

#loop through entire data

while start2+len(columns2) <= len(data2):
    player = []
    #use length of columns as iteration stop point to get list of info for 1 player
    for i in range(start2,start2+len(columns2)):
        player.append(data2[i])
    #add player row to list
    table2.append(player)
    #start at next player
    start2 += len(columns2)

df2 = pd.DataFrame(table2, columns = columns2, dtype = 'float').set_index('')

df2.head()

urlNHL3 = 'https://www.naturalstattrick.com/teamtable.php?fromseason=20242025&thruseason=20242025&stype=2&sit=pk&score=all&rate=y&team=all&loc=B&gpf=c&gp=25&fd=&td='
req3 = requests.get(urlNHL3, headers=headers)

soup3 = BeautifulSoup(req3.content, 'html5lib')

top3 = list(soup3.children)

body3 = soup3.find('body')

body3.find_all('th')

columns3 = [item.text for item in body3.find_all('th')]
columns3

body3.find_all('td')

data3 = [e.text for e in body3.find_all('td')]

start3 = 0
table3 = []
while start3+len(columns3) <= len(data3):
    player = []
    #use length of columns as iteration stop point to get list of info for 1 player
    for i in range(start3,start3+len(columns3)):
        player.append(data3[i])
    #add player row to list
    table3.append(player)
    #start at next player
    start3 += len(columns3)

df3 = pd.DataFrame(table3, columns = columns3, dtype = 'float').set_index('')

df3.head()

urlNHL4 = 'https://www.naturalstattrick.com/teamtable.php?fromseason=20242025&thruseason=20242025&stype=2&sit=sva&score=all&rate=y&team=all&loc=B&gpf=10&fd=&td='
req4 = requests.get(urlNHL4, headers=headers)

soup4 = BeautifulSoup(req4.content, 'html5lib')

top4 = list(soup4.children)

body4 = soup4.find('body')

body4.find_all('th')

columns4 = [item.text for item in body4.find_all('th')]
columns4

body4.find_all('td')

data4 = [e.text for e in body4.find_all('td')]

start4 = 0
table4 = []


#loop through entire data

while start4+len(columns4) <= len(data4):
    player = []
    #use length of columns as iteration stop point to get list of info for 1 player
    for i in range(start4,start4+len(columns4)):
        player.append(data4[i])
    #add player row to list
    table4.append(player)
    #start at next player
    start4 += len(columns4)

df4 = pd.DataFrame(table4, columns = columns4, dtype = 'float').set_index('')

df4.head()

urlNHL5 = 'https://www.naturalstattrick.com/teamtable.php?fromseason=20242025&thruseason=20242025&stype=2&sit=pp&score=all&rate=y&team=all&loc=B&gpf=25&fd=&td='
req5 = requests.get(urlNHL5, headers=headers)

soup5 = BeautifulSoup(req5.content, 'html5lib')

top5 = list(soup5.children)

body5 = soup5.find('body')

body5.find_all('th')

columns5 = [item.text for item in body5.find_all('th')]
columns5

body5.find_all('td')

data5 = [e.text for e in body5.find_all('td')]

start5 = 0
table5 = []



#loop through entire data

while start5+len(columns5) <= len(data5):
    player = []
    #use length of columns as iteration stop point to get list of info for 1 player
    for i in range(start5,start5+len(columns5)):
        player.append(data5[i])
    #add player row to list
    table5.append(player)
    #start at next player
    start5 += len(columns5)

df5 = pd.DataFrame(table5, columns = columns5, dtype = 'float').set_index('')

df5.head()

urlNHL6 = 'https://www.naturalstattrick.com/teamtable.php?fromseason=20242025&thruseason=20242025&stype=2&sit=pk&score=all&rate=y&team=all&loc=B&gpf=25&fd=&td='
req6 = requests.get(urlNHL6, headers=headers)

soup6 = BeautifulSoup(req6.content, 'html5lib')

top6 = list(soup6.children)

body6 = soup6.find('body')

body6.find_all('th')

columns6 = [item.text for item in body6.find_all('th')]
columns6

body6.find_all('td')

data6 = [e.text for e in body6.find_all('td')]

start6 = 0
table6 = []



#loop through entire data

while start6+len(columns6) <= len(data6):
    player = []
    #use length of columns as iteration stop point to get list of info for 1 player
    for i in range(start6,start6+len(columns6)):
        player.append(data6[i])
    #add player row to list
    table6.append(player)
    #start at next player
    start6 += len(columns6)

df6 = pd.DataFrame(table6, columns = columns6, dtype = 'float').set_index('')

df6.head()


# Set up API keys and endpoint
bearer_token = 'AAAAAAAAAAAAAAAAAAAAALipdwEAAAAAjEJzzmqWrvdlqLlu6XBTOxo%2Fjnc%3DyB7WhJ0vjDOJoano9MJFCO4s7R7aJTJuxjbU7tLGeoxSx9m4X3'
odds_api_key = '8be3ba1d05ea7d3cda1d4ec6953e78c9'
endpoint = 'https://api.twitter.com/2/tweets/search/recent'
odds_endpoint = 'https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds'


# Get the odds for today's NHL games
params = {
    'regions': 'us',
    'oddsFormat': 'decimal',
    'dateFormat': 'iso',
    'apiKey': odds_api_key,
    'sport': 'icehockey_nhl',
    'date': today,
    'Markets' : 'h2h',
    'Bookmakers' : 'fanduel'
}
response_odds = requests.get(odds_endpoint, params=params)

url = f"https://api-web.nhle.com/v1/schedule/{today}"
responseNHL = requests.get(url)
dataNHL = responseNHL.json()

print(dataNHL)


# Set up API request
url2 = f"https://api-web.nhle.com/v1/schedule/{today}"

# Make API request
responseNHL2 = requests.get(url2)
dataNHL2 = responseNHL2.json()['gameWeek'][0]['games']
# Analyser chaque équipe séparément

for game in dataNHL['gameWeek'][0]['games']:
    home_team = team_abbr_to_name.get(game['homeTeam']['abbrev'])
    away_team = team_abbr_to_name.get(game['awayTeam']['abbrev'])
    team_df_H = df.loc[df['Team'] == home_team]
    team_df_A = df.loc[df['Team'] == away_team]
    team_df_H2 = df2.loc[df2['Team'] == home_team]
    team_df_A2 = df2.loc[df2['Team'] == away_team]
    team_df_H3 = df3.loc[df3['Team'] == home_team]
    team_df_A3 = df3.loc[df3['Team'] == away_team]
    team_df_H4 = df4.loc[df3['Team'] == home_team]
    team_df_A4 = df4.loc[df3['Team'] == away_team]
    team_df_H5 = df5.loc[df3['Team'] == home_team]
    team_df_A5 = df5.loc[df3['Team'] == away_team]
    team_df_H6 = df6.loc[df3['Team'] == home_team]
    team_df_A6 = df6.loc[df3['Team'] == away_team]


    try:
      home_team.index('Canadiens')
    except ValueError:
      try:
        home_team.index('Blues')
      except ValueError:
        team_df_H = df.loc[df['Team'] == home_team]
      else:
        team_df_H = df.loc[df['Team'] == 'St Louis Blues']
    else:
      team_df_H = df.loc[df['Team'] == 'Montreal Canadiens']

    try:
        away_team.index('Canadiens')
    except ValueError:
        try:
            away_team.index('Blues')
        except ValueError:
            team_df_A = df.loc[df['Team'] == away_team]
        else:
            team_df_A = df.loc[df['Team'] == 'St Louis Blues']
    else:
        team_df_A = df.loc[df['Team'] == 'Montreal Canadiens']

    try:
      home_team.index('Canadiens')
    except ValueError:
      try:
        home_team.index('Blues')
      except ValueError:
        team_df_H2 = df2.loc[df2['Team'] == home_team]
      else:
        team_df_H2 = df2.loc[df2['Team'] == 'St Louis Blues']
    else:
      team_df_H2 = df2.loc[df2['Team'] == 'Montreal Canadiens']

    try:
        away_team.index('Canadiens')
    except ValueError:
        try:
            away_team.index('Blues')
        except ValueError:
            team_df_A2 = df2.loc[df2['Team'] == away_team]
        else:
            team_df_A2 = df2.loc[df2['Team'] == 'St Louis Blues']
    else:
        team_df_A2 = df2.loc[df2['Team'] == 'Montreal Canadiens']

    try:
      home_team.index('Canadiens')
    except ValueError:
      try:
        home_team.index('Blues')
      except ValueError:
        team_df_H3 = df3.loc[df3['Team'] == home_team]
      else:
        team_df_H3 = df3.loc[df3['Team'] == 'St Louis Blues']
    else:
      team_df_H3 = df3.loc[df3['Team'] == 'Montreal Canadiens']

    try:
        away_team.index('Canadiens')
    except ValueError:
        try:
            away_team.index('Blues')
        except ValueError:
            team_df_A3 = df3.loc[df3['Team'] == away_team]
        else:
            team_df_A3 = df3.loc[df3['Team'] == 'St Louis Blues']
    else:
        team_df_A3 = df3.loc[df3['Team'] == 'Montreal Canadiens']

    try:
      home_team.index('Canadiens')
    except ValueError:
      try:
        home_team.index('Blues')
      except ValueError:
        team_df_H4 = df4.loc[df4['Team'] == home_team]
      else:
        team_df_H4 = df4.loc[df4['Team'] == 'St Louis Blues']
    else:
      team_df_H4 = df4.loc[df4['Team'] == 'Montreal Canadiens']

    try:
        away_team.index('Canadiens')
    except ValueError:
        try:
            away_team.index('Blues')
        except ValueError:
            team_df_A4 = df4.loc[df4['Team'] == away_team]
        else:
            team_df_A4 = df4.loc[df4['Team'] == 'St Louis Blues']
    else:
        team_df_A4 = df4.loc[df4['Team'] == 'Montreal Canadiens']

    try:
      home_team.index('Canadiens')
    except ValueError:
      try:
        home_team.index('Blues')
      except ValueError:
        team_df_H5 = df5.loc[df5['Team'] == home_team]
      else:
        team_df_H5 = df5.loc[df5['Team'] == 'St Louis Blues']
    else:
      team_df_H5 = df5.loc[df5['Team'] == 'Montreal Canadiens']

    try:
        away_team.index('Canadiens')
    except ValueError:
        try:
            away_team.index('Blues')
        except ValueError:
            team_df_A5 = df5.loc[df5['Team'] == away_team]
        else:
            team_df_A5 = df5.loc[df5['Team'] == 'St Louis Blues']
    else:
        team_df_A5 = df5.loc[df5['Team'] == 'Montreal Canadiens']

    try:
      home_team.index('Canadiens')
    except ValueError:
      try:
        home_team.index('Blues')
      except ValueError:
        team_df_H6 = df6.loc[df6['Team'] == home_team]
      else:
        team_df_H6 = df6.loc[df6['Team'] == 'St Louis Blues']
    else:
      team_df_H6 = df6.loc[df6['Team'] == 'Montreal Canadiens']

    try:
        away_team.index('Canadiens')
    except ValueError:
        try:
            away_team.index('Blues')
        except ValueError:
            team_df_A6 = df6.loc[df6['Team'] == away_team]
        else:
            team_df_A6 = df6.loc[df6['Team'] == 'St Louis Blues']
    else:
        team_df_A6 = df6.loc[df6['Team'] == 'Montreal Canadiens']


    if response_odds.status_code == 200:
      data_odds = response_odds.json()
      for game in data_odds:

        away_team2 = game['away_team']
        home_team2 = game['home_team']



        if game['bookmakers'][0]['markets'][0]['outcomes'][0]['name'] == home_team2 :
          home_odds = game['bookmakers'][0]['markets'][0]['outcomes'][0]['price']
        else:
          home_odds = game['bookmakers'][0]['markets'][0]['outcomes'][1]['price']

        if game['bookmakers'][0]['markets'][0]['outcomes'][0]['name'] == away_team2 :
          away_odds = game['bookmakers'][0]['markets'][0]['outcomes'][0]['price']
        else:
           away_odds = game['bookmakers'][0]['markets'][0]['outcomes'][1]['price']
    else:
      print(f"Error getting odds: {response_odds.status_code}")

    HDCF_A = team_df_A['HDCF/60'].values
    HDCF_H = team_df_H['HDCF/60'].values
    HDCA_A = team_df_A['HDCA/60'].values
    HDCA_H = team_df_H['HDCA/60'].values

    HDSH_A = (team_df_A4['HDGF/60'].values*100)/team_df_A4['HDCF/60'].values
    HDSH_H = (team_df_H4['HDGF/60'].values*100)/team_df_H4['HDCF/60'].values

    HDSV_A = 100 - ((team_df_A4['HDGA/60'].values*100)/team_df_A4['HDCA/60'].values)
    HDSV_H = 100 - ((team_df_H4['HDGA/60'].values*100)/team_df_H4['HDCA/60'].values)

    MDCF_A = team_df_A['MDCF/60'].values
    MDCF_H = team_df_H['MDCF/60'].values
    MDCA_A = team_df_A['MDCA/60'].values
    MDCA_H = team_df_H['MDCA/60'].values

    MDSH_A = (team_df_A4['MDGF/60'].values*100)/team_df_A4['MDCF/60'].values
    MDSH_H = (team_df_H4['MDGF/60'].values*100)/team_df_H4['MDCF/60'].values

    MDSV_A = 100 - ((team_df_A4['MDGA/60'].values*100)/team_df_A4['MDCA/60'].values)
    MDSV_H = 100 - ((team_df_H4['MDGA/60'].values*100)/team_df_H4['MDCA/60'].values)

    XGF_A = team_df_A['xGF/60'].values
    XGF_H = team_df_H['xGF/60'].values

    XGA_A = team_df_A['xGA/60'].values
    XGA_H = team_df_H['xGA/60'].values

    HDCF_A2 = team_df_A2['HDCF/60'].values
    HDCF_H2 = team_df_H2['HDCF/60'].values
    HDCA_A2 = team_df_A2['HDCA/60'].values
    HDCA_H2 = team_df_H2['HDCA/60'].values

    HDSH_A2 = weird_division((team_df_A5['HDGF/60'].values*100),team_df_A5['HDCF/60'].values)
    HDSH_H2 = weird_division((team_df_H5['HDGF/60'].values*100),team_df_H5['HDCF/60'].values)

    HDSV_A2 = 100 - weird_division((team_df_A5['HDGA/60'].values*100),team_df_A5['HDCA/60'].values)
    HDSV_H2 = 100 - weird_division((team_df_H5['HDGA/60'].values*100),team_df_H5['HDCA/60'].values)

    MDCF_A2 = team_df_A2['MDCF/60'].values
    MDCF_H2 = team_df_H2['MDCF/60'].values
    MDCA_A2 = team_df_A2['MDCA/60'].values
    MDCA_H2 = team_df_H2['MDCA/60'].values

    MDSH_A2 = weird_division((team_df_A5['MDGF/60'].values*100),team_df_A5['MDCF/60'].values)
    MDSH_H2 = weird_division((team_df_H5['MDGF/60'].values*100),team_df_H5['MDCF/60'].values)

    MDSV_A2 = 100 - weird_division((team_df_A5['MDGA/60'].values*100),team_df_A5['MDCA/60'].values)
    MDSV_H2 = 100 - weird_division((team_df_H5['MDGA/60'].values*100),team_df_H5['MDCA/60'].values)

    XGF_A2 = team_df_A2['xGF/60'].values
    XGF_H2 = team_df_H2['xGF/60'].values

    XGA_A2 = team_df_A2['xGA/60'].values
    XGA_H2 = team_df_H2['xGA/60'].values

    HDCF_A3 = team_df_A3['HDCF/60'].values
    HDCF_H3 = team_df_H3['HDCF/60'].values
    HDCA_A3 = team_df_A3['HDCA/60'].values
    HDCA_H3 = team_df_H3['HDCA/60'].values

    HDSH_A3 = weird_division((team_df_A6['HDGF/60'].values*100),team_df_A6['HDCF/60'].values)
    HDSH_H3 = weird_division((team_df_H6['HDGF/60'].values*100),team_df_H6['HDCF/60'].values)

    HDSV_A3 = 100 - weird_division((team_df_A6['HDGA/60'].values*100),team_df_A6['HDCA/60'].values)
    HDSV_H3 = 100 - weird_division((team_df_H6['HDGA/60'].values*100),team_df_H6['HDCA/60'].values)

    MDCF_A3 = team_df_A3['MDCF/60'].values
    MDCF_H3 = team_df_H3['MDCF/60'].values
    MDCA_A3 = team_df_A3['MDCA/60'].values
    MDCA_H3 = team_df_H3['MDCA/60'].values

    MDSH_A3 = weird_division((team_df_A6['MDGF/60'].values*100),team_df_A6['MDCF/60'].values)
    MDSH_H3 = weird_division((team_df_H6['MDGF/60'].values*100),team_df_H6['MDCF/60'].values)

    MDSV_A3 = 100 - weird_division((team_df_A6['MDGA/60'].values*100),team_df_A6['MDCA/60'].values)
    MDSV_H3 = 100 - weird_division((team_df_H6['MDGA/60'].values*100),team_df_H6['MDCA/60'].values)

    XGF_A3 = team_df_A3['xGF/60'].values
    XGF_H3 = team_df_H3['xGF/60'].values

    XGA_A3 = team_df_A3['xGA/60'].values
    XGA_H3 = team_df_H3['xGA/60'].values

    PDO_A = team_df_A['PDO'].values
    PDO_H = team_df_H['PDO'].values

    time_array = team_df_A2['TOI/GP'].values
    time_str = time_array.astype(str)
    format_str = '%H:%M'
    time_in_hours = []
    for t in time_str:
      datetime_obj = datetime.datetime.strptime(t, format_str)
      time_in_hours.append(datetime_obj.hour + datetime_obj.minute/60)

    time_array2 = team_df_H2['TOI/GP'].values
    time_str2 = time_array2.astype(str)
    format_str2 = '%H:%M'
    time_in_hours2 = []
    for t in time_str2:
      datetime_obj = datetime.datetime.strptime(t, format_str2)
      time_in_hours2.append(datetime_obj.hour + datetime_obj.minute/60)

    time_array3 = team_df_A3['TOI/GP'].values
    time_str3 = time_array3.astype(str)
    format_str3 = '%H:%M'
    time_in_hours3 = []
    for t in time_str3:
      datetime_obj = datetime.datetime.strptime(t, format_str3)
      time_in_hours3.append(datetime_obj.hour + datetime_obj.minute/60)

    time_array4 = team_df_H3['TOI/GP'].values
    time_str4 = time_array4.astype(str)
    format_str4 = '%H:%M'
    time_in_hours4 = []
    for t in time_str4:
      datetime_obj = datetime.datetime.strptime(t, format_str4)
      time_in_hours4.append(datetime_obj.hour + datetime_obj.minute/60)
    PP_A = sum(time_in_hours)
    PP_H = sum(time_in_hours2)
    PK_A = sum(time_in_hours3)
    PK_H = sum(time_in_hours4)
    PP_total_A = (PP_A + PK_H)/2
    PP_total_H = (PP_H + PK_A)/2

    score1_A = ((((HDCF_A + HDCA_H)/2)*(((HDSH_A+(100-HDSV_H))/2)/100)) + (((MDCF_A + MDCA_H)/2)*(((MDSH_A+(100-MDSV_H))/2)/100))) + ((((((HDCF_A2 + HDCA_H3)/2)*(((HDSH_A2+(100-HDSV_H3))/2)/100)) + (((MDCF_A2 + MDCA_H3)/2)*(((MDSH_A2+(100-MDSV_H3))/2)/100)))/60)*PP_total_A)
    score1_H = ((((HDCF_H + HDCA_A)/2)*(((HDSH_H+(100-HDSV_A))/2)/100)) + (((MDCF_H + MDCA_A)/2)*(((MDSH_H+(100-MDSV_A))/2)/100))) + ((((((HDCF_H2 + HDCA_A3)/2)*(((HDSH_H2+(100-HDSV_A3))/2)/100)) + (((MDCF_H2 + MDCA_A3)/2)*(((MDSH_H2+(100-MDSV_A3))/2)/100)))/60)*PP_total_H)
    score2_A = ((XGF_A + XGA_H)/2) + ((((XGF_A2 + XGA_H3)/2)/60)*PP_total_A)
    score2_H = ((XGF_H + XGA_A)/2) + ((((XGF_H2 + XGA_A3)/2)/60)*PP_total_H)

    pred_A = (((score1_A*1.2) + (score2_A*0.8))/2)*PDO_A
    pred_H = (((score1_H*1.2) + (score2_H*0.8))/2)*PDO_H

    pred_total = pred_A+pred_H
    imp_odds_A = (pred_A / pred_total)*100
    imp_odds_B = (pred_H / pred_total)*100

    print(f"{away_team} vs {home_team}:")
    print("Score_A :", pred_A, "Win%:", imp_odds_A)
    print("Score_H :", pred_H, "Win%:", imp_odds_B)
    print("Total:", pred_total)
