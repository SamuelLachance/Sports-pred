import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def safe_divide(x, y):
    return np.divide(x, y, out=np.zeros_like(x), where=(y != 0))

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
    'UTA': 'Utah Hockey Club',
    'UTAH': 'Utah Hockey Club',
    'VAN': 'Vancouver Canucks',
    'VGK': 'Vegas Golden Knights',
    'VEG': 'Vegas Golden Knights',
    'WSH': 'Washington Capitals',
    'WAS': 'Washington Capitals',
    'WPG': 'Winnipeg Jets',
    'WIN': 'Winnipeg Jets'
}

today = datetime.date.today().strftime('%Y-%m-%d')
year = 20242025

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...',
    'Accept-Language': 'en-US,en;q=0.9',
}

def get_team_data(url):
    req = requests.get(url, headers=headers)
    soup = BeautifulSoup(req.content, 'html5lib')
    body = soup.find('body')
    columns = [item.text for item in body.find_all('th')]
    data = [e.text for e in body.find_all('td')]
    table = [data[i:i+len(columns)] for i in range(0, len(data), len(columns))]
    df = pd.DataFrame(table, columns=columns).set_index('').astype(float, errors='ignore')
    return df

urls = {
    'df': f'https://www.naturalstattrick.com/teamtable.php?fromseason={year}&thruseason={year}&stype=2&sit=sva&score=all&rate=y&team=all&loc=B&gpf=c&gp=10&fd=&td=',
    'df2': f'https://www.naturalstattrick.com/teamtable.php?fromseason={year}&thruseason={year}&stype=2&sit=pp&score=all&rate=y&team=all&loc=B&gpf=c&gp=25&fd=&td=',
    'df3': f'https://www.naturalstattrick.com/teamtable.php?fromseason={year}&thruseason={year}&stype=2&sit=pk&score=all&rate=y&team=all&loc=B&gpf=c&gp=25&fd=&td=',
    'df4': f'https://www.naturalstattrick.com/teamtable.php?fromseason={year}&thruseason={year}&stype=2&sit=sva&score=all&rate=y&team=all&loc=B&gpf=10&fd=&td=',
    'df5': f'https://www.naturalstattrick.com/teamtable.php?fromseason={year}&thruseason={year}&stype=2&sit=pp&score=all&rate=y&team=all&loc=B&gpf=25&fd=&td=',
    'df6': f'https://www.naturalstattrick.com/teamtable.php?fromseason={year}&thruseason={year}&stype=2&sit=pk&score=all&rate=y&team=all&loc=B&gpf=25&fd=&td='
}

df_dict = {key: get_team_data(url) for key, url in urls.items()}

odds_api_key = '8be3ba1d05ea7d3cda1d4ec6953e78c9'
odds_endpoint = 'https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds'

params = {
    'regions': 'us',
    'oddsFormat': 'decimal',
    'dateFormat': 'iso',
    'apiKey': odds_api_key,
    'sport': 'icehockey_nhl',
    'date': today,
    'markets': 'h2h',
    'bookmakers': 'fanduel'
}
response_odds = requests.get(odds_endpoint, params=params)
if response_odds.status_code == 200:
    data_odds = response_odds.json()
else:
    data_odds = []

nhl_schedule_url = f"https://api-web.nhle.com/v1/schedule/{today}"
responseNHL = requests.get(nhl_schedule_url)
dataNHL = responseNHL.json()
games = dataNHL.get('gameWeek', [{}])[0].get('games', [])

def get_team_dfs(team_name, df_dict):
    if 'Canadiens' in team_name:
        team_name_fixed = 'Montreal Canadiens'
    elif 'Blues' in team_name:
        team_name_fixed = 'St. Louis Blues'
    else:
        team_name_fixed = team_name
    team_dfs = {key: df[df['Team'] == team_name_fixed] for key, df in df_dict.items()}
    return team_dfs

for game in games:
    home_abbrev = game['homeTeam']['abbrev']
    away_abbrev = game['awayTeam']['abbrev']
    home_team = team_abbr_to_name.get(home_abbrev, home_abbrev)
    away_team = team_abbr_to_name.get(away_abbrev, away_abbrev)
    team_dfs_H = get_team_dfs(home_team, df_dict)
    team_dfs_A = get_team_dfs(away_team, df_dict)
    home_odds = away_odds = None
    for odds_game in data_odds:
        if odds_game['home_team'] == home_team and odds_game['away_team'] == away_team:
            for outcome in odds_game['bookmakers'][0]['markets'][0]['outcomes']:
                if outcome['name'] == home_team:
                    home_odds = outcome['price']
                elif outcome['name'] == away_team:
                    away_odds = outcome['price']
            break
    if home_odds is None or away_odds is None:
        continue
    try:
        HDCF_A = team_dfs_A['df']['HDCF/60'].values
        HDCF_H = team_dfs_H['df']['HDCF/60'].values
        HDCA_A = team_dfs_A['df']['HDCA/60'].values
        HDCA_H = team_dfs_H['df']['HDCA/60'].values
        HDSH_A = safe_divide(team_dfs_A['df4']['HDGF/60'].values * 100, team_dfs_A['df4']['HDCF/60'].values)
        HDSH_H = safe_divide(team_dfs_H['df4']['HDGF/60'].values * 100, team_dfs_H['df4']['HDCF/60'].values)
        HDSV_A = 100 - safe_divide(team_dfs_A['df4']['HDGA/60'].values * 100, team_dfs_A['df4']['HDCA/60'].values)
        HDSV_H = 100 - safe_divide(team_dfs_H['df4']['HDGA/60'].values * 100, team_dfs_H['df4']['HDCA/60'].values)
        MDCF_A = team_dfs_A['df']['MDCF/60'].values
        MDCF_H = team_dfs_H['df']['MDCF/60'].values
        MDCA_A = team_dfs_A['df']['MDCA/60'].values
        MDCA_H = team_dfs_H['df']['MDCA/60'].values
        MDSH_A = safe_divide(team_dfs_A['df4']['MDGF/60'].values * 100, team_dfs_A['df4']['MDCF/60'].values)
        MDSH_H = safe_divide(team_dfs_H['df4']['MDGF/60'].values * 100, team_dfs_H['df4']['MDCF/60'].values)
        MDSV_A = 100 - safe_divide(team_dfs_A['df4']['MDGA/60'].values * 100, team_dfs_A['df4']['MDCA/60'].values)
        MDSV_H = 100 - safe_divide(team_dfs_H['df4']['MDGA/60'].values * 100, team_dfs_H['df4']['MDCA/60'].values)
        XGF_A = team_dfs_A['df']['xGF/60'].values
        XGF_H = team_dfs_H['df']['xGF/60'].values
        XGA_A = team_dfs_A['df']['xGA/60'].values
        XGA_H = team_dfs_H['df']['xGA/60'].values
        HDCF_A2 = team_dfs_A['df2']['HDCF/60'].values
        HDCF_H2 = team_dfs_H['df2']['HDCF/60'].values
        HDCA_A2 = team_dfs_A['df2']['HDCA/60'].values
        HDCA_H2 = team_dfs_H['df2']['HDCA/60'].values
        HDSH_A2 = safe_divide(team_dfs_A['df5']['HDGF/60'].values * 100, team_dfs_A['df5']['HDCF/60'].values)
        HDSH_H2 = safe_divide(team_dfs_H['df5']['HDGF/60'].values * 100, team_dfs_H['df5']['HDCF/60'].values)
        HDSV_A2 = 100 - safe_divide(team_dfs_A['df5']['HDGA/60'].values * 100, team_dfs_A['df5']['HDCA/60'].values)
        HDSV_H2 = 100 - safe_divide(team_dfs_H['df5']['HDGA/60'].values * 100, team_dfs_H['df5']['HDCA/60'].values)
        MDCF_A2 = team_dfs_A['df2']['MDCF/60'].values
        MDCF_H2 = team_dfs_H['df2']['MDCF/60'].values
        MDCA_A2 = team_dfs_A['df2']['MDCA/60'].values
        MDCA_H2 = team_dfs_H['df2']['MDCA/60'].values
        MDSH_A2 = safe_divide(team_dfs_A['df5']['MDGF/60'].values * 100, team_dfs_A['df5']['MDCF/60'].values)
        MDSH_H2 = safe_divide(team_dfs_H['df5']['MDGF/60'].values * 100, team_dfs_H['df5']['MDCF/60'].values)
        MDSV_A2 = 100 - safe_divide(team_dfs_A['df5']['MDGA/60'].values * 100, team_dfs_A['df5']['MDCA/60'].values)
        MDSV_H2 = 100 - safe_divide(team_dfs_H['df5']['MDGA/60'].values * 100, team_dfs_H['df5']['MDCA/60'].values)
        XGF_A2 = team_dfs_A['df2']['xGF/60'].values
        XGF_H2 = team_dfs_H['df2']['xGF/60'].values
        XGA_A2 = team_dfs_A['df2']['xGA/60'].values
        XGA_H2 = team_dfs_H['df2']['xGA/60'].values
        HDCF_A3 = team_dfs_A['df3']['HDCF/60'].values
        HDCF_H3 = team_dfs_H['df3']['HDCF/60'].values
        HDCA_A3 = team_dfs_A['df3']['HDCA/60'].values
        HDCA_H3 = team_dfs_H['df3']['HDCA/60'].values
        HDSH_A3 = safe_divide(team_dfs_A['df6']['HDGF/60'].values * 100, team_dfs_A['df6']['HDCF/60'].values)
        HDSH_H3 = safe_divide(team_dfs_H['df6']['HDGF/60'].values * 100, team_dfs_H['df6']['HDCF/60'].values)
        HDSV_A3 = 100 - safe_divide(team_dfs_A['df6']['HDGA/60'].values * 100, team_dfs_A['df6']['HDCA/60'].values)
        HDSV_H3 = 100 - safe_divide(team_dfs_H['df6']['HDGA/60'].values * 100, team_dfs_H['df6']['HDCA/60'].values)
        MDCF_A3 = team_dfs_A['df3']['MDCF/60'].values
        MDCF_H3 = team_dfs_H['df3']['MDCF/60'].values
        MDCA_A3 = team_dfs_A['df3']['MDCA/60'].values
        MDCA_H3 = team_dfs_H['df3']['MDCA/60'].values
        MDSH_A3 = safe_divide(team_dfs_A['df6']['MDGF/60'].values * 100, team_dfs_A['df6']['MDCF/60'].values)
        MDSH_H3 = safe_divide(team_dfs_H['df6']['MDGF/60'].values * 100, team_dfs_H['df6']['MDCF/60'].values)
        MDSV_A3 = 100 - safe_divide(team_dfs_A['df6']['MDGA/60'].values * 100, team_dfs_A['df6']['MDCA/60'].values)
        MDSV_H3 = 100 - safe_divide(team_dfs_H['df6']['MDGA/60'].values * 100, team_dfs_H['df6']['MDCA/60'].values)
        XGF_A3 = team_dfs_A['df3']['xGF/60'].values
        XGF_H3 = team_dfs_H['df3']['xGF/60'].values
        XGA_A3 = team_dfs_A['df3']['xGA/60'].values
        XGA_H3 = team_dfs_H['df3']['xGA/60'].values
        PDO_A = team_dfs_A['df']['PDO'].values
        PDO_H = team_dfs_H['df']['PDO'].values
        time_array = team_dfs_A['df2']['TOI/GP'].values.astype(str)
        time_in_hours = [int(t.split(':')[0]) + int(t.split(':')[1])/60 for t in time_array]
        time_array2 = team_dfs_H['df2']['TOI/GP'].values.astype(str)
        time_in_hours2 = [int(t.split(':')[0]) + int(t.split(':')[1])/60 for t in time_array2]
        time_array3 = team_dfs_A['df3']['TOI/GP'].values.astype(str)
        time_in_hours3 = [int(t.split(':')[0]) + int(t.split(':')[1])/60 for t in time_array3]
        time_array4 = team_dfs_H['df3']['TOI/GP'].values.astype(str)
        time_in_hours4 = [int(t.split(':')[0]) + int(t.split(':')[1])/60 for t in time_array4]
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
        pred_A = score1_A
        pred_H = score1_H
        pred_A2 = score2_A * PDO_A
        pred_H2 = score2_H * PDO_H
        pred_total = pred_A + pred_H
        pred_total2 = pred_A2 + pred_H2
        imp_odds_A = (pred_A / pred_total)*100
        imp_odds_B = (pred_H / pred_total)*100
        print(f"{away_team} vs {home_team}:")
        print("Score_A :", pred_A, "Win%:", imp_odds_A)
        print("Score_H :", pred_H, "Win%:", imp_odds_B)
        print("Total:", pred_total2)
    except Exception as e:
        continue
