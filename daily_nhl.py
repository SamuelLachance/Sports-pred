import os
import time
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.metrics import log_loss
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from pandas import json_normalize
import warnings
warnings.filterwarnings("ignore")

class Team:
    def __init__(self, name):
        self.name = name
        self.team_game_list = []
        self.agd = 0
        self.opponent_power = []
        self.schedule = 0
        self.power = 0
        self.prev_power = 0
        self.goals_for = 0
        self.goals_against = 0
        self.record = '0-0-0'
        self.pct = 0

    def read_games(self):
        print(f'--{self.name.upper()} GAME LIST--')
        for game in self.team_game_list:
            print(f'{game.home_team.name}  {game.home_score}-{game.away_score}  {game.away_team.name}')

    def calc_agd(self):
        goal_differential = sum(
            (game.home_score - game.away_score) if self == game.home_team else (game.away_score - game.home_score)
            for game in self.team_game_list
        )
        agd = goal_differential / len(self.team_game_list) if self.team_game_list else 0
        return agd

    def calc_sched(self):
        self.opponent_power = [
            (game.away_team.prev_power if self == game.home_team else game.home_team.prev_power)
            for game in self.team_game_list
        ]
        return sum(self.opponent_power) / len(self.opponent_power) if self.opponent_power else 0

    def calc_power(self):
        return self.calc_sched() + self.agd

    def calc_pct(self):
        try:
            wins, losses, otl = map(int, self.record.split('-'))
            point_percentage = (wins * 2 + otl) / (len(self.team_game_list) * 2) if self.team_game_list else 0
            return point_percentage
        except (ValueError, ZeroDivisionError):
            return 0

    def calc_consistency(self):
        performance_list = [
            opponent.power + (game.home_score - game.away_score if self == game.home_team else game.away_score - game.home_score)
            for game in self.team_game_list
            for opponent in [game.away_team if self == game.home_team else game.home_team]
        ]
        variance = np.var(performance_list) if performance_list else 0
        return variance

class Game:
    def __init__(self, home_team, away_team, home_score, away_score, date):
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score
        self.date = date

def game_team_object_creation(games_metadf):
    total_game_list = []
    team_dict = {}

    for _, row in games_metadf.iterrows():
        try:
            home_score = float(row['Home Goals'])
            away_score = float(row['Away Goals'])
            home_team_name = row['Home Team']
            away_team_name = row['Away Team']

            # Get or create home team
            home_team = team_dict.setdefault(home_team_name, Team(home_team_name))

            # Get or create away team
            away_team = team_dict.setdefault(away_team_name, Team(away_team_name))

            game_obj = Game(home_team, away_team, home_score, away_score, row['Date'])

            home_team.team_game_list.append(game_obj)
            away_team.team_game_list.append(game_obj)
            home_team.goals_for += game_obj.home_score
            away_team.goals_against += game_obj.home_score
            home_team.goals_against += game_obj.away_score
            away_team.goals_for += game_obj.away_score
            home_team.record = row['Home Record']
            away_team.record = row['Away Record']
            total_game_list.append(game_obj)
        except ValueError:
            continue

    team_list = list(team_dict.values())
    return team_list, total_game_list

def scrape_nhl_data():
    data = []
    team_id_dict = {}

    team_metadata = requests.get("https://api.nhle.com/stats/rest/en/team").json()
    excluded_teams = {'Atlanta Thrashers', 'Hartford Whalers', 'Minnesota North Stars', 'Quebec Nordiques',
                      'Winnipeg Jets (1979)', 'Colorado Rockies', 'Ottawa Senators (1917)', 'Hamilton Tigers',
                      'Pittsburgh Pirates', 'Philadelphia Quakers', 'Detroit Cougars', 'Montreal Wanderers',
                      'Quebec Bulldogs', 'Montreal Maroons', 'New York Americans', 'St. Louis Eagles',
                      'Oakland Seals', 'Atlanta Flames', 'Kansas City Scouts', 'Cleveland Barons',
                      'Detroit Falcons', 'Brooklyn Americans', 'California Golden Seals', 'Toronto Arenas',
                      'Toronto St. Patricks', 'NHL'}

    for team in tqdm(team_metadata['data'], desc='Scraping Games', dynamic_ncols=True, colour='Green', bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'):
        team_name = team['fullName']
        if team_name in excluded_teams:
            continue

        team_id = team['id']
        team_id_dict[team_id] = team_name

        game_metadata = requests.get(f"https://api-web.nhle.com/v1/club-schedule-season/{team['triCode']}/20242025").json()

        for game in game_metadata.get('games', []):
            if game['gameType'] == 2 and game['gameState'] == 'OFF':
                data.append({
                    'GameID': game['id'],
                    'Date': game['gameDate'],
                    'Home Team': game['homeTeam']['id'],
                    'Home Goals': game['homeTeam']['score'],
                    'Away Goals': game['awayTeam']['score'],
                    'Away Team': game['awayTeam']['id'],
                    'FinalState': game['gameOutcome']['lastPeriodType']
                })

    scraped_df = pd.DataFrame(data)
    scraped_df['Home Team'] = scraped_df['Home Team'].replace(team_id_dict)
    scraped_df['Away Team'] = scraped_df['Away Team'].replace(team_id_dict)
    scraped_df = scraped_df.drop_duplicates(subset='GameID')
    scraped_df = scraped_df.sort_values(by=['GameID'])
    scraped_df = calculate_records(scraped_df)
    return scraped_df, team_id_dict

def calculate_records(df):
    all_teams = pd.concat([df['Home Team'], df['Away Team']]).unique()
    records = {team: {'wins': 0, 'losses': 0, 'ot_losses': 0} for team in all_teams}

    for index, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_goals = row['Home Goals']
        away_goals = row['Away Goals']
        final_state = row['FinalState']

        if home_goals > away_goals:
            records[home_team]['wins'] += 1
            if final_state == 'REG':
                records[away_team]['losses'] += 1
            else:
                records[away_team]['ot_losses'] += 1
        elif home_goals < away_goals:
            if final_state == 'REG':
                records[home_team]['losses'] += 1
            else:
                records[home_team]['ot_losses'] += 1
            records[away_team]['wins'] += 1
        else:
            print(f'Critical Error: Found Tie | Information: {home_team} {home_goals}-{away_goals} {away_team}')
            continue

        df.at[index, 'Home Record'] = f"{records[home_team]['wins']}-{records[home_team]['losses']}-{records[home_team]['ot_losses']}"
        df.at[index, 'Away Record'] = f"{records[away_team]['wins']}-{records[away_team]['losses']}-{records[away_team]['ot_losses']}"

    return df

def assign_power(team_list, iterations):
    for team in team_list:
        team.agd = team.calc_agd()
        team.pct = team.calc_pct()

    for _ in range(iterations):
        for team in team_list:
            team.schedule = team.calc_sched()
            team.power = team.calc_power()
        for team in team_list:
            team.prev_power = team.power

def prepare_power_rankings(team_list):
    power_data = [{
        'Team': team.name,
        'POWER': round(team.power, 2),
        'Record': team.record,
        'PCT': f"{team.calc_pct():.3f}",
        'Avg Goal Differential': round(team.calc_agd(), 2),
        'GF/Game': f"{team.goals_for / len(team.team_game_list):.2f}" if team.team_game_list else '0.00',
        'GA/Game': f"{team.goals_against / len(team.team_game_list):.2f}" if team.team_game_list else '0.00',
        'Strength of Schedule': f"{team.schedule:.3f}"
    } for team in team_list]

    power_df = pd.DataFrame(power_data)
    power_df.sort_values(by=['POWER'], inplace=True, ascending=False)
    power_df.reset_index(drop=True, inplace=True)
    power_df.index += 1
    return power_df

def logistic_regression(total_game_list):
    xpoints = []
    ypoints = []

    for game in total_game_list:
        rating_diff = game.home_team.power - game.away_team.power
        xpoints.append(rating_diff)

        if game.home_score > game.away_score:
            ypoints.append(1)
        elif game.home_score < game.away_score:
            ypoints.append(0)
        else:
            ypoints.append(0.5)

    parameters, _ = curve_fit(lambda t, param: 1 / (1 + np.exp(t / param)), [-x for x in xpoints], ypoints)
    param = -parameters[0]
    return xpoints, ypoints, param

def model_performance(xpoints, ypoints, param):
    x_fitted = np.linspace(np.min(xpoints) * 1.25, np.max(xpoints) * 1.25, 100)
    y_fitted = 1 / (1 + np.exp(x_fitted / param))

    r, _ = pearsonr(xpoints, ypoints)
    log_loss_value = log_loss(ypoints, 1 / (1 + np.exp(np.array(xpoints) / param)))

    print(f'Pearson Correlation of Independent and Dependent Variables: {r:.3f}')
    print(f'Log Loss of the Cumulative Distribution Function (CDF): {log_loss_value:.3f}')
    print(f'Regressed Sigmoid: 1/(1+exp((x)/{param:.3f}))')
    print(f'Precise Parameter: {param}')

    plt.plot(xpoints, ypoints, 'o', color='grey')
    plt.plot(x_fitted, y_fitted, color='black', alpha=1,
             label=f'CDF (Log Loss = {log_loss_value:.3f})')
    plt.legend()
    plt.title('Logistic Regression of Team Rating Difference vs Game Result')
    plt.xlabel('Rating Difference')
    plt.ylabel('Win Probability')
    plt.show()

def calc_prob(team, opponent, param):
    return 1 / (1 + np.exp((team.power - opponent.power) / param))

def calc_spread(team, opponent, param, lower_bound_spread, upper_bound_spread):
    lower_bound = float(lower_bound_spread) if lower_bound_spread != '-inf' else -np.inf
    upper_bound = float(upper_bound_spread) if upper_bound_spread != 'inf' else np.inf
    cdf_lower = 1 / (1 + np.exp((lower_bound - (team.power - opponent.power)) / param))
    cdf_upper = 1 / (1 + np.exp((upper_bound - (team.power - opponent.power)) / param))
    return cdf_upper - cdf_lower

def download_csv_option(df, filename):
    user_input = input('Would you like to download this as a CSV? (Y/N): ').strip().lower()
    if user_input in ['y', 'yes']:
        output_dir = os.path.join(os.path.dirname(__file__), 'Output CSV Data')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{filename}.csv')
        df.to_csv(output_path)
        print(f'{filename}.csv has been downloaded to: {output_path}')

def get_todays_games(param, team_list, team_id_dict):
    today_date = datetime.today().strftime('%Y-%m-%d')
    today_schedule = requests.get(f"https://api-web.nhle.com/v1/schedule/{today_date}").json()

    columns = [
        'GameID', 'Game State', 'Home Team', 'Home Goals', 'Away Goals',
        'Away Team', 'Pre-Game Home Win Probability', 'Pre-Game Away Win Probability',
        'Home Record', 'Away Record'
    ]
    rows = []

    try:
        date = today_schedule['gameWeek'][0]['date']
        for game in today_schedule['gameWeek'][0]['games']:
            home_team_id = game['homeTeam']['id']
            away_team_id = game['awayTeam']['id']
            home_team_name = team_id_dict.get(home_team_id)
            away_team_name = team_id_dict.get(away_team_id)
            home_team_obj = next((t for t in team_list if t.name == home_team_name), None)
            away_team_obj = next((t for t in team_list if t.name == away_team_name), None)

            if not home_team_obj or not away_team_obj:
                continue

            home_win_prob = calc_prob(home_team_obj, away_team_obj, param) * 100
            away_win_prob = 100 - home_win_prob

            game_state = game['gameState']
            if game_state == 'OFF':
                game_state_desc = 'Final'
                home_goals = game['homeTeam']['score']
                away_goals = game['awayTeam']['score']
            elif game_state == 'FUT':
                game_state_desc = 'Pre-Game'
                home_goals = 0
                away_goals = 0
            else:
                game_state_desc = f"Period {game.get('periodDescriptor', {}).get('number', '')}"
                home_goals = game['homeTeam'].get('score', 0)
                away_goals = game['awayTeam'].get('score', 0)

            row = {
                'GameID': game['id'],
                'Game State': game_state_desc,
                'Home Team': home_team_name,
                'Home Goals': home_goals,
                'Away Goals': away_goals,
                'Away Team': away_team_name,
                'Pre-Game Home Win Probability': f'{home_win_prob:.2f}%',
                'Pre-Game Away Win Probability': f'{away_win_prob:.2f}%',
                'Home Record': home_team_obj.record,
                'Away Record': away_team_obj.record
            }
            rows.append(row)

        today_games_df = pd.DataFrame(rows, columns=columns)
        today_games_df.index += 1
    except (IndexError, KeyError):
        today_games_df = None
        date = None

    return date, today_games_df

def custom_game_selector(param, team_list):
    def get_team(prompt):
        while True:
            team_input = input(prompt).strip().lower()
            for team in team_list:
                if team_input == team.name.lower().replace('é', 'e'):
                    return team
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    home_team = get_team('Enter the home team: ')
    away_team = get_team('Enter the away team: ')

    data = []
    columns = ['', home_team.name, away_team.name]

    data.append({'': 'Rating', home_team.name: f'{home_team.power:.3f}', away_team.name: f'{away_team.power:.3f}'})
    data.append({'': 'Record', home_team.name: home_team.record, away_team.name: away_team.record})
    data.append({'': 'Point PCT', home_team.name: f'{home_team.pct:.3f}', away_team.name: f'{away_team.pct:.3f}'})
    data.append({'': 'Win Probability', home_team.name: f'{calc_prob(home_team, away_team, param)*100:.2f}%', away_team.name: f'{calc_prob(away_team, home_team, param)*100:.2f}%'})

    spreads = [
        (0, 1.5, 'Win by 1'),
        (1.5, 2.5, 'Win by 2'),
        (2.5, 3.5, 'Win by 3'),
        (3.5, 4.5, 'Win by 4'),
        (4.5, 'inf', 'Win by 5+')
    ]

    for lower, upper, label in spreads:
        data.append({
            '': label,
            home_team.name: f'{calc_spread(home_team, away_team, param, lower, upper)*100:.2f}%',
            away_team.name: f'{calc_spread(away_team, home_team, param, lower, upper)*100:.2f}%'
        })

    game_probability_df = pd.DataFrame(data, columns=columns)
    game_probability_df.set_index('', inplace=True)

    return home_team, away_team, game_probability_df

def get_upsets(total_game_list):
    data = []

    for game in total_game_list:
        expected_score_diff = game.home_team.power - game.away_team.power
        actual_score_diff = game.home_score - game.away_score
        upset_rating = abs(actual_score_diff - expected_score_diff)

        data.append({
            'Home Team': game.home_team.name,
            'Home Goals': int(game.home_score),
            'Away Goals': int(game.away_score),
            'Away Team': game.away_team.name,
            'Date': game.date,
            'xGD': f'{expected_score_diff:.2f}',
            'GD': int(actual_score_diff),
            'Upset Rating': f'{upset_rating:.2f}'
        })

    upset_df = pd.DataFrame(data)
    upset_df.sort_values(by=['Upset Rating'], ascending=False, inplace=True)
    upset_df.reset_index(drop=True, inplace=True)
    upset_df.index += 1
    return upset_df

def get_best_performances(total_game_list):
    data = []

    for game in total_game_list:
        for team, opponent, gf, ga in [
            (game.home_team, game.away_team, game.home_score, game.away_score),
            (game.away_team, game.home_team, game.away_score, game.home_score)
        ]:
            performance = opponent.power + gf - ga
            data.append({
                'Team': team.name,
                'Opponent': opponent.name,
                'GF': int(gf),
                'GA': int(ga),
                'Date': game.date,
                'xGD': f'{team.power - opponent.power:.2f}',
                'Performance': round(performance, 2)
            })

    performance_df = pd.DataFrame(data)
    performance_df.sort_values(by=['Performance'], ascending=False, inplace=True)
    performance_df.reset_index(drop=True, inplace=True)
    performance_df.index += 1
    return performance_df

def get_team_consistency(team_list):
    data = []

    for team in team_list:
        variance = team.calc_consistency()
        data.append({
            'Team': team.name,
            'Rating': f'{team.power:.2f}',
            'Consistency (z-Score)': variance
        })

    consistency_df = pd.DataFrame(data)
    mean_var = consistency_df['Consistency (z-Score)'].mean()
    std_var = consistency_df['Consistency (z-Score)'].std()
    consistency_df['Consistency (z-Score)'] = ((consistency_df['Consistency (z-Score)'] - mean_var) / -std_var).round(2)
    consistency_df.sort_values(by=['Consistency (z-Score)'], ascending=False, inplace=True)
    consistency_df.reset_index(drop=True, inplace=True)
    consistency_df.index += 1
    return consistency_df

def team_game_log(team_list):
    input_team = input('Enter a team: ').strip().lower()
    team = next((t for t in team_list if t.name.lower().replace('é', 'e') == input_team), None)
    if not team:
        print('Sorry, I am not familiar with this team. Maybe check your spelling?')
        return None, None

    data = []

    for game in team.team_game_list:
        if team == game.home_team:
            goals_for = game.home_score
            opponent = game.away_team
            goals_against = game.away_score
        else:
            goals_for = game.away_score
            opponent = game.home_team
            goals_against = game.home_score

        performance = opponent.power + goals_for - goals_against
        data.append({
            'Date': game.date,
            'Opponent': opponent.name,
            'GF': int(goals_for),
            'GA': int(goals_against),
            'Performance': round(performance, 2)
        })

    game_log_df = pd.DataFrame(data)
    game_log_df.index += 1
    return team, game_log_df

def get_team_prob_breakdown(team_list, param):
    input_team = input('Enter a team: ').strip().lower()
    team = next((t for t in team_list if t.name.lower().replace('é', 'e') == input_team), None)
    if not team:
        print('Sorry, I am not familiar with this team. Maybe check your spelling?')
        return None, None

    data = []

    for opp_team in team_list:
        if opp_team == team:
            continue

        win_prob = calc_prob(team, opp_team, param) * 100
        spreads = [
            ('-inf', -4.5, 'Lose by 5+'),
            (-4.5, -3.5, 'Lose by 4'),
            (-3.5, -2.5, 'Lose by 3'),
            (-2.5, -1.5, 'Lose by 2'),
            (-1.5, 0, 'Lose by 1'),
            (0, 1.5, 'Win by 1'),
            (1.5, 2.5, 'Win by 2'),
            (2.5, 3.5, 'Win by 3'),
            (3.5, 4.5, 'Win by 4'),
            (4.5, 'inf', 'Win by 5+')
        ]

        spread_probs = {}
        for lower, upper, label in spreads:
            prob = calc_spread(team, opp_team, param, lower, upper) * 100
            spread_probs[label] = f'{prob:.2f}%'

        row = {
            'Opponent': opp_team.name,
            'Record': opp_team.record,
            'PCT': f'{opp_team.calc_pct():.3f}',
            'Win Probability': f'{win_prob:.2f}%'
        }
        row.update(spread_probs)
        data.append(row)

    prob_breakdown_df = pd.DataFrame(data)
    prob_breakdown_df.set_index('Opponent', inplace=True)
    prob_breakdown_df.sort_values(by=['PCT'], ascending=False, inplace=True)
    return team, prob_breakdown_df

def extra_menu(total_game_list, team_list, param):
    while True:
        print("""--EXTRAS MENU--
    1. Biggest Upsets
    2. Best Performances
    3. Most Consistent Teams
    4. Team Game Logs
    5. Team Probability Big Board
    6. Exit to Main Menu""")

        user_option = input('Enter a menu option: ').strip()
        if user_option == '1':
            upsets = get_upsets(total_game_list)
            print(upsets)
            download_csv_option(upsets, 'biggest_upsets')
        elif user_option == '2':
            performances = get_best_performances(total_game_list)
            print(performances)
            download_csv_option(performances, 'best_performances')
        elif user_option == '3':
            consistency = get_team_consistency(team_list)
            print(consistency)
            download_csv_option(consistency, 'most_consistent_teams')
        elif user_option == '4':
            team, game_log = team_game_log(team_list)
            if team and game_log is not None:
                print(game_log)
                download_csv_option(game_log, f'{team.name.replace(" ", "_").lower()}_game_log')
        elif user_option == '5':
            team, prob_breakdown_df = get_team_prob_breakdown(team_list, param)
            if team and prob_breakdown_df is not None:
                print(prob_breakdown_df)
                download_csv_option(prob_breakdown_df, f'{team.name.replace(" ", "_").lower()}_prob_breakdown')
        elif user_option == '6':
            break
        else:
            print(f'Invalid option: {user_option}')
        print()

def moneyline_to_proba(moneyline):
    moneyline = float(moneyline)
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)

def calculate_ev(model_prob, vegas_prob):
    potential_profit = (1 / vegas_prob) - 1
    prob_lose = 1 - model_prob
    ev = model_prob * potential_profit - prob_lose * 1
    return ev

def fetch_odds_data(date):
    base_url = f"https://www.oddsshark.com/api/scores/nhl/{date}?_format=json"
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.oddsshark.com/nhl/scores',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    response = requests.get(base_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = extract_team_data(data)
        df['Date'] = date
        return df
    else:
        print(f"Failed to fetch data for date: {date}")
        return None

def extract_team_data(json_data):
    extracted_data = []
    for game in json_data.get('scores', []):
        home_team = game['teams']['home']
        away_team = game['teams']['away']
        game_data = {
            'Home Name': home_team['names']['name'],
            'Home MoneyLine': home_team['moneyLine'],
            'Home Spread Price': home_team['spreadPrice'],
            'Home Score': home_team.get('score', 0),
            'Home Votes': home_team.get('votes', 0),
            'Home Spread': home_team['spread'],
            'Away Name': away_team['names']['name'],
            'Away MoneyLine': away_team['moneyLine'],
            'Away Spread Price': away_team['spreadPrice'],
            'Away Score': away_team.get('score', 0),
            'Away Votes': away_team.get('votes', 0),
            'Away Spread': away_team['spread'],
            'Under Price': game['underPrice'],
            'Over Price': game['overPrice'],
            'Over Votes': game.get('overVotes', 0),
            'Under Votes': game.get('underVotes', 0),
            'Total': game['total'],
            'Arena': game.get('stadium', '')
        }
        extracted_data.append(game_data)

    df = pd.DataFrame(extracted_data)
    df['Home Name'].replace({'Montreal Canadiens': 'Montréal Canadiens'}, inplace=True)
    df['Away Name'].replace({'Montreal Canadiens': 'Montréal Canadiens'}, inplace=True)
    return df

def merge_odds_and_projections(odds_df, projections_df):
    decimal_places = 3

    merged_df = pd.merge(
        odds_df,
        projections_df,
        left_on=['Home Name', 'Away Name'],
        right_on=['Home Team', 'Away Team'],
        how='inner'
    )

    merged_df = merged_df[[
        'Date', 'Home Name', 'Away Name', 'Home MoneyLine', 'Away MoneyLine',
        'Pre-Game Home Win Probability', 'Pre-Game Away Win Probability'
    ]]

    for col in ['Home MoneyLine', 'Away MoneyLine']:
        merged_df[col] = merged_df[col].apply(moneyline_to_proba)

    for col in ['Pre-Game Home Win Probability', 'Pre-Game Away Win Probability']:
        merged_df[col] = merged_df[col].str.rstrip('%').astype(float) / 100

    merged_df['Home EV'] = merged_df.apply(
        lambda x: calculate_ev(x['Pre-Game Home Win Probability'], x['Home MoneyLine']), axis=1) * 100
    merged_df['Away EV'] = merged_df.apply(
        lambda x: calculate_ev(x['Pre-Game Away Win Probability'], x['Away MoneyLine']), axis=1) * 100

    merged_df['Pre-Game Home Win Probability'] *= 100
    merged_df['Pre-Game Away Win Probability'] *= 100

    merged_df[['Home EV', 'Away EV']] = merged_df[['Home EV', 'Away EV']].round(1)
    merged_df[['Pre-Game Home Win Probability', 'Pre-Game Away Win Probability']] = merged_df[
        ['Pre-Game Home Win Probability', 'Pre-Game Away Win Probability']].round(1)

    merged_df.drop(columns=['Home MoneyLine', 'Away MoneyLine'], inplace=True)

    # Google Sheets Integration
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
    client = gspread.authorize(creds)

    spreadsheet_id = '1KgwFdqrRUs2fa5pSRmirj6ajyO2d14ONLsiksAYk8S8'
    spreadsheet = client.open_by_key(spreadsheet_id)

    sheet_name = 'BoomerModel'
    sheet = spreadsheet.worksheet(sheet_name)

    # Append headers if the first row is empty
    if not sheet.row_values(1):
        sheet.append_row(merged_df.columns.tolist())

    data = merged_df.values.tolist()
    sheet.append_rows(data)

    return merged_df

def menu(power_df, today_games_df, xpoints, ypoints, param, computation_time, total_game_list, team_list, date):
    while True:
        print("""--MAIN MENU--
    1. View Power Rankings
    2. View Today's Games
    3. Custom Game Selector
    4. View Model Performance
    5. View Program Performance
    6. Extra Options
    7. Quit""")

        user_option = input('Enter a menu option: ').strip()

        if user_option == '1':
            print(power_df)
            download_csv_option(power_df, 'power_rankings')
        elif user_option == '2':
            if today_games_df is not None:
                odds_df = fetch_odds_data(date)
                if odds_df is not None:
                    merged_df = merge_odds_and_projections(odds_df, today_games_df)
                    print(merged_df)
                else:
                    print("No odds data available.")
                download_csv_option(today_games_df, f'{date}_games')
            else:
                print('There are no games today!')
        elif user_option == '3':
            home_team, away_team, custom_game_df = custom_game_selector(param, team_list)
            print(custom_game_df)
            download_csv_option(custom_game_df, f'{home_team.name.replace(" ", "_").lower()}_vs_{away_team.name.replace(" ", "_").lower()}_game_probabilities')
        elif user_option == '4':
            model_performance(xpoints, ypoints, param)
        elif user_option == '5':
            print(f'Computation Time: {computation_time:.2f} seconds')
            print(f'Games Scraped: {len(total_game_list)}')
            print(f'Rate: {len(total_game_list)/computation_time:.1f} games/second')
        elif user_option == '6':
            extra_menu(total_game_list, team_list, param)
        elif user_option == '7':
            break
        else:
            print(f'Invalid option: {user_option}')
        input('Press ENTER to continue')
        print()

def main():
    start_time = time.time()

    games_metadf, team_id_dict = scrape_nhl_data()
    iterations = 10
    team_list, total_game_list = game_team_object_creation(games_metadf)
    assign_power(team_list, iterations)
    power_df = prepare_power_rankings(team_list)
    xpoints, ypoints, param = logistic_regression(total_game_list)
    date, today_games_df = get_todays_games(param, team_list, team_id_dict)

    computation_time = time.time() - start_time
    menu(power_df, today_games_df, xpoints, ypoints, param, computation_time, total_game_list, team_list, date)

if __name__ == '__main__':
    main()
