import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.metrics import log_loss
import os
import time
from datetime import datetime, timedelta
import requests
from fuzzywuzzy import process
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pandas import json_normalize

class Team():
    def __init__(self, name):
        self.name = name
        self.team_game_list = []
        self.apd = 0
        self.opponent_power = []
        self.schedule = 0
        self.power = 0
        self.prev_power = 0
        self.points_for = 0
        self.points_against = 0
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.pct = 0

    def read_games(self):
        print(f'--{self.name.upper()} GAME LIST--')
        for game in self.team_game_list:
            print(f'{game.home_team.name}  {game.home_score}-{game.away_score}  {game.away_team.name}')

    def calc_apd(self):
        point_differential = 0
        for game in self.team_game_list:
            if self == game.home_team: 
                point_differential += game.home_score - game.away_score
            else: 
                point_differential += game.away_score - game.home_score
        apd = point_differential / len(self.team_game_list)

        return apd

    def calc_sched(self):
        self.opponent_power = []
        for game in self.team_game_list:
            if self == game.home_team:
                self.opponent_power.append(game.away_team.prev_power)
            else:
                self.opponent_power.append(game.home_team.prev_power)

        return sum(self.opponent_power) / len(self.opponent_power)

    def calc_power(self):
        return self.calc_sched() + self.apd

    def calc_win_pct(self):
        win_percentage = self.wins/(self.wins + self.losses)
        return win_percentage

    def calc_consistency(self):
        performance_list = []
        for game in self.team_game_list:
            if self == game.home_team:
                performance_list.append(game.away_team.power + game.home_score - game.away_score)
            else:
                performance_list.append(game.away_team.power + game.home_score - game.away_score)
        
        variance = np.var(performance_list)
        return variance

class Game():
    def __init__(self, home_team, away_team, home_score, away_score):
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score

def retrieve_league_data():
    input_df = pd.read_csv(f'{os.path.dirname(__file__)}/game_data.csv')
    input_df.index += 1
    return input_df

def game_team_object_creation(games_metadf):
    total_game_list = []
    team_list = []

    for index, row in games_metadf.iterrows():
        try:
            row['Home Score'] = float(row['Home Score'])
            row['Visitor Score'] = float(row['Visitor Score'])

            team_in_list = False
            for team in team_list:
                if team.name == row['Home Team']:
                    team_in_list = True
                    home_team_obj = team
            if team_in_list == False: 
                home_team_obj = Team(row['Home Team'])
                team_list.append(home_team_obj)

            team_in_list = False
            for team in team_list:
                if team.name == row['Visiting Team']:
                    team_in_list = True
                    away_team_obj = team
            if team_in_list == False: 
                away_team_obj = Team(row['Visiting Team'])
                team_list.append(away_team_obj)

            game_obj = Game(home_team_obj, away_team_obj, row['Home Score'], row['Visitor Score'])

            home_team_obj.team_game_list.append(game_obj)
            away_team_obj.team_game_list.append(game_obj)
            home_team_obj.points_for += game_obj.home_score
            away_team_obj.points_against += game_obj.home_score
            home_team_obj.points_against += game_obj.away_score
            away_team_obj.points_for += game_obj.away_score

            if game_obj.home_score > game_obj.away_score:
                home_team_obj.wins += 1
                away_team_obj.losses += 1
            elif game_obj.home_score < game_obj.away_score:
                home_team_obj.losses += 1
                away_team_obj.wins += 1
            else:
                home_team_obj.ties += 1
                away_team_obj.ties += 1

            total_game_list.append(game_obj)
        except ValueError: 
            pass

    return team_list, total_game_list

def assign_power(team_list, iterations):
    for team in team_list:
        team.apd = team.calc_apd()
        team.pct = team.calc_win_pct()

    for iteration in range(iterations):
        # print(f'ITERATION {iteration+1}')
        for team in team_list:
            team.schedule = team.calc_sched()
            team.power = team.calc_power()
            # print(f'{team.name}\t\tAPD: {team.calc_apd():.2f}\tSCHEDULE: {team.schedule:.2f}\t\tPOWER: {team.power:.2f}')
        for team in team_list:
            team.prev_power = team.power

def prepare_power_rankings(team_list):
    power_df = pd.DataFrame()
    for team in team_list:
        power_df = pd.concat([power_df, pd.DataFrame([{
            'Team': team.name,
            'POWER': round(team.power, 2),
            'Record': f'{team.wins}-{team.losses}-{team.ties}',
            'Win%': f"{team.calc_win_pct():.3f}",
            'Avg PTS Diff': round(team.calc_apd(), 2),
            'Avg PTS For': f"{team.points_for/len(team.team_game_list):.2f}",
            'Avg PTS Against': f"{team.points_against/len(team.team_game_list):.2f}",
            'Strength of Schedule': f"{team.schedule:.3f}"
        }])], ignore_index=True)
    power_df.sort_values(by=['POWER'], inplace=True, ascending=False)
    power_df = power_df.reset_index(drop=True)
    power_df.index += 1 

    return power_df

def logistic_regression(total_game_list):
    xpoints = [] # Rating differential (Home - Away)
    ypoints = [] # Home Win/Loss Boolean (Win = 1, Tie = 0.5, Loss = 0)

    for game in total_game_list:
        xpoints.append(game.home_team.power - game.away_team.power)

        if game.home_score > game.away_score:
            ypoints.append(1)
        elif game.home_score < game.away_score:
            ypoints.append(0)
        else:
            ypoints.append(0.5)

    parameters, covariates = curve_fit(lambda t, param: 1/(1+np.exp((t)/param)), [-x for x in xpoints], ypoints) # Regression only works if parameter is positive.
    param = -parameters[0]
    
    return xpoints, ypoints, param

def model_performance(xpoints, ypoints, param):
    x_fitted = np.linspace(np.min(xpoints)*1.25, np.max(xpoints)*1.25, 100)
    y_fitted = 1/(1+np.exp((x_fitted)/param))

    ypoints_logloss_adj = []
    xpoints_logloss_adj = []
    for yindex in range(len(ypoints)):
        if ypoints[yindex] != 0.5:
            ypoints_logloss_adj.append(ypoints[yindex])
            xpoints_logloss_adj.append(xpoints[yindex])

    r, p = pearsonr(xpoints, ypoints)
    print(f'Pearson Correlation of Independent and Dependent Variables: {r:.3f}')
    print(f'Log Loss of the Cumulative Distribution Function (CDF): {log_loss(ypoints_logloss_adj, 1/(1+np.exp((xpoints_logloss_adj)/param))):.3f}')
    print(f'Regressed Sigmoid: 1/(1+exp((x)/{param:.3f}))')
    print(f'Precise Parameter: {param}')

    plt.plot(xpoints, ypoints, 'o', color='grey')
    plt.plot(x_fitted, y_fitted, color='black', alpha=1, label=f'CDF (Log Loss = {log_loss(ypoints_logloss_adj, 1/(1+np.exp((xpoints_logloss_adj)/param))):.3f})')
    plt.legend()
    plt.title('Logistic Regression of Team Rating Difference vs Game Result')
    plt.xlabel('Rating Difference')
    plt.ylabel('Win Probability')
    plt.show()

def calc_prob(team, opponent, param):
    return 1/(1+np.exp((team.power-opponent.power)/param))

def calc_spread(team, opponent, param, lower_bound_spread, upper_bound_spread):
    if lower_bound_spread == '-inf':
        if upper_bound_spread == 'inf':
            return 1
        return 1/(1+np.exp((upper_bound_spread-(team.power-opponent.power))/param))
    elif upper_bound_spread == 'inf': 
        return 1 - 1/(1+np.exp((lower_bound_spread-(team.power-opponent.power))/param))
    else: 
        return 1/(1+np.exp((upper_bound_spread-(team.power-opponent.power))/param)) - 1/(1+np.exp((lower_bound_spread-(team.power-opponent.power))/param))

def download_csv_option(df, filename):
    valid = False
    while valid == False:
        user_input = input('Would you like to download this as a CSV? (Y/N): ')
        if user_input.lower() in ['y', 'yes', 'y.', 'yes.']:
            valid = True
        elif user_input.lower() in ['n', 'no', 'n.', 'no.']:
            return
        else:
            print(f'Sorry, I could not understand "{user_input}". Please enter Y or N: ')

    if not os.path.exists(f'{os.path.dirname(__file__)}/Output CSV Data'):
        os.makedirs(f'{os.path.dirname(__file__)}/Output CSV Data')
    df.to_csv(f'{os.path.dirname(__file__)}/Output CSV Data/{filename}.csv')
    print(f'{filename}.csv has been downloaded to the following directory: {os.path.dirname(__file__)}/Output CSV Data')
    return

def custom_game_selector(param, team_list):
    valid = False
    while valid == False:
        home_team_input = input('Enter the home team: ')
        for team in team_list:
            if home_team_input.strip().lower() == team.name.lower().replace('é','e'):
                home_team = team
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    valid = False
    while valid == False:
        away_team_input = input('Enter the away team: ')
        for team in team_list:
            if away_team_input.strip().lower() == team.name.lower().replace('é','e'):
                away_team = team
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    game_probability_df = pd.DataFrame(columns = ['', home_team.name, away_team.name])

    game_probability_df = pd.concat([game_probability_df, pd.DataFrame([{'':'Rating', home_team.name:f'{home_team.power:.3f}', away_team.name:f'{away_team.power:.3f}'}])], ignore_index = True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame([{'':'Record', home_team.name:f'{home_team.wins}-{home_team.losses}-{home_team.ties}', away_team.name:f'{away_team.wins}-{away_team.losses}-{away_team.ties}'}])], ignore_index = True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame([{'':'Point PCT', home_team.name:f'{home_team.pct:.3f}', away_team.name:f'{away_team.pct:.3f}'}])], ignore_index = True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame([{'':'Win Probability', home_team.name:f'{calc_prob(home_team, away_team, param)*100:.2f}%', away_team.name:f'{(calc_prob(away_team, home_team, param))*100:.2f}%'}])], ignore_index = True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame([{'':'Win by 1-5 Points', home_team.name:f'{calc_spread(home_team, away_team, param, 0, 5.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 0, 5.5)*100:.2f}%'}])], ignore_index = True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame([{'':'Win by 6-10 Points', home_team.name:f'{calc_spread(home_team, away_team, param, 5.5, 10.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 5.5, 10.5)*100:.2f}%'}])], ignore_index = True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame([{'':'Win by 11-15 Points', home_team.name:f'{calc_spread(home_team, away_team, param, 10.5, 15.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 10.5, 15.5)*100:.2f}%'}])], ignore_index = True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame([{'':'Win by 16-20 Points', home_team.name:f'{calc_spread(home_team, away_team, param, 15.5, 20.5)*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 15.5, 20.5)*100:.2f}%'}])], ignore_index = True)
    game_probability_df = pd.concat([game_probability_df, pd.DataFrame([{'':'Win by 21+ Points', home_team.name:f'{calc_spread(home_team, away_team, param, 20.5, "inf")*100:.2f}%', away_team.name:f'{calc_spread(away_team, home_team, param, 20.5, "inf")*100:.2f}%'}])], ignore_index = True)
    game_probability_df = game_probability_df.set_index('')

    return home_team, away_team, game_probability_df

def get_upsets(total_game_list):
    upset_df = pd.DataFrame(columns = ['Home Team', 'Home points', 'Away points', 'Away Team', 'xGD', 'GD', 'Upset Rating'])

    for game in total_game_list:
        expected_score_diff = game.home_team.power - game.away_team.power #home - away
        actaul_score_diff = game.home_score - game.away_score
        upset_rating = actaul_score_diff - expected_score_diff #Positive score is an upset by the home team. Negative scores are upsets by the visiting team.

        upset_df = pd.concat([upset_df, pd.DataFrame([{'Home Team':game.home_team.name, 'Home points':int(game.home_score), 'Away points':int(game.away_score), 'Away Team':game.away_team.name,'xGD':f'{expected_score_diff:.2f}', 'GD':int(actaul_score_diff), 'Upset Rating':f'{abs(upset_rating):.2f}'}])], ignore_index = True)

    upset_df = upset_df.sort_values(by=['Upset Rating'], ascending=False)
    upset_df = upset_df.reset_index(drop=True)
    upset_df.index += 1
    return upset_df

def get_best_performances(total_game_list):
    performance_df = pd.DataFrame(columns = ['Team', 'Opponent', 'GF', 'GA', 'xGD', 'Performance'])

    for game in total_game_list:
        performance_df = pd.concat([performance_df, pd.DataFrame([{'Team':game.home_team.name, 'Opponent':game.away_team.name, 'GF':int(game.home_score), 'GA':int(game.away_score), 'xGD':f'{game.home_team.power-game.away_team.power:.2f}', 'Performance':round(game.away_team.power+game.home_score-game.away_score,2)}])], ignore_index = True)
        performance_df = pd.concat([performance_df, pd.DataFrame([{'Team':game.away_team.name, 'Opponent':game.home_team.name, 'GF':int(game.away_score), 'GA':int(game.home_score), 'xGD':f'{game.away_team.power-game.home_team.power:.2f}', 'Performance':round(game.home_team.power+game.away_score-game.home_score,2)}])], ignore_index = True)

    performance_df = performance_df.sort_values(by=['Performance'], ascending=False)
    performance_df = performance_df.reset_index(drop=True)
    performance_df.index += 1
    return performance_df

def get_team_consistency(team_list):
    consistency_df = pd.DataFrame(columns = ['Team', 'Rating', 'Consistency (z-Score)'])

    for team in team_list:
        consistency_df = pd.concat([consistency_df, pd.DataFrame([{'Team':team.name, 'Rating':f'{team.power:.2f}', 'Consistency (z-Score)':team.calc_consistency()}])], ignore_index = True)

    consistency_df['Consistency (z-Score)'] = consistency_df['Consistency (z-Score)'].apply(lambda x: (x-consistency_df['Consistency (z-Score)'].mean())/-consistency_df['Consistency (z-Score)'].std())

    consistency_df = consistency_df.sort_values(by=['Consistency (z-Score)'], ascending=False)
    consistency_df = consistency_df.reset_index(drop=True)
    consistency_df.index += 1
    consistency_df['Consistency (z-Score)'] = consistency_df['Consistency (z-Score)'].apply(lambda x: f'{x:.2f}')
    return consistency_df

def team_game_log(team_list):
    valid = False
    while valid == False:
        input_team = input('Enter a team: ')
        for team_obj in team_list:
            if input_team.strip().lower() == team_obj.name.lower().replace('é','e'):
                team = team_obj
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    game_log_df = pd.DataFrame(columns = ['Opponent', 'GF', 'GA', 'Performance'])
    for game in team.team_game_list:
        if team == game.home_team:
            points_for = game.home_score
            opponent = game.away_team
            points_against = game.away_score
        else:
            points_for = game.away_score
            opponent = game.home_team
            points_against = game.home_score
        game_log_df = pd.concat([game_log_df, pd.DataFrame([{'Opponent':opponent.name, 'GF':int(points_for), 'GA':int(points_against), 'Performance':round(opponent.power + points_for - points_against,2)}])], ignore_index = True)
    
    game_log_df.index += 1 
    return team, game_log_df

def get_team_prob_breakdown(team_list, param):
    valid = False
    while valid == False:
        input_team = input('Enter a team: ')
        for team_obj in team_list:
            if input_team.strip().lower() == team_obj.name.lower().replace('é','e'):
                team = team_obj
                valid = True
        if valid == False:
            print('Sorry, I am not familiar with this team. Maybe check your spelling?')

    prob_breakdown_df = pd.DataFrame()
    for opp_team in team_list:
        if opp_team is not team:
            prob_breakdown_df = pd.concat([prob_breakdown_df, pd.DataFrame([{'Opponent': opp_team.name, 
            'Record': f'{opp_team.wins}-{opp_team.losses}-{opp_team.ties}',
            'Win%': f'{opp_team.calc_win_pct():.3f}',
            'Win Probability':f'{calc_prob(team, opp_team, param)*100:.2f}%', 
            'Lose by 21+': f'{calc_spread(team, opp_team, param, "-inf", -20.5)*100:.2f}%',
            'Lose by 16-20': f'{calc_spread(team, opp_team, param, -20.5, -15.5)*100:.2f}%', 
            'Lose by 11-15': f'{calc_spread(team, opp_team, param, -15.5, -10.5)*100:.2f}%', 
            'Lose by 6-10': f'{calc_spread(team, opp_team, param, -10.5, -5.5)*100:.2f}%', 
            'Lose by 1-5': f'{calc_spread(team, opp_team, param, -5.5, 0)*100:.2f}%', 
            'Win by 1-5': f'{calc_spread(team, opp_team, param, 0, 5.5)*100:.2f}%', 
            'Win by 6-10': f'{calc_spread(team, opp_team, param, 5.5, 10.5)*100:.2f}%', 
            'Win by 10-15': f'{calc_spread(team, opp_team, param, 10.5, 15.5)*100:.2f}%', 
            'Win by 16-20': f'{calc_spread(team, opp_team, param, 15.5, 20.5)*100:.2f}%',
            'Win by 21+': f'{calc_spread(team, opp_team, param, 20.5, "inf")*100:.2f}%'}])], ignore_index = True)

    prob_breakdown_df = prob_breakdown_df.set_index('Opponent')
    prob_breakdown_df = prob_breakdown_df.sort_values(by=['Win%'], ascending=False)
    return team, prob_breakdown_df


def moneyline_to_proba(moneyline):
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)


def calculate_ev(model_prob, vegas_prob):
    """
    Calculate the Expected Value (EV) for betting based on the provided probabilities.

    Parameters:
    - model_prob: Model's probability of the team (home or away) winning.
    - vegas_prob: Sportsbook's implied probability based on the odds.

    Returns:
    - EV for betting on the team.
    """
    potential_profit = (1 / vegas_prob) - 1
    prob_lose = 1 - model_prob
    ev = model_prob * potential_profit - prob_lose * 1  # Assuming a 1 unit bet
    
    return ev

def fetch_odds_data(date, predict):
    base_url = f"https://www.oddsshark.com/api/scores/nfl/{date}?_format=json"
    print(base_url)
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.oddsshark.com/nhl/scores',
        'Sec-Ch-Ua': '"Chromium";v="118", "Microsoft Edge";v="118", "Not=A?Brand";v="99"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.44'
    }
    response = requests.get(base_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = extract_team_data(data, predict)
        df['Date'] = date
        df.to_csv(f"odds_data_{date}.csv", index=False)
        return df
    else:
        print(f"Failed to fetch data for date: {date}")
        return None

def extract_team_data(json_data, predict):
    extracted_data = []
    for game in json_data['scores']:
        game_data = {}
        
        home_team = game['teams']['home']
        away_team = game['teams']['away']
        game_data['Home Name'] = home_team['names']['name']
        if game_data['Home Name'] == 'Montreal Canadiens':
            game_data['Home Name'] = 'Montréal Canadiens'
        game_data['Home MoneyLine'] = home_team['moneyLine']
        game_data['Home Spread Price'] = home_team['spreadPrice']
        game_data['Home Score'] = home_team['score']
        game_data['Home Votes'] = home_team['votes']
        game_data['Home Spread'] = home_team['spread']

        if not predict:
            game_data['won_game'] = home_team['score'] > away_team['score']
        
        game_data['Away Name'] = away_team['names']['name']
        if game_data['Away Name'] == 'Montreal Canadiens':
            game_data['Away Name'] = 'Montréal Canadiens'
        game_data['Away MoneyLine'] = away_team['moneyLine']
        game_data['Away Spread Price'] = away_team['spreadPrice']
        game_data['Away Score'] = away_team['score']
        game_data['Away Votes'] = away_team['votes']
        game_data['Away Spread'] = away_team['spread']

        game_data['Under Price'] = game['underPrice']
        game_data['Over Price'] = game['overPrice']
        game_data['Over Votes'] = game['overVotes']
        game_data['Under Votes'] = game['underVotes']
        game_data['Total'] = game['total']
        if not predict:
            game_data['Totals'] = home_team['score'] + away_team['score']
        game_data['Arena'] = game['stadium']
        
        extracted_data.append(game_data)

    df = pd.DataFrame(extracted_data)
    return df

def get_closest_match(team_name, choices, threshold=70):
    match, score = process.extractOne(team_name, choices)
    return match if score >= threshold else None

def merge_odds_and_projections(odds_df, power_df):
    # Select relevant columns from odds_df
    odds_df = odds_df[['Date', 'Home Name', 'Away Name', 'Home MoneyLine', 'Away MoneyLine']]
    
    # Prepare lists of team names for fuzzy matching
    home_teams = power_df['Team'].tolist()
    
    # Create a new DataFrame to hold the merged results
    merged_rows = []

    for index, row in odds_df.iterrows():
        # Find closest match for home team
        home_team_match = get_closest_match(row['Home Name'], home_teams)
        # Find closest match for away team
        away_team_match = get_closest_match(row['Away Name'], home_teams)
        
        if home_team_match and away_team_match:
            # Extract relevant power data for matched teams
            home_team_data = power_df[power_df['Team'] == home_team_match].iloc[0]
            away_team_data = power_df[power_df['Team'] == away_team_match].iloc[0]

            # Create a merged row
            merged_row = {
                'Date': row['Date'],
                'Home Name': row['Home Name'],
                'Away Name': row['Away Name'],
                'Home MoneyLine': row['Home MoneyLine'],
                'Away MoneyLine': row['Away MoneyLine'],
                'Home POWER': home_team_data['POWER'],
                'Away POWER': away_team_data['POWER'],
                'Home Record': home_team_data['Record'],
                'Away Record': away_team_data['Record'],
                'Home Win%': home_team_data['Win%'],
                'Away Win%': away_team_data['Win%'],
                'Home Avg PTS Diff': home_team_data['Avg PTS Diff'],
                'Away Avg PTS Diff': away_team_data['Avg PTS Diff'],
                'Home Avg PTS For': home_team_data['Avg PTS For'],
                'Away Avg PTS For': away_team_data['Avg PTS For'],
                'Home Avg PTS Against': home_team_data['Avg PTS Against'],
                'Away Avg PTS Against': away_team_data['Avg PTS Against'],
                'Home Strength of Schedule': home_team_data['Strength of Schedule'],
                'Away Strength of Schedule': away_team_data['Strength of Schedule']
            }
            merged_rows.append(merged_row)
    
    # Convert merged rows to DataFrame
    merged_df = pd.DataFrame(merged_rows)
    
    return merged_df


def calc_prob_2(home_power, away_power, param):
    return 1 / (1 + np.exp((home_power - away_power) / param))

def calc_spread_2(home_power, away_power, param, lower_bound_spread, upper_bound_spread):
    if lower_bound_spread == '-inf':
        if upper_bound_spread == 'inf':
            return 1
        return 1 / (1 + np.exp((upper_bound_spread - (home_power - away_power)) / param))
    elif upper_bound_spread == 'inf':
        return 1 - 1 / (1 + np.exp((lower_bound_spread - (home_power - away_power)) / param))
    else:
        return (
            1 / (1 + np.exp((upper_bound_spread - (home_power - away_power)) / param)) -
            1 / (1 + np.exp((lower_bound_spread - (home_power - away_power)) / param)))

def custom_game_selector_from_df(param, team_df):
    print("Available games:")
    print(team_df[['Home Name', 'Away Name']])  # Display available games

    # Check if the DataFrame is empty
    if team_df.empty:
        print("No available games.")
        return None

    game_rows = []  # List to hold rows for each game

    # Iterate over each row in team_df to calculate probabilities for all games
    for index, row in team_df.iterrows():
        home_team_name = row['Home Name']
        away_team_name = row['Away Name']

        # Get the power ratings and other necessary stats
        home_power = row['Home POWER']  # Ensure this column exists
        away_power = row['Away POWER']  # Ensure this column exists

        # Prepare a dictionary for the current game's statistics
        game_row = {
            'Home Name': home_team_name,
            'Away Name': away_team_name,
            'Home POWER': home_power,
            'Away POWER': away_power,
            'Home Record': row['Home Record'],
            'Away Record': row['Away Record'],
            'Home Win %': float(row["Home Win%"]) if isinstance(row["Home Win%"], str) else row["Home Win%"],
            'Away Win %': float(row["Away Win%"]) if isinstance(row["Away Win%"], str) else row["Away Win%"],
            'Home MoneyLine': row['Home MoneyLine'],
            'Away MoneyLine': row['Away MoneyLine']
        }

        # Calculate win probabilities using the actual power values
        home_win_prob = calc_prob_2(home_power, away_power, param) * 100
        away_win_prob = calc_prob_2(away_power, home_power, param) * 100

        game_row['Home Win Probability'] = f'{home_win_prob:.2f}%'
        game_row['Away Win Probability'] = f'{away_win_prob:.2f}%'

        # Calculate probabilities for winning by specific point ranges
##        game_row['Win by 1-5 Points (Home)'] = f'{calc_spread_2(home_power, away_power, param, 0, 5.5) * 100:.2f}%'
##        game_row['Win by 1-5 Points (Away)'] = f'{calc_spread_2(away_power, home_power, param, 0, 5.5) * 100:.2f}%'
##        
##        game_row['Win by 6-10 Points (Home)'] = f'{calc_spread_2(home_power, away_power, param, 5.5, 10.5) * 100:.2f}%'
##        game_row['Win by 6-10 Points (Away)'] = f'{calc_spread_2(away_power, home_power, param, 5.5, 10.5) * 100:.2f}%'
##        
##        game_row['Win by 11-15 Points (Home)'] = f'{calc_spread_2(home_power, away_power, param, 10.5, 15.5) * 100:.2f}%'
##        game_row['Win by 11-15 Points (Away)'] = f'{calc_spread_2(away_power, home_power, param, 10.5, 15.5) * 100:.2f}%'
##        
##        game_row['Win by 16-20 Points (Home)'] = f'{calc_spread_2(home_power, away_power, param, 15.5, 20.5) * 100:.2f}%'
##        game_row['Win by 16-20 Points (Away)'] = f'{calc_spread_2(away_power, home_power, param, 15.5, 20.5) * 100:.2f}%'
##        
##        game_row['Win by 21+ Points (Home)'] = f'{calc_spread_2(home_power, away_power, param, 20.5, "inf") * 100:.2f}%'
##        game_row['Win by 21+ Points (Away)'] = f'{calc_spread_2(away_power, home_power, param, 20.5, "inf") * 100:.2f}%'

        # Append the current game's row to the list
        game_rows.append(game_row)

    # Create a DataFrame from the list of game rows
    final_game_probability_df = pd.DataFrame(game_rows)

    final_game_probability_df['Date'] = (datetime.today() + timedelta(days=0)).strftime('%Y-%m-%d')

    # Reorder columns to place 'Date' as the first column
    columns = ['Date'] + [col for col in final_game_probability_df.columns if col != 'Date']
    final_game_probability_df = final_game_probability_df[columns]

    cols_to_convert = ['Home MoneyLine', 'Away MoneyLine']

    for col in cols_to_convert:
        final_game_probability_df[col] = final_game_probability_df[col].apply(moneyline_to_proba)

    final_game_probability_df['Away Win Probability'] = final_game_probability_df['Away Win Probability'].str.rstrip('%').astype(float)
    final_game_probability_df['Home Win Probability'] = final_game_probability_df['Home Win Probability'].str.rstrip('%').astype(float)

    final_game_probability_df['Away Win Probability'] = pd.to_numeric(final_game_probability_df['Away Win Probability'], errors='coerce')

    final_game_probability_df['Home Win Probability'] = pd.to_numeric(final_game_probability_df['Home Win Probability'], errors='coerce')

    final_game_probability_df['Away Win Probability'] = round(final_game_probability_df['Away Win Probability']/100, 3)
    final_game_probability_df['Home Win Probability'] = round(final_game_probability_df['Home Win Probability']/100, 3)

    final_game_probability_df['Away EV'] = final_game_probability_df.apply(lambda x: calculate_ev(x['Away Win Probability'], x['Away MoneyLine']), axis=1)

    # Calculate Expected Value (EV) for Home Team
    final_game_probability_df['Home EV'] = final_game_probability_df.apply(lambda x: calculate_ev(x['Home Win Probability'], x['Home MoneyLine']), axis=1)

    final_game_probability_df['Away Win Probability'] = round(final_game_probability_df['Away Win Probability'] * 100,1)

    final_game_probability_df['Home Win Probability'] = round(final_game_probability_df['Home Win Probability'] * 100,1)

    final_game_probability_df['Home EV'] = final_game_probability_df['Home EV'] * 100

    final_game_probability_df['Home EV'] = round(final_game_probability_df['Home EV'], 1)

    final_game_probability_df['Away EV'] = final_game_probability_df['Away EV'] * 100

    final_game_probability_df['Away EV'] = round(final_game_probability_df['Away EV'], 1)

    final_game_probability_df.drop(columns=['Home MoneyLine', 'Away MoneyLine', 'Home POWER','Away POWER', 'Home Record', 'Away Record', 'Home Win %', 'Away Win %'], inplace=True)

    # Use credentials to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
    client = gspread.authorize(creds)

    # Open the spreadsheet by its key
    spreadsheet_id = '1UcmQOaDViWVQdH99G_YFJ1xW-nRo6fnqBgL4LD3yRSM'
    spreadsheet = client.open_by_key(spreadsheet_id)

    sheet_name = 'BoomerModel'
    
    # Select the first sheet
    sheet = spreadsheet.worksheet(sheet_name)

    # Clear existing data
    sheet.clear()

    # Append headers if the first row is empty
    if not sheet.row_values(1):
        sheet.append_row(final_game_probability_df.columns.tolist())  # Add headers

    # Convert DataFrame to a list of lists for the data rows
    data = final_game_probability_df.values.tolist()

    # Append the data rows to the sheet
    sheet.append_rows(data)  # Efficiently append the rows

    return final_game_probability_df



def extra_menu(total_game_list, team_list, param):
    
    while True:
        print("""--EXTRAS MENU--
    1. Biggest Upsets
    2. Best Performances
    3. Most Consistent Teams
    4. Team Game Logs
    5. Team Probability Big Board
    6. Exit to Main Menu""")

        valid = False
        while valid == False:
            user_option = input('Enter a menu option: ')
            try:
                user_option = int(user_option)
                if user_option >= 1 and user_option <= 6:
                    print()
                    valid = True
                else:
                    raise ValueError
            except ValueError:
                print(f'Your option "{user_option}" is invalid.', end=' ')

        if user_option == 1:
            upsets = get_upsets(total_game_list)
            print(upsets)
            download_csv_option(upsets, 'biggest_upsets')
        elif user_option == 2:
            performances = get_best_performances(total_game_list)
            print(performances)
            download_csv_option(performances, 'best_performances')
        elif user_option == 3:
            consistency = get_team_consistency(team_list)
            print(consistency)
            download_csv_option(consistency, 'most_consistent_teams')
        elif user_option == 4:
            team, game_log = team_game_log(team_list)
            print(game_log)
            download_csv_option(game_log, f'{team.name.replace(" ", "_").lower()}_game_log')
        elif user_option == 5:
            team, team_probabilities = get_team_prob_breakdown(team_list, param)
            print(team_probabilities)
            download_csv_option(team_probabilities, f'{team.name.replace(" ", "_").lower()}_game_log')
        elif user_option == 6:
            pass

        return

def menu(power_df, xpoints, ypoints, param, computation_time, total_game_list, team_list):
    date = (datetime.today() + timedelta(days=0)).strftime('%Y-%m-%d')
    
    while True:
        print("""--MAIN MENU--
    1. View Power Rankings
    2. Custom Game Selector
    3. View Model Performance
    4. View Program Performance
    5. Extra Options
    6. View Today's Games
    7. Quit""")

        valid = False
        while valid == False:
            user_option = input('Enter a menu option: ')
            try:
                user_option = int(user_option)
                if user_option >= 1 and user_option <= 6:
                    print()
                    valid = True
                else:
                    raise ValueError
            except ValueError:
                print(f'Your option "{user_option}" is invalid.', end=' ')

        if user_option == 1:
            print(power_df)
            download_csv_option(power_df, 'power_rankings')
        elif user_option == 2:
            home_team, away_team, custom_game_df = custom_game_selector(param, team_list)
            print(custom_game_df)
            download_csv_option(custom_game_df, f'{home_team.name.replace(" ", "_").lower()}_vs_{away_team.name.replace(" ", "_").lower()}_game_probabilities')
        elif user_option == 3:
            model_performance(xpoints, ypoints, param)
        elif user_option == 4:
            print(f'Computation Time: {computation_time:.2f} seconds')
            print(f'Games Scraped: {len(total_game_list)}')
            print(f'Rate: {len(total_game_list)/computation_time:.1f} games/second')
        elif user_option == 5:
            extra_menu(total_game_list, team_list, param)
            
        elif user_option == 6:
            odds_df = fetch_odds_data(date, True)
            merged_df = merge_odds_and_projections(odds_df,power_df)
            final_df = custom_game_selector_from_df(param, merged_df)
            
            download_csv_option(final_df, f'{date}_games')
                
        elif user_option == 7:
            return

        input('Press ENTER to continue\t\t')
        print()

def main():
    start_time = time.time()

    games_metadf = retrieve_league_data()
    iterations = 10
    team_list, total_game_list = game_team_object_creation(games_metadf)
    assign_power(team_list, iterations)
    power_df = prepare_power_rankings(team_list)
    xpoints, ypoints, param = logistic_regression(total_game_list)

    computation_time = time.time()-start_time
    menu(power_df, xpoints, ypoints, param, computation_time, total_game_list, team_list)

if __name__ == '__main__':
    main()
