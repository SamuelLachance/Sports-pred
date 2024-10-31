import requests
import pandas as pd
import numpy as np
from pandas import json_normalize
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime

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
    base_url = f"https://www.oddsshark.com/api/scores/nhl/{date}?_format=json"
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
        game_data['Home MoneyLine'] = home_team['moneyLine']
        game_data['Home Spread Price'] = home_team['spreadPrice']
        game_data['Home Score'] = home_team['score']
        game_data['Home Votes'] = home_team['votes']
        game_data['Home Spread'] = home_team['spread']

        if not predict:
            game_data['won_game'] = home_team['score'] > away_team['score']
        
        game_data['Away Name'] = away_team['names']['name']
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

def get_game_projections(date):
    url = "https://ozajxwcjhgjugnhluqcm.supabase.co/rest/v1/game_projections"
    headers = {
        "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im96YWp4d2NqaGdqdWduaGx1cWNtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTg2NzIxNDAsImV4cCI6MjAzNDI0ODE0MH0.V4rXuEbCloTCWeIAa-eCHcq9lvPi2mDhOMAaUL5oxGY",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im96YWp4d2NqaGdqdWduaGx1cWNtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTg2NzIxNDAsImV4cCI6MjAzNDI0ODE0MH0.V4rXuEbCloTCWeIAa-eCHcq9lvPi2mDhOMAaUL5oxGY",
        "Accept-Profile": "public",
    }
    params = {
        "select": "*",
        "date": f"eq.{date}",
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data)
        df.to_csv(f"game_projections_{date}.csv", index=False)
        return df
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Merging the odds and projections DataFrames
def merge_odds_and_projections(odds_df, projections_df):
    decimal_places = 3
    
    merged_df = pd.merge(
        odds_df, 
        projections_df,
        left_on=['Home Name', 'Away Name', 'Date'],
        right_on=['home_name', 'visitor_name', 'date'],
        how='inner'
    )
    
    # Select only the specified columns
    merged_df = merged_df[
        ['Date','Home Name', 'Away Name', 'Home MoneyLine', 'Away MoneyLine', 
         'home_score', 'visitor_score', 'home_prob', 'visitor_prob', 'Total']
    ]

    cols_to_convert = ['Home MoneyLine', 'Away MoneyLine']

    for col in cols_to_convert:
        merged_df[col] = merged_df[col].apply(moneyline_to_proba)

    merged_df['visitor_prob'] = round(merged_df['visitor_prob'], decimal_places)
    merged_df['home_prob'] = round(merged_df['home_prob'], decimal_places)

    # Calculate Expected Value (EV) for Away Team
    merged_df['Away EV'] = merged_df.apply(lambda x: calculate_ev(x['visitor_prob'], x['Away MoneyLine']), axis=1)

    # Calculate Expected Value (EV) for Home Team
    merged_df['Home EV'] = merged_df.apply(lambda x: calculate_ev(x['home_prob'], x['Home MoneyLine']), axis=1)

    merged_df['visitor_prob'] = round(merged_df['visitor_prob'] * 100,1)

    merged_df['home_prob'] = round(merged_df['home_prob'] * 100,1)

    merged_df['Home EV'] = merged_df['Home EV'] * 100

    merged_df['Home EV'] = round(merged_df['Home EV'], 1)

    merged_df['Away EV'] = merged_df['Away EV'] * 100

    merged_df['Away EV'] = round(merged_df['Away EV'], 1)

    merged_df['total_pred'] = round(merged_df['home_score'] + merged_df['visitor_score'], 3)

    merged_df.drop(columns=['Home MoneyLine', 'Away MoneyLine','home_score','visitor_score'], inplace=True)

    # Use credentials to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
    client = gspread.authorize(creds)

    # Open the spreadsheet by its key
    spreadsheet_id = '1KgwFdqrRUs2fa5pSRmirj6ajyO2d14ONLsiksAYk8S8'
    spreadsheet = client.open_by_key(spreadsheet_id)

    sheet_name = 'IceAnalytics'
    
    # Select the first sheet
    sheet = spreadsheet.worksheet(sheet_name)

    # Clear existing data
    #sheet.clear()

    # Append headers if the first row is empty
    if not sheet.row_values(1):
        sheet.append_row(merged_df.columns.tolist())  # Add headers

    # Convert DataFrame to a list of lists for the data rows
    data = merged_df.values.tolist()

    # Append the data rows to the sheet
    sheet.append_rows(data)  # Efficiently append the rows

        
    return merged_df

# Example usage
date = (datetime.date.today() + datetime.timedelta(days=0)).strftime('%Y-%m-%d') #"2024-10-30"
odds_df = fetch_odds_data(date, True)
game_projections_df = get_game_projections(date)

# Perform merge if both DataFrames are fetched successfully
if odds_df is not None and game_projections_df is not None:
    merged_df = merge_odds_and_projections(odds_df, game_projections_df)
    merged_df.to_csv(f"merged_odds_projections_{date}.csv", index=False)
    print(merged_df)
else:
    print("Data fetch failed; merge operation skipped.")
