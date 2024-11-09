import requests
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import pytz
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
import urllib3
from fuzzywuzzy import process
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pandas import json_normalize
from datetime import datetime, timedelta, date


def get_closest_match(team_name, choices, threshold=70):
    match, score = process.extractOne(team_name, choices)
    return match if score >= threshold else None

def dratings_nhl():
    url = 'https://www.dratings.com/predictor/nhl-hockey-predictions/'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find("tbody", class_="table-body")
    rows = table.find_all('tr')

    tips = []

    for row in rows:
        try:
            data = row.find_all("td")

            date = data[0].find('span').get_text()
            date_object = datetime.strptime(date, "%m/%d/%Y")
            formatted_date = date_object.strftime("%Y-%m-%d")

            teams = data[1].find_all('a')
            team_a = teams[0].get_text()
            team_b = teams[1].get_text()

            percentages = data[3].find_all('span')
            team_a_per = float(percentages[0].get_text().replace('%', "")) / 100
            team_b_per = float(percentages[1].get_text().replace('%', "")) / 100

            predicted_winner = team_b if team_b_per > team_a_per else team_a

            tip = {
                "date": formatted_date,
                "away_team": team_a,
                "home_team": team_b,
                "away_team_percentage": team_a_per,
                "home_team_percentage": team_b_per
            }
            tips.append(tip)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    return tips

def dratings_cbb():
    url = 'https://www.dratings.com/predictor/ncaa-basketball-predictions/'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find("tbody", class_="table-body")
    rows = table.find_all('tr')

    tips = []

    for row in rows:
        try:
            data = row.find_all("td")

            date = data[0].find('span').get_text()
            date_object = datetime.strptime(date, "%m/%d/%Y")
            formatted_date = date_object.strftime("%Y-%m-%d")

            teams = data[1].find_all('a')
            team_a = teams[0].get_text()
            team_b = teams[1].get_text()

            percentages = data[2].find_all('span')
            team_a_per = float(percentages[0].get_text().replace('%', "")) / 100
            team_b_per = float(percentages[1].get_text().replace('%', "")) / 100

            predicted_winner = team_b if team_b_per > team_a_per else team_a

            tip = {
                "date": formatted_date,
                "away_team": team_a,
                "home_team": team_b,
                "away_team_percentage": team_a_per,
                "home_team_percentage": team_b_per
            }
            tips.append(tip)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    return tips


def extract_team_data(json_data,predict):
    # List to store extracted data
    extracted_data = []
    
    # Iterate through the scores list
    for game in json_data['scores']:
        game_data = {}
        
        # Extract home team data
        home_team = game['teams']['home']
        away_team = game['teams']['away']
        game_data['Home Name'] = home_team['names']['name']
        game_data['Home MoneyLine'] = home_team['moneyLine']
        print(game_data['Home MoneyLine'])
        game_data['Home Spread Price'] = home_team['spreadPrice']
        game_data['Home Score'] = home_team['score']
        game_data['Home Votes'] = home_team['votes']
        game_data['Homes Spread'] = home_team['spread']

        if predict == False :
            game_data['won_game'] = home_team['score'] > away_team['score']
        
        # Extract away team data
        game_data['Away Name'] = away_team['names']['name']
        game_data['Away MoneyLine'] = away_team['moneyLine']
        game_data['Away Spread Price'] = away_team['spreadPrice']
        game_data['Away Score'] = away_team['score']
        game_data['Away Votes'] = away_team['votes']
        game_data['Away Spread'] = away_team['spread']
        
        # Extract shared data
        game_data['Under Price'] = game['underPrice']
        game_data['Over Price'] = game['overPrice']
        game_data['Over Votes'] = game['overVotes']
        game_data['Under Votes'] = game['underVotes']
        game_data['Total'] = game['total']
        if predict == False :
            game_data['Totals'] = home_team['score'] + away_team['score']
        game_data['Arena'] = game['stadium']
        
        extracted_data.append(game_data)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(extracted_data)
    return df

def fetch_odds_data(date, predict,sports):
    base_url = f"https://www.oddsshark.com/api/scores/{sports}/{date}?_format=json"

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
        df = extract_team_data(data,predict)
        df['Date'] = date
        return df
    else:
        print(f"Failed to fetch data for date: {date}")
        return None


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

def merge_dratings_and_odds(dratings_df, odds_df, sports):
    """
    Merge the dratings DataFrame with the odds DataFrame on date, home team, and away team names.
    
    Parameters:
    - dratings_df: DataFrame containing Dratings predictions
    - odds_df: DataFrame containing betting odds
    
    Returns:
    - Merged DataFrame
    """
    
    # Ensure date columns are in the same format
    dratings_df['date'] = pd.to_datetime(dratings_df['date'])
    odds_df['Date'] = pd.to_datetime(odds_df['Date'])
    
    # Initialize a list to store merged data
    merged_data = []
    
    # Loop through each row in the odds DataFrame
    for idx, odds_row in odds_df.iterrows():
        # Find closest match for home team and away team in the dratings data
        closest_home_team = get_closest_match(odds_row['Home Name'], dratings_df['home_team'].unique())
        closest_away_team = get_closest_match(odds_row['Away Name'], dratings_df['away_team'].unique())

        # Proceed only if both teams have a valid match
        if closest_home_team and closest_away_team:
            # Find matching row in dratings DataFrame
            match_row = dratings_df[
                (dratings_df['home_team'] == closest_home_team) &
                (dratings_df['away_team'] == closest_away_team)
            ]

            # If a matching row is found, merge the rows
            if not match_row.empty:
                combined_row = odds_row.to_dict()  # Convert odds row to a dictionary
                combined_row.update(match_row.iloc[0].to_dict())  # Update with dratings data
                merged_data.append(combined_row)  # Append combined data to the list

    # Convert the list of dictionaries to a DataFrame
    merged_df = pd.DataFrame(merged_data)

    merged_df['Date'] = (datetime.today() + timedelta(days=0)).strftime('%Y-%m-%d')

    # Reorder columns to place 'Date' as the first column
    columns = ['Date'] + [col for col in merged_df.columns if col != 'Date']
    merged_df = merged_df[columns]

    cols_to_convert = ['Home MoneyLine', 'Away MoneyLine']

    for col in cols_to_convert:
        merged_df[col] = merged_df[col].apply(moneyline_to_proba)

    merged_df['away_team_percentage'] = merged_df['away_team_percentage'].astype(float)
    merged_df['home_team_percentage'] = merged_df['home_team_percentage'].astype(float)

    merged_df['away_team_percentage'] = pd.to_numeric(merged_df['away_team_percentage'], errors='coerce')

    merged_df['home_team_percentage'] = pd.to_numeric(merged_df['home_team_percentage'], errors='coerce')

    merged_df['away_team_percentage'] = round(merged_df['away_team_percentage'], 3)
    merged_df['home_team_percentage'] = round(merged_df['home_team_percentage'], 3)

    merged_df['Away EV'] = merged_df.apply(lambda x: calculate_ev(x['away_team_percentage'], x['Away MoneyLine']), axis=1)

    # Calculate Expected Value (EV) for Home Team
    merged_df['Home EV'] = merged_df.apply(lambda x: calculate_ev(x['home_team_percentage'], x['Home MoneyLine']), axis=1)

    merged_df['away_team_percentage'] = round(merged_df['away_team_percentage'] * 100,1)

    merged_df['home_team_percentage'] = round(merged_df['home_team_percentage'] * 100,1)

    merged_df['Home EV'] = merged_df['Home EV'] * 100

    merged_df['Home EV'] = round(merged_df['Home EV'], 1)

    merged_df['Away EV'] = merged_df['Away EV'] * 100

    merged_df['Away EV'] = round(merged_df['Away EV'], 1)

    merged_df = merged_df[
        ['Home Name', 'Away Name','home_team_percentage', 'away_team_percentage','Home EV','Away EV']]

    if sports == "NHL" :
        # Use credentials to create a client to interact with the Google Drive API
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
        client = gspread.authorize(creds)

        # Open the spreadsheet by its key
        spreadsheet_id = '1KgwFdqrRUs2fa5pSRmirj6ajyO2d14ONLsiksAYk8S8'
        spreadsheet = client.open_by_key(spreadsheet_id)

        sheet_name = 'Dratings'
        
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

    if sports == "NCAAB" :
        # Use credentials to create a client to interact with the Google Drive API
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
        client = gspread.authorize(creds)

        # Open the spreadsheet by its key
        spreadsheet_id = '1KwURGRKr6iMgTTWK7mxev4ViUsxrhJE803ieCyB9Asw'
        spreadsheet = client.open_by_key(spreadsheet_id)

        sheet_name = 'Dratings'
        
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

def convert_to_dataframe(data):
    # Normalize the nested structure into a flat DataFrame
    df = json_normalize(data, sep='_')
    return df

def main():
    sports = "NCAAB"
    date = datetime.today().date()
    print(date)

    nhl_odds_df = fetch_odds_data(date, True, sports)
    if sports == "NHL":
        nhl_dratings_df = convert_to_dataframe(dratings_nhl())
    if sports == "NCAAB":
        nhl_dratings_df = convert_to_dataframe(dratings_cbb())
        

    # Save nhl_odds_df to CSV if it's not None
    if nhl_odds_df is not None:
        nhl_odds_df.to_csv(f"nhl_odds_{date}.csv", index=False)
        print(f"nhl_odds_{date}.csv saved successfully.")

    # Save nhl_dratings_df to CSV if it's not empty
    nhl_dratings_df_df = pd.DataFrame(nhl_dratings_df)  # Convert list of dicts to DataFrame
    if not nhl_dratings_df_df.empty:
        nhl_dratings_df_df.to_csv(f"nhl_dratings_{date}.csv", index=False)
        print(f"nhl_dratings_{date}.csv saved successfully.")
        merged_df = merge_dratings_and_odds(nhl_dratings_df, nhl_odds_df,sports)
        merged_df.to_csv(f"nhl_merged_{date}.csv", index=False)
        print(f"nhl_merged_{date}.csv saved successfully.")

if __name__ == "__main__":
    main()
