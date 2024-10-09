import requests
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import pytz
import pandas as pd

import gspread
from oauth2client.service_account import ServiceAccountCredentials

import numpy as np

from pandas import json_normalize

import urllib3

team_abbr_to_name_mlb = {
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
    'UTA': 'Utah Hockey Club',
    'TOR': 'Toronto Maple Leafs',
    'VAN': 'Vancouver Canucks',
    'VGK': 'Vegas Golden Knights',
    'VEG': 'Vegas Golden Knights',
    'WSH': 'Washington Capitals',
    'WAS': 'Washington Capitals',
    'WPG': 'Winnipeg Jets',
    'WIN': 'Winnipeg Jets'
}

# Headers for web scraping
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

source = 'covers'
def convert_to_dataframe(data):
    # Normalize the nested structure into a flat DataFrame
    df = json_normalize(data, sep='_')
    return df


def scrape_dratings():
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

def extract_sharp_data(data):
    # Prepare a list to store the extracted information
    game_data = []

    # Loop through the events (games)
    for event in data['data']['events']:
        game_id = event['gameId']
        
        # Find consensus odds (sportsbookId: 22)
        consensus_odds = next((odds for odds in event['odds']['pregame'] if odds['sportsbookId'] == 22), None)

        # If consensus odds exist, extract them
        if consensus_odds:
            away_moneyline = consensus_odds.get('awayMoneyLine', None)
            home_moneyline = consensus_odds.get('homeMoneyLine', None)
            away_spread = consensus_odds.get('awaySpread', None)
            home_spread = consensus_odds.get('homeSpread', None)
            over_under = consensus_odds.get('overUnder', None)
        else:
            away_moneyline = home_moneyline = away_spread = home_spread = over_under = None

        # Extract home and away team data
        away_team = next(team for team in event['teams'] if team['teamId'] == event['awayTeamId'])
        home_team = next(team for team in event['teams'] if team['teamId'] == event['homeTeamId'])

        # Append the data in the desired format
        game_data.append({
            'gameId': game_id,
            'awayTeamId': away_team['teamId'],
            'awayTeamKey': away_team['key'],
            'awayTeamName': away_team['displayName'],
            'homeTeamId': home_team['teamId'],
            'homeTeamKey': home_team['key'],
            'homeTeamName': home_team['displayName'],
            'awayMoneyLine': away_moneyline,
            'homeMoneyLine': home_moneyline,
            'awaySpread': away_spread,
            'homeSpread': home_spread,
            'overUnder': over_under
        })

    # Convert the list to a DataFrame
    df = pd.DataFrame(game_data)
    return df

def fetch_sharp_data(date):
    # Convert date to string in the correct format if needed
    start_date = f"{date.strftime('%Y-%m-%d')}T04:00:00.000Z"
    
    # Add one day to the input date to calculate end date
    next_day = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = f"{next_day}T03:59:59.000Z"
    
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")

    # Build the URL
    base_url = f"https://graph.sharp.app/operations/v1/events/LegacyByDates?wg_api_hash=0bd8d897&wg_variables={{\"league\":\"WNBA\",\"startAt\":\"{start_date}\",\"endAt\":\"{end_date}\"}}"
    print(base_url)

    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Sec-Ch-Ua': '"Chromium";v="118", "Microsoft Edge";v="118", "Not=A?Brand";v="99"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.44'
    }

    response = requests.get(base_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(data)
        df = extract_sharp_data(data)
        df['Date'] = date
        return df
    else:
        print(f"Failed to fetch data for date: {date}")
        return None

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

def fetch_odds_data(date, predict):
    base_url = f"https://www.oddsshark.com/api/scores/nhl/{date}?_format=json"

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

def fetch_mlb_consensus(url, tipser):
    # Create HTTP connection
    http = urllib3.PoolManager()

    # Get webpage
    response = http.request('GET', url)
    soup = BeautifulSoup(response.data, 'html.parser')

    # Extract tables using pandas
    data = pd.read_html(soup.prettify())

    # Combine all extracted tables into one DataFrame
    df = pd.concat(data, ignore_index=True)

    # Process 'Matchup' and 'Consensus' columns
    df['Matchup'] = df['Matchup'].str.split().str[1:]
    df['Consensus'] = df['Consensus'].str.split()

    # Split data into pairs
    df['Team_Consensus_Pairs'] = df.apply(lambda row: list(zip(row['Matchup'], row['Consensus'])), axis=1)

    # Create new columns for teams and consensus
    team_data = df['Team_Consensus_Pairs'].apply(lambda x: [i[0] for i in x]).tolist()
    consensus_data = df['Team_Consensus_Pairs'].apply(lambda x: [i[1].strip('%') for i in x]).tolist()  # Stripping '%' from percentages

    df[['away_team', 'home_team']] = pd.DataFrame(team_data)
    df[['away_team_percentage', 'home_team_percentage']] = pd.DataFrame(consensus_data).astype(float)  # Convert percentage strings to float

    # Assign game IDs and clean up DataFrame
    df['gameId'] = df.index
    df['datePull'] = datetime.today().date()
    df['tipser'] = tipser
    df['rowId'] = df.index

    columns_to_drop = ['Matchup', 'Consensus', 'Team_Consensus_Pairs', 'Date', 'Sides', 'Picks', 'Indepth', 'gameId', 'rowId', 'datePull']
    df.drop([col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)
    
    return df

def fetch_ai_predictions(game_id):
    url = f"https://graph.sharp.app/operations/v1/ai-predictions/ByGameId?wg_api_hash=0bd8d897&wg_variables=%7B%22gameId%22%3A{game_id}%7D"
    print(url)
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()['data']['predictions']
        
        # Extract the win percentages for both teams
        teams = data['teams']
        total = data['overUnder']
        away_team = teams[0]
        home_team = teams[1]
        
        away_team_id = away_team['teamId']
        away_win_percentage = away_team['winPercentage']
        away_spread = away_team['spread']
        home_team_id = home_team['teamId']
        home_spread = home_team['spread']
        home_win_percentage = home_team['winPercentage']
        
        return away_win_percentage, home_win_percentage, away_spread, home_spread, total
    else:
        # Return None if there was an error with the request
        return None, None

# Function to update the DataFrame with AI win percentages
def update_with_ai_predictions(df):
    # Initialize columns for win percentages
    df['awayWinPercentage'] = None
    df['homeWinPercentage'] = None
    df['away_spread'] = None
    df['home_spread'] = None
    df['total'] = None
    
    # Loop through each row in the DataFrame
    for i, row in df.iterrows():
        game_id = row['gameId']
        away_team_id = row['awayTeamId']
        home_team_id = row['homeTeamId']
        
        # Fetch the AI predictions for the current gameId
        away_win_percentage, home_win_percentage, away_spread, home_spread, total  = fetch_ai_predictions(game_id)
        
        # Update the DataFrame with the fetched win percentages
        df.at[i, 'awayWinPercentage'] = away_win_percentage
        df.at[i, 'homeWinPercentage'] = home_win_percentage
        df.at[i, 'away_spread'] = away_spread
        df.at[i, 'home_spread'] = home_spread
        df.at[i, 'total'] = total

    return df

def fetch_handles(game_id):
    url = f"https://graph.sharp.app/operations/v1/handles/ByGameId?wg_api_hash=0bd8d897&wg_variables=%7B%22gameId%22%3A{game_id}%7D"
    print(url)
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()['data']['handles']

        # Extract data for spread
        spread_data = data['spread']
        total_data = data['total']
        moneyline_data = data['moneyline']
        
        away_spread_info = next((item for item in spread_data if item['outcomeType'] == 'Away'), None)
        home_spread_info = next((item for item in spread_data if item['outcomeType'] == 'Home'), None)
        
        away_moneyline_info = next((item for item in moneyline_data if item['outcomeType'] == 'Away'), None)
        home_moneyline_info = next((item for item in moneyline_data if item['outcomeType'] == 'Home'), None)

        total_over_info = next((item for item in total_data if item['outcomeType'] == 'Over'), None)
        total_under_info = next((item for item in total_data if item['outcomeType'] == 'Under'), None)

        return {
            'away_spread': away_spread_info,
            'home_spread': home_spread_info,
            'away_moneyline': away_moneyline_info,
            'home_moneyline': home_moneyline_info,
            'total_over': total_over_info,
            'total_under': total_under_info
        }
    else:
        # Return None if there was an error with the request
        return None


# Function to update the DataFrame with handle percentages and spreads
def update_with_handles(df):
    # Initialize columns for handles data
    df['away_money'] = None
    df['home_money'] = None
    df['over_money'] = None
    df['under_money'] = None
    
    # Loop through each row in the DataFrame
    for i, row in df.iterrows():
        game_id = row['gameId']
        
        # Fetch the handles for the current gameId
        handles_data = fetch_handles(game_id)
        
        if handles_data:
            away_spread = handles_data['away_spread']
            home_spread = handles_data['home_spread']
            total_over = handles_data['total_over']
            total_under = handles_data['total_under']
            
            # Update the DataFrame with the fetched handle data
            if away_spread:
                df.at[i, 'away_money'] = away_spread['moneyPercentage']
            if home_spread:
                df.at[i, 'home_money'] = home_spread['moneyPercentage']
            if total_over:
                df.at[i, 'over_money'] = total_over['moneyPercentage']
            if total_under:
                df.at[i, 'under_money'] = total_under['moneyPercentage']

    return df
    
def main():

    today = datetime.today().date()

    decimal_places = 3

    # Fetch sharp data and AI predictions (assuming these functions exist and work properly)
    sharp_df = fetch_sharp_data(today)
    sharp_df = update_with_ai_predictions(sharp_df)
    #sharp_df = update_with_handles(sharp_df)

    cols_to_convert = ['awayMoneyLine', 'homeMoneyLine']

    for col in cols_to_convert:
        sharp_df[col] = sharp_df[col].apply(moneyline_to_proba)

    sharp_df.to_csv('sharp_df.csv', index=False)

    sharp_df['awayWinPercentage'] = pd.to_numeric(sharp_df['awayWinPercentage'], errors='coerce')

    sharp_df['homeWinPercentage'] = pd.to_numeric(sharp_df['homeWinPercentage'], errors='coerce')

    sharp_df['awayWinPercentage'] = round(sharp_df['awayWinPercentage'] / 100, decimal_places)
    sharp_df['homeWinPercentage'] = round(sharp_df['homeWinPercentage'] / 100, decimal_places)

    # Calculate Expected Value (EV) for Away Team
    sharp_df['Away EV'] = sharp_df.apply(lambda x: calculate_ev(x['awayWinPercentage'], x['awayMoneyLine']), axis=1)

    # Calculate Expected Value (EV) for Home Team
    sharp_df['Home EV'] = sharp_df.apply(lambda x: calculate_ev(x['homeWinPercentage'], x['homeMoneyLine']), axis=1)

    sharp_df.to_csv('sharp_df.csv', index=False)

    # Clean the DataFrame by dropping unwanted columns in a single line
    sharp_df.drop(columns=['homeSpread', 'awaySpread', 'homeMoneyLine', 'awayMoneyLine', 'homeTeamId', 'awayTeamId', 'overUnder', 'gameId', 'homeTeamName', 'awayTeamName'], inplace=True)

    # Ensure the 'Date' column is fully converted to string format
    sharp_df['Date'] = pd.to_datetime(sharp_df['Date']).dt.strftime('%Y-%m-%d')

    # Save the DataFrame to CSV (optional for debugging)
    sharp_df.to_csv('sharp_df_final.csv', index=False)


    sharp_df.rename(columns={
    'awayTeamKey': 'Away_Team',
    'homeTeamKey': 'Home_Team',
    'Date': 'Game_Date',
    'awayWinPercentage': 'Away_Win%',
    'homeWinPercentage': 'Home_Win%'
    }, inplace=True)


    sharp_df['Away_Win%'] = sharp_df['Away_Win%'] * 100

    sharp_df['Home_Win%'] = sharp_df['Home_Win%'] * 100

    sharp_df['Home EV'] = sharp_df['Home EV'] * 100

    sharp_df['Home EV'] = round(sharp_df['Home EV'], 1)

    sharp_df['Away EV'] = sharp_df['Away EV'] * 100

    sharp_df['Away EV'] = round(sharp_df['Away EV'], 1)

    # Use credentials to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
    client = gspread.authorize(creds)

    # Open the spreadsheet by its key
    spreadsheet_id = '1Spfeh_UBt3g9BNo_Zek9jekrsFLQOY-D4uJ9wKA8fCg'
    spreadsheet = client.open_by_key(spreadsheet_id)

    # Select the first sheet
    sheet = spreadsheet.sheet1

    # Clear existing data
    sheet.clear()

    # Append headers if the first row is empty
    if not sheet.row_values(1):
        sheet.append_row(sharp_df.columns.tolist())  # Add headers

    # Convert DataFrame to a list of lists for the data rows
    data = sharp_df.values.tolist()

    # Append the data rows to the sheet
    sheet.append_rows(data)  # Efficiently append the rows

if __name__ == "__main__":
    main()
