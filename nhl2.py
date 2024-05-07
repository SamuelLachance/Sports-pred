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
    'ARI': 'Arizona Coyotes',
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

    df.drop(['Matchup', 'Consensus', 'Team_Consensus_Pairs', 'Date', 'Sides', 'Picks', 'Indepth', 'gameId', 'rowId', 'datePull'], axis=1, inplace=True)
    
    return df
    
def main():

    today = datetime.today().date()

    today_2 = pd.Timestamp('now').floor('D')

    elo_url = 'https://raw.githubusercontent.com/Neil-Paine-1/NHL-Player-And-Team-Ratings/master/nhl_elo.csv'
    elo_df = pd.read_csv(elo_url).drop_duplicates()

    elo_df['away_team'] = elo_df.apply(lambda row: row['team2'] if row['is_home'] == 1 else row['team1'], axis=1)
    elo_df['home_team'] = elo_df.apply(lambda row: row['team1'] if row['is_home'] == 1 else row['team2'], axis=1)
    elo_df['away_team_elo'] = elo_df.apply(lambda row: row['elo2_pre'] if row['is_home'] == 1 else row['elo1_pre'], axis=1)
    elo_df['home_team_elo'] = elo_df.apply(lambda row: row['elo1_pre'] if row['is_home'] == 1 else row['elo2_pre'], axis=1)
    elo_df['away_team_percentage'] = elo_df.apply(lambda row: row['prob2'] if row['is_home'] == 1 else row['prob1'], axis=1)
    elo_df['home_team_percentage'] = elo_df.apply(lambda row: row['prob1'] if row['is_home'] == 1 else row['prob2'], axis=1)

    elo_df['away_team'] = elo_df['away_team'].map(team_abbr_to_name_mlb)
    elo_df['home_team'] = elo_df['home_team'].map(team_abbr_to_name_mlb)

    elo_df.rename(columns={'date': 'game_date'}, inplace=True)

    elo_df['game_date'] = pd.to_datetime(elo_df['game_date'])


    elo_df.to_csv('test.csv')


    elo_df = elo_df[elo_df['game_date'] == today_2]

    elo_df = elo_df[['home_team', 'away_team', 'home_team_elo', 'away_team_elo', 'home_team_percentage', 'away_team_percentage']]
    
    elo_df.to_csv('elo_df.csv')
    
    odds_df = fetch_odds_data(today, True)

    odds_df.drop('Arena', axis=1, inplace=True)
    odds_df.drop('Home Spread Price', axis=1, inplace=True)
    odds_df.drop('Away Spread Price', axis=1, inplace=True)
    odds_df.drop('Under Price', axis=1, inplace=True)
    odds_df.drop('Over Price', axis=1, inplace=True)
    odds_df.drop('Over Votes', axis=1, inplace=True)
    odds_df.drop('Under Votes', axis=1, inplace=True)
    odds_df.drop('Total', axis=1, inplace=True)
    odds_df.drop('Home Votes', axis=1, inplace=True)
    odds_df.drop('Homes Spread', axis=1, inplace=True)
    odds_df.drop('Away Score', axis=1, inplace=True)
    odds_df.drop('Away Votes', axis=1, inplace=True)
    odds_df.drop('Away Spread', axis=1, inplace=True)
    odds_df.drop('Home Score', axis=1, inplace=True)
    odds_df.drop('Date', axis=1, inplace=True)
    odds_df = odds_df.drop_duplicates()

    print(odds_df)
    
    odds_df.rename(columns={'Home Name': 'home_team', 'Away Name': 'away_team'}, inplace=True)

    cols_to_convert = ['Home MoneyLine', 'Away MoneyLine']

    for col in cols_to_convert:
        odds_df[col] = odds_df[col].apply(moneyline_to_proba)

    odds_df.to_csv('odds_df.csv')
    
    dratings_tips = scrape_dratings()
    
    # Scrape from different pages of the Covers website
    covers_urls = [
        (f'https://contests.covers.com/consensus/topconsensus/nhl/overall/{today}', 'Overall Bettors'),
        (f'https://contests.covers.com/consensus/topconsensus/nhl/expert/{today}', 'Team Leaders'),
        (f'https://contests.covers.com/consensus/topconsensus/nhl/top10pct/{today}', 'Top 10%')
    ]
    
    covers_tips = pd.DataFrame()
    for url, tipser in covers_urls:
        try:
            tip_df = fetch_mlb_consensus(url, tipser)
            if not tip_df.empty:
                covers_tips = pd.concat([covers_tips, tip_df], ignore_index=True)
        except Exception as e:
            print(f'Failed to extract tips from {url}: {e}')

    covers_df = covers_tips

    dratings_df = convert_to_dataframe(dratings_tips)

    print(dratings_df)

    dratings_df.dropna(inplace=True)

    dratings_df.to_csv('dratings_df.csv')

    print(covers_df)

    covers_df.to_csv('covers3_df.csv')

    covers_df['away_team'] = covers_df['away_team'].apply(lambda x: x.upper())

    # Convert the 'home_team' column to uppercase
    covers_df['home_team'] = covers_df['home_team'].apply(lambda x: x.upper())

    covers_df.to_csv('covers4_df.csv')
    
    decimal_places = 3  # You can adjust the number of decimal places as per your requirement

    # Convert percentage columns to float
    covers_df['away_team_percentage'] = covers_df['away_team_percentage'].astype(float)
    covers_df['home_team_percentage'] = covers_df['home_team_percentage'].astype(float)

    # Group by game and calculate the average percentages for each team
    covers_df = covers_df.groupby(['away_team', 'home_team']).agg({
        'away_team_percentage': 'mean',
        'home_team_percentage': 'mean'
    }).reset_index()

    covers_df['away_team_percentage'] = round(covers_df['away_team_percentage'] / 100, decimal_places)
    covers_df['home_team_percentage'] = round(covers_df['home_team_percentage'] / 100, decimal_places)

    covers_df.to_csv('covers2_df.csv')

    covers_df['away_team'] = covers_df['away_team'].map(team_abbr_to_name_mlb)
    covers_df['home_team'] = covers_df['home_team'].map(team_abbr_to_name_mlb)
    
    covers_df.to_csv('covers_df.csv')

    # Merge the two dataframes on the common columns (away_team and home_team)
    merged_df = pd.merge(covers_df, elo_df, on=['away_team', 'home_team'], how='outer')

    # Calculate the average team percentages
    merged_df['away_team_percentage'] = (merged_df['away_team_percentage_x'] + merged_df['away_team_percentage_y']) / 2
    merged_df['home_team_percentage'] = (merged_df['home_team_percentage_x'] + merged_df['home_team_percentage_y']) / 2

    # Drop unnecessary columns
    merged_df.drop(['away_team_percentage_x', 'away_team_percentage_y', 'home_team_percentage_x', 'home_team_percentage_y'], axis=1, inplace=True)

    merged_df.dropna(inplace=True)

    merged_df.to_csv('merged_df1.csv')

    merged_df = pd.merge(merged_df, dratings_df , on=['away_team', 'home_team'], how='outer')

    merged_df['away_team_percentage'] = (merged_df['away_team_percentage_x'] + merged_df['away_team_percentage_y']) / 2
    merged_df['home_team_percentage'] = (merged_df['home_team_percentage_x'] + merged_df['home_team_percentage_y']) / 2

    merged_df.drop(['away_team_percentage_x', 'away_team_percentage_y', 'home_team_percentage_x', 'home_team_percentage_y'], axis=1, inplace=True)

    merged_df = pd.merge(merged_df, odds_df, on=['away_team', 'home_team'], how='outer')

    # Calculate Expected Value (EV) for Away Team
    merged_df['Away EV'] = merged_df.apply(lambda x: calculate_ev(x['away_team_percentage'], x['Away MoneyLine']), axis=1)

    # Calculate Expected Value (EV) for Home Team
    merged_df['Home EV'] = merged_df.apply(lambda x: calculate_ev(x['home_team_percentage'], x['Home MoneyLine']), axis=1)

    merged_df.dropna(inplace=True)

    merged_df.drop('Home MoneyLine', axis=1, inplace=True)

    merged_df.drop('Away MoneyLine', axis=1, inplace=True)

    merged_df.drop('date', axis=1, inplace=True)

    merged_df['away_team_percentage'] = merged_df['away_team_percentage'] * 100

    merged_df['home_team_percentage'] = merged_df['home_team_percentage'] * 100

    merged_df['Home EV'] = merged_df['Home EV'] * 100

    merged_df['Home EV'] = round(merged_df['Home EV'], 1)

    merged_df['Away EV'] = merged_df['Away EV'] * 100

    merged_df['Away EV'] = round(merged_df['Away EV'], 1)

    merged_df = merged_df.drop_duplicates()
    
    merged_df.to_csv('final_df_nhl.csv')
    
    merged_df['game_date'] = today
    merged_df['game_date'] = merged_df['game_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        
    # Display the merged dataframe
    print("Final DataFrame:")
    print(merged_df.to_string(index=False, justify='center'))

    # Use credentials to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
    client = gspread.authorize(creds)

    # Open the spreadsheet by its key
    spreadsheet_id = '1KgwFdqrRUs2fa5pSRmirj6ajyO2d14ONLsiksAYk8S8'
    spreadsheet = client.open_by_key(spreadsheet_id)

    # Select the first sheet
    sheet = spreadsheet.sheet1

    # Clear existing data
    #sheet.clear()

    # Check if the first row is empty (indicating a need for headers)
    if not sheet.row_values(1):  # This checks the first row for any content
        # The sheet is empty, add the headers
        sheet.append_row(merged_df.columns.tolist())

    # Convert DataFrame to list of lists for the data rows
    data = merged_df.values.tolist()

    # Append the data to the sheet
    sheet.append_rows(data)  # Using append_rows for efficiency

if __name__ == "__main__":
    main()
