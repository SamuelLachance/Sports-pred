import requests
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import pytz
import pandas as pd

import numpy as np

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from pandas import json_normalize

team_abbr_to_name_mlb = {
    'ARI': 'Arizona Diamondbacks',
    'ATL': 'Atlanta Braves',
    'BAL': 'Baltimore Orioles',
    'BOS': 'Boston Red Sox',
    'CHC': 'Chicago Cubs',
    'CHW': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds',
    'CLE': 'Cleveland Guardians',
    'COL': 'Colorado Rockies',
    'DET': 'Detroit Tigers',
    'HOU': 'Houston Astros',
    'KC': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels',
    'LAD': 'Los Angeles Dodgers',
    'MIA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers',
    'MIN': 'Minnesota Twins',
    'NYM': 'New York Mets',
    'NYY': 'New York Yankees',
    'OAK': 'Oakland Athletics',
    'PHI': 'Philadelphia Phillies',
    'PIT': 'Pittsburgh Pirates',
    'SD': 'San Diego Padres',
    'SEA': 'Seattle Mariners',
    'SF': 'San Francisco Giants',
    'STL': 'St. Louis Cardinals',
    'TB': 'Tampa Bay Rays',
    'TEX': 'Texas Rangers',
    'TOR': 'Toronto Blue Jays',
    'WSH': 'Washington Nationals',  # Option 1
    'WAS': 'Washington Nationals'   # Option 2
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
    url = 'https://www.dratings.com/predictor/mlb-baseball-predictions/'
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

def scrape_covers(url, tipser_name):
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find('tbody')
    rows = table.find_all('tr')[1:]
    tips = []

    for row in rows:
        away_team = row.find('span', class_='covers-CoversConsensus-table--teamBlock').get_text().strip()
        home_team = row.find('span', class_='covers-CoversConsensus-table--teamBlock2').get_text().strip()
    
        date_cell = row.find_all('td')[1]
        dates = [value.strip() for value in date_cell.get_text(separator='|').split('|')]
        
   

        # Extract month and day
        date_str= f'{dates[0]}  {dates[1]}'

            # Extract date and time components
        date_part = date_str.split()[1:3]
        time_part = date_str.split()[3:5]

            # Create standard date and time strings
        formatted_date_str = f"{date_part[0]} {date_part[1]} {datetime.now().year}"
        formatted_time_str = ' '.join(time_part).replace('ET', '').strip()

        # Convert to datetime objects
        date_obj = datetime.strptime(formatted_date_str, '%b %d %Y')
        time_obj = datetime.strptime(formatted_time_str, '%I:%M %p')

        # Combine date and time into a single datetime object
        combined_datetime = datetime.combine(date_obj, time_obj.time())

        # Adjust for Eastern Time
        eastern = pytz.timezone('US/Eastern')
        localized_datetime = eastern.localize(combined_datetime)


            # Convert to GMT/UTC
        utc_datetime = localized_datetime.astimezone(pytz.utc)

        # Extract just the date
        utc_date = utc_datetime.date()

        # format date as YYYY-MM-DD

        utc_date = utc_date.strftime("%Y-%m-%d")


        game = {
            'game_date': utc_date,
            'away_team': away_team,
            'home_team': home_team, 
            "sport": "mlb"
        }

        


        cells = row.find_all('td')[4]
        # Split the text using <br> and strip any whitespace
        values = [value.strip() for value in cells.get_text(separator='|').split('|')]

        # Assign to separate variables
        away_team_picks, home_team_picks = values

        total_picks = int(away_team_picks) + int(home_team_picks)
        away_team_percentage = int(away_team_picks) / total_picks * 100
        home_team_percentage = int(home_team_picks) / total_picks * 100

        if int(away_team_picks) > int(home_team_picks):
            winner = away_team
        else:
            winner = home_team

        tipser = {"name": tipser_name, "source": source}

     
        tip = {"tipster": tipser, "game": game,"away_team": away_team,"home_team": home_team,"away_team_percentage": away_team_percentage,"home_team_percentage": home_team_percentage}

        tips.append(tip)

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
    base_url = f"https://www.oddsshark.com/api/scores/mlb/{date}?_format=json"

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
    
def main():

    today = datetime.today().date()

    today_2 = pd.Timestamp('now').floor('D')

    elo_url = 'https://raw.githubusercontent.com/Neil-Paine-1/MLB-WAR-data-historical/master/mlb-elo-latest.csv'
    elo_df = pd.read_csv(elo_url).drop_duplicates()

    is_home = elo_df['is_home'] == 1
    elo_df['home_team'] = np.where(is_home, elo_df['team1'], elo_df['team2'])
    elo_df['away_team'] = np.where(is_home, elo_df['team2'], elo_df['team1'])
    elo_df['home_team_elo'] = np.where(is_home, elo_df['elo1_pre'], elo_df['elo2_pre'])
    elo_df['away_team_elo'] = np.where(is_home, elo_df['elo2_pre'], elo_df['elo1_pre'])
    elo_df['home_team_percentage'] = np.where(is_home, elo_df['elo_prob1'], elo_df['elo_prob2'])
    elo_df['away_team_percentage'] = np.where(is_home, elo_df['elo_prob2'], elo_df['elo_prob1'])

    elo_df['away_team'] = elo_df['away_team'].map(team_abbr_to_name_mlb)
    elo_df['home_team'] = elo_df['home_team'].map(team_abbr_to_name_mlb)

    elo_df.rename(columns={'date': 'game_date'}, inplace=True)

    elo_df['game_date'] = pd.to_datetime(elo_df['game_date'])


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
        ('https://contests.covers.com/consensus/topconsensus/mlb/overall', 'Overall Bettors'),
        ('https://contests.covers.com/consensus/topconsensus/mlb/expert', 'Team Leaders'),
        ('https://contests.covers.com/consensus/topconsensus/mlb/top10pct', 'Top 10%')
    ]
    
    covers_tips = []
    for url, tipser in covers_urls:
        try:
            covers_tips += scrape_covers(url, tipser)
        except Exception as e:
            print(f'Failed to extract tips from {url}: {e}')

    covers_df = convert_to_dataframe(covers_tips)

    dratings_df = convert_to_dataframe(dratings_tips)

    print(dratings_df)

    dratings_df.to_csv('dratings_df.csv')

    print(covers_df)

    covers_df['away_team'] = covers_df['away_team'].apply(lambda x: x.upper())

    # Convert the 'home_team' column to uppercase
    covers_df['home_team'] = covers_df['home_team'].apply(lambda x: x.upper())
    
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
    
    merged_df.to_csv('final_df_mlb.csv')
    
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
    spreadsheet_id = '17bvX0sNEup_af6wBRlr9nYqR6ghSTDWnkLzO6-v4VBc'
    spreadsheet = client.open_by_key(spreadsheet_id)

    # Select the first sheet
    sheet = spreadsheet.sheet1

    # Clear existing data
    sheet.clear()

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
