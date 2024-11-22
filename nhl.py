import requests
import pandas as pd
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def moneyline_to_proba(moneyline):
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)

def calculate_ev(model_prob, vegas_prob):
    potential_profit = (1 / vegas_prob) - 1
    prob_lose = 1 - model_prob
    ev = model_prob * potential_profit - prob_lose * 1
    return ev

def fetch_sharp_data(date):
    start_date = f"{date.strftime('%Y-%m-%d')}T04:00:00.000Z"
    next_day = (date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = f"{next_day}T03:59:59.000Z"
    base_url = f"https://graph.sharp.app/operations/v1/events/LegacyByDates?wg_api_hash=0bd8d897&wg_variables={{\"league\":\"NHL\",\"startAt\":\"{start_date}\",\"endAt\":\"{end_date}\"}}"
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
        df = extract_sharp_data(data)
        df['Date'] = date
        return df
    else:
        print(f"Failed to fetch data for date: {date}")
        return pd.DataFrame()

def extract_sharp_data(data):
    game_data = []
    for event in data['data']['events']:
        game_id = event['gameId']
        consensus_odds = next((odds for odds in event['odds']['pregame'] if odds['sportsbookId'] == 22), None)
        if consensus_odds:
            away_moneyline = consensus_odds.get('awayMoneyLine', None)
            home_moneyline = consensus_odds.get('homeMoneyLine', None)
        else:
            away_moneyline = home_moneyline = None
        away_team = next(team for team in event['teams'] if team['teamId'] == event['awayTeamId'])
        home_team = next(team for team in event['teams'] if team['teamId'] == event['homeTeamId'])
        game_data.append({
            'gameId': game_id,
            'awayTeamId': away_team['teamId'],
            'awayTeamKey': away_team['key'],
            'awayTeamName': away_team['displayName'],
            'homeTeamId': home_team['teamId'],
            'homeTeamKey': home_team['key'],
            'homeTeamName': home_team['displayName'],
            'awayMoneyLine': away_moneyline,
            'homeMoneyLine': home_moneyline
        })
    df = pd.DataFrame(game_data)
    return df

def fetch_ai_predictions(game_id):
    url = f"https://graph.sharp.app/operations/v1/ai-predictions/ByGameId?wg_api_hash=0bd8d897&wg_variables=%7B%22gameId%22%3A{game_id}%7D"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['data']['predictions']
        teams = data['teams']
        away_team = teams[0]
        home_team = teams[1]
        away_win_percentage = away_team['winPercentage']
        home_win_percentage = home_team['winPercentage']
        return away_win_percentage, home_win_percentage
    else:
        return None, None

def update_with_ai_predictions(df):
    df['awayWinPercentage'] = None
    df['homeWinPercentage'] = None
    for i, row in df.iterrows():
        game_id = row['gameId']
        away_win_percentage, home_win_percentage = fetch_ai_predictions(game_id)
        df.at[i, 'awayWinPercentage'] = away_win_percentage
        df.at[i, 'homeWinPercentage'] = home_win_percentage
    return df

def fetch_handles(game_id):
    url = f"https://graph.sharp.app/operations/v1/handles/ByGameId?wg_api_hash=0bd8d897&wg_variables=%7B%22gameId%22%3A{game_id}%7D"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['data']['handles']
        moneyline_data = data['moneyline']
        away_moneyline_info = next((item for item in moneyline_data if item['outcomeType'] == 'Away'), None)
        home_moneyline_info = next((item for item in moneyline_data if item['outcomeType'] == 'Home'), None)
        return {
            'away_money': away_moneyline_info.get('moneyPercentage') if away_moneyline_info else None,
            'home_money': home_moneyline_info.get('moneyPercentage') if home_moneyline_info else None
        }
    else:
        return {'away_money': None, 'home_money': None}

def update_with_handles(df):
    df['away_money'] = None
    df['home_money'] = None
    for i, row in df.iterrows():
        game_id = row['gameId']
        handles_data = fetch_handles(game_id)
        df.at[i, 'away_money'] = handles_data['away_money']
        df.at[i, 'home_money'] = handles_data['home_money']
    return df

def main():
    today = datetime.today().date()
    decimal_places = 3
    sharp_df = fetch_sharp_data(today)
    if sharp_df.empty:
        print("No data fetched. Exiting.")
        return
    sharp_df = update_with_ai_predictions(sharp_df)
    sharp_df = update_with_handles(sharp_df)
    for col in ['awayMoneyLine', 'homeMoneyLine']:
        sharp_df[col] = pd.to_numeric(sharp_df[col], errors='coerce')
        sharp_df[col] = sharp_df[col].apply(moneyline_to_proba)
    sharp_df['awayWinPercentage'] = pd.to_numeric(sharp_df['awayWinPercentage'], errors='coerce')
    sharp_df['homeWinPercentage'] = pd.to_numeric(sharp_df['homeWinPercentage'], errors='coerce')
    sharp_df['awayWinPercentage'] = round(sharp_df['awayWinPercentage'] / 100, decimal_places)
    sharp_df['homeWinPercentage'] = round(sharp_df['homeWinPercentage'] / 100, decimal_places)
    sharp_df['Away EV'] = sharp_df.apply(lambda x: calculate_ev(x['awayWinPercentage'], x['awayMoneyLine']), axis=1)
    sharp_df['Home EV'] = sharp_df.apply(lambda x: calculate_ev(x['homeWinPercentage'], x['homeMoneyLine']), axis=1)
    sharp_df.drop(columns=['homeTeamId', 'awayTeamId', 'gameId', 'homeTeamName', 'awayTeamName'], inplace=True)
    sharp_df.rename(columns={
        'awayTeamKey': 'Away_Team',
        'homeTeamKey': 'Home_Team',
        'Date': 'Game_Date',
        'awayWinPercentage': 'Away_Win%',
        'homeWinPercentage': 'Home_Win%'
    }, inplace=True)
    sharp_df['Away_Win%'] = sharp_df['Away_Win%'] * 100
    sharp_df['Home_Win%'] = sharp_df['Home_Win%'] * 100
    sharp_df['Away EV'] = sharp_df['Away EV'] * 100
    sharp_df['Home EV'] = sharp_df['Home EV'] * 100
    sharp_df['Away EV'] = round(sharp_df['Away EV'], 1)
    sharp_df['Home EV'] = round(sharp_df['Home EV'], 1)
    # Convert 'Game_Date' column to string to avoid serialization issues
    sharp_df['Game_Date'] = sharp_df['Game_Date'].astype(str)
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
    client = gspread.authorize(creds)
    spreadsheet_id = '1KgwFdqrRUs2fa5pSRmirj6ajyO2d14ONLsiksAYk8S8'
    spreadsheet = client.open_by_key(spreadsheet_id)
    sheet = spreadsheet.sheet1
    sheet.clear()
    sheet.append_row(sharp_df.columns.tolist())
    data = sharp_df.values.tolist()
    sheet.append_rows(data)
if __name__ == "__main__":
    main()
