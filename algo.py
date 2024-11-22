import requests
from datetime import datetime, timedelta
import pandas as pd
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

nhl_team_name_mapping = {
    'ANA': 'Anaheim Ducks',
    'BOS': 'Boston Bruins',
    'BUF': 'Buffalo Sabres',
    'CGY': 'Calgary Flames',
    'CAL': 'Calgary Flames',
    'CAR': 'Carolina Hurricanes',
    'CHI': 'Chicago Blackhawks',
    'COL': 'Colorado Avalanche',
    'CBJ': 'Columbus Blue Jackets',
    'CLB': 'Columbus Blue Jackets',
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
    'WIN': 'Winnipeg Jets',
}

nba_team_name_mapping = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'GS': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NO': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'SAN': 'San Antonio Spurs',
    'SA': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'UTAH': 'Utah Jazz',
    'WSH': 'Washington Wizards',
}

class Algo:
    def __init__(self, league):
        self.league = league.lower()
        self.algorithm = {
            'nba': [10, 10, 5, 5, 8, 8, 3],
            'nhl': [10, 3, 3, 3, 0.5, 0.5, 3],
            'mlb': [10, 10, 5, 5, 8, 8, 3],
        }

    def calculate_V2(self, date, returned1, returned2):
        record_points = returned1['seasonal_records'] - returned2['seasonal_records']
        home_away_points = returned1['home_away_record'] - returned2['home_away_record']
        home_away_10_games_points = returned1['home_away_10_game_records'] - returned2['home_away_10_game_records']
        last_10_games_points = returned1['last_10_games'] - returned2['last_10_games']
        avg_points = returned1['avg_points'] - returned2['avg_points']
        avg_points_10_games = returned1['avg_points_10_games'] - returned2['avg_points_10_games']
        win_streak_home_away = returned1['win_streak_home_away'] - returned2['win_streak_home_away']

        algo_vars = [
            record_points,
            home_away_points,
            home_away_10_games_points,
            last_10_games_points,
            avg_points,
            avg_points_10_games,
            win_streak_home_away if self.league == 'nhl' else 0,
        ]

        dividers = []
        max_points = []

        if self.league == 'nba':
            dividers = [9, 6, 3, 3, 3, 3, 1]
            max_points = [10, 10, 7, 7, 8, 10, 0]
        elif self.league == 'nhl':
            dividers = [3, 3, 3, 3, 0.3, 0.6, 6]
            max_points = [10, 10, 7, 7, 10, 9, 7]
        elif self.league == 'mlb':
            dividers = [6, 6, 3, 3, 0.3, 1.5, 0]
            max_points = [10, 10, 7, 7, 10, 8, 0]

        for i in range(len(algo_vars)):
            algo_vars[i] /= dividers[i] if dividers[i] != 0 else 1
            if algo_vars[i] > max_points[i]:
                algo_vars[i] = max_points[i]
            elif algo_vars[i] < -max_points[i]:
                algo_vars[i] = -max_points[i]

        odds = {}
        if self.league == 'nba':
            odds['records'] = self.calculate_odds(abs(algo_vars[0]), -0.065, 5.4, 52.5)
            odds['home_away'] = self.calculate_odds(abs(algo_vars[1]), -0.42, 9, 50)
            odds['home_away_10_games'] = self.calculate_odds(abs(algo_vars[2]), -0.34, 10, 41.3)
            odds['last_10_games'] = self.calculate_odds(abs(algo_vars[3]), -0.39, 4.3, 51)
            odds['avg_points'] = self.calculate_odds(abs(algo_vars[4]), -0.44, 10.3, 44.2)
            odds['avg_points_10_games'] = self.calculate_odds(abs(algo_vars[5]), -0.009, 4.9, 49)
        elif self.league == 'nhl':
            odds['records'] = self.calculate_odds(abs(algo_vars[0]), 0.081, 0.41, 51.6)
            odds['home_away'] = self.calculate_odds(abs(algo_vars[1]), -0.16, 3.1, 49.6)
            odds['home_away_10_games'] = self.calculate_odds(abs(algo_vars[2]), -0.274, 4.4, 48)
            odds['last_10_games'] = self.calculate_odds(abs(algo_vars[3]), 0.64, -1.4, 53.93)
            odds['avg_points'] = self.calculate_odds(abs(algo_vars[4]), -0.21, 3.28, 49.9)
            odds['avg_points_10_games'] = self.calculate_odds(abs(algo_vars[5]), 0.69, -2.15, 54.1)
            odds['win_streak_home_away'] = self.calculate_odds(abs(algo_vars[6]), -0.63, 7.76, 65, cubic=True)
            if odds['win_streak_home_away'] < 60:
                odds['win_streak_home_away'] = 50
        elif self.league == 'mlb':
            odds['records'] = self.calculate_odds(abs(algo_vars[0]), -0.0378, 1.5474, 50.776)
            odds['home_away'] = self.calculate_odds(abs(algo_vars[1]), -0.2226, 3.8472, 47.282)
            odds['home_away_10_games'] = self.calculate_odds(abs(algo_vars[2]), 0.3025, 1.4568, 49.518)
            odds['last_10_games'] = self.calculate_odds(abs(algo_vars[3]), 0.3039, 0.1154, 51.6)
            odds['avg_points'] = self.calculate_odds(abs(algo_vars[4]), -0.1938, 3.1638, 49.105)
            odds['avg_points_10_games'] = self.calculate_odds(abs(algo_vars[5]), 0.0301, 0.5611, 51.278, cubic=True)

        for key in odds:
            if odds[key] < 50:
                odds[key] = 50

        for key, var in zip(odds.keys(), algo_vars):
            if var < 0:
                odds[key] *= -1

        total = sum(odds.values()) / len(odds)
        if total > 0:
            total += 50
        else:
            total -= 50

        return {'total': round(total, 2), **odds}

    @staticmethod
    def calculate_odds(x, a, b, c, cubic=False):
        if cubic:
            return round(-0.63 * (x ** 3) + 7.76 * (x ** 2) - 18.32 * x + 65, 2)
        else:
            return round(a * (x ** 2) + b * x + c, 2)

def espn_standings(sport="NHL"):
    if sport == "NHL":
        base_URL = "https://site.api.espn.com/apis/v2/sports/hockey/nhl/standings"
        group_ids = [7, 8]
    elif sport == "NBA":
        base_URL = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
        group_ids = [5, 6]
    else:
        print(f"Sport '{sport}' not supported.")
        return None

    stats = []
    for group_id in group_ids:
        sort_param = "points" if sport == "NHL" else "winPercent"
        standings_URL = f"{base_URL}?group={group_id}&sort={sort_param}"
        response = requests.get(standings_URL)
        if response.status_code != 200:
            print(f"Failed to fetch data from {standings_URL}, status code: {response.status_code}")
            return None
        data = response.json()
        standings_data = data.get("children", [])
        for division in standings_data:
            division_data = division.get("standings", {}).get("entries", [])
            for team_entry in division_data:
                team = team_entry["team"]
                stats_data = team_entry.get("stats", [])
                stats_lookup = {stat["name"]: stat for stat in stats_data}
                try:
                    if sport == "NHL":
                        name = team["abbreviation"]
                        overall = stats_lookup.get("overall", {})
                        record_summary = overall.get("summary", "0-0-0")
                        record_parts = record_summary.split("-")
                        wins, losses, otl = (
                            map(int, record_parts) if len(record_parts) == 3 else (0, 0, 0)
                        )
                        stats.append({
                            "name": name,
                            "wins": wins,
                            "losses": losses,
                            "ot_losses": otl,
                            "points": stats_lookup.get("points", {}).get("value", 0),
                            "games_played": stats_lookup.get("gamesPlayed", {}).get("value", 0),
                            "streak": stats_lookup.get("streak", {}).get("displayValue", "None"),
                            "home_record": stats_lookup.get("Home", {}).get("summary", "0-0-0"),
                            "away_record": stats_lookup.get("Road", {}).get("summary", "0-0-0"),
                            "last_ten_record": stats_lookup.get("Last Ten Games", {}).get("summary", "0-0-0"),
                        })
                    elif sport == "NBA":
                        name = team["abbreviation"]
                        wins = stats_lookup.get("wins", {}).get("value", 0)
                        losses = stats_lookup.get("losses", {}).get("value", 0)
                        win_percent = stats_lookup.get("winPercent", {}).get("value", 0)
                        home_record = stats_lookup.get("Home", {}).get("summary", "0-0")
                        away_record = stats_lookup.get("Road", {}).get("summary", "0-0")
                        last_ten = stats_lookup.get("Last Ten Games", {}).get("summary", "0-0")
                        streak = stats_lookup.get("streak", {}).get("displayValue", "None")
                        stats.append({
                            "name": name,
                            "wins": int(wins),
                            "losses": int(losses),
                            "win_percent": float(win_percent),
                            "home_record": home_record,
                            "away_record": away_record,
                            "last_ten_record": last_ten,
                            "streak": streak,
                        })
                except Exception as e:
                    print(f"Error processing team {team['abbreviation']}: {e}")
    return pd.DataFrame(stats)

def extract_team_data(json_data, predict):
    extracted_data = []
    for game in json_data.get('scores', []):
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
    return pd.DataFrame(extracted_data)

def fetch_odds_data(date, predict, sports):
    base_url = f"https://www.oddsshark.com/api/scores/{sports}/{date}?_format=json"
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Referer': f'https://www.oddsshark.com/{sports.lower()}/scores',
        'Sec-Ch-Ua': '"Chromium";v="118", "Microsoft Edge";v="118", "Not=A?Brand";v="99"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    }
    response = requests.get(base_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        df = extract_team_data(data, predict)
        df['Date'] = date
        return df
    else:
        print(f"Failed to fetch data for date: {date}")
        return None

def merge_espn_odds(espn_df, odds_df, team_name_mapping, sport):
    espn_df['full_name'] = espn_df['name'].map(team_name_mapping)
    merged_rows = []
    for _, odds_row in odds_df.iterrows():
        home_team = odds_row['Home Name']
        away_team = odds_row['Away Name']
        home_stats = espn_df[espn_df['full_name'] == home_team].to_dict('records')
        away_stats = espn_df[espn_df['full_name'] == away_team].to_dict('records')
        merged_row = {
            'Date': odds_row['Date'],
            'Home Team': home_team,
            'Away Team': away_team,
            'Home MoneyLine': odds_row['Home MoneyLine'],
            'Away MoneyLine': odds_row['Away MoneyLine'],
            'Home Spread Price': odds_row['Home Spread Price'],
            'Away Spread Price': odds_row['Away Spread Price'],
            'Home Score': odds_row.get('Home Score', None),
            'Away Score': odds_row.get('Away Score', None),
            'Over Price': odds_row['Over Price'],
            'Under Price': odds_row['Under Price'],
            'Total': odds_row['Total'],
            'Arena': odds_row['Arena']
        }
        if home_stats:
            home_stat = home_stats[0]
            if sport == "NHL":
                home_record = f"{home_stat.get('wins',0)}-{home_stat.get('losses',0)}-{home_stat.get('ot_losses',0)}"
            elif sport == "NBA":
                home_record = f"{home_stat.get('wins',0)}-{home_stat.get('losses',0)}"
            else:
                home_record = "0-0"
            merged_row.update({
                'Home Team Points': home_stat.get('points', 0),
                'Home Team Streak': home_stat.get('streak', "None"),
                'Home Team Record': home_record,
                'Home Team Games Played': home_stat.get('games_played', 0),
                'Home Team Home Record': home_stat.get('home_record', "0-0"),
                'Home Team Away Record': home_stat.get('away_record', "0-0"),
                'Home Team Last 10': home_stat.get('last_ten_record', "0-0")
            })
        else:
            merged_row.update({
                'Home Team Points': 0,
                'Home Team Streak': "None",
                'Home Team Record': "0-0" if sport == "NBA" else "0-0-0",
                'Home Team Games Played': 0,
                'Home Team Home Record': "0-0",
                'Home Team Away Record': "0-0",
                'Home Team Last 10': "0-0"
            })
        if away_stats:
            away_stat = away_stats[0]
            if sport == "NHL":
                away_record = f"{away_stat.get('wins',0)}-{away_stat.get('losses',0)}-{away_stat.get('ot_losses',0)}"
            elif sport == "NBA":
                away_record = f"{away_stat.get('wins',0)}-{away_stat.get('losses',0)}"
            else:
                away_record = "0-0"
            merged_row.update({
                'Away Team Points': away_stat.get('points', 0),
                'Away Team Streak': away_stat.get('streak', "None"),
                'Away Team Record': away_record,
                'Away Team Games Played': away_stat.get('games_played', 0),
                'Away Team Home Record': away_stat.get('home_record', "0-0"),
                'Away Team Away Record': away_stat.get('away_record', "0-0"),
                'Away Team Last 10': away_stat.get('last_ten_record', "0-0")
            })
        else:
            merged_row.update({
                'Away Team Points': 0,
                'Away Team Streak': "None",
                'Away Team Record': "0-0" if sport == "NBA" else "0-0-0",
                'Away Team Games Played': 0,
                'Away Team Home Record': "0-0",
                'Away Team Away Record': "0-0",
                'Away Team Last 10': "0-0"
            })
        merged_rows.append(merged_row)
    merged_df = pd.DataFrame(merged_rows)
    for col in ['Home Team Record', 'Away Team Record']:
        merged_df[col] = merged_df[col].apply(lambda x: x.split('/')[0] if isinstance(x, str) else x)
    return merged_df

def fetch_recent_games(days=30, sports="NHL"):
    recent_games = []
    today = datetime.today()
    sport_url_map = {
        "NHL": "hockey/nhl",
        "NBA": "basketball/nba",
    }
    if sports not in sport_url_map:
        raise ValueError(f"Unsupported sport: {sports}")
    sport_url_part = sport_url_map[sports]
    for day_offset in range(days):
        date = today - timedelta(days=day_offset)
        formatted_date = date.strftime('%Y%m%d')
        url = f"http://site.api.espn.com/apis/site/v2/sports/{sport_url_part}/scoreboard?dates={formatted_date}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            events = data.get('events', [])
            for event in events:
                game = {}
                competition = event.get('competitions', [])[0]
                game['Date'] = event.get('date', '').split('T')[0]
                competitors = competition.get('competitors', [])
                if len(competitors) == 2:
                    home_team = next((team for team in competitors if team['homeAway'] == 'home'), None)
                    away_team = next((team for team in competitors if team['homeAway'] == 'away'), None)
                    if home_team and away_team:
                        game['Home Team'] = home_team['team']['displayName']
                        game['Away Team'] = away_team['team']['displayName']
                        game['Home Score'] = int(home_team.get('score', '0'))
                        game['Away Score'] = int(away_team.get('score', '0'))
                        recent_games.append(game)
        else:
            print(f"Failed to fetch games for date: {formatted_date}")
    games_df = pd.DataFrame(recent_games)
    games_df['Date'] = pd.to_datetime(games_df['Date'])
    games_df = games_df[games_df['Date'] <= pd.Timestamp.today().normalize()]
    return games_df

def calculate_average_points(team_name, historical_games):
    team_games = historical_games[
        (historical_games['Home Team'] == team_name) | (historical_games['Away Team'] == team_name)
    ]
    if team_games.empty:
        return {'avg_points': 0, 'avg_points_10_games': 0}
    points_scored = team_games.apply(
        lambda game: game['Home Score'] if game['Home Team'] == team_name else game['Away Score'],
        axis=1
    )
    avg_points = points_scored.mean()
    last_10_games = team_games.sort_values(by='Date', ascending=False).head(10)
    last_10_points_scored = last_10_games.apply(
        lambda game: game['Home Score'] if game['Home Team'] == team_name else game['Away Score'],
        axis=1
    )
    avg_points_10_games = last_10_points_scored.mean()
    return {'avg_points': avg_points, 'avg_points_10_games': avg_points_10_games}

def calculate_win_loss_streaks(team_name, historical_games):
    team_games = historical_games[
        (historical_games['Home Team'] == team_name) | (historical_games['Away Team'] == team_name)
    ].sort_values(by='Date', ascending=False)
    if team_games.empty:
        return {'win_streak_home_away': 0}
    win_streak_home_away = 0
    last_game = team_games.iloc[0]
    location = 'Home' if last_game['Home Team'] == team_name else 'Away'
    for _, game in team_games.iterrows():
        current_location = 'Home' if game['Home Team'] == team_name else 'Away'
        if current_location != location:
            break
        if current_location == 'Home':
            won = game['Home Score'] > game['Away Score']
        else:
            won = game['Away Score'] > game['Home Score']
        if won:
            win_streak_home_away += 1
        else:
            break
    return {'win_streak_home_away': win_streak_home_away}

def prepare_team_data(row, team_type='Home', historical_games=None, sport='NBA'):
    team_data = {}
    team_name = row.get(f'{team_type} Team')
    if not team_name:
        team_data = {
            'seasonal_records': 0,
            'record_points': 0,
            'home_away_record': 0,
            'home_away_10_game_records': 0,
            'last_10_games': 0,
            'avg_points': 0,
            'avg_points_10_games': 0,
            'win_streak_home_away': 0
        }
        return team_data
    record = row.get(f'{team_type} Team Record')
    if record:
        try:
            record_parts = record.split('-')
            wins = int(record_parts[0]) if len(record_parts) >= 1 else 0
            losses = int(record_parts[1]) if len(record_parts) >= 2 else 0
            if sport == "NHL" and len(record_parts) >= 3:
                otl = int(record_parts[2])
                losses += otl
            record_points = wins - losses
            team_data['record_points'] = record_points
            team_data['seasonal_records'] = record_points
        except ValueError as e:
            team_data['record_points'] = 0
            team_data['seasonal_records'] = 0
    else:
        team_data['record_points'] = 0
        team_data['seasonal_records'] = 0
    home_record = row.get(f'{team_type} Team Home Record')
    away_record = row.get(f'{team_type} Team Away Record')
    if home_record and away_record and isinstance(home_record, str) and isinstance(away_record, str):
        try:
            home_parts = home_record.split('-')
            away_parts = away_record.split('-')
            home_wins = int(home_parts[0]) if len(home_parts) >= 1 else 0
            home_losses = int(home_parts[1]) if len(home_parts) >= 2 else 0
            away_wins = int(away_parts[0]) if len(away_parts) >= 1 else 0
            away_losses = int(away_parts[1]) if len(away_parts) >= 2 else 0
            if sport == "NHL":
                if len(home_parts) >= 3:
                    home_otl = int(home_parts[2])
                    home_losses += home_otl
                if len(away_parts) >= 3:
                    away_otl = int(away_parts[2])
                    away_losses += away_otl
            home_away = (away_wins - away_losses) - (home_wins - home_losses)
            team_data['home_away_record'] = home_away
        except ValueError as e:
            team_data['home_away_record'] = 0
    else:
        team_data['home_away_record'] = 0
    if historical_games is not None:
        try:
            if team_type == 'Home':
                last_10_home_games = historical_games[
                    (historical_games['Home Team'] == team_name)
                ].sort_values(by='Date', ascending=False).head(10)
                home_10_wins = sum(last_10_home_games['Home Score'] > last_10_home_games['Away Score'])
                home_10_losses = sum(last_10_home_games['Home Score'] <= last_10_home_games['Away Score'])
                home_away_10_games = home_10_wins - home_10_losses
            else:
                last_10_away_games = historical_games[
                    (historical_games['Away Team'] == team_name)
                ].sort_values(by='Date', ascending=False).head(10)
                away_10_wins = sum(last_10_away_games['Away Score'] > last_10_away_games['Home Score'])
                away_10_losses = sum(last_10_away_games['Away Score'] <= last_10_away_games['Home Score'])
                home_away_10_games = away_10_wins - away_10_losses
            team_data['home_away_10_game_records'] = home_away_10_games
        except Exception as e:
            team_data['home_away_10_game_records'] = 0
    else:
        team_data['home_away_10_game_records'] = 0
    if historical_games is not None:
        try:
            last_10_games = historical_games[
                (historical_games['Home Team'] == team_name) | (historical_games['Away Team'] == team_name)
            ].sort_values(by='Date', ascending=False).head(10)
            last_10_wins = 0
            last_10_losses = 0
            for _, game in last_10_games.iterrows():
                if game['Home Team'] == team_name:
                    won = game['Home Score'] > game['Away Score']
                else:
                    won = game['Away Score'] > game['Home Score']
                if won:
                    last_10_wins += 1
                else:
                    last_10_losses += 1
            last_10_games_points = last_10_wins - last_10_losses
            team_data['last_10_games'] = last_10_games_points
        except Exception as e:
            team_data['last_10_games'] = 0
    else:
        team_data['last_10_games'] = 0
    if historical_games is not None:
        try:
            avg_points_data = calculate_average_points(team_name, historical_games)
            team_data['avg_points'] = avg_points_data.get('avg_points', 0)
            team_data['avg_points_10_games'] = avg_points_data.get('avg_points_10_games', 0)
        except Exception as e:
            team_data['avg_points'] = 0
            team_data['avg_points_10_games'] = 0
    else:
        team_data['avg_points'] = 0
        team_data['avg_points_10_games'] = 0
    if historical_games is not None:
        try:
            streak_data = calculate_win_loss_streaks(team_name, historical_games)
            team_data['win_streak_home_away'] = streak_data.get('win_streak_home_away', 0)
        except Exception as e:
            team_data['win_streak_home_away'] = 0
    else:
        team_data['win_streak_home_away'] = 0
    if 'home_away_record' not in team_data:
        team_data['home_away_record'] = 0
    return team_data

def main():
    date = datetime.today().strftime('%Y-%m-%d')
    today_str = datetime.today().strftime("%Y-%m-%d")
    sports = "NBA"
    days = 30
    if sports == "NBA":
        team_name_mapping = nba_team_name_mapping
    elif sports == "NHL":
        team_name_mapping = nhl_team_name_mapping
    else:
        print(f"Unsupported sport: {sports}")
        return
    espn_df = espn_standings(sports)
    if espn_df is None or espn_df.empty:
        print("No ESPN standings data fetched.")
        return
    odds_df = fetch_odds_data(date, True, sports)
    if odds_df is None or odds_df.empty:
        print("No odds data fetched.")
        return
    merged_df = merge_espn_odds(espn_df, odds_df, team_name_mapping, sport=sports)
    if merged_df.empty:
        print("Merged DataFrame is empty.")
        return
    historical_games = fetch_recent_games(days, sports)
    if historical_games.empty:
        print("No historical games data fetched.")
    algo = Algo(sports)
    predictions = []
    for _, row in merged_df.iterrows():
        home_team_data = prepare_team_data(row, team_type='Home', historical_games=historical_games, sport=sports)
        away_team_data = prepare_team_data(row, team_type='Away', historical_games=historical_games, sport=sports)
        prediction = algo.calculate_V2(date, home_team_data, away_team_data)
        print(prediction)
        predictions.append({
            # Use 'today_str' if you want the current date, or keep row['Date'] if you want the game's date
            'Date': today_str,
            'Home Team': row['Home Team'],
            'Away Team': row['Away Team'],
            'Prediction': prediction.get('total', None),
        })

        predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(f"predictions_{date}.csv", index=False)

    # Update the 'Date' column if necessary
    predictions_df['Date'] = today_str

    # Optional: Ensure 'Date' is the first column
    cols = ['Date'] + [col for col in predictions_df.columns if col != 'Date']
    predictions_df = predictions_df[cols]
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
    client = gspread.authorize(creds)
    if sports == "NHL":
        spreadsheet_id = '1KgwFdqrRUs2fa5pSRmirj6ajyO2d14ONLsiksAYk8S8'
    elif sports == "NBA":
        spreadsheet_id = '1J_VjAUTF-0aSfd3ObPaIr1AJWYmdLhYRG-o8C-II4LA'
    else:
        print(f"Unsupported sport for Google Sheets upload: {sports}")
        return
    try:
        spreadsheet = client.open_by_key(spreadsheet_id)
    except Exception as e:
        print(f"Error opening spreadsheet: {e}")
        return
    sheet_name = 'Oracle'
    try:
        sheet = spreadsheet.worksheet(sheet_name)
    except Exception as e:
        print(f"Error accessing sheet '{sheet_name}': {e}")
        return
    sheet.clear()
    sheet.append_row(predictions_df.columns.tolist())
    data = predictions_df.values.tolist()
    sheet.append_rows(data, value_input_option='RAW')
    print(f"Data successfully uploaded to Google Sheets for {sports}.")

if __name__ == '__main__':
    main()
