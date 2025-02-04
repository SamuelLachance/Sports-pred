import datetime
from bs4 import BeautifulSoup
import utils
import requests
import csv

URLV2 = 'https://www.scoresandodds.com/ncaab?date=YEAR-MONTH-DAY'
DATA_FOLDER = utils.DATA_FOLDER
SR_NAMES_MAP = utils.read_two_column_csv_to_dict('Data/so_sr_mapping.csv')

def scrape_neutral_data():
    """Scrape neutral site information from TeamRankings"""
    schedule_url = 'https://www.teamrankings.com/ncb/schedules/season/'
    data = requests.get(schedule_url).content
    table_games_data = BeautifulSoup(data, 'html.parser').find_all("tr")
    all_rows = [i.text.split('\n') for i in table_games_data]

    neutral_map = {}
    this_date = utils.format_tr_dates(all_rows[0][1])
    for r in all_rows[1:]:
        val = r[1]
        if '@' in val:
            teams = val.split('  @  ')
            neutral_map[(teams[0], teams[1], this_date)] = 0
        elif 'vs.' in val:
            teams = val.split('  vs.  ')
            neutral_map[(teams[0], teams[1], this_date)] = 1
        else:
            this_date = utils.format_tr_dates(val)
    return neutral_map

def get_clean_team_name(team_element):
    """Extract and clean team name from HTML element"""
    team = team_element.find('span', {'class': 'team-name'})
    if team.find('a') is None:
        return team.find('span').text.strip(' 1234567890()')
    return team.find('a').text.strip(' 1234567890()')

def scrape_scores(date_obj):
    """Scrape game data for a specific date and format as dictionaries"""
    day, month, year = str(date_obj.day), str(date_obj.month), str(date_obj.year)
    url = URLV2.replace("DAY", day).replace("MONTH", month).replace("YEAR", year)
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f'ERROR: Response Code {response.status_code}')
        return []

    data = response.content
    soup = BeautifulSoup(data, 'html.parser')
    games = []

    for game in soup.find_all('tbody'):
        rows = game.find_all('tr')
        if len(rows) < 2:
            continue

        away_row, home_row = rows[0], rows[1]
        
        # Process away team
        away = SR_NAMES_MAP.get(
            get_clean_team_name(away_row), 
            get_clean_team_name(away_row)
        )
        away_score = away_row.find('td', {'class': 'event-card-score'})
        away_score = away_score.text.strip() if away_score else 'N/A'

        # Process home team
        home = SR_NAMES_MAP.get(
            get_clean_team_name(home_row), 
            get_clean_team_name(home_row)
        )
        home_score = home_row.find('td', {'class': 'event-card-score'})
        home_score = home_score.text.strip() if home_score else 'N/A'

        # Skip incomplete entries
        if 'N/A' in [away_score, home_score]:
            continue

        games.append({
            'Home Team': home,
            'Home Score': home_score,
            'Visitor Score': away_score,
            'Visiting Team': away
        })

    return games

def scrape_season(start_date, end_date):
    """Scrape an entire season's worth of data"""
    current_date = start_date
    all_games = []

    while current_date <= end_date:
        if current_date.month in [5, 6, 7, 8, 9, 10]:
            current_date += datetime.timedelta(days=1)
            continue
            
        print(f"Scraping {current_date.strftime('%Y-%m-%d')}")
        daily_games = scrape_scores(current_date)
        all_games.extend(daily_games)
        current_date += datetime.timedelta(days=1)

    return all_games

def save_to_csv(data, filename='game_data.csv'):
    """Save game data to CSV in the specified format"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Home Team', 'Home Score', 'Visitor Score', 'Visiting Team'])
        writer.writeheader()
        writer.writerows(data)

# Example usage:
if __name__ == "__main__":
    # Set your desired date range
    season_start = datetime.datetime(2024, 11, 1)
    season_end = datetime.datetime(2025, 3, 15)
    
    # Scrape the data
    game_data = scrape_season(season_start, season_end)
    
    # Save to CSV
    save_to_csv(game_data)
    print(f"Successfully saved {len(game_data)} games to ncaa_results.csv")
