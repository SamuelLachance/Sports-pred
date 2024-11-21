
# uses an algorithm to calculate sports betting odds


import urllib.request           #script for URL request handling
import urllib.parse             #script for URL handling
from urllib import request
import html.parser              #script for HTML handling
import os.path                  #script for directory/file handling
import csv                      #script for CSV file handling
import sys
from universal_functions import Universal_Functions
import requests
from datetime import datetime, timedelta
from fuzzywuzzy import process
import pandas as pd
from bs4 import BeautifulSoup
import csv
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Create a dictionary to map team abbreviations to full names
nhl_team_name_mapping = {
    # NHL Teams
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

	#can be nba, nhl, nfl, mlb
	league="nhl"

	universal=None


	algorithm={}
	



	def __init__(self, league):
		self.league=league

		self.algorithm['nba']=[10, 10, 5, 5,  8,  8,   3, 3]
		self.algorithm['nhl']=[10, 3,  3, 3, 0.5, 0.5, 3, 3]
		self.algorithm['mlb']=[10, 10, 5, 5,  8,  8,   3, 3]

		self.universal=Universal_Functions(league)


	#returns algo results
	def calculate(self, date, returned1, returned2):
		record_points             =self.calculate_points("seasonal_records",          returned1['seasonal_records'],  returned2['seasonal_records'])
		home_away_points          =self.calculate_points("home_away_records",         returned1['home_away_record'],  returned2['home_away_record'])
		home_away_10_games_points =self.calculate_points("home_away_10_game_records", returned1['home_away_10_game_records'],  returned2['home_away_10_game_records'])
		last_10_games_points      =self.calculate_points("last_10_games",             returned1['last_10_games'], returned2['last_10_games'])
		avg_points                =self.calculate_points("avg_points",                returned1['avg_points'],   returned2['avg_points'])
		avg_points_10_games       =self.calculate_points("avg_points_10_games",       returned1['avg_points_10_games'],   returned2['avg_points_10_games'])
		win_streak_home_away      =self.calculate_points("win_streak_home_away",      returned1['win_streak_home_away'], returned2['win_streak_home_away'])


		# print(self.algorithm)

		#doesn't count variables if they're negative (for testing algorithms)
		if self.algorithm[self.league][0]<0:
			record_points=             0
		if self.algorithm[self.league][1]<0:
			home_away_points=          0
		if self.algorithm[self.league][2]<0:
			home_away_10_games_points= 0
		if self.algorithm[self.league][3]<0:
			last_10_games_points=      0
		if self.algorithm[self.league][4]<0:
			avg_points=                0
		if self.algorithm[self.league][5]<0:
			avg_points_10_games=       0
		if self.algorithm[self.league][6]<0:
			win_streak_home_away=      0



		record_points             /=    self.algorithm[self.league][0]
		home_away_points          /=    self.algorithm[self.league][1]
		home_away_10_games_points /=    self.algorithm[self.league][2]
		last_10_games_points      /=    self.algorithm[self.league][3]
		avg_points                /=    self.algorithm[self.league][4]
		avg_points_10_games       /=    self.algorithm[self.league][5]
		win_streak_home_away      /=    self.algorithm[self.league][6]



		


		# record_points             /=    10
		# home_away_points          /=    5
		# home_away_10_games_points /=    4
		# last_10_games_points      /=    5
		# avg_points                /=    10
		# avg_points_10_games       /=    10
		# win_streak                /=    3
		# win_streak_home_away      /=    3



		total=record_points + home_away_points + home_away_10_games_points + last_10_games_points + avg_points + avg_points_10_games + win_streak_home_away

		# #always has home team win
		# total=-1


		avg_points=self.universal.convert_number(avg_points)
		avg_points_10_games=self.universal.convert_number(avg_points_10_games)
		total=self.universal.convert_number(total)

		to_return={}
		to_return['record_points']=             record_points
		to_return['home_away_points']=          home_away_points
		to_return['home_away_10_games_points']= home_away_10_games_points
		to_return['last_10_games_points']=      last_10_games_points
		to_return['avg_points']=                avg_points
		to_return['avg_points_10_games']=       avg_points_10_games
		to_return['win_streak_home_away']=      win_streak_home_away
		to_return['total']=                     total
		return to_return





	#calculates based off winning odds for every variable in algo
	#ALGO_V2
	def calculate_V2(self, date, returned1, returned2):
		record_points             =self.calculate_points("seasonal_records",          returned1['seasonal_records'],  returned2['seasonal_records'])
		home_away_points          =self.calculate_points("home_away_records",         returned1['home_away_record'],  returned2['home_away_record'])
		home_away_10_games_points =self.calculate_points("home_away_10_game_records", returned1['home_away_10_game_records'],  returned2['home_away_10_game_records'])
		last_10_games_points      =self.calculate_points("last_10_games",             returned1['last_10_games'], returned2['last_10_games'])
		avg_points                =self.calculate_points("avg_points",                returned1['avg_points'],   returned2['avg_points'])
		avg_points_10_games       =self.calculate_points("avg_points_10_games",       returned1['avg_points_10_games'],   returned2['avg_points_10_games'])
		win_streak_home_away      =self.calculate_points("win_streak_home_away",      returned1['win_streak_home_away'], returned2['win_streak_home_away'])


		algo_vars=[]
		algo_vars.append(record_points)
		algo_vars.append(home_away_points)
		algo_vars.append(home_away_10_games_points)
		algo_vars.append(last_10_games_points)
		algo_vars.append(avg_points)
		algo_vars.append(avg_points_10_games)
		# algo_vars.append(win_streak)
		if self.league=="nhl":
			algo_vars.append(win_streak_home_away)


		print("Record points: "+str(record_points))
		print("Home away points: "+str(home_away_points))
		print("Home away 10 game points: "+str(home_away_10_games_points))
		print("Last 10 games: "+str(last_10_games_points))
		print("Avg points: "+str(avg_points))
		print("Avg points 10 games: "+str(avg_points_10_games))

		#new algo divides by 3 to fit it into a levels chart that has increments of 3. Let's skip the middle man and just divide by 9.


		if self.league=="nba":
			dividers=[]
			dividers.append(9)
			dividers.append(6)
			dividers.append(3)
			dividers.append(3)
			dividers.append(3)
			dividers.append(3)
			# dividers.append(1)
			# dividers.append(1)

			max_points=[]
			max_points.append(10)
			max_points.append(10)
			max_points.append(7)
			max_points.append(7)
			max_points.append(8)
			max_points.append(10)
			# max_points.append(10)
			# max_points.append(10)



			#puts total points at a max of 10
			for x in range(0, len(algo_vars)):
				algo_vars[x] /= dividers[x]

				if algo_vars[x]>max_points[x]:
					algo_vars[x]=max_points[x]
				elif algo_vars[x]<max_points[x]*-1:
					algo_vars[x]=-max_points[x]


			odds={}

			#calculates odds from records stat
			x=abs(algo_vars[0])
			y=-0.065*(x**2) + 5.4*x + 52.5
			y=self.universal.convert_number(y)
			odds['records']=y
			print("Records: "+str(y))

			#calculates odds from home_away stat
			x=abs(algo_vars[1])
			y=-0.42*(x**2) + 9*x + 50
			y=self.universal.convert_number(y)
			odds['home_away']=y
			print("Home away: "+str(y))

			#calculates odds from home_away_10_games stat
			x=abs(algo_vars[2])
			y=-0.34*(x**2) + 10*x + 41.3
			y=self.universal.convert_number(y)
			odds['home_away_10_games']=y
			print("Home away 10 games: "+str(y))

			#calculates odds from last_10_games stat
			x=abs(algo_vars[3])
			y=-0.39*(x**2) + 4.3*x + 51
			y=self.universal.convert_number(y)
			odds['last_10_games']=y
			print("Last 10 games: "+str(y))

			#calculates odds from avg_points stat
			x=abs(algo_vars[4])
			y=-0.44*(x**2) + 10.3*x + 44.2
			y=self.universal.convert_number(y)
			odds['avg_points']=y
			print("Avg points: "+str(y))

			#calculates odds from avg_points_10_games stat
			x=abs(algo_vars[5])
			y=-0.009*(x**2) + 4.9*x + 49
			y=self.universal.convert_number(y)
			odds['avg_points_10_games']=y
			print("Avg points 10 games: "+str(y))

		elif self.league=="nhl":
			dividers=[]
			dividers.append(3)
			dividers.append(3)
			dividers.append(3) # dividers.append(6)
			dividers.append(3)
			dividers.append(0.3) #is 3/10 since algo_v1 would *10 then /3, so *10/3 or /(3/10)
			dividers.append(0.6) #is 3/5 since algo_v1 would *5 then /3, so *5/3 or /(3/5)
			# dividers.append(1)
			dividers.append(6)

			max_points=[]
			max_points.append(10)
			max_points.append(10)
			max_points.append(7) # max_points.append(4)
			max_points.append(7)
			max_points.append(10)
			max_points.append(9)
			# max_points.append(0)
			max_points.append(7)

			#puts total points at a max of 10
			for x in range(0, len(algo_vars)):
				algo_vars[x] /= dividers[x]

				if algo_vars[x]>max_points[x]:
					algo_vars[x]=max_points[x]
				elif algo_vars[x]<max_points[x]*-1:
					algo_vars[x]=-max_points[x]


			odds={}

			#calculates odds from records stat
			x=abs(algo_vars[0])
			y=0.081*(x**2) + 0.41*x + 51.6
			y=self.universal.convert_number(y)
			odds['records']=y
			print("Records: "+str(y))

			#calculates odds from home_away stat
			x=abs(algo_vars[1])
			y= -0.16*(x**2) + 3.1*x + 49.6
			y=self.universal.convert_number(y)
			odds['home_away']=y
			print("Home away: "+str(y))

			#calculates odds from home_away_10_games stat
			x=abs(algo_vars[2])
			# y=4.26*x + 49.8
			y= -0.274*(x**2) + 4.4*x + 48
			y=self.universal.convert_number(y)
			odds['home_away_10_games']=y
			print("Home away 10 games: "+str(y))

			#calculates odds from last_10_games stat
			x=abs(algo_vars[3])
			y=0.64*(x**2) - 1.4*x + 53.93
			y=self.universal.convert_number(y)
			odds['last_10_games']=y
			print("Last 10 games: "+str(y))

			#calculates odds from avg_points stat
			x=abs(algo_vars[4])
			y=-0.21*(x**2) + 3.28*x + 49.9
			y=self.universal.convert_number(y)
			odds['avg_points']=y
			print("Avg points: "+str(y))

			#calculates odds from avg_points_10_games stat
			x=abs(algo_vars[5])
			y=0.69*(x**2) - 2.15*x + 54.1
			y=self.universal.convert_number(y)
			odds['avg_points_10_games']=y
			print("Avg points 10 games: "+str(y))

			#calculates odds from win_streak_home_away stat
			x=abs(algo_vars[6])
			y=-0.63*(x**3) + 7.76*(x**2) - 18.32*x + 65
			y=self.universal.convert_number(y)
			odds['win_streak_home_away']=y
			print("Win Streak Home Away: "+str(y))

			if odds['win_streak_home_away']<60:
				odds['win_streak_home_away']=50



		elif self.league=="mlb":
			dividers=[]
			dividers.append(6)
			dividers.append(6)
			dividers.append(3)
			dividers.append(3)
			dividers.append(0.3) #is 3/10 since algo_v1 would /0.1 or *10 then /3, so *10/3 or /(3/10) | or 0.1*3...
			dividers.append(1.5) #OR dividers.append(3)
			# dividers.append(1) DOESN'T MEAN SHIT
			# dividers.append(6) DOESN'T MEAN SHIT

			max_points=[]
			max_points.append(10)
			max_points.append(10)
			max_points.append(7)
			max_points.append(7)
			max_points.append(10)
			max_points.append(8) #OR max_points.append(4)
			# max_points.append(0)
			# max_points.append(7)

			#puts total points at a max of 10
			for x in range(0, len(algo_vars)):
				algo_vars[x] /= dividers[x]

				if algo_vars[x]>max_points[x]:
					algo_vars[x]=max_points[x]
				elif algo_vars[x]<max_points[x]*-1:
					algo_vars[x]=-max_points[x]


			odds={}

			#calculates odds from records stat
			x=abs(algo_vars[0])
			y=-0.0378*(x**2) + 1.5474*x + 50.776
			y=self.universal.convert_number(y)
			odds['records']=y
			print("Records: "+str(y))


			


			#calculates odds from home_away stat
			x=abs(algo_vars[1])
			y=-0.2226*(x**2) + 3.8472*x + 47.282
			y=self.universal.convert_number(y)
			odds['home_away']=y
			print("Home away: "+str(y))

			#calculates odds from home_away_10_games stat
			x=abs(algo_vars[2])
			y=0.3025*(x**2) + 1.4568*x + 49.518
			y=self.universal.convert_number(y)
			odds['home_away_10_games']=y
			print("Home away 10 games: "+str(y))

			#calculates odds from last_10_games stat
			x=abs(algo_vars[3])
			y=0.3039*(x**2) + 0.1154*x + 51.6
			y=self.universal.convert_number(y)
			odds['last_10_games']=y
			print("Last 10 games: "+str(y))

			#calculates odds from avg_points stat
			x=abs(algo_vars[4])
			y=-0.1938*(x**2) + 3.1638*x + 49.105
			y=self.universal.convert_number(y)
			
			odds['avg_points']=y
			print("Avg points: "+str(y))

			#calculates odds from avg_points_10_games stat
			x=abs(algo_vars[5])
			y=0.0301*(x**3) + 0.5611*(x**2) - 0.6103*x + 51.278
			y=self.universal.convert_number(y)
			odds['avg_points_10_games']=y
			print("Avg points 10 games: "+str(y))








		print("Odds home_away before: "+str(odds["home_away"]))

		#corrects percentages <50
		for key in odds.keys():
			#if -49% or something
			# if odds[key]<0 and odds[key]>-50:
			# 	odds[key]=(odds[key] + 100)
			# elif odds[key]>0 and odds[key]<50:
				# odds[key]=(odds[key] - 100)
			if(odds[key]<50):
				odds[key] = 50

		print("Odds home_away after: "+str(odds["home_away"]))

		#subtracts 50 since 50 is origin
		for key in odds.keys():
			odds[key]-=50

		print("Odds home_away after2: "+str(odds["home_away"]))

		#reverses odds so that all values that get plugged in stay above 50%
		# if a favorable team is unfavorable, the parabola algo might be a problem.
		if algo_vars[0]<0:
			odds['records']*=-1

		if algo_vars[1]<0:
			odds['home_away']*=-1

		if algo_vars[2]<0:
			odds['home_away_10_games']*=-1

		if algo_vars[3]<0:
			odds['last_10_games']*=-1

		if algo_vars[4]<0:
			odds['avg_points']*=-1

		if algo_vars[5]<0:
			odds['avg_points_10_games']*=-1

		if self.league=="nhl" and algo_vars[6]<0:
			odds['win_streak_home_away']*=-1

		print("Odds home_away after3: "+str(odds["home_away"]))

		#can also have average equal highest odds. Or average equals average between two highest odds. 



		## Averages two highest ##
		# #gets 2 highest odds even if one is opposite sign
		# highest=0
		# highest2=0
		# for key in odds.keys():
		# 	if abs(odds[key])>abs(highest):
		# 		if abs(highest)>abs(highest2):
		# 			highest2=highest
		# 		highest=odds[key]

		# 	elif abs(odds[key])>abs(highest2):
		# 		highest2=odds[key]
		# average=(highest+highest2)/2



		## adds all favorites, adds all underdogs, then averages the two totals ##
		# favorite_total=0
		# underdog_total=0
		# for key in odds:
		# 	if odds[key]>0:
		# 		favorite_total+=odds[key]
		# 	else:
		# 		underdog_total+=odds[key]
		# average=(favorite_total+underdog_total)/2


		# print()
		# #adds all favorites, adds all underdogs, then averages the two totals
		# away_total=0
		# home_total=0
		# for key in odds:
		# 	if odds[key]>0:
		# 		print(key+" pos: "+str(odds[key]/100))
		# 		away_total+=(odds[key]/100)
		# 	else:
		# 		print(key+" neg: "+str(odds[key]/100))
		# 		home_total+=(odds[key]/100)
		# print("away total: "+str(away_total))
		# print("home total: "+str(home_total))
		# average=(away_total+home_total)/len(odds.keys())
		# print("average: "+str(average))
		# average = average/2*100




		# print()
		# #adds all favorites, adds all underdogs, then averages the two totals
		# differences_total=0
		# for key in odds:
		# 	if odds[key]>0:
		# 		differences_total+=(odds[key] - (100-odds[key]))
		# 	else:
		# 		differences_total+=(odds[key] + (100 + odds[key]))
		# average=(differences_total)/len(odds.keys())
		# print("average: "+str(average))


		print()
		#adds all favorites, adds all underdogs, then averages the two totals
		total=0
		for key in odds:
			total+=odds[key]
		average=(total)/len(odds.keys())
		print("average: "+str(average))



		# print()




		if average>0:
			average+=50
		else:
			average-=50



		print("Favorite: "+str(average))

		for key in odds.keys():
			if odds[key]<0:
				odds[key]-=50
			else:
				odds[key]+=50




		to_return={}
		to_return['record_points']=             odds['records']
		to_return['home_away_points']=          odds['home_away']
		to_return['home_away_10_games_points']= odds['home_away_10_games']
		to_return['last_10_games_points']=      odds['last_10_games']
		to_return['avg_points']=                odds['avg_points']
		to_return['avg_points_10_games']=       odds['avg_points_10_games']
		# to_return['win_streak']=                win_streak
		if self.league=="nhl":
			to_return['win_streak_home_away']=      odds['win_streak_home_away']
		to_return['total']=                     self.universal.convert_number(average)
		return to_return



	def calculate_points(self, calc_type, returned1, returned2):

		if calc_type=="seasonal_records":
			points = returned1 - returned2

		elif calc_type=="home_away_records":
			 points = returned1 - returned2

		elif calc_type=="home_away_10_game_records":
			points = returned1 - returned2

		elif calc_type=="last_10_games":
                        
			points = returned1 - returned2

		elif calc_type=="avg_points":
			 points = returned1 - returned2

		elif calc_type=="avg_points_10_games":
			 points = returned1 - returned2

		elif calc_type=="win_streak":
                        
			 points = returned1 - returned2

		elif calc_type=="win_streak_home_away":

			 points = returned1 - returned2



		return points

# ... [Previous code including class Algo remains unchanged] ...

def get_closest_match(team_name, choices, threshold=70):
    match, score = process.extractOne(team_name, choices)
    return match if score >= threshold else None

def espn_standings(sport="NHL"):
    # Define base URLs and group IDs based on the sport
    if sport == "NHL":
        base_URL = "https://site.api.espn.com/apis/v2/sports/hockey/nhl/standings"
        group_ids = [7, 8]  # Example group IDs for NHL divisions
    elif sport == "NBA":
        base_URL = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
        group_ids = [5, 6]  # Group IDs for NBA conferences (Eastern and Western)
    else:
        print(f"Sport '{sport}' not supported.")
        return None

    stats = []

    # Iterate through each group (divisions for NHL, conferences for NBA)
    for group_id in group_ids:
        # Adjust sorting based on sport
        sort_param = "points" if sport == "NHL" else "winPercent"
        standings_URL = f"{base_URL}?group={group_id}&sort={sort_param}"

        # Make the HTTP request to get the standings
        standings_rep = requests.get(standings_URL)

        if standings_rep.status_code != 200:
            print(f"Failed to fetch data from {standings_URL}, status code: {standings_rep.status_code}")
            return None

        standings_json = standings_rep.json()
        conf_name = standings_json.get("abbreviation", "Unknown")
        standings_data = standings_json.get("children", [])

        for div in standings_data:
            div_name = div.get("abbreviation", "Unknown")
            div_data = div.get("standings", {}).get("entries", [])
            stats_tmp = []

            for team in div_data:
                name = team["team"]["abbreviation"]
                team_stats = team.get("stats", [])

                # Create a dictionary for easy access to stats by name
                stat_names = {x["name"]: x for x in team_stats}

                if sport == "NHL":
                    # Extract NHL-specific statistics
                    try:
                        overall = stat_names.get("overall", {})
                        record_summary = overall.get("summary", "0-0-0")
                        # Split the record into Wins, Losses, and Overtime Losses
                        record_parts = record_summary.split("-")
                        if len(record_parts) >= 3:
                            wins, losses, otl = record_parts[:3]
                        else:
                            # Handle cases where OTL might be missing
                            wins, losses = record_parts[:2]
                            otl = "0"

                        pts_stat = stat_names.get("points")
                        pts = pts_stat["value"] if pts_stat else 0

                        GP_stat = stat_names.get("gamesPlayed")
                        GP = GP_stat["value"] if GP_stat else 0

                        streak_stat = stat_names.get("streak")
                        streak = streak_stat["displayValue"] if streak_stat else "None"

                        home_stat = stat_names.get("Home")
                        home_record = home_stat["summary"] if home_stat else "0-0-0"

                        away_stat = stat_names.get("Road")
                        away_record = away_stat["summary"] if away_stat else "0-0-0"

                        last_ten_stat = stat_names.get("Last Ten Games")
                        last_ten_record = last_ten_stat.get("summary") if last_ten_stat else "0-0"
                    except Exception as e:
                        print(f"Error processing team {name}: {e}")
                        continue

                    stats_tmp.append({
                        'name': name,
                        'wins': int(wins),
                        'losses': int(losses),
                        'ot_losses': int(otl),
                        'points': pts,
                        'games_played': GP,
                        'streak': streak,
                        'home_record': home_record,
                        'away_record': away_record,
                        'last_ten_record': last_ten_record
                    })
                    print(name)

                elif sport == "NBA":
                    # Extract NBA-specific statistics
                    try:
                        overall = stat_names.get("overall", {})
                        record_summary = overall.get("summary", "0-0")
                        # Split the record into Wins and Losses
                        record_parts = record_summary.split("-")
                        if len(record_parts) >= 2:
                            wins, losses = record_parts[:2]
                        else:
                            # Handle unexpected record formats
                            wins = record_parts[0] if len(record_parts) >=1 else "0"
                            losses = "0"

                        wins_stat = stat_names.get("wins")
                        wins_value = wins_stat["value"] if wins_stat else 0

                        losses_stat = stat_names.get("losses")
                        losses_value = losses_stat["value"] if losses_stat else 0

                        win_percent_stat = stat_names.get("winpercent")
                        win_percent = win_percent_stat["value"] if win_percent_stat else 0.0

                        streak_stat = stat_names.get("streak")
                        streak = streak_stat["displayValue"] if streak_stat else "None"

                        home_stat = stat_names.get("Home")
                        home_record = home_stat["summary"] if home_stat else "0-0"

                        away_stat = stat_names.get("Road")
                        away_record = away_stat["summary"] if away_stat else "0-0"

                        last_ten_stat = stat_names.get("Last Ten Games")
                        last_ten_record = last_ten_stat.get("summary") if last_ten_stat else "0-0"

                        # Calculate games_played if not directly available
                        games_played = wins_value + losses_value
                    except Exception as e:
                        print(f"Error processing team {name}: {e}")
                        continue

                    stats_tmp.append({
                        'name': name,
                        'wins': int(wins),
                        'losses': int(losses),
                        'win_percent': win_percent,
                        'games_played': games_played,
                        'streak': streak,
                        'home_record': home_record,
                        'away_record': away_record,
                        'last_ten_record': last_ten_record
                    })
                    print(name)

            # Sort teams based on relevant statistics
            if sport == "NHL":
                # Sort by points descending, then by streak
                team_sort = sorted(stats_tmp, key=lambda x: (x['points'], x['streak']), reverse=True)
            elif sport == "NBA":
                # Sort by win percentage descending, then by streak
                team_sort = sorted(stats_tmp, key=lambda x: (x['win_percent'], x['streak']), reverse=True)

            # Create a frame name based on conference/division
            frame_name = f"{conf_name} {div_name} ({'W-L-OTL' if sport == 'NHL' else 'W-L'})"
            stats.append({
                'name': frame_name,
                'data': team_sort
            })

    # Combine all teams into a single list
    all_teams = []
    for division in stats:
        all_teams.extend(division['data'])

    # Convert the list of teams into a pandas DataFrame
    return pd.DataFrame(all_teams)

def extract_team_data(json_data, predict):
    # List to store extracted data
    extracted_data = []

    # Iterate through the scores list
    for game in json_data.get('scores', []):
        game_data = {}

        # Extract home team data
        home_team = game['teams']['home']
        away_team = game['teams']['away']
        game_data['Home Name'] = home_team['names']['name']
        print(game_data['Home Name'])
        game_data['Home MoneyLine'] = home_team['moneyLine']
        game_data['Home Spread Price'] = home_team['spreadPrice']
        game_data['Home Score'] = home_team['score']
        game_data['Home Votes'] = home_team['votes']
        game_data['Home Spread'] = home_team['spread']

        if not predict:
            game_data['won_game'] = home_team['score'] > away_team['score']

        # Extract away team data
        game_data['Away Name'] = away_team['names']['name']
        print(game_data['Away Name'])
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
        if not predict:
            game_data['Totals'] = home_team['score'] + away_team['score']
        game_data['Arena'] = game['stadium']

        extracted_data.append(game_data)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(extracted_data)
    return df

def fetch_odds_data(date, predict, sports):
    base_url = f"https://www.oddsshark.com/api/scores/{sports}/{date}?_format=json"
    print(base_url)

    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Referer': f'https://www.oddsshark.com/{sports.lower()}/scores',
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
        return df
    else:
        print(f"Failed to fetch data for date: {date}")
        return None

def clean_record(record):
    return re.sub(r'/\d+$', '', record)

# Define a function to reformat the record to include letters
def fix_record(record):
    if pd.isnull(record):
        return record
    parts = str(record).split('-')
    if len(parts) == 3:
        wins, losses, otl = parts
        return f"{wins}W-{losses}L-{otl}OT"
    else:
        return record

def merge_espn_odds(espn_df, odds_df, team_name_mapping, sport):
    # Map team abbreviations to full names
    espn_df['full_name'] = espn_df['name'].map(team_name_mapping)

    merged_rows = []
    for _, odds_row in odds_df.iterrows():
        home_team = odds_row['Home Name']
        away_team = odds_row['Away Name']

        # Find matching team stats in espn_df
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

        # Include the team records directly if available
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
            # If home_stats is empty, set default values
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
            # If away_stats is empty, set default values
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

    # Clean up 'record' fields to remove any trailing data after '/'
    for col in ['Home Team Record', 'Away Team Record']:
        merged_df[col] = merged_df[col].apply(lambda x: x.split('/')[0] if isinstance(x, str) else x)

    return merged_df

def fetch_recent_games(days=30, sports="NHL"):
    recent_games = []
    today = datetime.today()

    # Map the sports to their corresponding API URL parts
    sport_url_map = {
        "NHL": "hockey/nhl",
        "NBA": "basketball/nba",
        # Additional sports can be mapped here
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
    # Convert 'Date' column to datetime to ensure no future dates are included
    games_df['Date'] = pd.to_datetime(games_df['Date'])
    # Filter out any entries with a date that is somehow in the future
    games_df = games_df[games_df['Date'] <= pd.Timestamp.today().normalize()]

    return games_df

def calculate_average_points(team_name, historical_games):
    """
    Calculate average points scored and average points scored in the last 10 games.

    Parameters:
        team_name (str): The name of the team.
        historical_games (pd.DataFrame): DataFrame containing historical game data.

    Returns:
        dict: Dictionary containing 'avg_points' and 'avg_points_10_games'.
    """
    # Filter games involving the team
    team_games = historical_games[
        (historical_games['Home Team'] == team_name) | (historical_games['Away Team'] == team_name)
    ]

    if team_games.empty:
        return {
            'avg_points': 0,
            'avg_points_10_games': 0
        }

    # Calculate average points scored across all games
    points_scored = team_games.apply(
        lambda game: game['Home Score'] if game['Home Team'] == team_name else game['Away Score'],
        axis=1
    )
    avg_points = points_scored.mean()
    print(avg_points)

    # Calculate average points scored in the last 10 games
    last_10_games = team_games.sort_values(by='Date', ascending=False).head(10)
    last_10_points_scored = last_10_games.apply(
        lambda game: game['Home Score'] if game['Home Team'] == team_name else game['Away Score'],
        axis=1
    )
    avg_points_10_games = last_10_points_scored.mean()

    # Return the relevant averages
    return {
        'avg_points': avg_points,                # Metric 5
        'avg_points_10_games': avg_points_10_games # Metric 6
    }

def calculate_win_loss_streaks(team_name, historical_games):
    """
    Calculate the current win streak and current win streak at home or away for a team.

    Parameters:
        team_name (str): The name of the team.
        historical_games (pd.DataFrame): DataFrame containing historical game data.

    Returns:
        dict: Dictionary containing current overall win streak and current home/away win streak.
    """
    # Sort games by date in descending order
    team_games = historical_games[
        (historical_games['Home Team'] == team_name) | (historical_games['Away Team'] == team_name)
    ].sort_values(by='Date', ascending=False)

    if team_games.empty:
        return {
            'win_streak_home_away': 0
        }

    # Calculate overall win streak
    win_streak = 0
    for _, game in team_games.iterrows():
        if game['Home Team'] == team_name:
            won = game['Home Score'] > game['Away Score']
        else:
            won = game['Away Score'] > game['Home Score']
        
        if won:
            win_streak += 1
        else:
            break  # Streak ended

    # Calculate home or away win streak based on last game type
    last_game = team_games.iloc[0]
    if last_game['Home Team'] == team_name:
        location = 'Home'
    else:
        location = 'Away'

    win_streak_home_away = 0
    for _, game in team_games.iterrows():
        current_location = 'Home' if game['Home Team'] == team_name else 'Away'
        if current_location != location:
            break  # Different location, streak specific to initial location
        if current_location == 'Home':
            won = game['Home Score'] > game['Away Score']
        else:
            won = game['Away Score'] > game['Home Score']
        
        if won:
            win_streak_home_away += 1
        else:
            break  # Streak ended

    print(win_streak_home_away)
    return {
        'win_streak_home_away': win_streak_home_away # Metric 7
    }

def prepare_team_data(row, team_type='Home', historical_games=None, sport='NBA'):
    """
    Prepare and calculate various metrics for a team based on game data.

    Parameters:
        row (pd.Series): A row from a DataFrame containing game and team information.
        team_type (str): 'Home' or 'Away' indicating the team's role in the game.
        historical_games (pd.DataFrame): DataFrame containing historical game data.
        sport (str): 'NHL' or 'NBA'

    Returns:
        dict: Dictionary containing calculated metrics for the team.
    """
    team_data = {}
    team_name = row.get(f'{team_type} Team')

    if not team_name:
        print(f"{team_type} Team name is missing in the row.")
        # Set all metrics to default
        team_data = {
            'seasonal_records': 0,          # Metric 1
            'record_points': 0,             # Metric 1
            'home_away_record': 0,          # Metric 2
            'home_away_10_game_records': 0, # Metric 3
            'last_10_games': 0,             # Metric 4
            'avg_points': 0,                # Metric 5
            'avg_points_10_games': 0,       # Metric 6
            'win_streak_home_away': 0       # Metric 7
        }
        return team_data  # Return default metrics

    # 1) Record_points = wins - losses
    record = row.get(f'{team_type} Team Record')
    if record:
        try:
            record_parts = record.split('-')
            wins = int(record_parts[0]) if len(record_parts) >= 1 else 0
            losses = int(record_parts[1]) if len(record_parts) >= 2 else 0
            # For NHL, there might be overtime losses; adjust if needed
            if sport == "NHL" and len(record_parts) >= 3:
                otl = int(record_parts[2])
                losses += otl  # Considering overtime losses as regular losses for Record_points
            # Calculate Record_points
            record_points = wins - losses
            team_data['record_points'] = record_points  # Metric 1
            team_data['seasonal_records'] = record_points  # Metric 1
        except ValueError as e:
            print(f"Error parsing {team_type} Team Record for {team_name}: {e}")
            team_data['record_points'] = 0  # Metric 1
            team_data['seasonal_records'] = 0  # Metric 1
    else:
        print(f"{team_type} Team Record is missing for {team_name}.")
        team_data['record_points'] = 0  # Metric 1
        team_data['seasonal_records'] = 0  # Metric 1

    # 2) Home_away = away_record - home_record
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
            # For NHL, handle overtime losses if present
            if sport == "NHL":
                if len(home_parts) >= 3:
                    home_otl = int(home_parts[2])
                    home_losses += home_otl
                if len(away_parts) >= 3:
                    away_otl = int(away_parts[2])
                    away_losses += away_otl
            # Calculate Home_away
            home_away = (away_wins - away_losses) - (home_wins - home_losses)
            team_data['home_away_record'] = home_away  # Metric 2
        except ValueError as e:
            print(f"Error parsing {team_type} Team Home/Away Record for {team_name}: {e}")
            team_data['home_away_record'] = 0  # Metric 2
    else:
        print(f"{team_type} Team Home/Away Record is missing or not a string for {team_name}.")
        team_data['home_away_record'] = 0  # Metric 2

    # 3) Home_away_10_games = away_record - home_record for the last 10 games
    if historical_games is not None:
        try:
            if team_type == 'Home':
                last_10_home_games = historical_games[
                    (historical_games['Home Team'] == team_name)
                ].sort_values(by='Date', ascending=False).head(10)
                home_10_wins = sum(last_10_home_games['Home Score'] > last_10_home_games['Away Score'])
                home_10_losses = sum(last_10_home_games['Home Score'] <= last_10_home_games['Away Score'])
                home_away_10_games = home_10_wins - home_10_losses
            else:  # Away
                last_10_away_games = historical_games[
                    (historical_games['Away Team'] == team_name)
                ].sort_values(by='Date', ascending=False).head(10)
                away_10_wins = sum(last_10_away_games['Away Score'] > last_10_away_games['Home Score'])
                away_10_losses = sum(last_10_away_games['Away Score'] <= last_10_away_games['Home Score'])
                home_away_10_games = away_10_wins - away_10_losses

            team_data['home_away_10_game_records'] = home_away_10_games  # Metric 3
        except Exception as e:
            print(f"Error calculating Home_away_10_games for {team_name}: {e}")
            team_data['home_away_10_game_records'] = 0  # Metric 3
    else:
        team_data['home_away_10_game_records'] = 0  # Metric 3

    # 4) Last_10_games_points = wins - losses of the last 10 games
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
            team_data['last_10_games'] = last_10_games_points  # Metric 4
        except Exception as e:
            print(f"Error calculating Last_10_games_points for {team_name}: {e}")
            team_data['last_10_games'] = 0  # Metric 4
    else:
        team_data['last_10_games'] = 0  # Metric 4

    # 5) Avg_points = total_goals / num_games
    # 6) Avg_points_10_games = total_goals / num_games of the last 10 games
    if historical_games is not None:
        try:
            avg_points_data = calculate_average_points(team_name, historical_games)
            team_data['avg_points'] = avg_points_data.get('avg_points', 0)  # Metric 5
            team_data['avg_points_10_games'] = avg_points_data.get('avg_points_10_games', 0)  # Metric 6
        except Exception as e:
            print(f"Error calculating average points for {team_name}: {e}")
            team_data['avg_points'] = 0  # Metric 5
            team_data['avg_points_10_games'] = 0  # Metric 6
    else:
        team_data['avg_points'] = 0  # Metric 5
        team_data['avg_points_10_games'] = 0  # Metric 6

    # 7) Win_streak_home_away = number of consecutive wins home or away
    if historical_games is not None:
        try:
            streak_data = calculate_win_loss_streaks(team_name, historical_games)
            team_data['win_streak_home_away'] = streak_data.get('win_streak_home_away', 0)  # Metric 7
        except Exception as e:
            print(f"Error calculating win streaks for {team_name}: {e}")
            team_data['win_streak_home_away'] = 0  # Metric 7
    else:
        team_data['win_streak_home_away'] = 0  # Metric 7

    # 2) Ensure 'home_away_record' is present
    if 'home_away_record' not in team_data:
        team_data['home_away_record'] = 0

    return team_data

def safe_to_lower(text):
    if isinstance(text, str):
        return text.lower()
    return text

def main():
    date = (datetime.today() + timedelta(days=0)).strftime('%Y-%m-%d')
    today_str = datetime.today().strftime("%Y-%m-%d")
    sports = "NBA"  # Change to "NHL" or "NBA" as needed
    days = 30

    # Select the appropriate team name mapping
    if sports == "NBA":
        team_name_mapping = nba_team_name_mapping
    elif sports == "NHL":
        team_name_mapping = nhl_team_name_mapping
    else:
        print(f"Unsupported sport: {sports}")
        return

    # Fetch ESPN standings
    espn_df = espn_standings(sports)
    if espn_df is None or espn_df.empty:
        print("No ESPN standings data fetched.")
        return
    print("ESPN Standings Data:")
    print(espn_df)

    # Map team abbreviations to full names
    espn_df['full_name'] = espn_df['name'].map(team_name_mapping)

    # Fetch Odds data
    odds_df = fetch_odds_data(date, True, sports)
    if odds_df is None or odds_df.empty:
        print("No odds data fetched.")
        return
    print("Odds Data:")
    print(odds_df)

    # Merge ESPN standings with Odds data
    merged_df = merge_espn_odds(espn_df, odds_df, team_name_mapping, sport=sports)
    if merged_df.empty:
        print("Merged DataFrame is empty.")
        return
    print("Merged Data:")
    print(merged_df)

    # Fetch historical games
    historical_games = fetch_recent_games(days, sports)
    if historical_games.empty:
        print("No historical games data fetched.")
    historical_games.to_csv(f"historical_games_{date}.csv", index=False)
    print("Historical Games Data:")
    print(historical_games)

    # Initialize the Algo class
    algo = Algo(safe_to_lower(sports))
    predictions = []

    for index, row in merged_df.iterrows():
        # Prepare data for home and away teams
        home_team_data = prepare_team_data(row, team_type='Home', historical_games=historical_games, sport=sports)
        away_team_data = prepare_team_data(row, team_type='Away', historical_games=historical_games, sport=sports)

        # Calculate predictions using the Algo class
        prediction = algo.calculate_V2(date, home_team_data, away_team_data)
        predictions.append({
            'Date': row['Date'],
            'Home Team': row['Home Team'],
            'Away Team': row['Away Team'],
            'Prediction': prediction.get('total', None),
            'Details': prediction
        })

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(f"predictions_{date}.csv", index=False)
    print("Predictions Data:")
    print(predictions_df)
    predictions_df.drop('Details', axis=1, inplace=True)
    predictions_df.drop('Date', axis=1, inplace=True)
    predictions_df.insert(0, 'Date', today_str)

    # Upload to Google Sheets
    # Use credentials to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('creds.json', scope)
    client = gspread.authorize(creds)

    if sports == "NHL":
        # Open the NHL spreadsheet by its key
        spreadsheet_id = '1KgwFdqrRUs2fa5pSRmirj6ajyO2d14ONLsiksAYk8S8'
    elif sports == "NBA":
        # Open the NBA spreadsheet by its key
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
        # Select the specified sheet
        sheet = spreadsheet.worksheet(sheet_name)
    except Exception as e:
        print(f"Error accessing sheet '{sheet_name}': {e}")
        return

    # Clear existing data
    sheet.clear()

    # Append headers
    sheet.append_row(predictions_df.columns.tolist())

    # Convert DataFrame to a list of lists for the data rows
    data = predictions_df.values.tolist()

    # Append the data rows to the sheet
    sheet.append_rows(data, value_input_option='RAW')  # Efficiently append the rows

    print(f"Data successfully uploaded to Google Sheets for {sports}.")

if __name__ == '__main__':
    main()
