import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, DMatrix, train
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier

import datetime
warnings.filterwarnings("ignore")

# Define today's date
today = datetime.date.today()

# Calculate yesterday by subtracting one day
yesterday = today - datetime.timedelta(days=2)

# Calculate tomorrow by adding one day
tomorrow = today + datetime.timedelta(days=1)

def calculate_recent_performance(df, games=10):
    """
    Calculate the team's recent performance metrics for home and away teams efficiently, including current winning
    and losing streaks.
    :param df: DataFrame containing the game data.
    :param games: Number of recent games to calculate metrics for.
    :return: DataFrame with calculated metrics for both home and away teams.
    """
    # Prepare DataFrame for rolling calculations
    # Calculate for home games
    home_df = df.copy()
    home_df['home_wins'] = home_df['home_win']
    home_df['home_goals_scored'] = home_df['score_home']
    home_df['home_goals_conceded'] = home_df['score_away']
    
    home_df.sort_values(['home_team', 'game_date'], inplace=True)

    # Rolling calculate win rate, goals scored, and goals conceded for home team
    home_df['home_recent_win_rate'] = home_df.groupby('home_team')['home_wins'].transform(lambda x: x.rolling(window=games, min_periods=1).mean())
    home_df['home_avg_goals_scored'] = home_df.groupby('home_team')['home_goals_scored'].transform(lambda x: x.rolling(window=games, min_periods=1).mean())
    home_df['home_avg_goals_conceded'] = home_df.groupby('home_team')['home_goals_conceded'].transform(lambda x: x.rolling(window=games, min_periods=1).mean())

    # Calculate current winning and losing streak for home teams
    home_df['home_winning_streak'] = home_df.groupby('home_team')['home_wins'].transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    home_df['home_losing_streak'] = home_df.groupby('home_team')['home_wins'].transform(lambda x: (1 - x).groupby((x != x.shift()).cumsum()).cumcount() + 1)

    # Calculate for away games
    away_df = df.copy()
    away_df['away_wins'] = away_df['home_win'].apply(lambda x: 1 if x == 0 else 0)  # Invert home_win for away team perspective
    away_df['away_goals_scored'] = away_df['score_away']
    away_df['away_goals_conceded'] = away_df['score_home']
    
    away_df.sort_values(['away_team', 'game_date'], inplace=True)

    # Rolling calculate win rate, goals scored, and goals conceded for away team
    away_df['away_recent_win_rate'] = away_df.groupby('away_team')['away_wins'].transform(lambda x: x.rolling(window=games, min_periods=1).mean())
    away_df['away_avg_goals_scored'] = away_df.groupby('away_team')['away_goals_scored'].transform(lambda x: x.rolling(window=games, min_periods=1).mean())
    away_df['away_avg_goals_conceded'] = away_df.groupby('away_team')['away_goals_conceded'].transform(lambda x: x.rolling(window=games, min_periods=1).mean())

    # Calculate current winning and losing streak for away teams
    away_df['away_winning_streak'] = away_df.groupby('away_team')['away_wins'].transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    away_df['away_losing_streak'] = away_df.groupby('away_team')['away_wins'].transform(lambda x: (1 - x).groupby((x != x.shift()).cumsum()).cumcount() + 1)

    # Merge the metrics back to the original dataframe
    df = df.merge(home_df[['game_date', 'home_team', 'home_recent_win_rate', 'home_avg_goals_scored', 'home_avg_goals_conceded', 'home_winning_streak', 'home_losing_streak']], on=['game_date', 'home_team'], how='left')
    df = df.merge(away_df[['game_date', 'away_team', 'away_recent_win_rate', 'away_avg_goals_scored', 'away_avg_goals_conceded', 'away_winning_streak', 'away_losing_streak']], on=['game_date', 'away_team'], how='left')

    return df


# Load NHL Elo ratings data
elo_url = 'https://raw.githubusercontent.com/Neil-Paine-1/NHL-Player-And-Team-Ratings/master/nhl_elo.csv'
elo_df = pd.read_csv(elo_url).drop_duplicates()


# Preprocess data to assign team names, ratings, probabilities, and scores
is_home = elo_df['is_home'] == 1
elo_df['home_team'] = np.where(is_home, elo_df['team1'], elo_df['team2'])
elo_df['away_team'] = np.where(is_home, elo_df['team2'], elo_df['team1'])
elo_df['home_team_elo'] = np.where(is_home, elo_df['elo1_pre'], elo_df['elo2_pre'])
elo_df['away_team_elo'] = np.where(is_home, elo_df['elo2_pre'], elo_df['elo1_pre'])
elo_df['home_team_prob'] = np.where(is_home, elo_df['prob1'], elo_df['prob2'])
elo_df['away_team_prob'] = np.where(is_home, elo_df['prob2'], elo_df['prob1'])
elo_df['home_team_pts'] = np.where(is_home, elo_df['exp_pts1'], elo_df['exp_pts2'])
elo_df['away_team_pts'] = np.where(is_home, elo_df['exp_pts2'], elo_df['exp_pts1'])
elo_df['score_home'] = np.where(is_home, elo_df['score1'], elo_df['score2'])
elo_df['score_away'] = np.where(is_home, elo_df['score2'], elo_df['score1'])
elo_df['elo_diff'] = elo_df['home_team_elo'] - elo_df['away_team_elo']
elo_df['home_win'] = (elo_df['score_home'] > elo_df['score_away']).astype(int)
elo_df.rename(columns={'date': 'game_date'}, inplace=True)
elo_df['game_date'] = pd.to_datetime(elo_df['game_date'])

# Calculate rest days and identify back-to-back games
elo_df.sort_values(['home_team', 'game_date'], inplace=True)
elo_df['previous_home_game'] = elo_df.groupby('home_team')['game_date'].shift(1)
elo_df['rest_days_home'] = (elo_df['game_date'] - elo_df['previous_home_game']).dt.days - 1
elo_df['rest_days_home'].fillna(-1, inplace=True)  # For the first game

elo_df.sort_values(['away_team', 'game_date'], inplace=True)
elo_df['previous_away_game'] = elo_df.groupby('away_team')['game_date'].shift(1)
elo_df['rest_days_away'] = (elo_df['game_date'] - elo_df['previous_away_game']).dt.days - 1
elo_df['rest_days_away'].fillna(-1, inplace=True)  # For the first game

elo_df['back_to_back_home'] = elo_df['rest_days_home'] == 0
elo_df['back_to_back_away'] = elo_df['rest_days_away'] == 0

elo_df.drop_duplicates(subset=['game_date', 'home_team', 'away_team'], keep='first', inplace=True)

elo_df.sort_values('game_date', inplace=True)

# Apply function to elo_df
elo_df = calculate_recent_performance(elo_df)

# Save DataFrame to CSV file
elo_df.to_csv('nhl.csv', index=False)

# Define features and target variable for model training
features = ['home_team_elo', 'away_team_elo', 'rest_days_home', 'rest_days_away',
            'back_to_back_home', 'back_to_back_away', 'away_team_prob', 'home_team_prob', 'elo_diff','home_recent_win_rate',
            'away_recent_win_rate','home_avg_goals_scored','home_avg_goals_conceded','away_avg_goals_scored','away_avg_goals_conceded','playoff',
            'home_winning_streak','away_winning_streak','home_losing_streak', 'away_losing_streak','is_home']
X = elo_df[features]
y = elo_df['home_win']

# Define a cutoff date for splitting the dataset into training and testing sets
# For example, choosing a date that separates the last 20% of data for testing
cutoff_date = elo_df['game_date'].quantile(0.8, interpolation='nearest')

today_2 = pd.Timestamp('now').floor('D')
yesterday_2 = today_2 - pd.Timedelta(days=1)
# Split the data based on the cutoff date
train_df = elo_df[elo_df['game_date'] < cutoff_date]
test_df = elo_df[(elo_df['game_date'] >= cutoff_date) & (elo_df['game_date'] < today_2)]

# Extract features and target from the training and testing sets
X_train = train_df[features]
y_train = train_df['home_win']
X_test = test_df[features]
y_test = test_df['home_win']


# Normalize features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate Logistic Regression model
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_scaled, y_train)
probabilities_lr = model_lr.predict_proba(X_test_scaled)[:, 1]
accuracy_lr = accuracy_score(y_test, model_lr.predict(X_test_scaled))
log_loss_lr = log_loss(y_test, probabilities_lr)
print(f"Logistic Regression - Accuracy: {accuracy_lr:.4f}, Log Loss: {log_loss_lr:.4f}")

# Train and evaluate Gradient Boosting Machine model
model_gbm = GradientBoostingClassifier()
model_gbm.fit(X_train, y_train)
probabilities_gbm = model_gbm.predict_proba(X_test)[:, 1]
accuracy_gbm = accuracy_score(y_test, model_gbm.predict(X_test))
log_loss_gbm = log_loss(y_test, probabilities_gbm)
print(f"Gradient Boosting Machine - Accuracy: {accuracy_gbm:.4f}, Log Loss: {log_loss_gbm:.4f}")

# Train and evaluate XGBoost model
# Initialize the XGBClassifier
model_xgb = XGBClassifier(n_estimators=25, use_label_encoder=False, eval_metric='logloss')

# Train the model using the high-level API which is compatible with scikit-learn
model_xgb.fit(X_train, y_train)

# Predict probabilities and class labels for the test set
probabilities_xgb = model_xgb.predict_proba(X_test)[:, 1]
predictions_xgb = (probabilities_xgb >= 0.5).astype(int)

# Calculate accuracy and log loss
accuracy_xgb = accuracy_score(y_test, predictions_xgb)
log_loss_xgb = log_loss(y_test, probabilities_xgb)

# Print the results
print(f"XGBoost - Accuracy: {accuracy_xgb:.4f}, Log Loss: {log_loss_xgb:.4f}")

# Function to predict today's games using the GBM model
def predict_today_games(model, date, games_df):
    # Ensure date formats match
    if not isinstance(date, datetime.date):
        try:
            date = pd.to_datetime(date).date()
        except ValueError as e:
            print(f"Date conversion error: {e}")
            return None

    # Filter for games on the specified date
    games_today_df = games_df[games_df['game_date'].dt.date == date].copy()
    if games_today_df.empty:
        print(f"No games on {date}")
        return None

    # Predict probabilities for the home team winning
    x_predict = games_today_df[features]
    # No need to convert x_predict to DMatrix, directly use the dataframe with XGBClassifier
    probabilities = model.predict_proba(x_predict)[:, 1]  # Using predict_proba to get probabilities

    games_today_df['predicted_prob_home_win'] = probabilities

    # Print predictions
    print("Predictions for today's games:")
    for index, row in games_today_df.iterrows():
        print(f"Game Date: {row['game_date'].strftime('%Y-%m-%d')}")
        print(f"Home Team: {row['home_team']}")
        print(f"Away Team: {row['away_team']}")
        print(f"Predicted Probability of Home Win: {row['predicted_prob_home_win']:.4f}")
        print("-" * 30)

    return games_today_df[['game_date', 'home_team', 'away_team', 'predicted_prob_home_win']]


# Predict today's games
today_predictions = predict_today_games(model_xgb, today, elo_df)
print(today_predictions)
