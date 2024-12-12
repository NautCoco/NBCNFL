import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score

import numpy as np
import matplotlib.pyplot as plt


#function to load and manipulate the data to the respective features and the output variable
def load_data(file_path, history_file_path):
    # load csv files
    data = pd.read_csv(file_path)
    history = pd.read_csv(history_file_path)

    # calculate cumulative stats for each team in the historical data
    teams = set(history['Winner/tie'].unique()).union(history['Loser/tie'].unique())
    historical_stats = {team: {
        'PointsForHistory': 0, 'PointsAgainstHistory': 0,
        'YardsForHistory': 0, 'YardsAgainstHistory': 0,
        'WinsHistory': 0, 'LossesHistory': 0,
        #'PlayoffWinsHistory': 0  #uncomment to use playoffs wins
        } for team in teams}

    for _, row in history.iterrows():
        winner = row['Winner/tie']
        loser = row['Loser/tie']

        # determine home and away based on '@' in data set
        home_team = loser if not pd.isna(row['Unnamed: 5']) else winner
        away_team = winner if not pd.isna(row['Unnamed: 5']) else loser

        # get game stats
        points_w = row['PtsW']
        points_l = row['PtsL']
        yards_w = row['YdsW']
        yards_l = row['YdsL']

        # uncomment this block to use playoffwins feature
        """
        if "WildCard" in row['Week']:
            historical_stats[winner]['PlayoffWinsHistory'] += 1
        elif "Division" in row['Week']:
            historical_stats[winner]['PlayoffWinsHistory'] += 2
        elif "ConfChamp" in row['Week']:
            historical_stats[winner]['PlayoffWinsHistory'] += 3
        elif "SuperBowl" in row['Week']:
            historical_stats[winner]['PlayoffWinsHistory'] += 4
        """
        # update historical stats for winner
        historical_stats[winner]['PointsForHistory'] += points_w
        historical_stats[winner]['PointsAgainstHistory'] += points_l
        historical_stats[winner]['YardsForHistory'] += yards_w
        historical_stats[winner]['YardsAgainstHistory'] += yards_l
        historical_stats[winner]['WinsHistory'] += 1

        # update historical stats for loser
        historical_stats[loser]['PointsForHistory'] += points_l
        historical_stats[loser]['PointsAgainstHistory'] += points_w
        historical_stats[loser]['YardsForHistory'] += yards_l
        historical_stats[loser]['YardsAgainstHistory'] += yards_w
        historical_stats[loser]['LossesHistory'] += 1

    # process current data
    teams = set(data['Winner/tie'].unique()).union(data['Loser/tie'].unique())
    stats = {team: {
        'PointsFor': 0, 'PointsAgainst': 0, 'YardsFor': 0, 'YardsAgainst': 0,
        'Wins': 0, 'Losses': 0,
        #'PlayoffWinsHisotry': 0 # uncomment to use playoffswins feature
    } for team in teams}

    rows = []

    for _, row in data.iterrows():
        winner = row['Winner/tie']
        loser = row['Loser/tie']

        # check home and away based on '@'
        home_team = loser if not pd.isna(row['Unnamed: 5']) else winner
        away_team = winner if not pd.isna(row['Unnamed: 5']) else loser

        # get game stats
        points_w = row['PtsW']
        points_l = row['PtsL']
        yards_w = row['YdsW']
        yards_l = row['YdsL']

        rows.append({
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'HomePlayoffWins': historical_stats[home_team]['PlayoffWinsHistory'],
            'AwayPlayoffWins': historical_stats[away_team]['PlayoffWinsHistory'],
            'HomePointsFor': stats[home_team]['PointsFor'],
            'HomePointsAgainst': stats[home_team]['PointsAgainst'],
            'HomeYardsFor': stats[home_team]['YardsFor'],
            'HomeYardsAgainst': stats[home_team]['YardsAgainst'],
            'HomeWins': stats[home_team]['Wins'],
            'HomeLosses': stats[home_team]['Losses'],
            'AwayPointsFor': stats[away_team]['PointsFor'],
            'AwayPointsAgainst': stats[away_team]['PointsAgainst'],
            'AwayYardsFor': stats[away_team]['YardsFor'],
            'AwayYardsAgainst': stats[away_team]['YardsAgainst'],
            'AwayWins': stats[away_team]['Wins'],
            'AwayLosses': stats[away_team]['Losses'],
            'HomePointsForHistory': historical_stats[home_team]['PointsForHistory'],
            'HomePointsAgainstHistory': historical_stats[home_team]['PointsAgainstHistory'],
            'HomeYardsForHistory': historical_stats[home_team]['YardsForHistory'],
            'HomeYardsAgainstHistory': historical_stats[home_team]['YardsAgainstHistory'],
            'HomeWinsHistory': historical_stats[home_team]['WinsHistory'],
            'HomeLossesHistory': historical_stats[home_team]['LossesHistory'],
            'AwayPointsForHistory': historical_stats[away_team]['PointsForHistory'],
            'AwayPointsAgainstHistory': historical_stats[away_team]['PointsAgainstHistory'],
            'AwayYardsForHistory': historical_stats[away_team]['YardsForHistory'],
            'AwayYardsAgainstHistory': historical_stats[away_team]['YardsAgainstHistory'],
            'AwayWinsHistory': historical_stats[away_team]['WinsHistory'],
            'AwayLossesHistory': historical_stats[away_team]['LossesHistory'],
            'HomeWin': 1 if home_team == winner else 0
        })

        # update stats for winner
        stats[winner]['PointsFor'] += points_w
        stats[winner]['PointsAgainst'] += points_l
        stats[winner]['YardsFor'] += yards_w
        stats[winner]['YardsAgainst'] += yards_l
        stats[winner]['Wins'] += 1

        # update stats for loser
        stats[loser]['PointsFor'] += points_l
        stats[loser]['PointsAgainst'] += points_w
        stats[loser]['YardsFor'] += yards_l
        stats[loser]['YardsAgainst'] += yards_w
        stats[loser]['Losses'] += 1

    processed_data = pd.DataFrame(rows)

    # features and the target to return
    features = processed_data[['HomePointsFor', 'HomePointsAgainst', 'HomeYardsFor', 'HomeYardsAgainst',
                               'HomeWins', 'HomeLosses', #'HomePlayoffWins', 'AwayPlayoffWins', #uncomment meaning remove '#' to use playoff wins feature
                               'AwayPointsFor', 'AwayPointsAgainst', 'AwayYardsFor', 'AwayYardsAgainst',
                               'AwayWins', 'AwayLosses',
                               'HomePointsForHistory', 'HomePointsAgainstHistory', 'HomeYardsForHistory',
                               'HomeYardsAgainstHistory', 'HomeWinsHistory', 'HomeLossesHistory',
                               'AwayPointsForHistory', 'AwayPointsAgainstHistory', 'AwayYardsForHistory',
                               'AwayYardsAgainstHistory', 'AwayWinsHistory', 'AwayLossesHistory']]
    target = processed_data['HomeWin']

    return features, target, stats, historical_stats

# train model function
def train_model(features, target):
    # split the data into training and test sets 
    testLatest = False
    #testLatest = true  # uncomment this line to use the latest 25% of games as test data
    
    
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=45)
    if (testLatest):
        split_index = int(0.75 * len(features))  
        x_train = features[:split_index]
        x_test = features[split_index:]
        y_train = target[:split_index]
        y_test = target[split_index:]

    # make the Naive Bayes model
    model = GaussianNB()
    # train the model
    model.fit(x_train, y_train)

    # evaluate the model on the test set
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # plot calibration curve
    prob_pos = model.predict_proba(x_test)[:, 1]
    ap_score = average_precision_score(y_test, prob_pos)
    print(f"Mean Average Precision (MAP): {ap_score:.4f}")
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()

    return model

# print records of each team (for test)
def visualize_records(stats):
    print("\nCurrent Team Records:")
    print("=====================")
    for team, record in stats.items():
        print(f"{team}: {record['Wins']} Wins, {record['Losses']} Losses")

# predict winner from the input
def predict_matchup(model, stats, historical_stats, home_team, away_team):
    if home_team not in stats or away_team not in stats:
        print("One or both teams not found in stats.")
        return

    features = pd.DataFrame([{
        'HomePointsFor': stats[home_team]['PointsFor'],
        'HomePointsAgainst': stats[home_team]['PointsAgainst'],
        'HomeYardsFor': stats[home_team]['YardsFor'],
        'HomeYardsAgainst': stats[home_team]['YardsAgainst'],
        'HomeWins': stats[home_team]['Wins'],
        'HomeLosses': stats[home_team]['Losses'],
        'HomePlayoffWins': historical_stats[home_team]['PlayoffWinsHistory'],
        'AwayPlayoffWins': historical_stats[away_team]['PlayoffWinsHistory'],
        'AwayPointsFor': stats[away_team]['PointsFor'],
        'AwayPointsAgainst': stats[away_team]['PointsAgainst'],
        'AwayYardsFor': stats[away_team]['YardsFor'],
        'AwayYardsAgainst': stats[away_team]['YardsAgainst'],
        'AwayWins': stats[away_team]['Wins'],
        'AwayLosses': stats[away_team]['Losses'],
        'HomePointsForHistory': historical_stats[home_team]['PointsForHistory'],  
        'HomePointsAgainstHistory': historical_stats[home_team]['PointsAgainstHistory'],
        'HomeYardsForHistory': historical_stats[home_team]['YardsForHistory'],
        'HomeYardsAgainstHistory': historical_stats[home_team]['YardsAgainstHistory'],
        'HomeWinsHistory': historical_stats[home_team]['WinsHistory'],
        'HomeLossesHistory': historical_stats[home_team]['LossesHistory'],
        'AwayPointsForHistory': historical_stats[away_team]['PointsForHistory'],
        'AwayPointsAgainstHistory': historical_stats[away_team]['PointsAgainstHistory'],
        'AwayYardsForHistory': historical_stats[away_team]['YardsForHistory'],
        'AwayYardsAgainstHistory': historical_stats[away_team]['YardsAgainstHistory'],
        'AwayWinsHistory': historical_stats[away_team]['WinsHistory'],
        'AwayLossesHistory': historical_stats[away_team]['LossesHistory']
    }])

    pred = model.predict(features)
    probs = model.predict_proba(features)
    print(f"Prediction: {'Home team wins' if pred[0] == 1 else 'Away team wins'}")
    print(f"Probability: Home Win: {probs[0][1]:.2f}, Away Win: {probs[0][0]:.2f}")

# main
if __name__ == "__main__":
    # file paths to the data
    file_path = "2024.csv"
    history_file_path = "history.csv"

    # load data and features from history and current season stats
    features, target, stats, historical_stats = load_data(file_path, history_file_path)
    #train model
    model = train_model(features, target)

    
    #visualize_records(stats)

    # ask for mathcup input 
    home_team = input("Enter the home team: ")
    away_team = input("Enter the away team: ")
    predict_matchup(model, stats, historical_stats, home_team, away_team)
