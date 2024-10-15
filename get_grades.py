"""
I mean it's called `get_grades`
"""
import numpy as np
import pandas as pd
import pickle
from consants import TOOL_INFO, GAME_TYPES
import pybaseball as pyb
import argparse
from etl import pull_data, format_data

def load_models() -> dict:
    """
    Load the models and other stuff needed for each tool

    :return: Dictionary of the models and friends
    """

    models = {}

    for info in TOOL_INFO.items():

        print(f"Loading {info[0]} model")

        with open(f"models/{info[0]}_model.pkl", "rb") as f:
            model = pickle.load(f)
            models[info[0]] = model

        with open(f"models/{info[0]}_features.pkl", "rb") as f:
            features = pickle.load(f)
            models[info[0] + "_features"] = features

    le = pickle.load(open('models/le.pkl', 'rb'))
    run_value = pd.read_csv('models/rv_table.csv', index_col=0)

    models['le'] = le
    models['run_value'] = run_value

    return models

def get_grades(data: pd.DataFrame, models: dict, year: int, q:int) -> pd.DataFrame:
    """
    Calculate grades
    :param data: cleaned statcast data
    :param models: dictionary of models and features
    :param year: year to pull sprint speed data
    :param q: minimum number of swings to be considered when standardizing grades
    :return: DataFrame of player grades
    """

    # Add swing probability
    data['xSw'] = models['Swing Decision'].predict_proba(data[models['Swing Decision' + "_features"]])[:, 1]

    # Add result probability
    probs = pd.DataFrame(models['Outcome Probability'].predict_proba(data[models['Outcome Probability' + "_features"]]))

    data[models['Outcome Probability' + "_features"]].to_csv('pls3.csv')

    probs.to_csv('pls1.csv')

    # Calculate run value for each event
    for col in probs.columns:
        probs[col] = probs[col] * models['run_value'].iloc[int(col)]

    probs.fillna(0, inplace=True)

    probs.to_csv('pls.csv')

    # Calculate swing run value
    data['swingRv'] = list(probs.sum(axis=1))

    # Add Ball and Strike Values
    data['ballRv'] = data.loc[data['cResult'] == (data['count'] + 'ball')]['cRV'].mean()
    data['strikeRv'] = data.loc[data['cResult'] == (data['count'] + 'called_strike')]['cRV'].mean()

    # Calculate swing/called strike/ball probabilities and turn into descion score
    data['pSwing'] = models['Swing Decision'].predict_proba(data[models['Swing Decision' + "_features"]])[:, 1]
    data['pStrike'] = (models['Strike Probability'].predict_proba(data[models['Strike Probability' + "_features"]])[:, 1]) * (1-data['pSwing'])
    data['pBall'] = 1 - data['pSwing'] - data['pStrike']
    data['xPitchScore'] = data['xSw'] * data['pSwing'] + data['strikeRv'] * data['pStrike'] + data['ballRv'] * data['pBall']
    data['TakeScore'] = (data['strikeRv'] * (data['pStrike'] / (data['pStrike'] + data['pBall']))) + (
                data['ballRv'] * (data['pBall'] / (data['pStrike'] + data['pBall'])))
    data['decScore'] = (np.where(data['decision'] == 1, data['swingRv'], data['TakeScore']) - data['xPitchScore'])

    # Calculated xEV and EV above expected
    data['xEV'] = models['xEV'].predict(data[models['xEV' + "_features"]])
    data['xEV'] = np.where(data['xEV'] > 0, data['xEV'], 0)
    data['xEV'] = np.where(data['xEV'] < 120, data['xEV'], 120)
    data['xEV'] = np.where(pd.isna(data['launch_speed']), np.nan, data['xEV'])
    data['delta_ev'] = data['launch_speed'] - data['xEV']

    # project bat speed from normalizing delta_ev, mean is 72 and 1 std is 6.5
    data['pBat_speed'] = (data['delta_ev'] - data['delta_ev'].mean()) / data['delta_ev'].std() * 6.5 + 72

    # Calculate contact score
    data['xCon'] = models['Bat to Ball'].predict_proba(data[models['Bat to Ball' + "_features"]])[:, 1]
    data['conScore'] = np.where(data['decision'] == 1, data['contact'] - data['xCon'], np.nan)

    # 1 + (Exit Velocity – Bat Speed)/(Pitch Speed + Bat Speed)
    data['smash_factor'] = 1 + (data['launch_speed'] - data['pBat_speed']) / (
                data['release_speed'] + data['pBat_speed'])

    # Smash Factor is 0 for whiffs and fouls
    data['smash_factor'] = np.where(data['result'].isin(['foul', 'swinging_strike', 'swinging_strike_blocked', 'foul_tip']), 0,
                                    data['smash_factor'])

    # Calculate the 95th percentile of a players bat speed and the std of swings between their 90th and max bat speed and EV
    data['EV95'] = data['batter'].map(data.groupby('batter')['launch_speed'].quantile(0.95))
    data['95th_pBat_speed'] = data['batter'].map(data.groupby('batter')['pBat_speed'].quantile(0.95))

    # Calculate 90th and max bat speed for each player
    data['90th_EV'] = data.groupby('batter')['launch_speed'].transform(lambda x: x.quantile(0.90))
    data['90th_pBat_speed'] = data.groupby('batter')['pBat_speed'].transform(lambda x: x.quantile(0.90))

    data['max_EV'] = data.groupby('batter')['launch_speed'].transform('max')
    data['max_pBat_speed'] = data.groupby('batter')['pBat_speed'].transform('max')

    # Filter swings between 90th and max bat speed for each player
    mask = (data['pBat_speed'] >= data['90th_pBat_speed']) & (data['pBat_speed'] <= data['max_pBat_speed'])

    # Calculate the standard deviation of bat speed for swings between 90th and max for each player
    data['std_pBat_speed'] = data[mask].groupby('batter')['pBat_speed'].transform('std')

    # Filter swings between 90th and max EV for each player
    mask = (data['launch_speed'] >= data['90th_EV']) & (data['launch_speed'] <= data['max_EV'])
    data['std_EV'] = data[mask].groupby('batter')['launch_speed'].transform('std')

    # Aggregate the data by batter
    grades = data.groupby('batter').agg(
        {'decScore': 'mean', 'pBat_speed': 'mean', 'smash_factor': 'mean', '95th_pBat_speed': 'mean',
         'std_pBat_speed': 'mean', 'conScore': 'mean', 'xRV': 'mean', 'std_EV':'mean', 'EV95': 'mean',
         'count': 'count'}).sort_values('decScore', ascending=False)

    # Get sprint speed from statcast
    sprint_grade = pyb.statcast_sprint_speed(year, 0)
    grades['sprint_speed'] = grades.index.map(sprint_grade.set_index('player_id')['hp_to_1b'])
    grades['sprint_speed'] = grades['sprint_speed'].fillna(grades['sprint_speed'].mean())

    # Standardize grades with a minimum of q swings
    qualifiers = grades.query('count >= @q').copy()

    # Make decGrade mean 50 and std 10
    grades['decGrade'] = (grades['decScore'] - qualifiers['decScore'].mean()) / qualifiers['decScore'].std() * 10 + 50
    grades['powGrade'] = (grades['pBat_speed'] - qualifiers['pBat_speed'].mean()) / qualifiers['pBat_speed'].std() * 10 + 50
    grades['altPowGrade'] = (grades['std_pBat_speed'] - qualifiers['std_pBat_speed'].mean()) / qualifiers['std_pBat_speed'].std() * 10 + 50
    grades['conGrade'] = (grades['conScore'] - qualifiers['conScore'].mean()) / qualifiers['conScore'].std() * 10 + 50
    grades['SFGrade'] = (grades['smash_factor'] - qualifiers['smash_factor'].mean()) / qualifiers['smash_factor'].std() * 10 + 50
    grades['speedGrade'] = (grades['sprint_speed'] - qualifiers['sprint_speed'].mean()) / qualifiers['sprint_speed'].std() * 10 + 50
    grades['95thPowGrade'] = (grades['95th_pBat_speed'] - qualifiers['95th_pBat_speed'].mean()) / qualifiers['95th_pBat_speed'].std() * 10 + 50
    grades['EV95Grade'] = (grades['EV95'] - qualifiers['EV95'].mean()) / qualifiers['EV95'].std() * 10 + 50
    grades['stdEVGrade'] = (grades['std_EV'] - qualifiers['std_EV'].mean()) / qualifiers['std_EV'].std() * 10 + 50

    batter_names = pyb.playerid_reverse_lookup(grades.index, key_type='mlbam')

    batter_names['name'] = (batter_names['name_first'].str.title() + ' ' + batter_names['name_last'].str.title())
    batter_names.set_index('key_mlbam', inplace=True)

    grades['Name'] = grades.index.map(batter_names['name'])

    grades['IDfg'] = grades.index.map(batter_names['key_fangraphs'])

    grades.set_index('Name', inplace=True)

    return grades, data

def get_grades_args() -> argparse.Namespace:
    """
    Get the arguments from the command line

    :return: Namespace of the arguments
    """

    parser = argparse.ArgumentParser(description="Get grades for players")

    parser.add_argument("--suffix", type=str, help="Suffix for the output files", required=True)
    parser.add_argument("--start_date", type=str, help="Start date for grades", required=True)
    parser.add_argument("--end_date", type=str, help="End date for grades", required=True)
    parser.add_argument("--year", type=int, help="Year to pull sprint speed data", required=True)
    parser.add_argument("--q", type=int, help="Minimum number of swings to be considered when standardizing grades", default=1000)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_grades_args()

    print("Pulling data...")
    data = pull_data(args.start_date, args.end_date, GAME_TYPES)

    print("Formatting data...")
    data = format_data(data)

    print("Loading models...")
    models = load_models()

    print("Calculating grades...")
    grades, data = get_grades(data, models, args.year, args.q)

    print("Saving grades...")
    grades.to_csv(f"results/grades_{args.suffix}.csv")
    data.to_csv(f"results/data_{args.suffix}.csv")
