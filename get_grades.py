"""
I mean it's called `get_grades`
"""
import numpy as np
import pandas as pd
import pickle
from constants import TOOL_INFO, GAME_TYPES
import pybaseball as pyb
import argparse
from etl import pull_data, format_data
import pymc as pm
import arviz as az

def load_models(args) -> dict:
    """
    Load the models and other stuff needed for each tool

    :return: Dictionary of the models and friends
    """

    models = {}

    for info in TOOL_INFO.items():

        print(f"Loading {info[0]} model")

        with open(f"models/{info[0]}_lgbm.pkl", "rb") as f:
            model = pickle.load(f)
            models[info[0]] = model

        with open(f"models/{info[0]}_features.pkl", "rb") as f:
            features = pickle.load(f)
            models[info[0] + "_features"] = features

    return models

def estimate_true_power(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate true power potential using hierarchical model for exit velocities"""
    power_data = data[['batter', 'powScore']].dropna()
    
    # Only include players with enough observations
    valid_players = power_data.groupby('batter').size()
    valid_players = valid_players[valid_players >= 10].index
    
    player_stats = power_data[power_data['batter'].isin(valid_players)].groupby('batter').agg({
        'powScore': ['mean', 'std', 'count', 
                    lambda x: np.percentile(x, 95)]
    })
    player_stats.columns = ['mean', 'std', 'n', 'p95']
    
    # Standardize for numerical stability
    overall_mean = player_stats['mean'].mean()
    overall_std = player_stats['mean'].std()
    player_stats['mean_standardized'] = (player_stats['mean'] - overall_mean) / overall_std

    # Get 20-80 scale of player stats
    player_stats['grade'] = (player_stats['mean_standardized'] - player_stats['mean_standardized'].mean()) / player_stats['mean_standardized'].std() * 10 + 50

    return pd.Series(player_stats['grade'], index=valid_players)

def get_grades(data: pd.DataFrame, models: dict, year: int, q:int) -> pd.DataFrame:
    """
    Calculate grades
    :param data: cleaned statcast data
    :param models: dictionary of models and features
    :param year: year to pull sprint speed data
    :param q: minimum number of swings to be considered when standardizing grades
    :return: DataFrame of player grades
    """

    # Add swing probability - calibration is already built in
    data['pSwing'] = models['Swing Decision'].predict_proba(data[models['Swing Decision' + "_features"]])[:, 1]
    
    # Calculate run value for each event
    data['swingRv'] = models['Outcome Probability'].predict(data[models['Outcome Probability' + "_features"]])

    # Add Ball and Strike Values
    strikes = data.query('result == "called_strike"')
    balls = data.query('result == "ball"')

    data['ballRv'] = data['count'].map(balls.groupby('count')['delta_run_exp'].mean())

    data['strikeRv'] = data['count'].map(strikes.groupby('count')['delta_run_exp'].mean())

    # Calculate swing/called strike/ball probabilities and turn into decision score
    data['pSwing'] = models['Swing Decision'].predict_proba(data[models['Swing Decision' + "_features"]])[:, 1]
    data['pStrike'] = (models['Strike Probability'].predict_proba(data[models['Strike Probability' + "_features"]])[:, 1]) * (1-data['pSwing'])
    data['pBall'] = 1 - data['pSwing'] - data['pStrike']
    
    data['xPitchScore'] = data['swingRv'] * data['pSwing'] + data['strikeRv'] * data['pStrike'] + data['ballRv'] * data['pBall']
    
    take_denom = data['pStrike'] + data['pBall']
    data['TakeScore'] = np.where(
        take_denom > 0,
        (data['strikeRv'] * data['pStrike'] + data['ballRv'] * data['pBall']) / take_denom,
        0 
    )
    
    # Calculate final decision score
    data['decScore'] = np.where(data['decision'] == 1, data['swingRv'], data['TakeScore']) - data['xPitchScore']

    # Calculated xEV and EV above expected
    data['xEV'] = models['xEV'].predict(data[models['xEV' + "_features"]])
    data['xEV'] = np.where(data['xEV'] > 0, data['xEV'], 0)
    data['xEV'] = np.where(data['xEV'] < 120, data['xEV'], 120)
    data['xEV'] = np.where(pd.isna(data['launch_speed']), np.nan, data['xEV'])
    data['powScore'] = data['launch_speed'] - data['xEV']

    # project bat speed from normalizing delta_ev, mean is 72 and 1 std is 6.5
    data['pBat_speed'] = (data['powScore'] - data['powScore'].mean()) / data['powScore'].std() * 6.5 + 72

    # Calculate contact score
    data['xCon'] = models['Bat to Ball'].predict_proba(data[models['Bat to Ball' + "_features"]])[:, 1]
    data['conScore'] = np.where(data['decision'] == 1, data['contact'] - data['xCon'], np.nan)

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
    data['prepScore'] = data[mask].groupby('batter')['powScore'].transform('std')

    # Aggregate the data by batter
    grades = data.groupby('batter').agg(
        {'decScore': 'mean', 'pBat_speed': 'mean', '95th_pBat_speed': 'mean',
         'std_pBat_speed': 'mean', 'conScore': 'mean', 'xRV': 'mean', 'EV95': 'mean',
         'powScore':'mean', 'prepScore':'mean', 'count': 'count'}).sort_values('decScore', ascending=False)

    # Get sprint speed from statcast
    sprint_grade = pyb.statcast_sprint_speed(year, 0)
    grades['sprint_speed'] = grades.index.map(sprint_grade.set_index('player_id')['hp_to_1b'])
    grades['sprint_speed'] = grades['sprint_speed'].fillna(grades['sprint_speed'].mean())

    # Standardize grades with a minimum of q swings
    qualifiers = grades.query('count >= @q').copy()

    # Make decGrade mean 50 and std 10
    grades['decGrade'] = (grades['decScore'] - qualifiers['decScore'].mean()) / qualifiers['decScore'].std() * 10 + 50
    grades['mPowGrade'] = (grades['powScore'] - qualifiers['powScore'].mean()) / qualifiers['powScore'].std() * 10 + 50
    grades['prepScore'] = (qualifiers['prepScore'].mean() - grades['prepScore']) / qualifiers['prepScore'].std() * 10 + 50
    grades['altPowGrade'] = (grades['std_pBat_speed'] - qualifiers['std_pBat_speed'].mean()) / qualifiers['std_pBat_speed'].std() * 10 + 50
    grades['conGrade'] = (grades['conScore'] - qualifiers['conScore'].mean()) / qualifiers['conScore'].std() * 10 + 50
    grades['speedGrade'] = (grades['sprint_speed'] - qualifiers['sprint_speed'].mean()) / qualifiers['sprint_speed'].std() * 10 + 50
    grades['powGrade'] = (grades['95th_pBat_speed'] - qualifiers['95th_pBat_speed'].mean()) / qualifiers['95th_pBat_speed'].std() * 10 + 50
    grades['EV95Grade'] = (grades['EV95'] - qualifiers['EV95'].mean()) / qualifiers['EV95'].std() * 10 + 50

    # Define league-wide priors based on historical data
    league_priors = {
        'decScore': {'mean': 0, 'std': 0.015},
        'powScore': {'mean': .5, 'std':1.5},
        'prepScore': {'mean': 50, 'std': 21.5},
        'conScore': {'mean': 0, 'std': 0.09}
    }

    # Replace direct 95th percentile calculation with Bayesian estimate
    print("Estimating true power potential...")
    true_power = estimate_true_power(data)
    grades['powScore'] = grades.index.map(true_power)

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
    parser.add_argument("--ovr_model", type=str, help="Calculate overall grades", default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_grades_args()

    print("Pulling data...")
    data = pull_data(args.start_date, args.end_date, GAME_TYPES)

    print("Formatting data...")
    data = format_data(data)

    print("Loading models...")
    models = load_models(args)

    print("Calculating grades...")
    grades, data = get_grades(data, models, args.year, args.q)

    if args.ovr_model is not None:
        print("Calculating overall grade predictions...")

        with open(f"models/{args.ovr_model}_predictor.pkl", "rb") as f:
            ovr_model = pickle.load(f)
            
        features = ['decScore', 'powScore', 'prepScore', 'conScore', 'speedGrade']
        X = grades[features]
        grades['OVRGrade'] = ovr_model.predict(X)
    else:
        grades['OVRGrade'] = np.nan
    print("Saving grades...")
    grades.to_csv(f"results/grades_{args.suffix}.csv")
    data.to_csv(f"results/data_{args.suffix}.csv")

    print("Saved! We ballin'")