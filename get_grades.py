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

def load_models() -> dict:
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

    le = pickle.load(open('models/le.pkl', 'rb'))
    run_value = pd.read_csv('models/rv_table.csv', index_col=0)

    models['le'] = le
    models['run_value'] = run_value

    return models

def standardize_to_20_80(values: pd.Series) -> pd.Series:
    """
    Standardize values to a 20-80 scale where 50 is mean and 10 points = 1 standard deviation
    """
    return (values - values.mean()) / values.std() * 10 + 50

def calculate_bayesian_grades(data: pd.DataFrame, league_priors: dict) -> pd.DataFrame:
    """
    Calculate Bayesian estimates for player grades using league-wide priors
    """
    bayesian_grades = pd.DataFrame()
    
    for metric in ['decScore', '95th_pBat_speed', 'smash_factor', 'conScore']:
        # Get relevant data and filter out missing values
        metric_data = data[['batter', metric]].dropna()
        
        # Only include players with at least 10 valid observations
        valid_players = metric_data.groupby('batter').size()
        valid_players = valid_players[valid_players >= 10].index
        
        if len(valid_players) == 0:
            continue
            
        metric_data = metric_data[metric_data['batter'].isin(valid_players)]
        
        # Calculate stats and standardize data
        player_stats = metric_data.groupby('batter').agg({
            metric: ['mean', 'std', 'count']
        })
        player_stats.columns = ['mean', 'std', 'n']
        
        # Handle zero variance cases and standardize
        player_stats['std'] = player_stats['std'].fillna(0.001)
        player_stats['std'] = player_stats['std'].clip(lower=0.001)
        
        # Standardize means for numerical stability
        overall_mean = player_stats['mean'].mean()
        overall_std = player_stats['mean'].std()
        player_stats['mean_standardized'] = (player_stats['mean'] - overall_mean) / overall_std
        
        # Pre-calculate observation standard deviations
        obs_stds = np.sqrt((player_stats['std'].values**2 / player_stats['n'].values) / overall_std**2)
        obs_stds = np.clip(obs_stds, 0.001, None)
        
        # Adjust priors to standardized scale
        prior_mean = (league_priors[metric]['mean'] - overall_mean) / overall_std
        prior_std = league_priors[metric]['std'] / overall_std
        
        with pm.Model() as model:
            # Hierarchical priors with improved numerical stability
            mu = pm.Normal('mu', mu=prior_mean, sigma=max(prior_std, 0.1))
            sigma = pm.HalfNormal('sigma', sigma=max(prior_std, 0.1))
            
            # Player effects with stable variance
            player_effects = pm.Normal('player_effects',
                                     mu=mu,
                                     sigma=sigma,
                                     shape=len(valid_players))
            
            # Likelihood with pre-calculated observation standard deviations
            obs = pm.Normal('obs',
                          mu=player_effects,
                          sigma=obs_stds,
                          observed=player_stats['mean_standardized'])
            
            # Inference with increased tuning
            trace = pm.sample(2000, tune=2000, target_accept=0.9)
        
        # Transform results back to original scale and create grades
        summary = az.summary(trace, var_names=['player_effects'])
        raw_vals = summary['mean'].values * overall_std + overall_mean
        ci_lower = summary['hdi_3%'].values * overall_std + overall_mean
        ci_upper = summary['hdi_97%'].values * overall_std + overall_mean
        
        # Create grades from raw values
        grades = standardize_to_20_80(pd.Series(raw_vals))
        ci_lower_grade = standardize_to_20_80(pd.Series(ci_lower))
        ci_upper_grade = standardize_to_20_80(pd.Series(ci_upper))
        
        # Store both raw values and grades
        summary_df = pd.DataFrame({
            f'{metric}_bayes': raw_vals,
            f'{metric}_bayes_grade': grades,
            f'{metric}_ci_lower_grade': ci_lower_grade,
            f'{metric}_ci_upper_grade': ci_upper_grade
        }, index=valid_players)
        
        if bayesian_grades.empty:
            bayesian_grades = summary_df
        else:
            bayesian_grades = bayesian_grades.join(summary_df)
    
    return bayesian_grades

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

    # Calculate run value for each event
    for col in probs.columns:
        probs[col] = probs[col] * models['run_value'].iloc[int(col)]

    probs.fillna(0, inplace=True)

    # Calculate swing run value
    data['swingRv'] = list(probs.sum(axis=1))

    # Add Ball and Strike Values
    data['ballRv'] = data.loc[data['cResult'] == (data['count'] + 'ball')]['cRV'].mean()
    data['strikeRv'] = data.loc[data['cResult'] == (data['count'] + 'called_strike')]['cRV'].mean()

    # Calculate swing/called strike/ball probabilities and turn into decision score
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

    # Define league-wide priors based on historical data
    league_priors = {
        'decScore': {'mean': -0.45, 'std': 0.05},
        '95th_pBat_speed': {'mean': 79.0, 'std': 1.25},
        'smash_factor': {'mean': .40, 'std': 0.09},
        'conScore': {'mean': -0.15, 'std': 0.075}
    }
    
    # Calculate Bayesian grades
    bayesian_estimates = calculate_bayesian_grades(data, league_priors)

    grades.to_csv('results/pre_grades.csv')
    bayesian_estimates.to_csv('results/pre_baes.csv')
    
    # Add Bayesian estimates to grades DataFrame
    grades = grades.join(bayesian_estimates)
    
    # Add confidence intervals and standardized Bayesian grades
    for metric in ['decScore', '95th_pBat_speed', 'smash_factor', 'conScore']:
        grades[f'{metric}_ci_lower'] = bayesian_estimates[f'{metric}_ci_lower']
        grades[f'{metric}_ci_upper'] = bayesian_estimates[f'{metric}_ci_upper']
        grades[f'{metric}_bayes_grade'] = bayesian_estimates[f'{metric}_bayes_grade']

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