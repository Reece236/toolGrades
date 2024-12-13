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
    
    for metric in ['decScore', 'powScore', 'prepScore', 'conScore']:
        
        with pm.Model() as model:
            # Population parameters
            mu = pm.Normal('mu', mu=league_priors[metric]['mean'], sigma=league_priors[metric]['std'])
            sigma = pm.HalfNormal('sigma', sigma=league_priors[metric]['std'])
            
            # Player-specific parameters
            player_mu = pm.Normal('player_mu', mu=mu, sigma=sigma, shape=len(data['batter'].unique()))
            
            # Likelihood
            obs = pm.Normal(metric, mu=player_mu, sigma=sigma, observed=data.groupby('batter')[metric].mean())
            
            # Sample
            trace = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True)

        # Extract posterior samples
        posterior_samples = trace.posterior['player_mu'].values.reshape(-1, len(data['batter'].unique()))

        # Calculate 95% credible intervals
        ci_lower = np.percentile(posterior_samples, 2.5, axis=0)
        ci_upper = np.percentile(posterior_samples, 97.5, axis=0)

        # Calculate Bayesian grades
        bayes_grade = standardize_to_20_80(data.groupby('batter')[metric].mean())

        bayesian_grades[f'{metric}_ci_lower'] = ci_lower
        bayesian_grades[f'{metric}_ci_upper'] = ci_upper
        bayesian_grades[f'{metric}_bayes_grade'] = bayes_grade

    return bayesian_grades

def estimate_true_power(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate true power potential using hierarchical model for exit velocities"""
    power_data = data[['batter', 'powScore']].dropna()
    
    # Only include players with enough observations
    valid_players = power_data.groupby('batter').size()
    valid_players = valid_players[valid_players >= 10].index
    
    player_stats = power_data[power_data['batter'].isin(valid_players)].groupby('batter').agg({
        'powScore': ['mean', 'std', 'count', 
                    lambda x: np.percentile(x, 95)]  # Observed 95th
    })
    player_stats.columns = ['mean', 'std', 'n', 'p95']
    
    # Standardize for numerical stability
    overall_mean = player_stats['mean'].mean()
    overall_std = player_stats['mean'].std()
    player_stats['mean_standardized'] = (player_stats['mean'] - overall_mean) / overall_std
    
    with pm.Model() as model:
        # Population parameters
        mu = pm.Normal('mu', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Player-specific parameters
        player_mu = pm.Normal('player_mu', 
                            mu=mu, 
                            sigma=sigma,
                            shape=len(valid_players))
        
        # Player-specific scale (for fat tails)
        player_scale = pm.HalfNormal('player_scale',
                                   sigma=1,
                                   shape=len(valid_players))
        
        # Student's T distribution for fat tails
        obs = pm.StudentT('obs',
                         nu=3,  # Degrees of freedom
                         mu=player_mu,
                         sigma=player_scale,
                         observed=player_stats['mean_standardized'])
        
        # Sample
        trace = pm.sample(2000, 
                         tune=1000,
                         target_accept=0.9,
                         return_inferencedata=True)
    
    # Extract posterior predictions
    posterior_samples = trace.posterior['player_mu'].values.reshape(-1, len(valid_players))
    posterior_scales = trace.posterior['player_scale'].values.reshape(-1, len(valid_players))
    
    # Calculate 95th percentile estimates
    true_95th = np.zeros(len(valid_players))
    for i in range(len(valid_players)):
        samples = np.random.standard_t(df=3, size=10000)
        samples = samples * posterior_scales[:100, i].mean() + posterior_samples[:100, i].mean()
        true_95th[i] = np.percentile(samples, 95)
    
    # Transform back to original scale
    true_95th = true_95th * overall_std + overall_mean
    
    return pd.Series(true_95th, index=valid_players)

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

    # Calculate Bayesian grades
    bayesian_estimates = calculate_bayesian_grades(data, league_priors)
    
    # Add Bayesian estimates to grades DataFrame
    grades = grades.join(bayesian_estimates)
    
    # Add confidence intervals and standardized Bayesian grades
    for metric in ['decScore', 'powScore', 'prepScore', 'conScore']:
        grades[f'{metric}_ci_lower'] = bayesian_estimates[f'{metric}_ci_lower']
        grades[f'{metric}_ci_upper'] = bayesian_estimates[f'{metric}_ci_upper']
        grades[f'{metric}_bayes_grade'] = bayesian_estimates[f'{metric}_bayes_grade']

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
            
        features = ['decScore_bayes', 'powScore_bayes', 'prepScore_bayes', 'conScore_bayes', 'speedGrade']
        X = grades[features]
        grades['OVRGrade'] = ovr_model.predict(X)
    else:
        grades['OVRGrade'] = np.nan
    print("Saving grades...")
    grades.to_csv(f"results/grades_{args.suffix}.csv")
    data.to_csv(f"results/data_{args.suffix}.csv")

    print("Saved! We ballin'")