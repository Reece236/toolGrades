"""
Train LGB Classifier to predict second half results for a given statistic
"""

import lightgbm as lgb
import pandas as pd
import pybaseball as pyb
from etl import pull_data, format_data
from get_grades import get_grades, load_models
from constants import GAME_TYPES, OVR_YEAR, OVR_METRIC, QUALIFIER, REG_PARAMS, RANDOM_STATE, TEST_SIZE
import pickle
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import tqdm
import pymc as pm
import arviz as az
import warnings

def generate_tool_samples(grades: pd.DataFrame, n_samples: int = 200) -> pd.DataFrame:
    """Generate MCMC samples with proper chain handling - 200 samples per player"""
    tool_metrics = ['decScore', 'powScore', 'prepScore', 'conScore']
    valid_players = grades.dropna(subset=[f'{m}_bayes' for m in tool_metrics] + 
                                       [f'{m}_ci_lower' for m in tool_metrics] + 
                                       [f'{m}_ci_upper' for m in tool_metrics])
    
    samples_list = []
    failed_players = 0
    max_failures = len(valid_players) * 0.2
    samples_per_player = n_samples  # Force n_samples per player instead of dividing
    
    warnings.filterwarnings('ignore', category=UserWarning)
    
    pbar = tqdm.tqdm(valid_players.iterrows(), total=len(valid_players))
    
    for idx, player in pbar:
        try:
            with pm.Model() as model:
                # Set up variables
                vars_dict = {}
                for metric in tool_metrics:
                    mean = float(player[f'{metric}_bayes'])
                    lower = float(player[f'{metric}_ci_lower'])
                    upper = float(player[f'{metric}_ci_upper'])
                    std = max((upper - lower) / (2 * 1.96), 0.001)
                    
                    vars_dict[metric] = pm.TruncatedNormal(
                        metric,
                        mu=mean,
                        sigma=std,
                        lower=max(lower, mean - 4*std),
                        upper=min(upper, mean + 4*std)
                    )
                
                # Proper MCMC sampling - increase draws to ensure enough samples
                trace = pm.sample(
                    draws=2000,          # Increased draws to ensure enough samples
                    tune=1000,
                    chains=4,
                    cores=2,
                    return_inferencedata=True,
                    target_accept=0.95 
                )
                
                # More comprehensive convergence diagnostics
                summary = az.summary(trace)
                if not (
                    np.all(summary['r_hat'] < 1.05) and  # Stricter R-hat
                    np.all(summary['ess_bulk'] > 400) and # Check bulk ESS
                    np.all(summary['ess_tail'] > 400) and # Check tail ESS
                    trace.sample_stats.diverging.sum() == 0  # No divergences
                ):
                    failed_players += 1
                    continue
                
                # Ensure exactly n_samples per player
                chain_samples = []
                for metric in tool_metrics:
                    samples = trace.posterior[metric].values.flatten()
                    # Take evenly spaced samples to reach desired count
                    indices = np.linspace(0, len(samples)-1, n_samples, dtype=int)
                    chain_samples.append(samples[indices])
                
                # Create exactly n_samples samples per player
                for i in range(n_samples):
                    sample = {
                        'Name': idx,
                        'speedGrade': player['speedGrade'] - (player['speedGrade'] % 5)
                    }
                    for j, metric in enumerate(tool_metrics):
                        sample[f'{metric}_grade'] = chain_samples[j][i]
                    if OVR_METRIC in valid_players.columns:
                        sample[OVR_METRIC] = player[OVR_METRIC]
                    samples_list.append(sample)
                
        except Exception as e:
            failed_players += 1
            if failed_players >= max_failures:
                raise RuntimeError(f"Too many sampling failures: {failed_players}/{pbar.n}")
            continue

    if not samples_list:
        raise RuntimeError("No valid samples generated")
    
    return pd.DataFrame(samples_list)

def train_overall_model(samples: pd.DataFrame, metric: str) -> lgb.LGBMRegressor:
    """
    Train LightGBM model using posterior means and uncertainty-based sample weights
    """
    # Aggregate samples to get posterior means and standard deviations per player
    player_stats = samples.groupby('Name').agg({
        'decScore_grade': ['mean', 'std'],
        'powScore_grade': ['mean', 'std'],
        'prepScore_grade': ['mean', 'std'],
        'conScore_grade': ['mean', 'std'],
        'speedGrade': 'first',  # Speed grade is constant per player
        metric: 'first'  # Target metric is constant per player
    })
    
    # Flatten column names
    player_stats.columns = ['dec_mean', 'dec_std', 
                          'pow_mean', 'pow_std',
                          'prep_mean', 'prep_std',
                          'con_mean', 'con_std',
                          'speed', 'target']
    
    # Calculate weights based on posterior uncertainty
    # Higher weight for more certain predictions
    weights = 1 / (player_stats[['dec_std', 'pow_std', 'prep_std', 'con_std']].mean(axis=1) + 1e-6)
    weights = weights / weights.sum()  # Normalize weights
    
    # Prepare features using posterior means
    X = pd.DataFrame({
        'decScore_grade': player_stats['dec_mean'],
        'powGrade': player_stats['pow_mean'],
        'prepScore_grade': player_stats['prep_mean'],
        'conScore_grade': player_stats['con_mean'],
        'speedGrade': player_stats['speed']
    })
    y = player_stats['target']
    
    # Split incorporating weights
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    params = {
        'n_estimators': 500,
        'max_depth': 10,
        'learning_rate': 0.05,
        'verbosity': -1,
        'random_state': RANDOM_STATE
    }
    
    model = lgb.LGBMRegressor(**params)
    
    # Train with sample weights
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse'
    )
    
    return model

def main():
    
    print("Pulling data...")
    try:
        grades = pd.read_csv(f'results/grades_{OVR_YEAR}Grades.csv')
    except:
        print(f'No grades found for {OVR_YEAR}')
        return
    
    try:
        stats = pd.read_csv(f'data/secondhalf_splits/splits_{OVR_YEAR}.csv')
        stats = stats.query('PA > @QUALIFIER/4')
    except:
        print(f'No stats found for {OVR_YEAR}')
        return
    
    
    print('Calculating grades...')
    grades[OVR_METRIC] = grades['IDfg'].map(stats.set_index('PlayerId')[OVR_METRIC])
    grades = grades.dropna(subset=[OVR_METRIC])
    
    print('Generating MCMC samples...')
    #samples = generate_tool_samples(grades, n_samples=1000)  # More samples for better posterior estimates

    samples = pd.read_csv('samples.csv')

    # Save raw samples for analysis
    #samples.to_csv('samples.csv')
    
    print(f"Training {OVR_METRIC} prediction model...")
    ovr_model = train_overall_model(samples, OVR_METRIC)
    
    print("Saving model...")
    with open(f"models/{OVR_METRIC}_predictor.pkl", "wb") as f:
        pickle.dump(ovr_model, f)

if __name__ == "__main__":
    main()