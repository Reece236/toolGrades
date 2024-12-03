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

def generate_tool_samples(grades: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
    """
    Generate MCMC samples with optimized parallel sampling
    """
    tool_metrics = ['decScore', 'powScore', 'prepScore', 'conScore']
    valid_players = grades.dropna(subset=[f'{m}_bayes' for m in tool_metrics] + 
                                       [f'{m}_ci_lower' for m in tool_metrics] + 
                                       [f'{m}_ci_upper' for m in tool_metrics])
    
    samples_list = []
    failed_players = 0
    max_failures = len(valid_players) * 0.2
    
    warnings.filterwarnings('ignore', category=UserWarning)
    
    pbar = tqdm.tqdm(valid_players.iterrows(), total=len(valid_players), 
                     desc="Processing players", 
                     postfix={'failed': 0, 'success_rate': '100%'})
    
    for idx, player in pbar:
        try:
            with pm.Model() as model:
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
                        lower=max(lower, mean - 4*std),  # Limit extreme values
                        upper=min(upper, mean + 4*std)
                    )
                
                # Improved sampling parameters for better ESS
                trace = pm.sample(
                    draws=500,          # Increased from 100
                    tune=2000,          # Increased tuning period
                    chains=4,
                    cores=2,
                    progressbar=False,
                    return_inferencedata=True,
                    compute_convergence_checks=True,
                    target_accept=0.85,  # Slightly reduced for better mixing
                    init='advi',         # Changed initialization
                    random_seed=RANDOM_STATE
                )
                
                summary = az.summary(trace)
                r_hat = summary['r_hat'].max()
                n_eff = summary['ess_bulk'].min()
                divergences = trace.sample_stats.diverging.sum()
                
                # Stricter convergence criteria
                if r_hat > 1.03 or n_eff < 400 or divergences > n_samples * 0.005:
                    failed_players += 1
                    pbar.set_postfix({'failed': failed_players, 
                                    'success_rate': f'{(1 - failed_players/pbar.n)*100:.1f}%',
                                    'min_ess': f'{n_eff:.0f}'})
                    continue
                
                # Sample extraction - take every nth sample to reduce autocorrelation
                thin = max(1, trace.posterior.sizes["draw"] // n_samples)
                for chain in range(trace.posterior.sizes["chain"]):
                    for i in range(0, trace.posterior.sizes["draw"], thin):
                        if len(samples_list) >= n_samples * 2:  # Ensure we don't exceed desired samples
                            break
                        sample = {'Name': idx}
                        for metric in tool_metrics:
                            sample[f'{metric}_grade'] = trace.posterior[metric][chain, i].item()
                        
                        sample['speedGrade'] = player['speedGrade'] - (player['speedGrade'] % 5)
                        if OVR_METRIC in valid_players.columns:
                            sample[OVR_METRIC] = player[OVR_METRIC]
                        
                        samples_list.append(sample)
                
        except Exception as e:
            failed_players += 1
            pbar.set_postfix({'failed': failed_players, 
                            'success_rate': f'{(1 - failed_players/pbar.n)*100:.1f}%'})
            if failed_players >= max_failures:
                raise RuntimeError(f"Too many sampling failures: {failed_players}/{pbar.n}")
            continue
    
    if not samples_list:
        raise RuntimeError("No valid samples generated")
    
    return pd.DataFrame(samples_list)

def train_overall_model(samples: pd.DataFrame, metric: str) -> lgb.LGBMRegressor:
    """
    Train LightGBM model to predict target metric from tool grades
    """
    features = [col for col in samples.columns if col.endswith('_grade') or col.endswith('Grade')]
    X = samples[features]
    y = samples[metric]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    params = {
        'n_estimators': 500,
        'max_depth': 10,
        'learning_rate': 0.05,
        'verbosity': -1,
        'random_state': RANDOM_STATE
    }
    
    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
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
        stats = stats.query('PA > @QUALIFIER/20')
    except:
        print(f'No stats found for {OVR_YEAR}')
        return
    
    
    print('Calculating grades...')
    grades[OVR_METRIC] = grades['IDfg'].map(stats.set_index('PlayerId')[OVR_METRIC])
    
    print('Generating MCMC samples...')
    samples = generate_tool_samples(grades)

    samples.to_csv('samples.csv')
    
    print(f"Training {OVR_METRIC} prediction model...")
    ovr_model = train_overall_model(samples, OVR_METRIC)
    
    print("Saving model...")
    with open(f"models/{OVR_METRIC}_predictor.pkl", "wb") as f:
        pickle.dump(ovr_model, f)

if __name__ == "__main__":
    main()