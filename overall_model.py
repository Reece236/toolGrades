"""
Train LGB Classifier to predict second half results for a given statistic
"""

import lightgbm as lgb
import pandas as pd
from constants import OVR_YEAR, OVR_METRIC, QUALIFIER, RANDOM_STATE, TEST_SIZE
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tqdm
import pymc as pm
import arviz as az

def generate_tool_samples(grades: pd.DataFrame, n_samples: int = 200) -> pd.DataFrame:
    """
    Generate samples for each tool
    """
    tool_metrics = ['decScore', 'powScore', 'prepScore', 'conScore']
    required_columns = [f'{m}_grade' for m in tool_metrics] + \
                       [f'{m}_lower' for m in tool_metrics] + \
                       [f'{m}_upper' for m in tool_metrics]
    valid_players = grades.dropna(subset=required_columns)

    samples_list = []
    for idx, player in valid_players.iterrows():
        with pm.Model() as model:
            vars_dict = {}
            for metric in tool_metrics:
                mean = float(player[f'{metric}_grade'])
                lower = float(player[f'{metric}_lower'])
                upper = float(player[f'{metric}_upper'])
                std = max((upper - lower) / (2 * 1.96), 0.001)
                vars_dict[metric] = pm.TruncatedNormal(
                    metric,
                    mu=mean,
                    sigma=std,
                    lower=max(lower, mean - 4*std),
                    upper=min(upper, mean + 4*std)
                )
            trace = pm.sample(
                draws=2000,
                tune=1000,
                chains=4,
                cores=2,
                return_inferencedata=True,
                target_accept=0.95
            )

            summary = az.summary(trace)
            convergence = (
                np.all(summary['r_hat'] < 1.05) and
                np.all(summary['ess_bulk'] > 400) and
                np.all(summary['ess_tail'] > 400) and
                trace.sample_stats.diverging.sum() == 0
            )
            if not convergence:
                raise RuntimeError("Convergence failed")

            chain_samples = []
            for metric in tool_metrics:
                samples = trace.posterior[metric].values.flatten()
                indices = np.linspace(0, len(samples)-1, n_samples, dtype=int)
                chain_samples.append(samples[indices])

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
    return pd.DataFrame(samples_list)

def train_overall_model(samples: pd.DataFrame, metric: str) -> lgb.LGBMRegressor:
    """
    Train LightGBM model using sample means and standard deviations
    """

    player_stats = samples.groupby('Name').agg({
        'decScore_grade': ['mean', 'std'],
        'powScore_grade': ['mean', 'std'],
        'prepScore_grade': ['mean', 'std'],
        'conScore_grade': ['mean', 'std'],
        'speedGrade': 'first',
        metric: 'first'
    })
    
    player_stats.columns = ['dec_mean', 'dec_std', 
                          'pow_mean', 'pow_std',
                          'prep_mean', 'prep_std',
                          'con_mean', 'con_std',
                          'speed', 'target']
    
    # Calculate weights based on posterior uncertainty
    weights = 1 / (player_stats[['dec_std', 'pow_std', 'prep_std', 'con_std']].mean(axis=1) + 1e-6)
    weights = weights / weights.sum()
    
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
    
    samples = generate_tool_samples(grades, n_samples=1000)

    print(f"Training {OVR_METRIC} prediction model...")
    ovr_model = train_overall_model(samples, OVR_METRIC)
    
    print("Saving model...")
    with open(f"models/{OVR_METRIC}_predictor.pkl", "wb") as f:
        pickle.dump(ovr_model, f)

if __name__ == "__main__":
    main()