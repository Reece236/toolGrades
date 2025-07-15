import pymc as pm
import numpy as np
import pandas as pd


def hierarchical_normal(data: pd.DataFrame, metric: str, prior_mean: float, prior_std: float):
    """Fit hierarchical normal model for a metric"""
    players = data['batter'].unique()
    player_index = {p: i for i, p in enumerate(players)}
    idx = data['batter'].map(player_index).values
    values = data[metric].values

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=prior_mean, sigma=prior_std * 2)
        tau = pm.HalfNormal('tau', sigma=prior_std)
        sigma = pm.HalfNormal('sigma', sigma=prior_std)
        player_mu = pm.Normal('player_mu', mu=mu, sigma=tau, shape=len(players))
        pm.Normal('obs', mu=player_mu[idx], sigma=sigma, observed=values)
        trace = pm.sample(1000, tune=1000, chains=2, cores=1,
                          target_accept=0.9,
                          return_inferencedata=True)

    samples = trace.posterior['player_mu'].values.reshape(-1, len(players))
    mean = samples.mean(axis=0)
    ci_lower = np.percentile(samples, 2.5, axis=0)
    ci_upper = np.percentile(samples, 97.5, axis=0)

    return pd.DataFrame({
        'batter': players,
        f'{metric}_bayes': mean,
        f'{metric}_ci_lower': ci_lower,
        f'{metric}_ci_upper': ci_upper
    }).set_index('batter')
