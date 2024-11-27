# We have 4 tools to grade. Bat to Ball, Contact Quality, Power, and Swing Decision
# We will use Bayesian methods to grade each tool, using the lightgbm model as the prior

from constants import RANDOM_STATE
import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import pickle as pkl
from pymc.gp.mean import Zero
from sklearn.preprocessing import StandardScaler

class PredictionMean:
    """Mean function wrapper for predictions"""
    def __init__(self, predictions):
        self.predictions = predictions
    
    def __call__(self, X):
        return self.predictions

def bat_to_ball_tool(data: pd.DataFrame, model: dict) -> Tuple[az.InferenceData, pm.Model, Dict]:
    """
    Calculate grades for bat to ball tool using PyMC
    :param data: cleaned statcast data
    :param model: dictionary of model, feature, and target
    :return: Posterior distribution and PyMC model
    """
    # Get the features and lightgbm model
    features = model['features']
    target = model['target']
    lgb_model = pkl.load(open('models/Bat to Ball_lgbm.pkl', 'rb'))
    X = data[features].values
    y = data[target].values

    # Get lightgbm predictions
    lgb_predictions = lgb_model.predict(X)

    # Standardize features and target
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    with pm.Model() as bat_to_ball_model:
        # Length scale and amplitude priors for GP kernel
        ls = pm.Gamma('ls', alpha=2, beta=1)
        eta = pm.HalfNormal('eta', sigma=1)
        
        # Define GP kernel (RBF/squared exponential)
        cov_func = eta ** 2 * pm.gp.cov.ExpQuad(input_dim=X_scaled.shape[1], ls=ls)
        
        # GP prior using lightgbm predictions as mean function
        mean_func = PredictionMean(lgb_predictions)
        gp = pm.gp.Marginal(cov_func=cov_func, mean_func=mean_func)
        
        # Add observation noise
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Likelihood (use scaled y)
        y_ = gp.marginal_likelihood('y', X=X_scaled, y=y_scaled, sigma=sigma)
        
        # Sample from the posterior
        trace = pm.sample(500, tune=250, random_seed=RANDOM_STATE, cores=2)

    # Convert to ArviZ InferenceData
    posterior = az.from_pymc(trace)

    # Record trace plot
    az.plot_trace(posterior)
    plt.savefig('/fig/bat_to_ball_trace.png')

    return posterior, bat_to_ball_model, {'X_scaler': X_scaler, 'y_scaler': y_scaler}


def contact_quality_tool(data: pd.DataFrame, model: dict) -> Tuple[az.InferenceData, pm.Model, Dict]:
    """
    Create a PyMC model for contact quality tool using Gaussian Process regression
    :param data: cleaned statcast data
    :param model: dictionary of model, feature, and target
    :return: Posterior distribution and PyMC model
    """
    # Get the features and lightgbm model
    features = model['features']
    target = model['target']
    lgb_model = pkl.load(open('models/Outcome Probability_lgbm.pkl', 'rb'))
    X = data[features].values
    y = data[target].values

    # Get lightgbm predictions
    lgb_predictions = lgb_model.predict(X)

    # Standardize features and target
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    with pm.Model() as contact_quality_model:
        # Length scale and amplitude priors for GP kernel
        ls = pm.Gamma('ls', alpha=2, beta=1)
        eta = pm.HalfNormal('eta', sigma=1)
        
        # Define GP kernel (RBF/squared exponential)
        cov_func = eta ** 2 * pm.gp.cov.ExpQuad(input_dim=X_scaled.shape[1], ls=ls)
        
        # GP prior using lightgbm predictions as mean function
        mean_func = PredictionMean(lgb_predictions)
        gp = pm.gp.Marginal(cov_func=cov_func, mean_func=mean_func)
        
        # Add observation noise
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Likelihood
        y_ = gp.marginal_likelihood('y', X=X_scaled, y=y_scaled, sigma=sigma)
        
        # Sample from the posterior
        trace = pm.sample(500, tune=250, random_seed=RANDOM_STATE, cores=2)

    # Convert to ArviZ InferenceData
    posterior = az.from_pymc(trace)

    # Record trace plot
    az.plot_trace(posterior)
    plt.savefig('/fig/contact_quality_trace.png')

    return posterior, contact_quality_model, {'X_scaler': X_scaler, 'y_scaler': y_scaler}

def power_tool(data: pd.DataFrame, model: dict) -> Tuple[az.InferenceData, pm.Model, Dict]:
    """
    Create a PyMC model for power tool using Gaussian Process regression
    
    :param data: cleaned statcast data
    :param model: dictionary of model, feature, and target
    :return: Posterior distribution and PyMC model
    """
    # Get the features and lightgbm model
    features = model['features']
    target = model['target']
    lgb_model = pkl.load(open('models/xEV_lgbm.pkl', 'rb'))
    X = data[features].values
    y = data[target].values

    # Get lightgbm predictions
    lgb_predictions = lgb_model.predict(X)

    # Standardize features and target
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    with pm.Model() as power_model:
        # Kernel hyperparameters
        ls_rbf = pm.Gamma('ls_rbf', alpha=2, beta=1)
        eta_rbf = pm.HalfNormal('eta_rbf', sigma=1)
        eta_linear = pm.HalfNormal('eta_linear', sigma=1)
        
        # Add scaling parameter for Linear kernel
        c = pm.HalfNormal('c', sigma=1)
        
        # Combined kernel: RBF + Linear (good for power metrics that may have both local and global trends)
        rbf_kernel = eta_rbf ** 2 * pm.gp.cov.ExpQuad(input_dim=X_scaled.shape[1], ls=ls_rbf)
        linear_kernel = eta_linear ** 2 * pm.gp.cov.Linear(input_dim=X_scaled.shape[1], c=c)
        cov_func = rbf_kernel + linear_kernel
        
        # GP prior with lightgbm predictions as mean function
        mean_func = PredictionMean(lgb_predictions)
        gp = pm.gp.Marginal(cov_func=cov_func, mean_func=mean_func)
        
        # Add observation noise with gamma prior (power metrics often have heteroscedastic noise)
        sigma = pm.Gamma('sigma', alpha=2, beta=1)
        
        # Likelihood
        y_ = gp.marginal_likelihood('y', X=X_scaled, y=y_scaled, sigma=sigma)
        
        # Sample from the posterior
        trace = pm.sample(500, tune=250, random_seed=RANDOM_STATE, cores=2)

    # Convert to ArviZ InferenceData
    posterior = az.from_pymc(trace)

    # Record trace plot
    az.plot_trace(posterior)
    plt.savefig('/fig/power_trace.png')

    return posterior, power_model, {'X_scaler': X_scaler, 'y_scaler': y_scaler}

def swing_decision_tool(data: pd.DataFrame, model: dict) -> Tuple[az.InferenceData, pm.Model, Dict]:
    """
    Create a PyMC model for swing decision tool using Gaussian Process regression
    
    :param data: cleaned statcast data
    :param model: dictionary of model, feature, and target
    :return: Posterior distribution and PyMC model
    """
    # Get the features and lightgbm model
    features = model['features']
    target = model['target']
    lgb_model = pkl.load(open('models/Swing Decision_lgbm.pkl', 'rb'))
    X = data[features].values
    y = data[target].values

    # Get lightgbm predictions
    lgb_predictions = lgb_model.predict(X)

    # Standardize features and target
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    with pm.Model() as swing_decision_model:
        # Length scale and amplitude priors for GP kernel
        ls = pm.Gamma('ls', alpha=2, beta=1)
        eta = pm.HalfNormal('eta', sigma=1)
        
        # Define GP kernel (RBF/squared exponential)
        cov_func = eta ** 2 * pm.gp.cov.ExpQuad(input_dim=X_scaled.shape[1], ls=ls)
        
        # GP prior using lightgbm predictions as mean function
        mean_func = PredictionMean(lgb_predictions)
        gp = pm.gp.Marginal(cov_func=cov_func, mean_func=mean_func)
        
        # Add observation noise
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Likelihood
        y_ = gp.marginal_likelihood('y', X=X_scaled, y=y_scaled, sigma=sigma)
        
        # Sample from the posterior
        trace = pm.sample(500, tune=250, random_seed=RANDOM_STATE, cores=2)

    # Convert to ArviZ InferenceData
    posterior = az.from_pymc(trace)

    # Record trace plot
    az.plot_trace(posterior)
    plt.savefig('/fig/swing_decision_trace.png')

    return posterior, swing_decision_model, {'X_scaler': X_scaler, 'y_scaler': y_scaler}