"""
Idk, Goal is to objectify hitter tool grades and do so in a way that is independent of the other grades.
Train the respective models for each tool grade

Author: Reece Calvin

"""

import pandas as pd
import numpy as np
from consants import GAME_TYPES, TRAIN_START, TRAIN_END, TOOL_INFO, CLF_MODEL_TYPES, REG_MODEL_TYPES, CLF_PARAMS, REG_PARAMS, MAX_EVALS
from sklearn.model_selection import cross_val_score
import pickle
import argparse
from etl import pull_data, format_data
import lightgbm as lgbm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def train_all_models(tool_info: dict, start_date: str, end_date: str, clf_model_types: list, reg_model_types: list, clf_params: dict, reg_params: dict, max_evals: int = 100):
    """
    Train all models for each tool grade

    :param tool_info: Dictionary of required models and their respective features and targets
    :param targets: List of targets for each model
    :param start_date: Date to start training
    :param end_date: Date to end training
    :param clf_model_types: List of classifier model types to test
    :param reg_model_types: List of regressor model types to test
    :param clf_params: Dictionary of hyperparameters for classifier models
    :param reg_params: Dictionary of hyperparameters for regressor models
    :param max_evals: Maximum number of evaluations for hyperparameter tuning

    """

    # Pull and format the data
    print("Pulling data...")
    data = pull_data(start_date=start_date, end_date=end_date, game_types=GAME_TYPES)
    data = format_data(data)

    for tool_model in tool_info.keys():
        info = tool_info[tool_model]

        # Record the best model
        best_score = float('-inf')

        # Set up the model types
        model_types = clf_model_types if info["model_type"] == "classifier" else reg_model_types

        for model_type in model_types:

            print(f"Training model {tool_model} with model type {model_type}")

            if model_type == "LGBM":
                model = lgbm.LGBMClassifier() if info["model_type"] == "classifier" else lgbm.LGBMRegressor()

            elif model_type == "XGB":
                model = xgb.XGBClassifier() if info["model_type"] == "classifier" else xgb.XGBRegressor()

            elif model_type == "RF":
                model = RandomForestClassifier() if info["model_type"] == "classifier" else RandomForestRegressor()

            # Get info from the dictionary
            features = info["features"]
            target = info["target"]
            scoring = info["scoring"]

            # Set up the data
            X = data[features]
            y = data[target]

            # Fit the hyperparameters
            print("Fitting hyperparameters...")
            params = clf_params[model_type] if info["model_type"] == "classifier" else reg_params[model_type]

            def objective(params):
                model.set_params(**params)
                score = cross_val_score(model, X, y, cv=5, scoring=scoring)
                return {"loss": -score.mean(), "status": STATUS_OK}

            trials = Trials()
            best = fmin(objective, params, algo=tpe.suggest, max_evals=max_evals, trials=trials)

            # Set the best hyperparameters
            model.set_params(**best)

            # Cross validate the model
            print("Cross validating model...")
            scores = cross_val_score(model, X, y, cv=5, scoring=scoring)

            print(f"Cross validation scores: {scores}")

            if np.mean(scores) > best_score:
                best_model = model
                best_score = np.mean(scores)

        # Save the best model and features
        with open(f"{target}_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        with open(f"{target}_features.pkl", "wb") as f:
            pickle.dump(features, f)

def get_state_exp_args() -> argparse.Namespace:
    """
    Get the arguments from the command line

    :return: Namespace of the arguments
    """

    parser = argparse.ArgumentParser(description="Train models needed for tool grades")

    parser.add_argument("--start_date", type=str, help="Date to start training", default=TRAIN_START)
    parser.add_argument("--end_date", type=str, help="Date to end training", default=TRAIN_END)
    parser.add_argument("--game_types", type=list, help="List of game types to pull data from", default=GAME_TYPES)
    parser.add_argument("--clf_model_types", type=list, help="List of model types to test", default=CLF_MODEL_TYPES)
    parser.add_argument("--reg_model_types", type=list, help="List of model types to test", default=REG_MODEL_TYPES)
    parser.add_argument("--clf_params", type=dict, help="Dictionary of hyperparameters for classifier models", default=CLF_PARAMS)
    parser.add_argument("--reg_params", type=dict, help="Dictionary of hyperparameters for regressor models", default=REG_PARAMS)
    parser.add_argument("--tool_info", type=dict, help="Dictionary of required models and their respective features and targets", default=TOOL_INFO)
    parser.add_argument("--max_evals", type=int, help="Maximum number of evaluations for hyperparameter tuning", default=MAX_EVALS)

    return parser.parse_args()

def main():

    # Get the arguments
    args = get_state_exp_args()

    # Train the models
    train_all_models(args.tool_info, args.start_date, args.end_date, args.clf_model_types, args.reg_model_types, args.clf_params, args.reg_params, args.max_evals)

if __name__ == "__main__":
    main()
