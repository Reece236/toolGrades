"""
Idk, Goal is to objectify hitter tool grades and do so in a way that is independent of the other grades.
Train the respective models for each tool grade

Author: Reece Calvin

"""

import pandas as pd
from constants import GAME_TYPES, TRAIN_START, TRAIN_END, TOOL_INFO, MAX_EVALS, RANDOM_STATE
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import pickle
import argparse
from etl import pull_data, format_data
import lightgbm as lgbm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
from bayes import bat_to_ball_tool, contact_quality_tool, power_tool, swing_decision_tool
from sklearn.calibration import CalibratedClassifierCV

def train_all_models(tool_info: dict, data: pd.DataFrame, max_evals: int, folds: int = 5):
    """
    Train all models using LightGBM
    """
    for tool_model in tool_info.items():
        
        info = tool_model[1]
        print(f"Training model {tool_model[0]}")

        # Initialize LightGBM model
        model = lgbm.LGBMClassifier() if info["model_type"] == "classifier" else lgbm.LGBMRegressor()

        # Get info from the dictionary
        features = info["features"]
        target = info["target"]
        scoring = info["scoring"]
        query = info["query"]

        # Set up the data
        subset = data.query(query).dropna(subset=features + [target])
        X = subset[features]
        y = subset[target]

        # Basic LightGBM parameters
        params = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
            'num_leaves': hp.quniform('num_leaves', 20, 200, 10),
            'min_child_samples': hp.quniform('min_child_samples', 10, 100, 5)
        }

        def objective(params):
            # Convert float parameters to integers where needed
            params['n_estimators'] = int(params['n_estimators'])
            params['num_leaves'] = int(params['num_leaves'])
            params['min_child_samples'] = int(params['min_child_samples'])
            
            model.set_params(**params)
            score = cross_val_score(model, X, y, cv=folds, scoring=scoring)
            return {"loss": -score.mean(), "status": STATUS_OK}

        # Find best parameters
        trials = Trials()
        best = fmin(objective, params, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        # Set the best parameters and fit
        best['n_estimators'] = int(best['n_estimators'])
        best['num_leaves'] = int(best['num_leaves'])
        best['min_child_samples'] = int(best['min_child_samples'])
        model.set_params(**best)
        
        # Fit final model
        if info["model_type"] == "classifier":
            # Calibration is built into the saved model
            model = CalibratedClassifierCV(model, cv=5, method='isotonic')
            model.fit(X, y)
        else:
            model.fit(X, y)

        # Save model and features
        with open(f"models/{tool_model[0]}_lgbm.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(f"models/{tool_model[0]}_features.pkl", "wb") as f:
            pickle.dump(features, f)

        print(f"Saved {tool_model[0]} model\n")

def get_state_exp_args() -> argparse.Namespace:
    """
    Get the arguments from the command line

    :return: Namespace of the arguments
    """

    parser = argparse.ArgumentParser(description="Train models needed for tool grades")

    parser.add_argument("--start_date", type=str, help="Date to start training", default=TRAIN_START)
    parser.add_argument("--end_date", type=str, help="Date to end training", default=TRAIN_END)
    parser.add_argument("--game_types", type=list, help="List of game types to pull data from", default=GAME_TYPES)
    parser.add_argument("--tool_info", type=dict, help="Dictionary of required models and their respective features and targets", default=TOOL_INFO)
    parser.add_argument("--max_evals", type=int, help="Maximum number of evaluations for hyperparameter tuning", default=MAX_EVALS)

    return parser.parse_args()

def main():

    # Get the arguments
    args = get_state_exp_args()

    # Pull and format the data
    print("Pulling data...")
    data = pull_data(args.start_date, args.end_date, args.game_types)
    data = format_data(data, build_rv_table=True)

    # Train the models
    train_all_models(args.tool_info, data, args.max_evals)

if __name__ == "__main__":
    main()
