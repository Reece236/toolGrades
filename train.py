"""
Idk, Goal is to objectify hitter tool grades and do so in a way that is independent of the other grades.
Train the respective models for each tool grade

Author: Reece Calvin

"""

import pandas as pd
from constants import GAME_TYPES, TRAIN_START, TRAIN_END, TOOL_INFO, CLF_MODEL_TYPES, REG_MODEL_TYPES, CLF_PARAMS, REG_PARAMS, MAX_EVALS, RANDOM_STATE
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import pickle
import argparse
from etl import pull_data, format_data
import lightgbm as lgbm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from bayes import bat_to_ball_tool, contact_quality_tool, power_tool, swing_decision_tool

def train_gp_models(data: pd.DataFrame, model: dict):
    """
    Train the Gaussian Process models for each tool grade

    :param data: DataFrame of the data
    :param model: Dictionary of the model, features, and target
    """

    for tool_model in model.items():
        info = tool_model[1]

        print(f"Training model {tool_model[0]}")

        if tool_model[0] == "Bat to Ball":
            posterior, bat_to_ball_model = bat_to_ball_tool(data, info)

        elif tool_model[0] == "Outcome Probability":
            posterior, contact_quality_model = contact_quality_tool(data, info)

        elif tool_model[0] == "xEV":
            posterior, power_model = power_tool(data, info)

        elif tool_model[0] == "Swing Decision":
            posterior, swing_decision_model = swing_decision_tool(data, info)

        # Save the model
        with open(f"models/{tool_model[0]}_model.pkl", "wb") as f:
            pickle.dump(posterior, f)

        print(f"Saved {tool_model[0]} model \n")

def train_all_models(tool_info: dict, data: pd.DataFrame, clf_model_types: list, reg_model_types: list, clf_params: dict, reg_params: dict, max_evals: int, folds: int = 5):
    """
    Train all models for each tool grade

    :param tool_info: Dictionary of required models and their respective features and targets
    :param targets: List of targets for each model
    :param data: DataFrame of the data
    :param clf_model_types: List of classifier model types to test
    :param reg_model_types: List of regressor model types to test
    :param clf_params: Dictionary of hyperparameters for classifier models
    :param reg_params: Dictionary of hyperparameters for regressor models
    :param max_evals: Maximum number of evaluations for hyperparameter tuning
    :param folds: Number of folds for cross validation

    """

    for tool_model in tool_info.items():
        info = tool_model[1]

        # Record the best model
        best_score = float('-inf')

        # Set up the model types
        model_types = clf_model_types if info["model_type"] == "classifier" else reg_model_types

        for model_type in model_types:

            print(f"Training model {tool_model[0]} with model type {model_type} {info['model_type']}")

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
            query = info["query"]

            # Set up the data
            subset = data.query(query).dropna(subset=features + [target])
            X = subset[features]
            y = subset[target]

            # Fit the hyperparameters
            print("Fitting hyperparameters...")
            params = clf_params[model_type] if info["model_type"] == "classifier" else reg_params[model_type]

            def objective(params):
                model.set_params(**params)
                if info["model_type"] == "classifier":
                    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
                else:
                    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

                score = cross_val_score(model, X, y, cv=kf, scoring=scoring)
                return {"loss": -score.mean(), "status": STATUS_OK}

            trials = Trials()
            best = fmin(objective, params, algo=tpe.suggest, max_evals=max_evals, trials=trials)

            # Retrieve the best trial
            best_trial = min(trials.results, key=lambda x: x['loss'])
            score = best_trial['loss']

            # Save the best model
            if score > best_score:

                # Set the best hyperparameters
                model.set_params(**best)

                # Fit the model
                print("Fitting the model...")
                model.fit(X, y)

                best_score = score
                best_model = model

        # Save the best model and features
        with open(f"models/{tool_model[0]}_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        with open(f"models/{tool_model[0]}_features.pkl", "wb") as f:
            pickle.dump(features, f)

        print(f"Saved {tool_model[0]} model \n")

def get_state_exp_args() -> argparse.Namespace:
    """
    Get the arguments from the command line

    :return: Namespace of the arguments
    """

    parser = argparse.ArgumentParser(description="Train models needed for tool grades")

    parser.add_argument("--train_lgb", action='store_false', help="Train the lightgbm models", default=True)
    parser.add_argument("--train_gp", action='store_false', help="Train the Gaussian Process models", default=True)
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

    # Pull and format the data
    print("Pulling data...")
    data = pull_data(args.start_date, args.end_date, args.game_types)
    data = format_data(data, build_rv_table=True)


    # Train the models
    if args.train_lgb:
        train_all_models(args.tool_info, data, args.clf_model_types, args.reg_model_types, args.clf_params, args.reg_params, args.max_evals)

    if args.train_gp:
        train_gp_models(data, args.tool_info)

if __name__ == "__main__":
    main()
