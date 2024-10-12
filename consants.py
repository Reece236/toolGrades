import numpy as np
from hyperopt import hp

TRAIN_START = "2018-01-01"
TRAIN_END = "2021-12-31"
GAME_TYPES = ["R", "F", "D", "L", "W"]
TEST_SIZE = .3
RANDOM_STATE = 6
CLF_MODEL_TYPES = ["LGBM"]
REG_MODEL_TYPES = ["LGBM"]
CLF_PARAMS = {"LGBM": {"n_estimators": hp.choice("n_estimators", np.arange(100, 1000, 100)), "max_depth": hp.choice("max_depth", np.arange(2, 10, 1)), "learning_rate": hp.choice("learning_rate", np.arange(.01, .1, .01)), "verbosity": -1}}
REG_PARAMS = {"LGBM": {"n_estimators": hp.choice("n_estimators", np.arange(100, 1000, 100)), "max_depth": hp.choice("max_depth", np.arange(2, 10, 1)), "learning_rate": hp.choice("learning_rate", np.arange(.01, .1, .01)), "verbosity": -1}}
MAX_EVALS = 10

TAKES = ['ball','called_strike', 'blocked_ball', 'hit_by_pitch', 'pitchout']
CONTACT = ['hit_into_play', 'foul', 'foul_tip', 'foul_bunt', 'bunt_foul_tip', 'foul_pitchout']

# Models needed for each tool
strike_prob = {
    "features": ['pfx_x','pfx_z','plate_x','plate_z','release_speed'],
    "target": "cStrike",
    "query" : "decision == 0",
    "model_type": "classifier",
    "scoring": "neg_log_loss"
    }

b2b_model = {
    "features": ['pfx_x','pfx_z','plate_x','plate_z','release_speed', 'balls', 'strikes'],
    "target": "contact",
    "query" : "decision == 1",
    "model_type": "classifier",
    "scoring": "neg_log_loss"
    }

decision = {
    "features": ['pfx_x','pfx_z','plate_x','plate_z','release_speed', 'balls', 'strikes'],
    "target": "decision",
    "query" : "pitch_type != '-'",
    "model_type": "classifier",
    "scoring": "neg_log_loss"
    }

xev = {
    "features": ['pfx_x','pfx_z','plate_x','plate_z','release_speed', 'balls', 'strikes', 'launch_angle'],
    "target": "launch_speed",
    "query" : "launch_speed != np.nan",
    "model_type": "regressor",
    "scoring": "neg_mean_squared_error"
    }

res_prob = {
    "features": ['pfx_x','pfx_z','plate_x','plate_z','release_speed', 'balls', 'strikes'],
    "target": "bb_barrels",
    "query" : "bb_barrels != np.nan",
    "model_type": "classifier",
    "scoring": "neg_mean_squared_error"
    }

TOOL_INFO = {'Strike Probability': strike_prob, 'Bat to Ball': b2b_model, 'Swing Decision': decision, 'xEV': xev, 'Outcome Probability': res_prob}
