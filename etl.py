"""
Nothing to see here. Just a little pre-processing
"""
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from constants import TAKES, CONTACT, GAME_TYPES
from toolgrade_sql import savant_query
import sqlite3


def pull_data(start_date: str, end_date: str, game_types: list) -> pd.DataFrame:
    """
    :param start_date: Date to start pulling data from
    :param end_date: Date to end pulling data from
    :param game_types: List of game types to pull data from
    :return: DataFrame of the data
    """

    # Gotta make sure the start date is before the end date
    if start_date >= end_date:
        print("Invalid date range, but I'll swap them for you champ!")
        start_date, end_date = end_date, start_date

    # Connect to database statcast.db
    conn = sqlite3.connect('data/statcast.db')

    # Pull the data
    data = pd.read_sql_query(savant_query.format(takes=TAKES, contact=CONTACT, game_types=GAME_TYPES, start_date=start_date, end_date=end_date), conn)

    # Close the connection
    conn.close()

    return data

def format_data(data: pd.DataFrame, build_rv_table: bool = False) -> pd.DataFrame:
    """
    Format the data for analysis
    :param data: statcast data
    :param build_rv_table: boolean to build rv table or use existing
    :return: formatted data
    """

    # Label encode bb_barrels and calculate rv_table then save
    if build_rv_table:
        le = LabelEncoder()
        data['bb_barrels'] = le.fit_transform(data['bb_barrels'])
        rv_table = data.groupby('bb_barrels')['delta_run_exp'].mean()

        rv_table.to_csv('models/rv_table.csv')
        with open('models/le.pkl', 'wb') as f:
            pickle.dump(le, f)

    else:
        with open('models/le.pkl', 'rb') as f:
            le = pickle.load(f)

        data['bb_barrels'] = le.transform(data['bb_barrels'])


    # Replace hit_into_play with bb_barrels and calculate xRV and cRV (count dependent xRV)
    data['result'] = data['description'].replace({'hit_into_play': np.nan}).fillna(data['bb_barrels'])
    data['xRV'] = data['result'].map(data.groupby('result')['delta_run_exp'].mean())
    data['cResult'] = data['count'].astype(str) + data['result'].astype(str)
    data['cRV'] = data['cResult'].map(data.groupby('cResult')['delta_run_exp'].mean())

    # Replace any bb_barrels that occur fewer than 10 times with nan
    data['bb_barrels'] = data['bb_barrels'].mask(data['bb_barrels'].map(data['bb_barrels'].value_counts()) < 10)

    return data
