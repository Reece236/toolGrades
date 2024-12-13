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


def pull_data(start_date: str, end_date: str, game_types: list = False) -> pd.DataFrame:
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

    # Set the query
    query = savant_query.format(takes=TAKES, contact=CONTACT, game_types=GAME_TYPES, start_date=start_date, end_date=end_date)
    
    # Pull the data
    data = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    return data

def format_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Format the data for analysis
    :param data: statcast data
    :return: formatted data
    """

    # Replace hit_into_play with bb_barrels and calculate xRV and cRV (count dependent xRV)
    data['result'] = data['description'].replace({'hit_into_play': np.nan}).fillna(data['bb_barrels'])
    data['xRV'] = data['result'].map(data.groupby('result')['delta_run_exp'].mean())
    data['cResult'] = data['count'].astype(str) + data['result'].astype(str)
    data['cRV'] = data['cResult'].map(data.groupby('cResult')['delta_run_exp'].mean())

    # Replace any bb_barrels that occur fewer than 10 times with nan
    data['bb_barrels'] = data['bb_barrels'].mask(data['bb_barrels'].map(data['bb_barrels'].value_counts()) < 10)

    return data
