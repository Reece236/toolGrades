import pandas as pd
import numpy as np
from consants import TAKES, CONTACT


def pull_data(start_date: str, end_date: str, game_types: list):
    """
    While I don't have access to a database to pull from, I am good at downloading CSVs from Savant

    :param start_date: Date to start pulling data from
    :param end_date: Date to end pulling data from
    :param game_types: List of game types to pull data from
    :return: DataFrame of the data
    """

    # Read in the data
    data = pd.read_csv('data/all_statcast.csv', on_bad_lines='skip')

    # Gotta make sure the start date is before the end date
    if start_date >= end_date:
        print("Invalid date range, but I'll swap them for you champ!")
        start_date, end_date = end_date, start_date

    # Make sure the dates are in range
    data_start = data["game_date"].min()
    data_end = data["game_date"].max()

    if start_date < data_start or end_date > data_end:
        print(f"Invalid date range, data is from {data_start} to {data_end}")

        start_date = max(start_date, data_start)
        end_date = min(end_date, data_end)

        print(f"Setting the date range to {start_date} to {end_date}")

    # Filter the data
    data = data[
        (data["game_date"] >= start_date) & (data["game_date"] <= end_date) & (data["game_type"].isin(game_types))]

    return data

def format_data(data: pd.DataFrame):

    # Add column for swing decision
    data['decision'] = np.where(data['description'].isin(TAKES), 0, 1)

    # Add column for contact
    data['contact'] = np.where(data['description'].isin(CONTACT), 1, 0)

    # Add column if called strike
    data['cStrike'] = np.where(data['description'] == 'called_strike', 1, 0)

    # Add count column
    data['count'] = data['balls'].astype(str) + '-' + data['strikes'].astype(str)

    # Add specific event by batted ball and hit zone (i.e. 'barrel' or 'weak')
    data['bb_barrels'] = data['bb_type'].astype(str) + data['launch_speed_angle'].astype(str)

    # Replace hit_into_play with bb_barrels and calculate xRV and cRV (count dependent xRV)
    data['result'] = data['description'].replace({'hit_into_play': np.nan}).fillna(data['bb_barrels'])
    data['xRV'] = data['result'].map(data.groupby('result')['delta_run_exp'].mean())
    data['cResult'] = data['count'] + data['result']
    data['cRV'] = data['cResult'].map(data.groupby('cResult')['delta_run_exp'].mean())

    return data
