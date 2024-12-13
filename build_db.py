"""
Since the 3GB database file is too large, I will build a new database file with the same data.
Really the only reason I'm doing this at all is so I can show I've worked with SQL
"""

import sqlite3
import glob
import pandas as pd

def build_db():
    """
    Build a database file with the statcast data
    """
    conn = sqlite3.connect('statcast.db')
    c = conn.cursor()

    all_data = pd.DataFrame()

    for file in glob.glob('archive/*.csv'):
        df = pd.read_csv(file)
        all_data = pd.concat([all_data, df], ignore_index=True)
        print(f'Read {file}')

    all_data.to_sql('statcast', conn, index=False, if_exists='replace')
    print('All data added to statcast table in statcast.db')

    conn.close()

if __name__ == '__main__':
    build_db()