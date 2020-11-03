import sqlite3
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from zipfile import ZipFile
import numpy as np


def get_dataframe_dwd_data():
    """ Downloads data from dwd and returns a pandas dataframe with the dwd data
    """

    # get dwd data
    url = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes' \
          '/air_temperature/recent/10minutenwerte_TU_01048_akt.zip'
    request = requests.get(url)

    # extract zipfile
    with open('old.zip', 'wb') as f:
        f.write(request.content)
    with ZipFile('old.zip', 'r') as z:
        files = z.namelist()
        open('old.txt', 'wb').write(z.read(files[0]))

    # return dataframe
    return pd.read_csv('old.txt', delimiter=';')


def get_dataframe_old_db():
    """ Downloads given database and returns a pandas dataframe with the old data
    """

    # information about database
    url = 'https://wwwdb.inf.tu-dresden.de/misc/WS2021/PDS/the.db'
    login = HTTPBasicAuth('tud', 'dbs')

    # create GET requests
    request = requests.get(url, auth=login)

    # store content of GET requests
    open('db.sql', 'wb').write(request.content)

    # explore db
    db = sqlite3.connect('db.sql')
    cursor = db.cursor()
    cursor.execute('SELECT name FROM sqlite_master')
    table_name = cursor.fetchone()[0]
    # columns = [description[0] for description in cursor.description]

    # query data
    query = 'SELECT * FROM {}'.format(table_name)
    cursor.execute(query)
    data = cursor.fetchall()
    # print('Old database has {} entries'.format(len(data)))

    # export to dataframe
    df = pd.read_sql_query(query, db)
    db.close()

    return df


def adapt_dwd_data(df, min_timestamp, max_timestamp):
    """ Implementation of task 3:
        matching schema of dwd data to given database and retrieve only data from end of old.csv to end of 2020-10-23

    :param df: pandas dataframe
    :param min_timestamp: most recent timestamp from the given database
    :param max_timestamp: last timestamp that should be included
    :return: pandas data frame
    """

    # drop unnecessary columns
    df = df.drop(['  QN', 'eor'], axis=1)

    # rename columns
    df = df.rename(columns={'STATIONS_ID': 'id', 'MESS_DATUM': 'timestamp', 'PP_10': 'airpressure',
                            'TT_10': 'temperature', 'TM5_10': 'temperature_ground', 'RF_10': 'humidity',
                            'TD_10': 'temperature_dew'})

    # switch columns
    new_column_head = ['timestamp', 'id', 'temperature', 'temperature_ground',
                       'temperature_dew', 'humidity', 'airpressure']
    df = df.reindex(columns=new_column_head)

    # filter data by timestamp
    df.drop(df[df.timestamp <= min_timestamp].index, inplace=True)
    df.drop(df[df.timestamp > max_timestamp].index, inplace=True)

    return df


def concat_dataframes(old, dwd):
    """ Task 4: concatenates both dataframes

    :param old: dataframe of given db
    :param dwd: dataframe of dwd
    :return: concatenated dataframe
    """

    # concatenate and reset indexing
    concat = pd.concat([old, dwd], ignore_index=True)

    return concat


def calc_average_temp(df, db_path):
    """ Task 4: calculate average temperature per hour and add to existing database

    :param db_path: path to database
    :param df: dataframe
    :return: dataframe
    """

    db = sqlite3.connect(db_path)

    # Create dataframe with every 6th timestamp
    times = df[['timestamp']].iloc[::6].reset_index(drop=True)

    # Create dataframe with average of every 6 temperature values
    temps = df[['temperature']].groupby(np.arange(len(df)) // 6).mean()

    # Concatenate columns and create new table 'hourly_avg'
    avg_df = pd.concat([times, temps], axis=1)
    avg_df.to_sql('hourly_avg', db, if_exists='replace', index=False)

    db.close()


def main():
    # specify output path
    output_json = 'output.json'

    # task 1 and 2
    old = get_dataframe_old_db()
    dwd = get_dataframe_dwd_data()

    # task 3
    # parameters: last timestamp of the given table and last available timestamp for 2020-10-23
    dwd = adapt_dwd_data(dwd, 202010131050, 202010232350)

    # task 4
    # append tables
    concat = concat_dataframes(old, dwd)
    concat.to_json(output_json)

    # calculate averages and store in the database
    calc_average_temp(concat, 'db.sql')


if __name__ == '__main__':
    main()
