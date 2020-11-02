import sqlite3
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from zipfile import ZipFile
import io
import re


def txt_to_csv(source, destination):
    """Converts the text file (source) of the zip archive to a csv file (destination)

    :param source: path of source file
    :param destination: output path of csv file
    """

    # open necessary files
    txt = open(source, 'r')
    csv = open(destination, 'w')
    lines = txt.readlines()

    # write lines to csv
    for line in lines:
        # split line at whitespace and ; and remove empty strings from the result
        elements = list(filter(None, re.split('[; \s]', line.strip())))
        # write to the csv file
        csv.write(elements[0])
        for i in range(1, len(elements)):
            csv.write(',{}'.format(elements[i]))
        csv.write('\n')

    # close files
    txt.close()
    csv.close()


def get_csv_files_from_source():
    """ Implementation of task 1 and 2: download from both data sources and convert to csv to make it work for pandas
    """
    # information about databases
    db_old_url = 'https://wwwdb.inf.tu-dresden.de/misc/WS2021/PDS/the.db'
    db_old_login = HTTPBasicAuth('tud', 'dbs')

    db_new_url = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes' \
                 '/air_temperature/recent/10minutenwerte_TU_01048_akt.zip'

    # create GET requests
    db_old_request = requests.get(db_old_url, auth=db_old_login)
    db_new_request = requests.get(db_new_url)

    # store content of GET requests
    open('old_db.sql', 'wb').write(db_old_request.content)
    zip_file = io.BytesIO(db_new_request.content)

    # convert sql to csv
    db_old = sqlite3.connect('old_db.sql')
    table_old = pd.read_sql_query("SELECT * from dwd", db_old)
    table_old.to_csv('old.csv', index_label='index')

    # extract txt file from zip archive
    ZipFile(zip_file).extractall()

    # convert txt to csv
    txt_to_csv('produkt_zehn_min_tu_20190501_20201031_01048.txt', 'dwd.csv')


def adapt_dwd_data(csv_file, min_timestamp, max_timestamp):
    """ Implementation of task 3:
        matching schema of dwd data to given database and retrieve only data from end of old.csv to end of 2020-10-23

    :param csv_file: Path to csv file
    :param min_timestamp: most recent timestamp from the given database
    :param max_timestamp: last timestamp that should be included
    :return: pandas data frame
    """

    # load dataframe from csv
    df = pd.read_csv(csv_file)

    # drop unnecessary columns
    df = df.drop(['QN', 'eor'], axis=1)

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


def concat_dataframes(given, dwd):
    """ Task 4: concatenates both dataframes

    :param given: dataframe of given db
    :param dwd: dataframe of dwd
    :return: concatenated dataframe
    """

    # drop index column of given df
    given.drop('index', axis=1, inplace=True)

    # concatenate and reset indexing
    concat = pd.concat([given, dwd])
    concat.reset_index(drop=True, inplace=True)

    return concat


def calc_average_temp(df):
    """ Task 4: calculate average temperature per hour

    :param df: dataframe
    :return: dataframe
    """

    avg_list = []
    timestamp_list = []
    current_index = 0

    # iterate over all rows of df
    while current_index < len(df.index):
        # add timestamp of hour to the timestamp list
        timestamp_list.append(int(df.iloc[current_index]['timestamp']))

        # calculate average over next 6 rows
        current_sum = 0
        for i in range(6):
            try:
                current_sum += df.iloc[current_index]['temperature']
                # increment current_index
                current_index += 1
            except IndexError:
                continue

        # add average value to avg_list
        avg_list.append(current_sum/6)

    # create new dataframe
    avg_df = pd.DataFrame(data={'timestamp': timestamp_list, 'avg_tmp': avg_list})

    return avg_df


def main():
    # specify output paths
    output_json = 'output.json'
    output_avg = 'avg_tmp.csv'

    # task 1 and 2
    get_csv_files_from_source()
    old = pd.read_csv('old.csv')

    # task 3
    dwd = adapt_dwd_data('dwd.csv', 202010131050, 202010232350)

    # task 4
    # append tables
    concat = concat_dataframes(old, dwd)
    concat.to_json(output_json)

    # calculate averages and store
    avg_table = calc_average_temp(concat)
    # It was not clear to me how the average temperature table should be saved. I chose a csv file
    avg_table.to_csv(output_avg)


if __name__ == '__main__':
    main()
