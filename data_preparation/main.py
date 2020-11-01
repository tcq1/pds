import sqlite3
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from zipfile import ZipFile
import io
import re


def txt_to_csv(source, destination):
    # Converts the text file (source) of the zip archive to a csv file (destination)
    txt = open(source, 'r')
    csv = open(destination, 'w')
    lines = txt.readlines()

    for line in lines:
        elements = list(filter(None, re.split('[; \W+]', line.strip())))
        csv.write(elements[0])
        for i in range(1, len(elements)):
            csv.write(',{}'.format(elements[i]))
        csv.write('\n')

    txt.close()
    csv.close()


def get_csv_files_from_source():
    """ Implementation of task 1 and 2: download from both data sources and make data manageable
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

    # convert sql to csv --- nicht sicher ob notwendig, evtl. kann man direkt mit der sql file arbeiten
    db_old = sqlite3.connect('old_db.sql')
    table_old = pd.read_sql_query("SELECT * from dwd", db_old)
    table_old.to_csv('old.csv', index_label='index')

    # extract txt file from zip archive
    ZipFile(zip_file).extractall()

    # convert txt to csv
    txt_to_csv('produkt_zehn_min_tu_20190501_20201031_01048.txt', 'dwd.csv')


def main():
    # task 1 and 2
    get_csv_files_from_source()


if __name__ == '__main__':
    main()
