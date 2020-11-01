import sqlite3
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
from zipfile import ZipFile
import io


def get_databases_from_source():
    # information about databases
    db_old_url = 'https://wwwdb.inf.tu-dresden.de/misc/WS2021/PDS/the.db'
    db_old_login = HTTPBasicAuth('tud', 'dbs')

    db_new_url = 'https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes' \
                 '/air_temperature/recent/10minutenwerte_TU_01048_akt.zip'

    # create GET requests
    db_old_request = requests.get(db_old_url, auth=db_old_login)
    db_new_request = requests.get(db_new_url)

    # store content of GET requests in a file
    open('old_db.sql', 'wb').write(db_old_request.content)
    zip_file = io.BytesIO(db_new_request.content)

    ZipFile(zip_file).extractall()


def main():
    get_databases_from_source()
    db_old = sqlite3.connect('old_db.sql')
    print("Database created and Successfully Connected to SQLite")

    table_old = pd.read_sql_query("SELECT * from dwd", db_old)
    table_old.to_csv('table.csv', index_label='index')


if __name__ == '__main__':
    main()
