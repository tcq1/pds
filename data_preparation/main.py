import sqlite3
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import zipfile


def get_databases_from_source():
    # information about databases
    db_old_url = 'https://wwwdb.inf.tu-dresden.de/misc/WS2021/PDS/the.db'
    db_old_login = HTTPBasicAuth('tud', 'dbs')

    # db_new_url = 'fill_this_out'

    # create GET requests
    db_old_request = requests.get(db_old_url, auth=db_old_login)
    # db_new_request = requests.get(db_new_url)

    # store content of GET requests in a sql file
    open('old_db.sql', 'wb').write(db_old_request.content)
    # open('new_db.sql', 'wb').write(db_new_request.content)


def main():
    get_databases_from_source()
    db_old = sqlite3.connect('old_db.sql')
    print("Database created and Successfully Connected to SQLite")

    table_old = pd.read_sql_query("SELECT * from dwd", db_old)
    table_old.to_csv('table.csv', index_label='index')

    

if __name__ == '__main__':
    main()

