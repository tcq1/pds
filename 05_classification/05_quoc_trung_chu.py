import numpy as np
import pandas as pd


def split_columns(df):
    """ Takes the data frame and returns a table with the input data and a table with the labels
    """
    input_data = df.loc[:, df.columns != 'tennis']
    label_data = df.loc[:, df.columns == 'tennis']

    return input_data, label_data


def calc_information_gain():
    """ Calculates the information gain of an attribute
    """
    pass


def calc_entropy():
    pass


def main():
    file_path = 'data-cls.csv'
    df = pd.read_csv(file_path)
    input_data, label_data = split_columns(df)


if __name__ == '__main__':
    main()
