"""
Various preprocessing functions
"""

# 3rd party imports
import pandas as pd
import numpy as np


def get_feature_labels(data):
    """
    Converts raw data from CSV file (in dataframe or numpy array)
    into two numpy arrays:
    x -> features
    y -> labels
    """
    if type(data) == np.ndarray:
        return data[:, :-1], data[:, -1]
    elif type(data) == pd.DataFrame:
        data_arr = data.to_numpy()
        return data_arr[:, :-1], data_arr[:, -1]
    else:
        print("Error: data parameter is of unknown datatype: {0}".format(type(data)))
        exit()


if __name__ == "__main__":
    """
    This code is just some testing to ensure that the functions work correctly.
    This won't run unless running preprocessing.py directly
    """
    import read_data

    data_np = read_data.get_data_numpy()
    x, y = get_feature_labels(data_np)
    print(data_np)
    print(x)
    print(y)


    data_df = read_data.get_data_df()
    x, y = get_feature_labels(data_df)
    print(data_df)
    print(x)
    print(y)
