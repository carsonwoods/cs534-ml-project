"""
Handles reading the data using a consistent interface.
"""

# 1st party imports
from pathlib import Path

# 3rd party imports
import pandas as pd


def get_data_df():
    """
    Reads CSV into pandas dataframe from data directory
    """
    data_path = Path("./data/diabetic_data.csv")
    return pd.read_csv(data_path, na_values=['?'])


def get_data_numpy():
    """
    Reads CSV into numpy array from data directory
    """
    data_path = Path("./data/diabetic_data.csv")
    return pd.read_csv(data_path, na_values=['?']).to_numpy()


if __name__ == "__main__":
    print(get_data_numpy())
