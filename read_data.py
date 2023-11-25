"""
Handles reading the data using a consistent interface.
"""

from pathlib import Path

import pandas as pd


def get_data_df():
    """
    Reads CSV into pandas dataframe from data directory
    """
    data_path = Path("./data/diabetic_data.csv")
    return pd.read_csv(data_path)


if __name__ == "__main__":
    print(get_data_df())
