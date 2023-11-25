"""
Handles reading the data using a consistent interface.
"""

from pathlib import Path

import pandas as pd
import numpy as np


def get_data_df():
    """
    Reads CSV into pandas dataframe from data directory
    """
    data_path = Path("./data/diabetic_data.csv")
    return pd.read_csv(data_path)


def get_data_numpy():
    """
    Reads CSV into numpy array from data directory
    """
    data_path = Path("./data/diabetic_data.csv")
    return pd.read_csv(data_path).to_numpy()


if __name__ == "__main__":
    print(get_data_numpy())
