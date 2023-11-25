from pathlib import Path

import pandas as pd

def get_data_df():
    data_path = Path("./data/diabetic_data.csv")
    return pd.read_csv(data_path)

if __name__ == "__main__":
    print(get_data_df())
