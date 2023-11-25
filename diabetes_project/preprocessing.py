"""
Various preprocessing functions
"""
# 1st party imports
import sys

# 3rd party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_feature_labels(data):
    """
    Converts raw data from CSV file (in dataframe or numpy array)
    into two numpy arrays:
    x -> features
    y -> labels
    """
    if isinstance(data, np.ndarray):
        return data[:, :-1], data[:, -1]
    if isinstance(data, pd.DataFrame):
        data_arr = data.to_numpy()
        return data_arr[:, :-1], data_arr[:, -1]

    print(f"Error: data parameter is of unsupported datatype: {type(data)}")
    sys.exit(1)


def factorize(x):
    """
    Handles preprocessing of non-numeric datatypes
    this will not impact numeric data
    Returns:
        x_factorized -> ndarray in the original shape of x with
                        non-numeric features encoded via factorization
    """
    if isinstance(x, np.ndarray):
        df = pd.DataFrame(x)
    elif isinstance(x, pd.DataFrame):
        df = x
    else:
        print(f"Error: x parameter is of unsupported datatype: {type(x)}")
        sys.exit(1)

    df = df.convert_dtypes()
    for col_idx in df:
        if (df[col_idx].dtype) == "string":
            df[col_idx] = pd.factorize(df[col_idx])[0]

    return df.to_numpy()


def standardize_features(x, return_scaler=True):
    """
    Uses sklearn to perform standard scaling on features.
    With scaler being returned as well to apply transformation to
    a different dataset (fit model on train, apply to test, etc.)
    Params:
        x             -> the features of the data to be scaled
        return_scaler -> determines if the scaler object should be returned
    Returns:
        x_transformed -> the standardizes input data
        scaler        -> the scaling model
    """
    scaler = StandardScaler()
    scaler.fit(x)
    if return_scaler:
        return scaler.transform(x), scaler

    return scaler.transform(x)


if __name__ == "__main__":
    # This code is just testing to ensure that the functions work correctly.
    # This won't run unless running preprocessing.py directly

    # imports from other parts of the project
    from diabetes_project.read_data import get_data_df

    data_df = get_data_df()
    data_x, data_y = get_feature_labels(data_df)
    print(data_df)
    print(data_x)
    print(data_y)

    # try factorizing
    data_x = factorize(data_x)

    x_transform = standardize_features(data_x, return_scaler=False)
    print(x_transform)
