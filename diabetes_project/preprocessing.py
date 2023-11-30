"""
Various preprocessing functions
"""
# 1st party imports
import sys

# 3rd party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr
from scipy.stats import kstest


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


def factorize(data):
    """
    Handles preprocessing of non-numeric datatypes
    this will not impact numeric data.
    If a 1-D array (such as labels) is passed in, the output
    will be flattened again to a single dimension.
    Returns:
        x_factorized -> ndarray in the original shape of x with
                        non-numeric features encoded via factorization
    """
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        print(f"Error: data parameter is of unsupported datatype: {type(data)}")
        sys.exit(1)

    df = df.convert_dtypes()
    for col_idx in df:
        if (df[col_idx].dtype) == "string":
            df[col_idx] = pd.factorize(df[col_idx])[0]

    array = df.to_numpy()
    if array.shape[1] == 1:
        return array.flatten()
    return array


def filter_pairwise_corr_vars(features, corrtype, threshold=0.60):
    """
    Removes one of any two features which has a
    correlation score exceeding >= threshold
    Params:
        features -> numpy array of features only
    Returns:
        x_fs -> numpy array containing only the features
                which survived the selection
    """
    if corrtype not in ("pearson", "kendall", "spearman"):
        print("Error: must choose from pearson, kendall, or spearman corrtype")
        sys.exit(1)

    x_df = pd.DataFrame(features).convert_dtypes()

    corr_matrix = x_df.corr(
        method=corrtype,
        min_periods=1,
        numeric_only=False,
    ).to_numpy()

    remove_list = set()
    for index, corr_score in np.ndenumerate(corr_matrix):
        # don't check on diagonals or on previously removed features
        if index[0] != index[1] and index[0] not in remove_list:
            if abs(corr_score) >= threshold:
                remove_list.add(index[1])

    return np.delete(x, list(remove_list), 1)


def filter_most_corr(x, y, rank_type="correlation", threshold=0.80):
    """
    Removes features with a correlation or mutual information
    score of >= threshold from the list of features in x
    Params:
        x -> original features
        y -> original labels
        rank_type -> "correlation" or "mutual"
        threshold -> "cutoff for feature selection"
    Returns:
        x_fs -> ndarray of x with filtered features removed
    """
    rankings = []
    if rank_type == "correlation":
        corr_results = {}
        # ensures that the type is an integer
        # so a classifier can be applied
        # y = y.astype(int)

        # iterate across all features (columns) in x
        for col_idx in range(0, len(x[0])):
            corr = abs(spearmanr(x[:, col_idx], y).statistic)
            if corr not in corr_results:
                corr_results[corr] = [col_idx]
            else:
                corr_results[corr].extend([col_idx])

        for key in sorted(corr_results, reverse=True):
            rankings.extend(corr_results[key])
    elif rank_type == "mutual":
        corr_results = {}

        # ensures that the type is an integer
        # so a classifier can be applied
        y = y.astype(int)

        # iterate across all features (columns) in x
        for col_idx in range(0, len(x[0])):
            corr = abs(mutual_info_score(x[:, col_idx], y))
            if corr not in corr_results:
                corr_results[corr] = [col_idx]
            else:
                corr_results[corr].extend([col_idx])

        for key in sorted(corr_results, reverse=True):
            rankings.extend(corr_results[key])
    else:
        print("Error: must choose correlation or mutual for rank_type")
        sys.exit(1)

    if threshold > 1 or threshold < 0:
        print("Error: threshold must be between 0 and 1")

    new_rankings = []
    for idx in range(0, int(len(rankings) * threshold) - 1):
        new_rankings.append(rankings[idx])

    delete_list = set()
    for feature_idx in range(0, x.shape[1]):
        if feature_idx not in new_rankings:
            delete_list.add(feature_idx)

    return np.delete(x, list(delete_list), 1)


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


def remove_missing_features(features, threshold=.5, return_type="ndarray"):
    """
    Removes any feature column where the number of
    missing values exceeds (>= threshold)
    Params:
        features  -> pandas df or numpy array of features
        threshold -> float value in range (0,1)
    Returns:
        x_fs      -> removes
    """
    if isinstance(features, np.ndarray):
        df = pd.DataFrame(features)
    elif isinstance(features, pd.DataFrame):
        df = features
    else:
        print(f"Error: features parameter is of unsupported datatype: {type(features)}")
        sys.exit(1)

    if not isinstance(threshold, float) or threshold >= 1 or threshold <= 0:
        print("Error: threshold must be a float between 0 and 1")
        sys.exit(1)

    if return_type not in ["dataframe", "ndarray"]:
        print("Error: return_type must be either \"dataframe\" or \"ndarray\"")
        sys.exit(1)

    remove_list = []

    for column in df:
        count_nan = df[column].isnull().sum()
        if count_nan/df.shape[0] >= threshold:
            remove_list.append(column)

    df.drop(remove_list, axis=1, inplace=True)

    if return_type == "dataframe":
        return df

    return df.to_numpy()


def impute_missing_value(features):
    """
    Imputes missing values with the mean/median for a column or
    "not recorded" for categorical/string values
    Params:
        features  -> pandas df or numpy array of features
    Returns:
        new_features -> features with missing values replaced with imputed values
    """
    if isinstance(features, np.ndarray):
        df = pd.DataFrame(features)
    elif isinstance(features, pd.DataFrame):
        df = features
    else:
        print(f"Error: features parameter is of unsupported datatype: {type(features)}")
        sys.exit(1)

    df = df.convert_dtypes()
    for col_idx in df:
        if (df[col_idx].dtype) != "string":
            df[col_idx].fillna((df[col_idx].median()), inplace=True)
        else:
            df[col_idx].fillna("not recorded", inplace=True)


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

    remove_missing_features(data_x)

    # try factorizing
    data_x = factorize(data_x)
    data_y = factorize(data_y)

    x_transform = standardize_features(data_x, return_scaler=False)
    print(x_transform.shape)
    x_transform = filter_most_corr(x_transform, data_y, "mutual", 0.8)
    print(x_transform.shape)
