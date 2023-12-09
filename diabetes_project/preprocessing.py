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
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, NMF


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
        print(
            f"Error: data parameter is of unsupported datatype: {type(data)}"
        )
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

    return np.delete(features, list(remove_list), 1)


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


def standardize_features(data):
    """
    Uses sklearn to perform standard scaling on features.
    With scaler being returned as well to apply transformation to
    a different dataset (fit model on train, apply to test, etc.)
    Params:
        data          -> the features of the data to be scaled
    Returns:
        x_transformed -> the standardizes input data
    """
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        print(
            f"Error: data parameter is of unsupported datatype: {type(data)}"
        )
        sys.exit(1)

    df = df.convert_dtypes()
    for col_idx in df:
        if (df[col_idx].dtype) != "string":
            scaler = StandardScaler()
            scaler.fit(df[col_idx].to_numpy().reshape(-1,1))
            df[col_idx] = scaler.transform(df[col_idx].to_numpy().reshape(-1,1))

    return df.to_numpy()


def remove_missing_features(features, threshold=0.5, return_type="ndarray"):
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
        print(
            f"""Error: features parameter is
            of unsupported datatype: {type(features)}"""
        )
        sys.exit(1)

    if not isinstance(threshold, float) or threshold >= 1 or threshold <= 0:
        print("Error: threshold must be a float between 0 and 1")
        sys.exit(1)

    if return_type not in ["dataframe", "ndarray"]:
        print('Error: return_type must be either "dataframe" or "ndarray"')
        sys.exit(1)

    remove_list = []

    for column in df:
        count_nan = df[column].isnull().sum()
        if count_nan / df.shape[0] >= threshold:
            remove_list.append(column)

    df.drop(remove_list, axis=1, inplace=True)

    if return_type == "dataframe":
        return df

    return df.to_numpy()


def remove_repeat_patients(features, new_feature=False, return_type="ndarray"):
    """
    Removes samples from patients who have already been seen
    Params:
        features    -> pandas dataframe of features
        new_feature -> specify whether to add new feature corresponding
                       to how frequently a patient reappears in the dataset
        return_type -> specify if you want the features returned as ndarray
                       or as a dataframe
    Returns:
        x_fs      -> features with repeat patients removed
                     (ndarray or dataframe)
    """
    if isinstance(features, np.ndarray):
        df = pd.DataFrame(features)
    elif isinstance(features, pd.DataFrame):
        df = features
    else:
        print(
            f"""Error: features parameter is of
                unsupported datatype: {type(features)}"""
        )
        sys.exit(1)

    if return_type not in ["dataframe", "ndarray"]:
        print('Error: return_type must be either "dataframe" or "ndarray"')
        sys.exit(1)

    seen_list = []
    remove_list = []
    count_list = []

    for index, row in df.iterrows():
        if row.iloc[1] not in seen_list:
            seen_list.append(row.iloc[1])
            count_list.append(1)
        else:
            remove_list.append(index)
            count_list[seen_list.index(row.iloc[1])] += 1

    df.drop(remove_list, axis=0, inplace=True)

    # if specified, add number of time patient appears in dataset
    if new_feature:
        df.insert(len(df.columns) - 1, "repeat_patient_count", count_list)

    if return_type == "dataframe":
        return df

    return df.to_numpy()


def impute_missing_value(features, return_type="ndarray"):
    """
    Imputes missing values with the mean/median for a column or
    "not recorded" for categorical/string values
    Params:
        features  -> pandas df or numpy array of features
    Returns:
        new_features -> features with missing
                        values replaced with imputed values
    """
    if isinstance(features, np.ndarray):
        df = pd.DataFrame(features)
    elif isinstance(features, pd.DataFrame):
        df = features
    else:
        print(
            f"""Error: features parameter is of
                unsupported datatype: {type(features)}"""
        )
        sys.exit(1)

    if return_type not in ["dataframe", "ndarray"]:
        print('Error: return_type must be either "dataframe" or "ndarray"')
        sys.exit(1)

    df = df.convert_dtypes()
    for col_idx in df:
        if (df[col_idx].dtype) != "string":
            df[col_idx].fillna((df[col_idx].median()), inplace=True)
        else:
            df[col_idx].fillna("not recorded", inplace=True)

    if return_type == "dataframe":
        return df

    return df.to_numpy()


def remove_constant_features(features, return_type="ndarray"):
    """
    Removes any features with only one possible value
    (constant value across observations)
    Params:
        features  -> pandas df or numpy array of features
        return_type -> format to return transformed feature set
    Returns:
        new_features -> features with constant variables dropped
    """

    if isinstance(features, np.ndarray):
        df = pd.DataFrame(features)
    elif isinstance(features, pd.DataFrame):
        df = features
    else:
        print(
            f"""Error: features parameter is of
                unsupported datatype: {type(features)}"""
        )
        sys.exit(1)

    if return_type not in ["dataframe", "ndarray"]:
        print('Error: return_type must be either "dataframe" or "ndarray"')
        sys.exit(1)

    for col_idx in df:
        if len(df[col_idx].unique()) == 1:
            df.drop(columns=[col_idx], inplace=True)

    if return_type == "dataframe":
        return df

    return df.to_numpy()

def run_pca(train_x, test_x):

    # Z-score normalize input data
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    pca = PCA()
    pca.fit(train_x)

    # Select number of components that explain at least 95% of the variance
    sum = 0
    n_components = 0
    for i in range(0,len(pca.explained_variance_ratio_)):
        sum += pca.explained_variance_ratio_[i]
        if sum > 0.95:
            n_components = i + 1
            break

    # Transform training set and select features up to n_components
    train_transformed = pca.transform(train_x)
    train_transformed = train_transformed[:,0:n_components]

    # Transform test set and select features up to n_components
    test_transformed = pca.transform(test_x)
    test_transformed = test_transformed[:,0:n_components]

    return (train_transformed, test_transformed)


def default_preprocessing(data_df):
    """
    Conducts default preprocessing for this data
        - Remove duplicate records for same patient
        - Remove features with constant value
        - Remove features with > 40% missing values
        - Split features and target
        - Impute missing features with mean (numerical)
          or "Not recorded" (categorical)
        - Factorize categorical features/target
        - Filter correlated features with corr > 0.75
        - Split train/test in 70/30 ratio, uses random
          seed for reproducible split across model training runs
    Params:
        data_df  -> pandas df, output from get_data_df
    Returns:
        train_x, test_x, train_y, test_y -> numpy arrays for model fitting
    """
    # removes repeat patients and generates new files
    data_df = remove_repeat_patients(data_df, new_feature=True)

    # removes any features with constant values
    data_df = remove_constant_features(data_df)

    # remove features with > 40% missing values
    data_df = remove_missing_features(data_df, threshold=0.4)

    # splits data into train and test splits
    data_x, data_y = get_feature_labels(data_df)

    # imputs missing values
    data_x = impute_missing_value(data_x)

    # factorizes features
    data_x = factorize(data_x)
    data_y = factorize(data_y)

    # filter highly correlated features
    data_x = filter_most_corr(data_x, data_y, "correlation", 0.75)

    # gets test and training data
    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, test_size=0.30, random_state=42
    )

    return (train_x, test_x, train_y, test_y)

def default_preprocessing_pca(data_df):
    """
    Conducts default preprocessing for this data above plus PCA dimensionality reduction
    Params:
        data_df  -> pandas df, output from get_data_df
    Returns:
        train_x, test_x, train_y, test_y -> numpy arrays for model fitting
    """
    # removes repeat patients and generates new files
    data_df = remove_repeat_patients(data_df, new_feature=True)

    # removes any features with constant values
    data_df = remove_constant_features(data_df)

    # remove features with > 40% missing values
    data_df = remove_missing_features(data_df, threshold=0.4)

    # splits data into train and test splits
    data_x, data_y = get_feature_labels(data_df)

    # imputs missing values
    data_x = impute_missing_value(data_x)

    # factorizes features
    data_x = factorize(data_x)
    data_y = factorize(data_y)

    # filter highly correlated features
    data_x = filter_most_corr(data_x, data_y, "correlation", 0.75)

    # gets test and training data
    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, test_size=0.30, random_state=42
    )

    train_x, test_x = run_pca(train_x, test_x)
    return (train_x, test_x, train_y, test_y)


if __name__ == "__main__":
    # This code is just testing to ensure that the functions work correctly.
    # This won't run unless running preprocessing.py directly

    # imports from other parts of the project
    from read_data import get_data_df

    data_df = get_data_df()
    data_x, data_y = get_feature_labels(data_df)

    print(data_x)
    data_x = standardize_features(data_x)
    print(data_x)
    exit(0)

    print(f"Before: {data_x.shape}")
    remove_missing_features(data_x)
    print(f"After: {data_x.shape}")

    # try factorizing
    data_x = factorize(data_x)
    data_y = factorize(data_y)

    # x_transform = standardize_features(data_x, return_scaler=False)
    print(data_x.shape)
    x_transform = filter_most_corr(data_x, data_y, "correlation", 0.8)
    print(data_x.shape)
