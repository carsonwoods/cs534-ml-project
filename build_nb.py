"""
Builds NaiveBayes model
"""

# 3rd party imports
from sklearn.naive_bayes import GaussianNB

# imports from other parts of the project
from diabetes_project.read_data import get_data_df
from diabetes_project.preprocessing import (
    default_preprocessing,
    default_preprocessing_pca,
    default_preprocessing_zscore
)
from diabetes_project.model import build_generic_model


# reads data into dataframe
data_df = get_data_df()

# Run full set of preprocessing steps
train_x, test_x, train_y, test_y = default_preprocessing(data_df)

results = {}
results["params"] = {}
results["params"]["var_smoothing"] = [0.75, 0.1, 0.15, 0.5, 1.0, 3.0, 5.0]


xgboost_model = GaussianNB()
results = build_generic_model(
    xgboost_model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)

print(results)