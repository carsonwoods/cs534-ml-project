"""
Builds Random Forest model using GridSearchCV
"""

# 3rd party imports
from sklearn.ensemble import RandomForestClassifier

# imports from other parts of the project
from diabetes_project.read_data import get_data_df
from diabetes_project.preprocessing import (
    default_preprocessing,
    default_preprocessing_pca,
    default_preprocessing_zscore,
)
from diabetes_project.model import build_generic_model

"""
# reads data into dataframe
data_df = get_data_df()

# Run full set of preprocessing steps
train_x, test_x, train_y, test_y = default_preprocessing(data_df)

# trains RandomForest classifier
results = {}
results["params"] = {}
results["params"]["n_estimators"] = [50, 100, 150, 250]
results["params"]["max_depth"] = [2, 5, 10, 25, 50]
results["params"]["min_samples_leaf"] = [1, 5, 10, 25, 50]
model = RandomForestClassifier()
results = build_generic_model(
    model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)
print(results)


# Run again with zscoring - load df from scratch to avoid any accidental inplace modifications from calls above
data_df = get_data_df()

# Run full set of preprocessing steps + zscoring
train_x, test_x, train_y, test_y = default_preprocessing_zscore(data_df)

model = RandomForestClassifier()
results = {}
results["params"] = {}
results["params"]["n_estimators"] = [50, 100, 150, 250]
results["params"]["max_depth"] = [2, 5, 10, 25, 50]
results["params"]["min_samples_leaf"] = [1, 5, 10, 25, 50]
results = build_generic_model(
    model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)
print(results)

"""
# Run again with PCA transformed data - load df from scratch to avoid any accidental inplace modifications from calls above
data_df = get_data_df()

# Run full set of preprocessing steps + PCA
train_x, test_x, train_y, test_y = default_preprocessing_pca(data_df)

model = RandomForestClassifier()
results = {}
results["params"] = {}
results["params"]["n_estimators"] = [50, 100, 150, 250]
results["params"]["max_depth"] = [2, 5, 10, 25, 50]
results["params"]["min_samples_leaf"] = [1, 5, 10, 25, 50]
results = build_generic_model(
    model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)
print(results)
