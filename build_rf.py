"""
Builds Random Forest model using GridSearchCV
"""
# imports from other parts of the project
from diabetes_project.read_data import get_data_df
from diabetes_project.preprocessing import (
    factorize,
    filter_most_corr,
    get_feature_labels,
    impute_missing_value,
    remove_repeat_patients,
    default_preprocessing
)
from diabetes_project.model import build_generic_model

# 3rd party imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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

