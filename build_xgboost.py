"""
Builds XGBoost model using GridSearchCV
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
from xgboost import XGBClassifier

# reads data into dataframe
data_df = get_data_df()

# Run full set of preprocessing steps
train_x, test_x, train_y, test_y = default_preprocessing(data_df)

results = {}
results["params"] = {}
results["params"]["n_estimators"] = [2, 3, 4, 5, 10, 20]
results["params"]["max_depth"] = [2, 3, 4, 5, 10, 20]
results["params"]["max_leaves"] = [0, 5, 10, 25, 50]
results["params"]["learning_rate"] = [.5, 1, 2]
results["params"]["objective"] = ['multi:softprob']

model = XGBClassifier()
results = build_generic_model(
    model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)

print(results)



