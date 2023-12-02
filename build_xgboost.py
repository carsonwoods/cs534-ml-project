"""
Builds XGBoost model using GridSearchCV
"""

# 3rd party imports
from xgboost import XGBClassifier

# imports from other parts of the project
from diabetes_project.read_data import get_data_df
from diabetes_project.preprocessing import (
    default_preprocessing,
)
from diabetes_project.model import build_generic_model


# reads data into dataframe
data_df = get_data_df()

# Run full set of preprocessing steps
train_x, test_x, train_y, test_y = default_preprocessing(data_df)

results = {}
results["params"] = {}
results["params"]["n_estimators"] = [2, 3, 4, 5, 10, 20]
results["params"]["max_depth"] = [2, 3, 4, 5, 10, 20]
results["params"]["max_leaves"] = [0, 5, 10, 25, 50]
results["params"]["learning_rate"] = [0.5, 1, 2]
results["params"]["objective"] = ["multi:softprob"]

xgboost_model = XGBClassifier()
results = build_generic_model(
    xgboost_model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)

print(results)
