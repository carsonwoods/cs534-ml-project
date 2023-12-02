"""
Builds K-NN model using GridSearchCV
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
from sklearn.neighbors import KNeighborsClassifier

# reads data into dataframe
data_df = get_data_df()

# Run full set of preprocessing steps
train_x, test_x, train_y, test_y = default_preprocessing(data_df)

# trains K-NN classifier
results = {}
results["params"] = {}
results["params"]["n_neighbors"] = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100]
results["params"]["leaf_size"] = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
results["params"]["weights"] = ["uniform", "distance"]
results["params"]["metric"] = ["manhattan", "euclidean", "cosine", "haversine", "minkowski"]

# instantiates model to tune
model = KNeighborsClassifier()
results = build_generic_model(
    model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)

print(results)

