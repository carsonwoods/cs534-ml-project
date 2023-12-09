"""
Builds K-NN model using GridSearchCV
"""

# 3rd party imports
from sklearn.neighbors import KNeighborsClassifier

# imports from other parts of the project
from diabetes_project.read_data import get_data_df
from diabetes_project.preprocessing import (
    default_preprocessing, default_preprocessing_pca
)
from diabetes_project.model import build_generic_model


# reads data into dataframe
data_df = get_data_df()

# Run full set of preprocessing steps
train_x, test_x, train_y, test_y = default_preprocessing(data_df)

# trains K-NN classifier
results = {}
results["params"] = {}
results["params"]["n_neighbors"] = [5, 10, 15, 20, 25, 50, 100]
results["params"]["metric"] = [
    "manhattan",
    "euclidean",
    "cosine",
    "haversine",
    "minkowski",
]

# instantiates model to tune
knn_model = KNeighborsClassifier()
results = build_generic_model(
    knn_model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)

print(results)

'''
# Run again with PCA transformed data - load df from scratch to avoid any accidental inplace modifications from calls above
data_df = get_data_df()

# Run full set of preprocessing steps + PCA
train_x, test_x, train_y, test_y = default_preprocessing_pca(data_df)

# trains K-NN classifier
results = {}
results["params"] = {}
results["params"]["n_neighbors"] = [5, 10, 15, 20, 25, 50, 100]
results["params"]["metric"] = [
    "manhattan",
    "euclidean",
    "cosine",
    "haversine",
    "minkowski",
]

# instantiates model to tune
knn_model = KNeighborsClassifier()
results = build_generic_model(
    knn_model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)

print(results)
'''