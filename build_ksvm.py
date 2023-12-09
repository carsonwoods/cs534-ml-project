"""
Builds K-SVM model using GridSearchCV
"""

# 3rd party imports
from sklearn.svm import SVC

# imports from other parts of the project
from diabetes_project.read_data import get_data_df
from diabetes_project.preprocessing import (
    remove_repeat_patients,
    default_preprocessing,
    default_preprocessing_pca,
    default_preprocessing_zscore
)
from diabetes_project.model import build_generic_model

# reads data into dataframe
data_df = get_data_df()

# removes repeat patients and generates new files
data_df = remove_repeat_patients(data_df, new_feature=True)

# Run full set of preprocessing steps
train_x, test_x, train_y, test_y = default_preprocessing(data_df)

# trains K-NN classifier
results = {}
results["params"] = {}
results["params"]["C"] = [0.1, 0.25, 0.5, 1, 2, 5]
results["params"]["degree"] = [1, 2, 3, 4, 5, 7, 8, 9, 10]
results["params"]["kernel"] = ["linear", "poly", "rbf"]

# instantiates model to tune
svm_model = SVC(max_iter=20000, probability=True)
results = build_generic_model(
    svm_model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)

print(results)

'''
# Run again with z-scoring - load df from scratch to avoid any accidental inplace modifications from calls above
data_df = get_data_df()

# Run full set of preprocessing steps + z-scoring
train_x, test_x, train_y, test_y = default_preprocessing_zscore(data_df)

# trains K-NN classifier
results = {}
results["params"] = {}
results["params"]["C"] = [0.1, 0.25, 0.5, 1, 2, 5]
results["params"]["degree"] = [1, 2, 3, 4, 5, 7, 8, 9, 10]
results["params"]["kernel"] = ["linear", "poly", "rbf"]

# instantiates model to tune
svm_model = SVC(max_iter=20000, probability=True)
results = build_generic_model(
    svm_model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)

print(results)

# Run again with PCA transformed data - load df from scratch to avoid any accidental inplace modifications from calls above
data_df = get_data_df()

# Run full set of preprocessing steps + PCA
train_x, test_x, train_y, test_y = default_preprocessing_pca(data_df)

# trains K-NN classifier
results = {}
results["params"] = {}
results["params"]["C"] = [0.1, 0.25, 0.5, 1, 2, 5]
results["params"]["degree"] = [1, 2, 3, 4, 5, 7, 8, 9, 10]
results["params"]["kernel"] = ["linear", "poly", "rbf"]

# instantiates model to tune
svm_model = SVC(max_iter=20000, probability=True)
results = build_generic_model(
    svm_model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)

print(results)
'''