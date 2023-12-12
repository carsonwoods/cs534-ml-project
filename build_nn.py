"""
Builds Neural Network model using GridSearchCV
"""


# 3rd party imports
from sklearn.neural_network import MLPClassifier

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

# trains NeuralNetwork classifier
results = {}
results["params"] = {}
results["params"]["hidden_layer_sizes"] = [
    (25),
    (100),
    (300),
    (25, 25, 25),
    (100, 100, 100),
    (300, 300, 300),
    (25, 25, 25, 25, 25),
    (100, 100, 100, 100, 100),
    (300, 300, 300, 300, 300),
]
results["params"]["activation"] = ["logistic", "tanh", "relu"]
results["params"]["alpha"] = [0, 0.01, 0.1, 1, 5]
# results["params"]["learning_rate"] = [0, 0.001, 0.01, 0.1, 1, 5]


# instantiates model to tune
nn_model = MLPClassifier(random_state=42)
results = build_generic_model(
    nn_model,
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

# trains NeuralNetwork classifier
results = {}
results["params"] = {}
results["params"]["hidden_layer_sizes"] = [
    (25),
    (100),
    (300),
    (25, 25, 25),
    (100, 100, 100),
    (300, 300, 300),
    (25, 25, 25, 25, 25),
    (100, 100, 100, 100, 100),
    (300, 300, 300, 300, 300),
]
results["params"]["activation"] = ["logistic", "tanh", "relu"]
results["params"]["alpha"] = [0, 0.01, 0.1, 1, 5]
# results["params"]["learning_rate"] = [0, 0.001, 0.01, 0.1, 1, 5]


# instantiates model to tune
nn_model = MLPClassifier(random_state=42)
results = build_generic_model(
    nn_model,
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

# trains NeuralNetwork classifier
results = {}
results["params"] = {}
results["params"]["hidden_layer_sizes"] = [
    (25),
    (100),
    (300),
    (25, 25, 25),
    (100, 100, 100),
    (300, 300, 300),
    (25, 25, 25, 25, 25),
    (100, 100, 100, 100, 100),
    (300, 300, 300, 300, 300),
]
results["params"]["activation"] = ["logistic", "tanh", "relu"]
results["params"]["alpha"] = [0, 0.01, 0.1, 1, 5]
# results["params"]["learning_rate"] = [0, 0.001, 0.01, 0.1, 1, 5]


# instantiates model to tune
nn_model = MLPClassifier(random_state=42)
results = build_generic_model(
    nn_model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)
print(results)
