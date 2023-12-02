"""
Builds Neural Network model using GridSearchCV
"""

# 3rd party imports
from sklearn.neural_network import MLPClassifier

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

# trains NeuralNetwork classifier
results = {}
results["params"] = {}
results["params"]["hidden_layer_sizes"] = [
    (25),
    (100),
    (300),
    (25, 25),
    (100, 100),
    (300, 300),
]
results["params"]["activation"] = ["logistic", "tanh", "relu"]
results["params"]["alpha"] = [0, 0.01, 0.1, 1, 5]

# instantiates model to tune
nn_model = MLPClassifier()
results = build_generic_model(
    nn_model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)

print(results)
