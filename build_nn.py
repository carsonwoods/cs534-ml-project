"""
Builds Neural Network model using GridSearchCV
"""

# imports from other parts of the project
from diabetes_project.read_data import get_data_df
from diabetes_project.preprocessing import (
    factorize,
    filter_most_corr,
    get_feature_labels,
    impute_missing_value,
    remove_repeat_patients,
)
from diabetes_project.model import build_generic_model

# 3rd party imports
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# reads data into dataframe
data_df = get_data_df()

# removes repeat patients and generates new files
data_df = remove_repeat_patients(data_df, new_feature=True)

# splits data into train and test splits
data_x, data_y = get_feature_labels(data_df)

# imputs missing values
data_x = impute_missing_value(data_x)


# factorizes features
data_x = factorize(data_x)
data_y = factorize(data_y)

# filter highly correlated features
data_x = filter_most_corr(data_x, data_y, "correlation", 0.75)

# gets test and training data
train_x, test_x, train_y, test_y = train_test_split(
    data_x, data_y, test_size=0.30, random_state=42
)



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
model = MLPClassifier()
results = build_generic_model(
    model,
    results["params"],
    train_x,
    train_y,
    test_x,
    test_y,
)

print(results)

