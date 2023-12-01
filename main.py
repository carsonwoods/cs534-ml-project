"""
Test code for the module
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

# This code is just testing to ensure that the functions work correctly.
# This won't run unless running preprocessing.py directly

data_df = get_data_df()
data_x, data_y = get_feature_labels(data_df)
# print(data_df)
# print(data_x)
# print(data_y)

data_x = impute_missing_value(data_x)
data_x = remove_repeat_patients(data_x, new_feature=True)

# try factorizing
data_x = factorize(data_x)
data_y = factorize(data_y)

print(data_x.shape)
x_transform = filter_most_corr(data_x, data_y, "correlation", 0.75)
print(x_transform.shape)
