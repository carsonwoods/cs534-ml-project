# Diabetes Hospital Readmission Project

### Reading Data
To get a Pandas dataframe or Numpy ndarray of the raw data (no preprocessing), you can do the following:

```python
import read_data

# reads data as a pandas dataframe array
data_df = read_data.get_data_df()

# reads data as a numpy ndarray
data_np = read_data.get_data_numpy()
```

### Preprocessing
Preprocessing is handled in the `preprocessing` module and has functions for handling the data format, various approaches to preprocessing, and some basic data manipulation functions.
They are documented below:

```python
import preprocessing

# parses data from either pandas dataframe or numpy ndarray into
# feature and label ndarrays.
x, y = preprocessing.get_feature_labels(data_df) # this function accepts ndarrays or dataframes

# encodes the non-numeric data in a factorized format
x = factorize(x)

# performs standard scaling on x features
# if return_scaler is set to False, just the transformed data is returned
# if return_scaler is set to True, the transformed data is returned as well
# in a tuple alongside the scaler model from sklearn so it can be applied to other data splits
x_transform = standardize_features(data_x, return_scaler=False)
```