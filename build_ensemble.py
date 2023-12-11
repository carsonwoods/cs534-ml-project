"""
Builds Ensemble model, combining Random Forest and XGBoost
"""
# 3rd party imports
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, fbeta_score

# imports from other parts of the project
from diabetes_project.read_data import get_data_df
from diabetes_project.preprocessing import default_preprocessing
from diabetes_project.model import build_generic_model

# Function to build an ensemble model of Random Forest and XGBoost
def build_ensemble_model(
    rf_model,
    xgb_model,
    rf_params,
    xgb_params,
    train_x,
    train_y,
    test_x,
    test_y
):
    """
    Combining best performing models 
    """
    # Build Random Forest model
    rf_results = build_generic_model(rf_model, rf_params, train_x, train_y, test_x, test_y)

    # Build XGBoost model
    xgb_results = build_generic_model(xgb_model, xgb_params, train_x, train_y, test_x, test_y)

    # Combine predictions using averaging (you can also experiment with other ensemble methods)
    ensemble_predictions = (rf_results['best-model'].predict_proba(test_x) +
                            xgb_results['best-model'].predict_proba(test_x)) / 2

    # Evaluate accuracy of the ensemble model
    ensemble_accuracy = (rf_results['best-model'].score(test_x, test_y) +
                         xgb_results['best-model'].score(test_x, test_y)) / 2

    # Calculate ensemble AUC
    ensemble_auc = roc_auc_score(test_y, ensemble_predictions, multi_class="ovr")

    # Calculate ensemble F1 and F2
    ensemble_f1 = f1_score(test_y, ensemble_predictions.argmax(axis=1), average='weighted')
    ensemble_f2 = fbeta_score(test_y, ensemble_predictions.argmax(axis=1), beta=2, average='weighted')

    ensemble_results = {
        'train_auc': rf_results['train-auc'],
        'train_accuracy': rf_results['train-acc'],
        'val_auc': rf_results['val-auc'],
        'val_accuracy': rf_results['val-acc'],
        'test_auc': ensemble_auc,
        'test_accuracy': ensemble_accuracy,
        'test_f1': ensemble_f1,
        'test_f2': ensemble_f2,
        'train_time_seconds': rf_results['fit_time'],
        'rf_results': rf_results,
        'xgb_results': xgb_results,
        'ensemble_predictions': ensemble_predictions,
        'ensemble_accuracy': ensemble_accuracy
    }

    return ensemble_results

# Read data into a dataframe
data_df = get_data_df()

# Run full set of preprocessing steps
train_x, test_x, train_y, test_y = default_preprocessing(data_df)

# Random Forest parameters for tuning
rf_params = {
    'n_estimators': [50, 100, 150, 250],
    'max_depth': [2, 5, 10, 25, 50],
    'min_samples_split': [1, 5, 10, 25, 50],
}

# XGBoost parameters for tuning
xgb_params = {
    'n_estimators':  [2, 3, 4, 5, 10, 20],
    'max_depth': [2, 3, 4, 5, 10, 20],
    'max_leaves':[0, 5, 10, 25, 50],
    'learning_rate':  [0.5, 1, 2],
    'objective': ['multi:softprob'],
}

# Build and evaluate the ensemble model
random_forest_model = RandomForestClassifier()
xgboost_model = XGBClassifier()

ensemble_results = build_ensemble_model(random_forest_model, xgboost_model, rf_params, xgb_params, train_x, train_y, test_x, test_y)

print(ensemble_results)
