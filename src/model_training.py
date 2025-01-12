"""
Model training module for Employee Attrition prediction.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

def train_base_models(X_train, y_train, models_dir):
    """
    Train multiple base models.
    
    Args:
        X_train: Training features
        y_train: Training target
        models_dir: Directory to save trained models
    
    Returns:
        dict: Dictionary of trained models
    """
    # Convert data to float for XGBoost
    X_train_float = X_train.astype(float)
    
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBClassifier(random_state=42)
    }
    
    # Train models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == 'xgboost':
            model.fit(X_train_float, y_train)
        else:
            model.fit(X_train, y_train)
        
        # Save model
        save_model(model, name, models_dir)
    
    return models

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using various metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Convert data to float for XGBoost
    if isinstance(model, xgb.XGBClassifier):
        X_test = X_test.astype(float)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def optimize_model(best_model, param_grid, X_train, y_train):
    """
    Perform hyperparameter optimization using GridSearchCV.
    
    Args:
        best_model: Base model to optimize
        param_grid (dict): Parameter grid for optimization
        X_train (np.array): Training features
        y_train (np.array): Training target
        
    Returns:
        object: Optimized model
    """
    grid_search = GridSearchCV(
        best_model,
        param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def save_model(model, model_name, output_dir):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        output_dir: Directory to save the model
    """
    import joblib
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_name}.joblib')
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Example hyperparameter grids for different models
    param_grids = {
        'logistic': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgboost': {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
