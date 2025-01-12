"""
Main script for Employee Attrition prediction project.
"""

import os
import pandas as pd
from data_preprocessing import load_data, preprocess_data, create_train_test_split, save_processed_data
from model_training import train_base_models, evaluate_model, optimize_model
from visualization import (
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_correlation_matrix,
    plot_categorical_distribution
)

def main():
    """
    Main function to run the analysis pipeline.
    """
    # Set up absolute paths
    base_dir = 'c:/Users/prana/CascadeProjects/employee_attrition'
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    figures_dir = os.path.join(reports_dir, 'figures')
    processed_dir = os.path.join(base_dir, 'data/processed')
    
    # Create directories
    for directory in [models_dir, reports_dir, figures_dir, processed_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(os.path.join(base_dir, 'data/raw/MFG10YearTerminationData.csv'))
    X, y, feature_names, cat_cols, num_cols = preprocess_data(df)
    
    # Save processed data
    save_processed_data(X, y, feature_names, cat_cols, num_cols, processed_dir)
    
    # Split data
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    
    # Train models
    print("\nTraining base models...")
    models = train_base_models(X_train, y_train, models_dir)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f"\n{name} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot feature importance for Random Forest
    plot_feature_importance(
        models['random_forest'],
        feature_names,
        save_path=os.path.join(figures_dir, 'feature_importance.png')
    )
    
    # Plot confusion matrix for best model (using XGBoost with float data)
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    X_test_float = X_test.astype(float) if best_model_name == 'xgboost' else X_test
    plot_confusion_matrix(
        y_test,
        models[best_model_name].predict(X_test_float),
        save_path=os.path.join(figures_dir, 'confusion_matrix.png')
    )
    
    # Plot ROC curves
    plot_roc_curves(
        {name: model for name, model in models.items()},
        X_test.astype(float),  # Convert to float for all models
        y_test,
        save_path=os.path.join(figures_dir, 'roc_curves.png')
    )
    
    print("\nAnalysis completed! Check the reports/figures directory for visualizations.")

if __name__ == "__main__":
    main()
