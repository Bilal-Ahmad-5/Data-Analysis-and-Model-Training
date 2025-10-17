import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

def train_models(X_train, y_train, X_test, y_test, models_to_train, perform_tuning=False, cv_folds=5, metrics=None):
    """
    Train multiple classification models and evaluate their performance
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target variable
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target variable
    models_to_train : dict
        Dictionary with model names as keys and boolean values indicating whether to train them
    perform_tuning : bool, optional
        Whether to perform hyperparameter tuning
    cv_folds : int, optional
        Number of cross-validation folds for hyperparameter tuning
    metrics : dict, optional
        Dictionary specifying which metrics to calculate
        
    Returns:
    --------
    trained_models : dict
        Dictionary of trained models
    results : pandas.DataFrame
        DataFrame with model performance metrics
    feature_importance : dict
        Dictionary with feature importance information for each model
    """
    # Default metrics if none provided
    if metrics is None:
        metrics = {
            'accuracy': True,
            'precision': True,
            'recall': True,
            'f1': True,
            'auc': True,
            'confusion_matrix': True
        }
    
    # Initialize containers for results
    trained_models = {}
    results_dict = {}
    feature_importance = {}
    
    # Define the models and their parameters
    model_definitions = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l1', 'l2']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    }
    
    # Train the selected models
    for model_name, train_model in models_to_train.items():
        if not train_model:
            continue
        
        if model_name not in model_definitions:
            continue
        
        print(f"Training {model_name}...")
        
        model_info = model_definitions[model_name]
        model = model_info['model']
        
        # Perform hyperparameter tuning if requested
        if perform_tuning:
            param_grid = model_info['params']
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=cv_folds,
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
        else:
            model.fit(X_train, y_train)
        
        # Store the trained model
        trained_models[model_name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # For AUC, we need predict_proba
        y_pred_proba = None
        if metrics.get('auc', False) and hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                # Fall back if predict_proba fails
                pass
        
        # Calculate the selected metrics
        model_results = {}
        
        if metrics.get('accuracy', False):
            model_results['accuracy'] = accuracy_score(y_test, y_pred)
        
        if metrics.get('precision', False):
            try:
                model_results['precision'] = precision_score(y_test, y_pred, average='weighted')
            except:
                model_results['precision'] = np.nan
        
        if metrics.get('recall', False):
            try:
                model_results['recall'] = recall_score(y_test, y_pred, average='weighted')
            except:
                model_results['recall'] = np.nan
        
        if metrics.get('f1', False):
            try:
                model_results['f1'] = f1_score(y_test, y_pred, average='weighted')
            except:
                model_results['f1'] = np.nan
        
        if metrics.get('auc', False) and y_pred_proba is not None:
            try:
                # For binary classification
                if y_pred_proba.shape[1] == 2:
                    model_results['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                # For multiclass
                else:
                    model_results['auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                model_results['auc'] = np.nan
        
        if metrics.get('confusion_matrix', False):
            model_results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        # Store results
        results_dict[model_name] = model_results
        
        # Extract feature importance if available
        importance_df = None
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importance = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': importance
            })
            
            importance_df = importance_df.set_index('Feature')
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            
        elif hasattr(model, 'coef_'):
            # For linear models
            importance = np.abs(model.coef_)
            
            # For multiclass, we take the mean importance across classes
            if len(importance.shape) > 1 and importance.shape[0] > 1:
                importance = np.mean(importance, axis=0)
            else:
                # For binary classification or single-coefficient models
                importance = importance.ravel()
            
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': importance
            })
            
            importance_df = importance_df.set_index('Feature')
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        feature_importance[model_name] = importance_df
    
    # Convert results to DataFrame
    results = pd.DataFrame.from_dict(results_dict, orient='index')
    
    # Remove the confusion matrix from the DataFrame (it's stored but not displayed)
    if 'confusion_matrix' in results.columns:
        cm_dict = {model: results.loc[model, 'confusion_matrix'] for model in results.index}
        results = results.drop('confusion_matrix', axis=1)
    else:
        cm_dict = None
    
    return trained_models, results, feature_importance
