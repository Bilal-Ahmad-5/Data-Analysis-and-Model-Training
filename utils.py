import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import StringIO
import datetime
import re

def preprocess_data(
    df, 
    target_column, 
    excluded_features=None, 
    numerical_imputation='mean', 
    categorical_imputation='mode',
    handle_categorical=True,
    scaling_method='StandardScaler'
):
    """
    Preprocess the dataset for model training
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    target_column : str
        The name of the target variable column
    excluded_features : list, optional
        List of features to exclude
    numerical_imputation : str, optional
        Method for imputing numerical missing values ('mean', 'median', 'zero', 'none')
    categorical_imputation : str, optional
        Method for imputing categorical missing values ('mode', 'new category', 'none')
    handle_categorical : bool, optional
        Whether to encode categorical features
    scaling_method : str, optional
        Method for feature scaling ('StandardScaler', 'MinMaxScaler', 'RobustScaler', 'None')
        
    Returns:
    --------
    X : pandas.DataFrame
        Preprocessed features
    y : pandas.Series
        Target variable
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Extract target variable
    y = data[target_column]
    
    # Convert target to numeric if it's categorical
    if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Exclude specified features and target
    if excluded_features:
        data = data.drop(excluded_features, axis=1)
    X = data.drop(target_column, axis=1)
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing for numerical features
    numerical_transformer = None
    if numerical_cols:
        num_steps = []
        
        # Imputation for numerical features
        if numerical_imputation != 'none' and numerical_imputation is not None:
            if numerical_imputation == 'mean':
                num_steps.append(('imputer', SimpleImputer(strategy='mean')))
            elif numerical_imputation == 'median':
                num_steps.append(('imputer', SimpleImputer(strategy='median')))
            elif numerical_imputation == 'zero':
                num_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=0)))
        
        # Scaling for numerical features
        if scaling_method.lower() != 'none' and scaling_method is not None:
            if scaling_method == 'StandardScaler':
                num_steps.append(('scaler', StandardScaler()))
            elif scaling_method == 'MinMaxScaler':
                num_steps.append(('scaler', MinMaxScaler()))
            elif scaling_method == 'RobustScaler':
                num_steps.append(('scaler', RobustScaler()))
        
        if num_steps:
            numerical_transformer = Pipeline(steps=num_steps)
    
    # Preprocessing for categorical features
    categorical_transformer = None
    if categorical_cols and handle_categorical:
        cat_steps = []
        
        # Imputation for categorical features
        if categorical_imputation != 'none' and categorical_imputation is not None:
            if categorical_imputation == 'mode':
                cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            elif categorical_imputation == 'new category':
                cat_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
        
        # Encoding for categorical features
        cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
        
        if cat_steps:
            categorical_transformer = Pipeline(steps=cat_steps)
    
    # Combine preprocessing steps
    transformers = []
    
    if numerical_transformer is not None and numerical_cols:
        transformers.append(('numerical', numerical_transformer, numerical_cols))
    
    if categorical_transformer is not None and categorical_cols:
        transformers.append(('categorical', categorical_transformer, categorical_cols))
    
    # If we have transformers, apply them
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers)
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        feature_names = []
        
        # Get numerical feature names
        if numerical_transformer is not None and numerical_cols:
            feature_names.extend(numerical_cols)
        
        # Get one-hot encoded feature names
        if categorical_transformer is not None and categorical_cols:
            encoder = categorical_transformer.named_steps['encoder']
            encoded_cols = []
            for i, col in enumerate(categorical_cols):
                # Get the categories for this column
                if hasattr(encoder, 'categories_'):
                    categories = encoder.categories_[i]
                    encoded_cols.extend([f"{col}_{cat}" for cat in categories])
                else:
                    # Fallback if categories_ is not available
                    unique_vals = X[col].dropna().unique()
                    encoded_cols.extend([f"{col}_{val}" for val in unique_vals])
            feature_names.extend(encoded_cols)
        
        # Convert to dataframe with proper column names
        # If sparse matrix, convert to dense first
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
        
        # If we have too many features, use numbered columns
        if len(feature_names) != X_processed.shape[1]:
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        X_processed = pd.DataFrame(X_processed, columns=feature_names)
        
        return X_processed, y
    else:
        # If no transformations, return original X (except excluded and target)
        return X, y

def perform_feature_engineering(X, y):
    """
    Perform feature engineering on the preprocessed data
    
    Parameters:
    -----------
    X : pandas.DataFrame
        The preprocessed features
    y : pandas.Series
        The target variable
        
    Returns:
    --------
    X_engineered : pandas.DataFrame
        Features after engineering
    """
    # For now, we return the input features
    # This function can be expanded for more sophisticated feature engineering
    return X

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and test sets
    
    Parameters:
    -----------
    X : pandas.DataFrame
        The features
    y : pandas.Series
        The target variable
    test_size : float, optional
        The proportion of the dataset to include in the test split
    random_state : int, optional
        Random state for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : Split data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def generate_report(
    title, 
    author, 
    objective, 
    findings, 
    data, 
    target_column, 
    preprocessing_steps, 
    models, 
    results, 
    feature_importance, 
    best_model_name, 
    X_test, 
    y_test
):
    """
    Generate an HTML report of the classification analysis
    
    Parameters:
    -----------
    Various inputs related to the analysis
        
    Returns:
    --------
    html : str
        HTML string of the report
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3, h4 {{
                color: #2c3e50;
            }}
            h1 {{
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                border-bottom: 1px solid #3498db;
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .header-container {{
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .header-info {{
                text-align: right;
                font-size: 0.9em;
                color: #7f8c8d;
            }}
            .chart-container {{
                margin: 20px 0;
                max-width: 800px;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                text-align: center;
                font-size: 0.8em;
                color: #7f8c8d;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .highlight {{
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-container">
                <h1>{title}</h1>
                <div class="header-info">
                    <p>Author: {author}</p>
                    <p>Date: {now}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>1. Executive Summary</h2>
                <div class="highlight">
                    <h3>Objective</h3>
                    <p>{objective}</p>
                    
                    <h3>Key Findings</h3>
                    <p>{findings}</p>
                    
                    <h3>Best Model</h3>
                    <p>The analysis identified <strong>{best_model_name}</strong> as the best performing model for this classification task.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>2. Data Description</h2>
                <p>The analysis was performed on a dataset with {data.shape[0]} rows and {data.shape[1]} columns. 
                The target variable for classification was <strong>"{target_column}"</strong>.</p>
                
                <h3>Dataset Overview</h3>
                <table>
                    <tr>
                        <th>Attribute</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Number of Instances</td>
                        <td>{data.shape[0]}</td>
                    </tr>
                    <tr>
                        <td>Number of Features</td>
                        <td>{data.shape[1] - 1}</td>
                    </tr>
                    <tr>
                        <td>Target Variable</td>
                        <td>{target_column}</td>
                    </tr>
                    <tr>
                        <td>Target Classes</td>
                        <td>{data[target_column].nunique()}</td>
                    </tr>
                </table>
                
                <h3>Data Types Summary</h3>
                <table>
                    <tr>
                        <th>Data Type</th>
                        <th>Count</th>
                    </tr>
    """
    
    # Add data types summary
    for dtype, count in data.dtypes.value_counts().items():
        html += f"""
                    <tr>
                        <td>{dtype}</td>
                        <td>{count}</td>
                    </tr>
        """
    
    html += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>3. Data Preprocessing</h2>
                <p>The following preprocessing steps were applied to prepare the data for model training:</p>
                
                <ul>
    """
    
    # Add preprocessing steps
    for step in preprocessing_steps:
        html += f"<li>{step}</li>\n"
    
    html += f"""
                </ul>
            </div>
            
            <div class="section">
                <h2>4. Model Comparison</h2>
                <p>Several classification models were trained and evaluated. The performance metrics are summarized below:</p>
                
                <table>
                    <tr>
                        <th>Model</th>
    """
    
    # Add metric columns
    for col in results.columns:
        html += f"<th>{col.capitalize()}</th>\n"
    
    html += f"""
                    </tr>
    """
    
    # Add model results
    for model_name, row in results.iterrows():
        html += f"<tr><td>{model_name}</td>\n"
        
        for col in results.columns:
            # Format percentages for certain metrics
            if col in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                value = f"{row[col]:.2%}"
            else:
                value = f"{row[col]:.4f}"
            
            # Highlight the best model
            if model_name == best_model_name:
                html += f"<td><strong>{value}</strong></td>\n"
            else:
                html += f"<td>{value}</td>\n"
        
        html += "</tr>\n"
    
    html += f"""
                </table>
                
                <h3>Best Model Selection</h3>
                <p>Based on the evaluation metrics, <strong>{best_model_name}</strong> was selected as the best model 
                for this classification task. The selection was based primarily on overall performance across accuracy, precision, recall, and F1 score.</p>
            </div>
            
            <div class="section">
                <h2>5. Key Findings and Insights</h2>
    """
    
    # Add feature importance if available
    if feature_importance is not None and best_model_name in feature_importance:
        best_fi = feature_importance[best_model_name]
        
        if best_fi is not None and len(best_fi) > 0:
            html += f"""
                <h3>Feature Importance</h3>
                <p>The following features were identified as most influential for predicting {target_column}:</p>
                
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
            """
            
            # Sort by importance and show top 10 features
            sorted_fi = best_fi.sort_values(by='Importance', ascending=False).head(10)
            
            for idx, row in sorted_fi.iterrows():
                html += f"""
                    <tr>
                        <td>{row.name}</td>
                        <td>{row['Importance']:.4f}</td>
                    </tr>
                """
            
            html += "</table>\n"
            
            # Add insights based on top features
            html += f"""
                <h3>Interpretation of Key Features</h3>
                <div class="highlight">
                    <p>The top features driving the classification model are:</p>
                    <ul>
            """
            
            # Add top 3 features with interpretation
            for idx, row in sorted_fi.head(3).iterrows():
                html += f"<li><strong>{row.name}</strong>: A strong predictor with an importance score of {row['Importance']:.4f}</li>\n"
            
            html += f"""
                    </ul>
                    <p>These features should be the focus of attention for business decisions and further data collection efforts.</p>
                </div>
            """
    
    # Add model performance insights
    html += f"""
                <h3>Model Performance Insights</h3>
                <div class="highlight">
    """
    
    # Add accuracy insights if available
    if 'accuracy' in results.loc[best_model_name]:
        accuracy = results.loc[best_model_name, 'accuracy']
        html += f"<p>The model achieves an accuracy of {accuracy:.2%}, "
        
        if accuracy > 0.9:
            html += "which is excellent and indicates strong predictive power.</p>\n"
        elif accuracy > 0.8:
            html += "which is good and suitable for many business applications.</p>\n"
        elif accuracy > 0.7:
            html += "which is moderate. There may be room for improvement.</p>\n"
        else:
            html += "which is relatively low. The problem may require more complex modeling or additional features.</p>\n"
    
    # Add precision-recall insights if available
    if 'precision' in results.loc[best_model_name] and 'recall' in results.loc[best_model_name]:
        precision = results.loc[best_model_name, 'precision']
        recall = results.loc[best_model_name, 'recall']
        
        html += f"<p>The model shows a precision of {precision:.2%} and a recall of {recall:.2%}, "
        
        if precision > 0.8 and recall > 0.8:
            html += "indicating a balanced model that correctly identifies most positive cases while minimizing false positives.</p>\n"
        elif precision > 0.8 and recall < 0.6:
            html += "suggesting the model is conservative and may miss some positive cases, but is reliable when it does make a positive prediction.</p>\n"
        elif precision < 0.6 and recall > 0.8:
            html += "suggesting the model captures most positive cases but with a higher rate of false positives.</p>\n"
        else:
            html += "representing a moderate balance between false positives and false negatives.</p>\n"
    
    html += f"""
                </div>
            </div>
            
            <div class="section">
                <h2>6. Limitations and Future Work</h2>
                
                <h3>Current Limitations</h3>
                <ul>
                    <li>The analysis is based on the available data, which may not capture all relevant factors affecting the target variable.</li>
                    <li>The models were evaluated on a single train-test split, which may not fully represent the model's performance on new data.</li>
    """
    
    # Check for class imbalance
    target_counts = data[target_column].value_counts()
    imbalance_ratio = target_counts.max() / target_counts.min()
    
    if imbalance_ratio > 3:
        html += f"<li>The dataset shows class imbalance (ratio of {imbalance_ratio:.1f}:1), which may affect model performance and interpretation.</li>\n"
    
    html += f"""
                </ul>
                
                <h3>Recommendations for Future Work</h3>
                <ul>
                    <li>Collect additional data to improve model generalization and robustness.</li>
                    <li>Explore more advanced feature engineering techniques to capture complex relationships in the data.</li>
                    <li>Consider ensemble methods or more sophisticated models to potentially improve prediction accuracy.</li>
                    <li>Implement cross-validation for more reliable model evaluation.</li>
                    <li>Conduct a deeper analysis of misclassified instances to identify patterns and improve the model.</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>Generated on {now}</p>
                <p>Classification Model Analysis Report</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html
