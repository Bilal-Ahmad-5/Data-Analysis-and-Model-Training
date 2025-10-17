import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import base64
import pickle
import os
from datetime import datetime

from utils import (preprocess_data, perform_feature_engineering, 
                  split_data, generate_report)
from models import train_models
from visualization import (plot_correlation_matrix, plot_class_distribution, 
                          plot_feature_importance, plot_confusion_matrices, 
                          plot_roc_curves, plot_precision_recall_curves)
from database import Dataset, ModelResult, SampleDataset

# Set page config
st.set_page_config(
    page_title="Classification Model Builder & Analyzer",
    page_icon="📊",
    layout="wide"
)

# Application title and description
st.title("Classification Model Builder & Analyzer")
st.markdown("""
This application helps you build, compare, and analyze classification models to produce insightful reports.
Upload your dataset, select your target variable, and explore various classification models to gain valuable insights.
""")

# Initialize session state for storing data and results
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'preprocessing_steps' not in st.session_state:
    st.session_state.preprocessing_steps = []
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None

# Create tabs for different stages of the workflow
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Data Upload & Exploration", 
    "2. Data Preprocessing", 
    "3. Model Training & Evaluation", 
    "4. Model Insights", 
    "5. Report Generation"
])

# Tab 1: Data Upload & Exploration
with tab1:
    st.header("Data Upload & Exploration")
    
    # Add tabs for data source selection
    data_source_tab1, data_source_tab2, data_source_tab3 = st.tabs(["Upload Data", "Use Sample Data", "Saved Datasets"])
    
    with data_source_tab1:
        # File upload widget
        uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                # Attempt to read the file based on file type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Option to save dataset to database
                save_dataset = st.checkbox("Save this dataset to the database for future use")
                
                if save_dataset:
                    dataset_name = st.text_input("Dataset Name", value=uploaded_file.name)
                    dataset_description = st.text_area("Dataset Description (optional)")
                    
                    if st.button("Save Dataset"):
                        if dataset_name:
                            try:
                                # Save to database
                                dataset_id = Dataset.store_dataframe(
                                    name=dataset_name,
                                    df=df,
                                    description=dataset_description
                                )
                                st.success(f"Dataset saved successfully with ID: {dataset_id}")
                            except Exception as e:
                                st.error(f"Error saving dataset: {e}")
                        else:
                            st.warning("Please provide a name for the dataset")
                
                # Store the dataframe in session state
                st.session_state.data = df
            except Exception as e:
                st.error(f"Error loading the dataset: {e}")
    
    with data_source_tab2:
        st.subheader("Sample Datasets")
        st.write("Choose from these sample datasets to explore the application's features:")
        
        # Get available sample datasets
        sample_datasets = SampleDataset.get_sample_datasets()
        
        if sample_datasets:
            # Create selection for datasets
            sample_options = {f"{dataset[1]} (ID: {dataset[0]})": dataset[0] for dataset in sample_datasets}
            selected_sample = st.selectbox("Select a sample dataset:", list(sample_options.keys()))
            
            if selected_sample:
                selected_id = sample_options[selected_sample]
                
                # Show dataset description
                for dataset in sample_datasets:
                    if dataset[0] == selected_id:
                        st.markdown(f"**Description:** {dataset[2]}")
                        break
                
                # Load sample dataset button
                if st.button("Load Sample Dataset"):
                    with st.spinner("Loading dataset..."):
                        try:
                            df, name, description = SampleDataset.retrieve_sample_dataframe(selected_id)
                            if df is not None:
                                st.session_state.data = df
                                st.success(f"Sample dataset '{name}' loaded successfully!")
                                st.experimental_rerun()
                            else:
                                st.error("Could not load the selected dataset")
                        except Exception as e:
                            st.error(f"Error loading dataset: {e}")
        else:
            st.info("No sample datasets available")
    
    with data_source_tab3:
        st.subheader("Your Saved Datasets")
        
        # Get all saved datasets
        saved_datasets = Dataset.get_all_datasets()
        
        if saved_datasets:
            # Create a DataFrame for better display
            datasets_df = pd.DataFrame(saved_datasets, columns=["ID", "Name", "Description", "Created At"])
            st.dataframe(datasets_df)
            
            # Create selection for datasets
            dataset_options = {f"{dataset[1]} (ID: {dataset[0]}, Created: {dataset[3]})": dataset[0] for dataset in saved_datasets}
            selected_dataset = st.selectbox("Select a dataset to load:", list(dataset_options.keys()))
            
            if selected_dataset:
                selected_id = dataset_options[selected_dataset]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Load dataset button
                    if st.button("Load Dataset"):
                        with st.spinner("Loading dataset..."):
                            try:
                                df, name, description = Dataset.retrieve_dataframe(selected_id)
                                if df is not None:
                                    st.session_state.data = df
                                    st.success(f"Dataset '{name}' loaded successfully!")
                                    st.experimental_rerun()
                                else:
                                    st.error("Could not load the selected dataset")
                            except Exception as e:
                                st.error(f"Error loading dataset: {e}")
                
                with col2:
                    # Delete dataset button
                    if st.button("Delete Dataset", key="delete_dataset"):
                        if st.session_state.get('confirm_delete', False):
                            # Delete the dataset
                            if Dataset.delete_dataset(selected_id):
                                st.success("Dataset deleted successfully!")
                                st.session_state.confirm_delete = False
                                st.experimental_rerun()
                            else:
                                st.error("Could not delete the dataset")
                        else:
                            st.session_state.confirm_delete = True
                            st.warning("Are you sure you want to delete this dataset? Click Delete Dataset again to confirm.")
        else:
            st.info("No saved datasets available. Upload and save a dataset to see it here.")
    
    # Display dataset info if it's loaded
    if st.session_state.data is not None:
        df = st.session_state.data
        try:
            # Display basic information about the dataset
            st.subheader("Dataset Overview")
            st.write(f"**Number of rows:** {df.shape[0]}")
            st.write(f"**Number of columns:** {df.shape[1]}")
            
            # Display first few rows of the dataset
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Display column information
            st.subheader("Column Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types:**")
                st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))
            
            with col2:
                st.write("**Missing Values:**")
                st.dataframe(pd.DataFrame(df.isnull().sum(), columns=["Missing Values"]))
            
            # Allow user to select target variable
            st.subheader("Target Variable Selection")
            st.session_state.target_column = st.selectbox(
                "Select your target variable (classification outcome):",
                df.columns.tolist()
            )
            
            if st.session_state.target_column:
                # Display distribution of target variable
                st.subheader(f"Distribution of Target Variable: {st.session_state.target_column}")
                
                if df[st.session_state.target_column].dtype in ['int64', 'float64'] and df[st.session_state.target_column].nunique() <= 10:
                    # For numerical targets with few unique values (likely classification)
                    fig = plot_class_distribution(df, st.session_state.target_column)
                    st.pyplot(fig)
                elif df[st.session_state.target_column].dtype == 'object' or df[st.session_state.target_column].dtype.name == 'category':
                    # For categorical targets
                    fig = plot_class_distribution(df, st.session_state.target_column)
                    st.pyplot(fig)
                else:
                    st.warning("Please select a categorical target variable for classification.")
                
                # Display correlation matrix for numerical columns
                st.subheader("Correlation Matrix (Numerical Features)")
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if len(numeric_cols) > 1:  # Need at least 2 columns for correlation
                    fig = plot_correlation_matrix(df[numeric_cols])
                    st.pyplot(fig)
                else:
                    st.info("Not enough numerical columns to generate a correlation matrix.")
        except Exception as e:
            st.error(f"Error displaying dataset information: {e}")

# Tab 2: Data Preprocessing
with tab2:
    st.header("Data Preprocessing")
    
    if st.session_state.data is not None and st.session_state.target_column is not None:
        df = st.session_state.data
        
        # Feature selection
        st.subheader("Feature Selection")
        
        # Let user select features to exclude
        all_columns = df.columns.tolist()
        if st.session_state.target_column in all_columns:
            all_columns.remove(st.session_state.target_column)
        
        excluded_features = st.multiselect(
            "Select features to exclude from the model:",
            all_columns
        )
        
        # Handling missing values
        st.subheader("Missing Value Handling")
        
        numerical_imputation = st.selectbox(
            "Imputation method for numerical features:",
            ["Mean", "Median", "Zero", "None"]
        )
        
        categorical_imputation = st.selectbox(
            "Imputation method for categorical features:",
            ["Mode", "New category", "None"]
        )
        
        # Feature engineering
        st.subheader("Feature Engineering")
        
        handle_categorical = st.checkbox("Encode categorical features", value=True)
        scaling_method = st.selectbox(
            "Feature scaling method:",
            ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
        )
        
        # Data splitting
        st.subheader("Data Splitting")
        
        test_size = st.slider("Test set size (%):", 10, 50, 20) / 100
        random_state = st.number_input("Random state (for reproducibility):", 0, 1000, 42)
        
        # Process button
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                try:
                    # Record preprocessing steps for the report
                    st.session_state.preprocessing_steps = [
                        f"Excluded features: {', '.join(excluded_features) if excluded_features else 'None'}",
                        f"Numerical imputation: {numerical_imputation}",
                        f"Categorical imputation: {categorical_imputation}",
                        f"Categorical encoding: {'Yes' if handle_categorical else 'No'}",
                        f"Scaling method: {scaling_method}",
                        f"Test size: {test_size}",
                        f"Random state: {random_state}"
                    ]
                    
                    # Preprocess data
                    X, y = preprocess_data(
                        df,
                        target_column=st.session_state.target_column,
                        excluded_features=excluded_features,
                        numerical_imputation=numerical_imputation.lower(),
                        categorical_imputation=categorical_imputation.lower(),
                        handle_categorical=handle_categorical,
                        scaling_method=scaling_method
                    )
                    
                    # Feature engineering
                    X = perform_feature_engineering(X, y)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = split_data(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    st.success("Data preprocessing completed successfully!")
                    
                    # Display preprocessed data info
                    st.subheader("Preprocessed Dataset Information")
                    st.write(f"**Training set shape:** {X_train.shape}")
                    st.write(f"**Test set shape:** {X_test.shape}")
                    
                    # Display class distribution after split
                    st.subheader("Target Distribution in Train/Test Sets")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Training set:**")
                        st.dataframe(pd.Series(y_train).value_counts().rename_axis('Class').reset_index(name='Count'))
                    
                    with col2:
                        st.write("**Test set:**")
                        st.dataframe(pd.Series(y_test).value_counts().rename_axis('Class').reset_index(name='Count'))
                
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
    else:
        st.warning("Please upload a dataset and select a target variable in the 'Data Upload & Exploration' tab first.")

# Tab 3: Model Training & Evaluation
with tab3:
    st.header("Model Training & Evaluation")
    
    if (st.session_state.X_train is not None and 
        st.session_state.y_train is not None and 
        st.session_state.X_test is not None and 
        st.session_state.y_test is not None):
        
        # Model selection
        st.subheader("Select Models to Train")
        
        col1, col2 = st.columns(2)
        
        with col1:
            logistic_regression = st.checkbox("Logistic Regression", value=True)
            decision_tree = st.checkbox("Decision Tree", value=True)
            random_forest = st.checkbox("Random Forest", value=True)
        
        with col2:
            svm = st.checkbox("Support Vector Machine", value=False)
            knn = st.checkbox("K-Nearest Neighbors", value=False)
            gradient_boosting = st.checkbox("Gradient Boosting", value=True)
        
        # Hyperparameter tuning
        st.subheader("Hyperparameter Tuning")
        perform_tuning = st.checkbox("Perform hyperparameter tuning", value=False)
        
        if perform_tuning:
            st.info("Note: Hyperparameter tuning may take some time depending on the dataset size.")
            cv_folds = st.slider("Cross-validation folds:", 3, 10, 5)
        else:
            cv_folds = 5  # Default value
        
        # Evaluation metrics
        st.subheader("Evaluation Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_accuracy = st.checkbox("Accuracy", value=True)
            use_precision = st.checkbox("Precision", value=True)
            use_recall = st.checkbox("Recall", value=True)
        
        with col2:
            use_f1 = st.checkbox("F1 Score", value=True)
            use_auc = st.checkbox("AUC-ROC", value=True)
            use_confusion_matrix = st.checkbox("Confusion Matrix", value=True)
        
        # Training button
        if st.button("Train Models"):
            # Check if at least one model is selected
            if not any([logistic_regression, decision_tree, random_forest, svm, knn, gradient_boosting]):
                st.error("Please select at least one model to train.")
            else:
                with st.spinner("Training models... This may take a few minutes."):
                    try:
                        # Prepare model selection dictionary
                        models_to_train = {
                            'Logistic Regression': logistic_regression,
                            'Decision Tree': decision_tree,
                            'Random Forest': random_forest,
                            'SVM': svm,
                            'KNN': knn,
                            'Gradient Boosting': gradient_boosting
                        }
                        
                        # Prepare metrics selection
                        metrics = {
                            'accuracy': use_accuracy,
                            'precision': use_precision,
                            'recall': use_recall,
                            'f1': use_f1,
                            'auc': use_auc,
                            'confusion_matrix': use_confusion_matrix
                        }
                        
                        # Train models
                        trained_models, results, feature_importance = train_models(
                            st.session_state.X_train, 
                            st.session_state.y_train,
                            st.session_state.X_test,
                            st.session_state.y_test,
                            models_to_train,
                            perform_tuning=perform_tuning,
                            cv_folds=cv_folds,
                            metrics=metrics
                        )
                        
                        # Store results in session state
                        st.session_state.models = trained_models
                        st.session_state.results = results
                        st.session_state.feature_importance = feature_importance
                        
                        # Find best model based on F1 score or accuracy
                        if use_f1:
                            best_metric = 'f1'
                        elif use_accuracy:
                            best_metric = 'accuracy'
                        else:
                            best_metric = next((m for m, selected in metrics.items() if selected), None)
                        
                        if best_metric and best_metric in results.columns:
                            best_idx = results[best_metric].idxmax()
                            st.session_state.best_model_name = best_idx
                            st.session_state.best_model = trained_models[best_idx]
                        
                        st.success("Models trained successfully!")
                        
                        # Display results table
                        st.subheader("Model Performance Comparison")
                        st.dataframe(results)
                        
                        # Display ROC curves if AUC was selected
                        if use_auc:
                            st.subheader("ROC Curves")
                            fig = plot_roc_curves(trained_models, st.session_state.X_test, st.session_state.y_test)
                            st.pyplot(fig)
                        
                        # Display confusion matrices if selected
                        if use_confusion_matrix:
                            st.subheader("Confusion Matrices")
                            fig = plot_confusion_matrices(trained_models, st.session_state.X_test, st.session_state.y_test)
                            st.pyplot(fig)
                        
                        # Highlight best model
                        if st.session_state.best_model_name:
                            st.subheader("Best Performing Model")
                            st.info(f"The best performing model based on {best_metric} is: **{st.session_state.best_model_name}**")
                            st.write("Performance metrics:")
                            best_model_results = results.loc[st.session_state.best_model_name]
                            st.dataframe(pd.DataFrame(best_model_results).T)
                    
                    except Exception as e:
                        st.error(f"Error during model training: {e}")
    else:
        st.warning("Please preprocess your data in the 'Data Preprocessing' tab first.")

# Tab 4: Model Insights
with tab4:
    st.header("Model Insights and Analysis")
    
    if st.session_state.models is not None and st.session_state.best_model_name is not None:
        # Feature importance
        st.subheader("Feature Importance Analysis")
        
        if st.session_state.feature_importance is not None and st.session_state.best_model_name in st.session_state.feature_importance:
            # Get feature importance for the best model
            best_fi = st.session_state.feature_importance[st.session_state.best_model_name]
            
            if best_fi is not None and len(best_fi) > 0:
                # Plot feature importance
                fig = plot_feature_importance(
                    best_fi, 
                    title=f"Feature Importance - {st.session_state.best_model_name}"
                )
                st.pyplot(fig)
                
                # Display feature importance as a table
                st.subheader("Feature Importance Table")
                st.dataframe(best_fi)
            else:
                st.info(f"Feature importance not available for {st.session_state.best_model_name}.")
        else:
            st.info("Feature importance information not available for the selected model.")
        
        # Precision-Recall Curve
        st.subheader("Precision-Recall Analysis")
        fig = plot_precision_recall_curves(
            {st.session_state.best_model_name: st.session_state.models[st.session_state.best_model_name]},
            st.session_state.X_test,
            st.session_state.y_test
        )
        st.pyplot(fig)
        
        # Key findings and insights
        st.subheader("Key Findings and Insights")
        
        # Model performance summary
        st.write("**Model Performance Summary:**")
        results_df = st.session_state.results.loc[st.session_state.best_model_name]
        
        # Create insights based on model results
        st.write("**Model Insights:**")
        
        # Accuracy analysis
        if 'accuracy' in results_df:
            accuracy = results_df['accuracy']
            if accuracy > 0.9:
                st.write(f"- The model achieves high accuracy ({accuracy:.2%}), indicating strong overall performance.")
            elif accuracy > 0.8:
                st.write(f"- The model achieves good accuracy ({accuracy:.2%}), but there might be room for improvement.")
            elif accuracy > 0.7:
                st.write(f"- The model achieves moderate accuracy ({accuracy:.2%}). Consider more features or advanced models.")
            else:
                st.write(f"- The model achieves low accuracy ({accuracy:.2%}). The problem may require more complex modeling or additional features.")
        
        # Precision-Recall analysis
        if 'precision' in results_df and 'recall' in results_df:
            precision = results_df['precision']
            recall = results_df['recall']
            
            if precision > 0.8 and recall > 0.8:
                st.write("- The model has both high precision and recall, indicating it correctly identifies most positive cases while minimizing false positives.")
            elif precision > 0.8 and recall < 0.6:
                st.write("- The model has high precision but low recall, indicating it's conservative and misses many positive cases.")
            elif precision < 0.6 and recall > 0.8:
                st.write("- The model has high recall but low precision, indicating it captures most positive cases but with many false positives.")
            else:
                st.write("- Both precision and recall are moderate, suggesting a balanced approach between false positives and false negatives.")
        
        # Feature importance insights
        if st.session_state.feature_importance is not None and st.session_state.best_model_name in st.session_state.feature_importance:
            best_fi = st.session_state.feature_importance[st.session_state.best_model_name]
            
            if best_fi is not None and len(best_fi) > 0:
                # Get top 3 features
                top_features = best_fi.sort_values(by='Importance', ascending=False).head(3)
                
                st.write("**Top Predictive Features:**")
                for idx, row in top_features.iterrows():
                    st.write(f"- {row.name}: {row['Importance']:.4f}")
                
                st.write("These features have the strongest influence on model predictions. Consider focusing on these variables for further analysis or data collection.")
        
        # Model comparison insights
        st.write("**Model Comparison Insights:**")
        
        if st.session_state.results is not None and len(st.session_state.results) > 1:
            # Get the second best model
            if 'f1' in st.session_state.results.columns:
                second_best = st.session_state.results[st.session_state.results.index != st.session_state.best_model_name]['f1'].idxmax()
                metric = 'f1'
            elif 'accuracy' in st.session_state.results.columns:
                second_best = st.session_state.results[st.session_state.results.index != st.session_state.best_model_name]['accuracy'].idxmax()
                metric = 'accuracy'
            else:
                second_best = None
                metric = None
            
            if second_best and metric:
                best_score = st.session_state.results.loc[st.session_state.best_model_name, metric]
                second_score = st.session_state.results.loc[second_best, metric]
                
                difference = best_score - second_score
                
                if difference > 0.05:
                    st.write(f"- {st.session_state.best_model_name} significantly outperforms {second_best} with a {metric} difference of {difference:.2%}")
                else:
                    st.write(f"- {st.session_state.best_model_name} slightly outperforms {second_best} with a {metric} difference of {difference:.2%}")
                    
                    # Suggest ensemble if difference is small
                    st.write("- The small performance difference suggests that an ensemble approach combining these models might improve results further.")
        
        # Potential limitations and improvements
        st.subheader("Potential Limitations and Improvements")
        
        st.write("**Current Limitations:**")
        
        # Check for class imbalance
        if hasattr(st.session_state, 'y_train'):
            class_counts = pd.Series(st.session_state.y_train).value_counts()
            total = len(st.session_state.y_train)
            
            # Check if any class represents less than 20% of the data
            imbalance = any((count / total) < 0.2 for count in class_counts)
            
            if imbalance:
                st.write("- The dataset shows class imbalance, which may affect model performance. Consider resampling techniques or specialized metrics.")
        
        # Check for overfitting
        if 'accuracy' in results_df and hasattr(st.session_state, 'models') and st.session_state.best_model_name:
            model = st.session_state.models[st.session_state.best_model_name]
            
            if hasattr(model, 'predict'):
                train_preds = model.predict(st.session_state.X_train)
                from sklearn.metrics import accuracy_score
                train_acc = accuracy_score(st.session_state.y_train, train_preds)
                test_acc = results_df['accuracy']
                
                if train_acc - test_acc > 0.1:
                    st.write(f"- The model may be overfitting, with training accuracy ({train_acc:.2%}) significantly higher than test accuracy ({test_acc:.2%}).")
                    st.write("  Consider regularization, simpler models, or more training data to address this issue.")
        
        st.write("**Suggested Improvements:**")
        st.write("- Collect more data to improve model generalization")
        st.write("- Try more advanced feature engineering techniques")
        st.write("- Explore ensemble methods combining multiple models")
        
        if st.session_state.best_model_name == 'Logistic Regression':
            st.write("- Consider more complex models like Neural Networks or XGBoost for potentially better performance")
        elif st.session_state.best_model_name in ['Random Forest', 'Gradient Boosting']:
            st.write("- Fine-tune hyperparameters with more extensive grid search")
        
    else:
        st.warning("Please train models in the 'Model Training & Evaluation' tab first.")

# Tab 5: Report Generation
with tab5:
    st.header("Report Generation")
    
    if (st.session_state.data is not None and 
        st.session_state.models is not None and 
        st.session_state.results is not None):
        
        st.subheader("Report Settings")
        
        # Report title
        report_title = st.text_input("Report Title", "Classification Model Analysis Report")
        
        # Author name
        author_name = st.text_input("Author Name", "Data Scientist")
        
        # Report objectives
        report_objective = st.text_area(
            "Report Objective",
            "This analysis aims to develop and evaluate classification models to predict outcomes and provide insights for business decision-making."
        )
        
        # Main findings
        main_findings = st.text_area(
            "Main Findings (optional)",
            "The analysis identified key predictors and achieved good performance with the selected classification model."
        )
        
        # Generate report button
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                try:
                    # Generate report
                    report_html = generate_report(
                        title=report_title,
                        author=author_name,
                        objective=report_objective,
                        findings=main_findings,
                        data=st.session_state.data,
                        target_column=st.session_state.target_column,
                        preprocessing_steps=st.session_state.preprocessing_steps,
                        models=st.session_state.models,
                        results=st.session_state.results,
                        feature_importance=st.session_state.feature_importance,
                        best_model_name=st.session_state.best_model_name,
                        X_test=st.session_state.X_test,
                        y_test=st.session_state.y_test
                    )
                    
                    # Create a download link for the HTML report
                    report_download = io.BytesIO()
                    report_download.write(report_html.encode())
                    report_download.seek(0)
                    
                    # Generate a filename with the current date/time
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"classification_report_{now}.html"
                    
                    # Create download button
                    st.download_button(
                        label="Download Report (HTML)",
                        data=report_download,
                        file_name=filename,
                        mime="text/html"
                    )
                    
                    # Display the report in an iframe
                    st.subheader("Report Preview")
                    st.components.v1.html(report_html, height=600, scrolling=True)
                    
                except Exception as e:
                    st.error(f"Error generating report: {e}")
    else:
        st.warning("Please complete the model training process before generating a report.")

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Classification Model Builder & Analyzer | Created with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
