import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import itertools

def plot_correlation_matrix(df):
    """
    Plot a correlation matrix for the numerical features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing numerical features
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    # Create a correlation matrix
    corr = df.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the heatmap with the mask
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8}
    )
    
    # Set the title
    plt.title('Correlation Matrix of Numerical Features', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_class_distribution(df, target_column):
    """
    Plot the distribution of classes in the target variable
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the target variable
    target_column : str
        Name of the target variable column
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    # Create a count plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the value counts
    value_counts = df[target_column].value_counts()
    
    # Create the bar plot
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
    
    # Set labels and title
    ax.set_xlabel(target_column, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of {target_column}', fontsize=14)
    
    # Add count labels on top of each bar
    for i, count in enumerate(value_counts.values):
        ax.text(i, count + 5, str(count), ha='center', fontsize=10)
    
    # Rotate x-axis labels if there are many classes
    if len(value_counts) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_feature_importance(feature_importance_df, title="Feature Importance"):
    """
    Plot feature importance
    
    Parameters:
    -----------
    feature_importance_df : pandas.DataFrame
        DataFrame containing feature importance values
    title : str, optional
        Title for the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    # Sort by importance
    sorted_df = feature_importance_df.sort_values(by='Importance', ascending=True)
    
    # Take top 15 features for readability
    if len(sorted_df) > 15:
        sorted_df = sorted_df.tail(15)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the horizontal bar plot
    sns.barplot(x='Importance', y=sorted_df.index, data=sorted_df, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_confusion_matrices(models, X_test, y_test):
    """
    Plot confusion matrices for all models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target variable
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    # Determine the number of rows and columns for the subplot grid
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Create the figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Flatten axes for easy iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # For each model, plot the confusion matrix
    for i, (model_name, model) in enumerate(models.items()):
        if i < len(axes):
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot the confusion matrix
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
            
            # Set title and labels
            axes[i].set_title(f'Confusion Matrix - {model_name}')
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
        
    # Hide any unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_roc_curves(models, X_test, y_test):
    """
    Plot ROC curves for all models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target variable
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Check if the problem is multiclass or binary
    n_classes = len(np.unique(y_test))
    is_binary = (n_classes == 2)
    
    # Colors for different models
    colors = plt.cm.get_cmap('tab10', len(models))
    
    # For each model, plot the ROC curve
    for i, (model_name, model) in enumerate(models.items()):
        if hasattr(model, 'predict_proba'):
            try:
                # Get probability predictions
                y_score = model.predict_proba(X_test)
                
                if is_binary:
                    # Binary classification
                    y_score = y_score[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})',
                           color=colors(i))
                    
                else:
                    # Multiclass classification
                    # Compute micro-average ROC curve and ROC area
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    for j in range(n_classes):
                        fpr[j], tpr[j], _ = roc_curve((y_test == j).astype(int), y_score[:, j])
                        roc_auc[j] = auc(fpr[j], tpr[j])
                    
                    # Compute micro-average ROC curve and ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(pd.get_dummies(y_test).values.ravel(), 
                                                            y_score.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    
                    ax.plot(fpr["micro"], tpr["micro"], lw=2, 
                           label=f'{model_name} (micro-avg AUC = {roc_auc["micro"]:.2f})',
                           color=colors(i))
            except:
                # Skip if predict_proba fails
                continue
    
    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    
    # Add legend
    ax.legend(loc="lower right")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_precision_recall_curves(models, X_test, y_test):
    """
    Plot Precision-Recall curves for all models
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target variable
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Check if the problem is multiclass or binary
    n_classes = len(np.unique(y_test))
    is_binary = (n_classes == 2)
    
    # Colors for different models
    colors = plt.cm.get_cmap('tab10', len(models))
    
    # For each model, plot the Precision-Recall curve
    for i, (model_name, model) in enumerate(models.items()):
        if hasattr(model, 'predict_proba'):
            try:
                # Get probability predictions
                y_score = model.predict_proba(X_test)
                
                if is_binary:
                    # Binary classification
                    y_score = y_score[:, 1]
                    precision, recall, _ = precision_recall_curve(y_test, y_score)
                    avg_precision = average_precision_score(y_test, y_score)
                    
                    ax.plot(recall, precision, lw=2, 
                           label=f'{model_name} (AP = {avg_precision:.2f})',
                           color=colors(i))
                    
                else:
                    # Multiclass classification
                    # Compute micro-average Precision-Recall curve and average precision
                    precision = dict()
                    recall = dict()
                    avg_precision = dict()
                    
                    for j in range(n_classes):
                        precision[j], recall[j], _ = precision_recall_curve(
                            (y_test == j).astype(int), y_score[:, j])
                        avg_precision[j] = average_precision_score(
                            (y_test == j).astype(int), y_score[:, j])
                    
                    # Compute micro-average precision-recall curve
                    y_test_bin = pd.get_dummies(y_test).values
                    precision["micro"], recall["micro"], _ = precision_recall_curve(
                        y_test_bin.ravel(), y_score.ravel())
                    avg_precision["micro"] = average_precision_score(
                        y_test_bin, y_score, average="micro")
                    
                    ax.plot(recall["micro"], precision["micro"], lw=2, 
                           label=f'{model_name} (micro-avg AP = {avg_precision["micro"]:.2f})',
                           color=colors(i))
            except:
                # Skip if predict_proba fails
                continue
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Set labels and title
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14)
    
    # Add legend
    ax.legend(loc="best")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig
