🧠 Data Analysis & Model Training

A Streamlit-based web application that helps in data analysis, cleaing, and training on multiple classification machine learning models in one place.
Upload your data, choose preprocessing options, train different models, and get detailed analytics and reports — all without writing a single line of code.

🚀 Features

Easy Data Upload – Load CSV or Excel files and explore them interactively.

Configurable Preprocessing – Choose imputation, scaling, and encoding methods.

Feature Engineering – Add or select important features.

Multiple Models – Train Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, SVM, and KNN.

Performance Comparison – Evaluate models using multiple metrics and side-by-side visualizations.

Visual Insights – View correlations, class distributions, ROC curves, and more.

Automated Reports – Generate stakeholder-ready analytical summaries.

Database Support – Save and retrieve datasets and model results from an SQL database.

🧩 System Architecture
Frontend

Framework: Streamlit

UI Pattern: Single-page layout with sidebar controls and main content area

State Management: Uses Streamlit’s session_state to store datasets, models, and results

Layout: Wide layout optimized for displaying data visualizations

Backend

The backend follows a modular design for clean separation of responsibilities:

Module Responsibility
utils.py Data preprocessing, feature engineering, train-test splitting, and report generation
models.py Model training and evaluation for multiple algorithms
visualization.py Plot generation (correlation, feature importance, ROC, confusion matrix, etc.)
⚙️ Data Processing Workflow

Upload Data: Import CSV or Excel files and validate structure

Configure Preprocessing: Choose imputation, encoding, and scaling strategies

Feature Engineering: Optionally add or remove features

Model Training: Train multiple algorithms in parallel

Evaluation: Compare accuracy, precision, recall, F1-score, and ROC-AUC

Visualization: Explore plots for performance and feature importance

Reporting: Generate professional summaries and export results

🏗️ Data Storage

Database: SQLAlchemy ORM (supports SQLite, PostgreSQL, etc.)

Environment Variable: DATABASE_URL defines connection settings

Data Models:

Dataset: Stores uploaded datasets and metadata

ModelResult: Saves model performance metrics and parameters

SampleDataset: Provides sample datasets for quick demos

Serialization: Pickle used for efficient DataFrame storage

🧮 Machine Learning Pipeline

Preprocessing Options:

Imputation for missing values (numerical & categorical)

Scaling: StandardScaler, MinMaxScaler, or RobustScaler

Encoding: One-Hot or Label Encoding

Algorithms Supported:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Hyperparameter Tuning: Optional GridSearchCV with adjustable cross-validation folds

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

📊 Visualization Capabilities

Correlation heatmaps

Class distribution plots

Feature importance rankings

Confusion matrices

ROC and Precision-Recall curves

🧱 Design Principles

Modular Architecture: Separate layers for data, modeling, visualization, and persistence

Configuration-Driven: Fully customizable steps through the UI

State Preservation: Keeps progress and results in session state

Batch Processing: Trains multiple models in parallel for comparison

🧰 Tech Stack
Core Libraries

pandas – Data manipulation

numpy – Numerical operations

scikit-learn – ML algorithms and metrics

Visualization

matplotlib, seaborn – Static plots

plotly.express – Interactive charts

Web Framework

Streamlit – Frontend framework

Database & Persistence

SQLAlchemy – ORM for database interaction

pickle – Object serialization for DataFrames

⚙️ Environment Setup
1️⃣ Clone the Repository
git clone https://github.com/Bilal-Ahmad-5/Data-Analysis-and-Model-Training.git
cd classification-model-builder

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure the Database

Create a .env file in the root folder:

DATABASE_URL=sqlite:///data.db

(For production, use PostgreSQL or another supported backend.)

5️⃣ Run the Application
streamlit run app.py

📘 Example Use Cases

Quickly benchmark multiple classification algorithms on a dataset

Generate performance comparison reports for business presentations

Explore feature importance and visual insights without coding

Save and reload models and datasets for continuous analysis

🧑‍💻 Future Enhancements

Add support for AutoML pipelines

Expand algorithm options (e.g., XGBoost, LightGBM)

Implement model explainability (SHAP/LIME)

Enable export to PDF/Excel reports

📄 License

This project is released under the MIT License.
You are free to use, modify, and distribute it as long as you include the license notice
