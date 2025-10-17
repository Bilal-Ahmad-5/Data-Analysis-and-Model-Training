import os
import pandas as pd
import numpy as np
import json
import pickle
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, LargeBinary, MetaData, Table, select, insert, delete, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import base64
from io import BytesIO

# Get the database URL from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")

# Create database engine with error handling
def create_db_engine():
    """Create database engine with fallback to SQLite"""
    db_url = DATABASE_URL
    
    if db_url:
        try:
            # Try to create engine and test connection
            test_engine = create_engine(db_url)
            # Test the connection
            with test_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Successfully connected to PostgreSQL database")
            return test_engine
        except Exception as e:
            print(f"Warning: Could not connect to PostgreSQL: {e}")
            print("Falling back to SQLite database")
    else:
        print("Warning: DATABASE_URL not found, using SQLite instead")
    
    # Fallback to SQLite
    return create_engine('sqlite:///classification_models.db')

engine = create_db_engine()
Base = declarative_base()
metadata = MetaData()

# Create a session factory
Session = sessionmaker(bind=engine)

# Define models
class Dataset(Base):
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    data = Column(LargeBinary, nullable=False)  # Pickled dataframe
    created_at = Column(String(50), nullable=False)
    
    @staticmethod
    def store_dataframe(name, df, description=""):
        """Store a pandas DataFrame in the database"""
        from datetime import datetime
        
        # Pickle the dataframe
        pickled_df = pickle.dumps(df)
        
        # Create a new dataset record
        dataset = Dataset(
            name=name,
            description=description,
            data=pickled_df,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Add to database
        session = Session()
        session.add(dataset)
        session.commit()
        dataset_id = dataset.id
        session.close()
        
        return dataset_id
    
    @staticmethod
    def retrieve_dataframe(dataset_id):
        """Retrieve a pandas DataFrame from the database"""
        session = Session()
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        
        if dataset:
            # Unpickle the dataframe
            df = pickle.loads(dataset.data)
            session.close()
            return df, dataset.name, dataset.description
        else:
            session.close()
            return None, None, None
    
    @staticmethod
    def get_all_datasets():
        """Get a list of all datasets"""
        session = Session()
        datasets = session.query(Dataset.id, Dataset.name, Dataset.description, Dataset.created_at).all()
        session.close()
        return datasets
    
    @staticmethod
    def delete_dataset(dataset_id):
        """Delete a dataset from the database"""
        session = Session()
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        
        if dataset:
            session.delete(dataset)
            session.commit()
            session.close()
            return True
        else:
            session.close()
            return False


class ModelResult(Base):
    __tablename__ = 'model_results'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    model_data = Column(LargeBinary, nullable=False)  # Pickled model and related info
    created_at = Column(String(50), nullable=False)
    
    @staticmethod
    def store_model_results(dataset_id, name, models, results, feature_importance, description=""):
        """Store model results in the database"""
        from datetime import datetime
        
        # Package all model data
        model_data = {
            'models': models,
            'results': results,
            'feature_importance': feature_importance
        }
        
        # Pickle the model data
        pickled_data = pickle.dumps(model_data)
        
        # Create a new model result record
        model_result = ModelResult(
            dataset_id=dataset_id,
            name=name,
            description=description,
            model_data=pickled_data,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Add to database
        session = Session()
        session.add(model_result)
        session.commit()
        result_id = model_result.id
        session.close()
        
        return result_id
    
    @staticmethod
    def retrieve_model_results(result_id):
        """Retrieve model results from the database"""
        session = Session()
        model_result = session.query(ModelResult).filter(ModelResult.id == result_id).first()
        
        if model_result:
            # Unpickle the model data
            model_data = pickle.loads(model_result.model_data)
            session.close()
            return model_data, model_result.name, model_result.description
        else:
            session.close()
            return None, None, None
    
    @staticmethod
    def get_model_results_by_dataset(dataset_id):
        """Get all model results for a specific dataset"""
        session = Session()
        results = session.query(ModelResult.id, ModelResult.name, ModelResult.description, ModelResult.created_at)\
            .filter(ModelResult.dataset_id == dataset_id).all()
        session.close()
        return results
    
    @staticmethod
    def delete_model_result(result_id):
        """Delete a model result from the database"""
        session = Session()
        result = session.query(ModelResult).filter(ModelResult.id == result_id).first()
        
        if result:
            session.delete(result)
            session.commit()
            session.close()
            return True
        else:
            session.close()
            return False


class SampleDataset(Base):
    __tablename__ = 'sample_datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    data = Column(LargeBinary, nullable=False)  # Pickled dataframe
    
    @staticmethod
    def get_sample_datasets():
        """Get a list of all sample datasets"""
        session = Session()
        datasets = session.query(SampleDataset.id, SampleDataset.name, SampleDataset.description).all()
        session.close()
        return datasets
    
    @staticmethod
    def retrieve_sample_dataframe(dataset_id):
        """Retrieve a sample pandas DataFrame from the database"""
        session = Session()
        dataset = session.query(SampleDataset).filter(SampleDataset.id == dataset_id).first()
        
        if dataset:
            # Unpickle the dataframe
            df = pickle.loads(dataset.data)
            session.close()
            return df, dataset.name, dataset.description
        else:
            session.close()
            return None, None, None


# Create all tables
def initialize_database():
    """Create all database tables if they don't exist"""
    Base.metadata.create_all(engine)
    
    # Add some sample datasets
    session = Session()
    
    # Check if we already have sample datasets
    if session.query(SampleDataset).count() == 0:
        # Iris dataset
        iris_df = create_iris_dataset()
        iris_description = """
        The famous Iris dataset, often used for classification tasks.
        It contains measurements of sepals and petals of three species of iris flowers.
        Target variable: Species (Setosa, Versicolor, or Virginica)
        """
        iris_pickled = pickle.dumps(iris_df)
        iris_sample = SampleDataset(
            name="Iris Flower Dataset",
            description=iris_description,
            data=iris_pickled
        )
        
        # Titanic dataset
        titanic_df = create_titanic_dataset()
        titanic_description = """
        The Titanic passenger survival dataset.
        Contains passenger information and whether they survived the sinking of the Titanic.
        Target variable: Survived (0 = No, 1 = Yes)
        """
        titanic_pickled = pickle.dumps(titanic_df)
        titanic_sample = SampleDataset(
            name="Titanic Survival Dataset",
            description=titanic_description,
            data=titanic_pickled
        )
        
        # Wine dataset
        wine_df = create_wine_dataset()
        wine_description = """
        The Wine dataset for wine classification.
        Contains chemical attributes of different wines and their class.
        Target variable: Class (0, 1, or 2)
        """
        wine_pickled = pickle.dumps(wine_df)
        wine_sample = SampleDataset(
            name="Wine Classification Dataset",
            description=wine_description,
            data=wine_pickled
        )
        
        # Add all sample datasets
        session.add_all([iris_sample, titanic_sample, wine_sample])
        session.commit()
    
    session.close()

# Helper functions to create sample datasets
def create_iris_dataset():
    """Create a pandas DataFrame for the Iris dataset"""
    # Create a sample Iris dataset
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    return iris_df

def create_titanic_dataset():
    """Create a pandas DataFrame for the Titanic dataset"""
    # Create a simplified Titanic dataset
    data = {
        'PassengerId': range(1, 892),
        'Survived': np.random.choice([0, 1], size=891, p=[0.6, 0.4]),
        'Pclass': np.random.choice([1, 2, 3], size=891, p=[0.25, 0.25, 0.5]),
        'Name': [f"Passenger_{i}" for i in range(1, 892)],
        'Sex': np.random.choice(['male', 'female'], size=891),
        'Age': np.clip(np.random.normal(29, 14, size=891), 0.5, 80),
        'SibSp': np.random.choice(range(9), size=891, p=[0.65, 0.2, 0.1, 0.02, 0.01, 0.01, 0.005, 0.005, 0.01]),
        'Parch': np.random.choice(range(10), size=891, p=[0.75, 0.12, 0.08, 0.02, 0.01, 0.005, 0.005, 0.005, 0.005, 0.0]),
        'Fare': np.clip(np.random.exponential(32.2, size=891), 0, 512),
        'Embarked': np.random.choice(['C', 'Q', 'S'], size=891, p=[0.2, 0.1, 0.7])
    }
    
    # Create correlation between features and target
    for i in range(len(data['Survived'])):
        # Higher class, female, and younger age have higher survival probability
        if data['Pclass'][i] == 1 and data['Sex'][i] == 'female' and data['Age'][i] < 30:
            data['Survived'][i] = np.random.choice([0, 1], p=[0.1, 0.9])
        elif data['Pclass'][i] == 3 and data['Sex'][i] == 'male' and data['Age'][i] > 30:
            data['Survived'][i] = np.random.choice([0, 1], p=[0.95, 0.05])
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    df.loc[np.random.choice(df.index, size=int(0.2*len(df))), 'Age'] = np.nan
    df.loc[np.random.choice(df.index, size=int(0.05*len(df))), 'Embarked'] = np.nan
    
    return df

def create_wine_dataset():
    """Create a pandas DataFrame for the Wine dataset"""
    from sklearn.datasets import load_wine
    
    wine = load_wine()
    wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    wine_df['Class'] = wine.target
    
    return wine_df

# Initialize the database when this module is imported
try:
    initialize_database()
except Exception as e:
    print(f"Warning: Could not initialize database: {e}")
    print("The application will continue but database features may not work.")