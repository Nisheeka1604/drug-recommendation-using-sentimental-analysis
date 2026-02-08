"""
Machine learning models for the Drug Recommendation System.
Includes model training, evaluation, and prediction functions.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

class DrugRecommendationModel:
    def __init__(self, model_path='../models/sentiment_model.pkl',
                 vectorizer_path='../models/tfidf_vectorizer.pkl'):
        """
        Initialize the Drug Recommendation Model.
        
        Args:
            model_path (str): Path to save/load the trained model
            vectorizer_path (str): Path to save/load the TF-IDF vectorizer
        """
        self.model = LinearSVC(random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    def train(self, X_train, y_train):
        """
        Train the model on the given training data.
        
        Args:
            X_train: Training text data
            y_train: Training labels
        """
        # Vectorize the text data
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_vec, y_train)
        
        # Save the trained model and vectorizer
        self.save_model()
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Text data to make predictions on
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test text data
            y_test: True labels for test data
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'classification_report': report
        }
    
    def save_model(self):
        """Save the model and vectorizer to disk."""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
    
    def load_model(self):
        """Load the model and vectorizer from disk."""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            return True
        return False

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train and evaluate the drug recommendation model.
    
    Args:
        X: Input features (text data)
        y: Target labels
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (model, metrics) where metrics is a dictionary of evaluation metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train the model
    model = DrugRecommendationModel()
    model.train(X_train, y_train)
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics
