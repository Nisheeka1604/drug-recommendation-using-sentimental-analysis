"""
Data preprocessing module for the Drug Recommendation System.
Handles loading, cleaning, and preparing the dataset for model training.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(filepath):
    """
    Load the dataset from the given filepath.
    
    Args:
        filepath (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # TODO: Implement data loading logic
    # Example: return pd.read_csv(filepath)
    pass

def clean_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def preprocess_data(df, text_column, label_column=None):
    """
    Preprocess the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Name of the column containing text data
        label_column (str, optional): Name of the label column
        
    Returns:
        tuple: (X, y) where X is the preprocessed text and y are the labels (if available)
    """
    # Clean text data
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    X = df['cleaned_text']
    y = df[label_column] if label_column else None
    
    return (X, y) if y is not None else X
