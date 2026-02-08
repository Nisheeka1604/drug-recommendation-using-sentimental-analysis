""
Utility functions for the Drug Recommendation System.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union

def ensure_dir(directory: str) -> None:
    """Ensure that a directory exists, create it if it doesn't.
    
    Args:
        directory: Path to the directory
    """
    os.makedirs(directory, exist_ok=True)

def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        filepath: Path to save the metrics file
    """
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load evaluation metrics from a JSON file.
    
    Args:
        filepath: Path to the metrics file
        
    Returns:
        Dictionary containing the loaded metrics
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def get_most_common_drugs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get the most common drugs in the dataset.
    
    Args:
        df: DataFrame containing drug data
        n: Number of top drugs to return
        
    Returns:
        DataFrame with the most common drugs and their counts
    """
    return df['drug_name'].value_counts().head(n).reset_index()

def get_drug_recommendations(model: Any, vectorizer: Any, 
                           drug_reviews: pd.DataFrame, 
                           condition: str, top_n: int = 5) -> pd.DataFrame:
    """Get top drug recommendations for a given condition.
    
    Args:
        model: Trained sentiment analysis model
        vectorizer: Fitted TF-IDF vectorizer
        drug_reviews: DataFrame containing drug reviews
        condition: Medical condition to get recommendations for
        top_n: Number of top recommendations to return
        
    Returns:
        DataFrame with top drug recommendations
    """
    # Filter reviews for the given condition
    condition_reviews = drug_reviews[drug_reviews['condition'] == condition]
    
    if len(condition_reviews) == 0:
        return pd.DataFrame()
    
    # Get average sentiment for each drug
    X = vectorizer.transform(condition_reviews['review'])
    condition_reviews['sentiment'] = model.predict(X)
    
    # Group by drug and calculate average sentiment and number of reviews
    drug_sentiments = condition_reviews.groupby('drug_name').agg({
        'sentiment': 'mean',
        'review': 'count'
    }).rename(columns={'review': 'review_count'})
    
    # Sort by sentiment and review count
    drug_sentiments = drug_sentiments.sort_values(
        by=['sentiment', 'review_count'], 
        ascending=[False, False]
    )
    
    return drug_sentiments.head(top_n)

def print_evaluation_metrics(metrics: Dict[str, Any]) -> None:
    """Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    print("\n=== Classification Report ===")
    report = metrics['classification_report']
    for label, scores in report.items():
        if isinstance(scores, dict):
            print(f"{label}:")
            for metric, value in scores.items():
                if metric != 'support':
                    print(f"  {metric}: {value:.4f}")
