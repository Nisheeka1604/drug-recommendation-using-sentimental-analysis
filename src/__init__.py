"""
Drug Recommendation System using Sentiment Analysis

This package provides functionality for analyzing drug reviews and making
personalized drug recommendations based on sentiment analysis.
"""

from .data_preprocessing import load_data, clean_text, preprocess_data
from .model import DrugRecommendationModel, train_model
from .utils import (
    ensure_dir,
    save_metrics,
    load_metrics,
    get_most_common_drugs,
    get_drug_recommendations,
    print_evaluation_metrics
)

__all__ = [
    'load_data',
    'clean_text',
    'preprocess_data',
    'DrugRecommendationModel',
    'train_model',
    'ensure_dir',
    'save_metrics',
    'load_metrics',
    'get_most_common_drugs',
    'get_drug_recommendations',
    'print_evaluation_metrics'
]
