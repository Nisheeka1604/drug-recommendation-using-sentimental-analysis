# Drug Recommendation System Using Sentiment Analysis

A machine learning-based system that analyzes patient reviews to provide personalized drug recommendations through sentiment analysis.

## Features

- Sentiment analysis of drug reviews
- Personalized drug recommendations
- Multiple ML models for sentiment classification
- Performance evaluation metrics

## Technologies

- **Python 3.8+**
- **Machine Learning**: Scikit-learn, TensorFlow
- **NLP**: NLTK, Word2Vec, TF-IDF
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Nisheeka1604/drug-recommendation-using-sentimental-analysis.git
   cd drug-recommendation-using-sentimental-analysis
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Jupyter notebook:
   ```bash
   jupyter notebook
   ```

2. Open and run `drug_recommendation.ipynb` for the complete workflow:
   - Data preprocessing
   - Model training
   - Sentiment analysis
   - Drug recommendations

## Project Structure

```
drug-recommendation-using-sentimental-analysis/                 # Saved models
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
│   ├── data_preprocessing.py
│   ├── model.py
│   └── utils.py
├── requirements.txt         # Python dependencies
└── README.md
```

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Linear SVC (TF-IDF) | 93% | 0.92 | 0.93 | 0.92 |
| Decision Tree (Word2Vec) | 78% | 0.77 | 0.78 | 0.77 |

