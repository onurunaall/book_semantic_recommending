import pytest
import pandas as pd
from semantic_book_recommender.sentiment_analysis import calculate_max_emotion_scores, process_books

def test_calculate_max_emotion_scores():
    predictions = [[{"label": "anger", "score": 0.1},
                    {"label": "disgust", "score": 0.2},
                    {"label": "fear", "score": 0.3},
                    {"label": "joy", "score": 0.4},
                    {"label": "sadness", "score": 0.5},
                    {"label": "surprise", "score": 0.6},
                    {"label": "neutral", "score": 0.7}
                   ],
                   [
                     {"label": "anger", "score": 0.2},
                     {"label": "disgust", "score": 0.3},
                     {"label": "fear", "score": 0.4},
                     {"label": "joy", "score": 0.5},
                     {"label": "sadness", "score": 0.6},
                     {"label": "surprise", "score": 0.7},
                     {"label": "neutral", "score": 0.8}
                   ]
                  ]
    expected = {
        "anger": 0.2,
        "disgust": 0.3,
        "fear": 0.4,
        "joy": 0.5,
        "sadness": 0.6,
        "surprise": 0.7,
        "neutral": 0.8
    }
  
    result = calculate_max_emotion_scores(predictions)
    assert result == expected

class DummyClassifier:
    def __call__(self, sentences, candidate_labels=None):
        # Return a fixed prediction for each sentence.
        return [{"labels": candidate_labels, "scores": [0.2, 0.8]} for _ in sentences]

def test_process_books():
    df = pd.DataFrame({
        "isbn13": [9876543210123],
        "description": ["Sentence one. Sentence two."],
        "published_year": [2005],
        "num_pages": [250],
        "average_rating": [4.3],
        "subtitle": ["Test Subtitle"],
        "title": ["Test Book"],
        "categories": ["Fiction"]
    })
  
    classifier = DummyClassifier()
    
    emotions_df = process_books(classifier, df)
  
    expected_scores = {label: 0.8 for label in ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]}
    row = emotions_df.iloc[0].to_dict()
    row.pop("isbn13", None)
    assert row == expected_scores
