import pandas as pd
import pytest

from semantic_book_recommender.sentiment_analysis import (
    calculate_max_emotion_scores,
    process_books
)

def test_calculate_max_emotion_scores():
    # Two sentences with varying scores per emotion
    predictions = [
        [
            {"label": "anger", "score": 0.1},
            {"label": "disgust", "score": 0.2},
            {"label": "fear", "score": 0.3},
            {"label": "joy", "score": 0.4},
            {"label": "sadness", "score": 0.5},
            {"label": "surprise", "score": 0.6},
            {"label": "neutral", "score": 0.7},
        ],
        [
            {"label": "anger", "score": 0.2},
            {"label": "disgust", "score": 0.3},
            {"label": "fear", "score": 0.4},
            {"label": "joy", "score": 0.5},
            {"label": "sadness", "score": 0.6},
            {"label": "surprise", "score": 0.7},
            {"label": "neutral", "score": 0.8},
        ],
    ]
    expected = {
        "anger": 0.2,
        "disgust": 0.3,
        "fear": 0.4,
        "joy": 0.5,
        "sadness": 0.6,
        "surprise": 0.7,
        "neutral": 0.8,
    }
    result = calculate_max_emotion_scores(predictions)
    assert result == expected

class DummyClassifier:
    """
    Dummy classifier that returns, for each sentence, the same list of emotion scores.
    """
    def __call__(self, sentences):
        # One prediction-list per sentence
        return [
            [
                {"label": label, "score": 0.8}
                for label in ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
            ]
            for _ in sentences
        ]

def test_process_books_batch_and_max_scores(monkeypatch):
    # Single book with two sentences
    df = pd.DataFrame({
        "isbn13": [9876543210123],
        "description": ["First sentence. Second sentence."],
        "published_year": [2005],
        "num_pages": [250],
        "average_rating": [4.3],
        "subtitle": ["Test Subtitle"],
        "title": ["Test Book"],
        "categories": ["Fiction"]
    })

    classifier = DummyClassifier()

    emotions_df = process_books(classifier, df)

    # Expect one row with isbn13 and 7 emotion columns
    assert list(emotions_df.columns) == [
        "anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral", "isbn13"
    ]

    row = emotions_df.iloc[0].to_dict()
    # Remove isbn13 to check only emotion scores
    assert row.pop("isbn13") == 9876543210123

    # All emotion max scores should be 0.8
    expected_scores = {label: 0.8 for label in ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]}
    assert row == expected_scores
