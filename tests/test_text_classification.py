import pandas as pd
import numpy as np
import pytest
from semantic_book_recommender.text_classification import map_categories, generate_prediction, predict_missing_categories

def test_map_categories():
    df = pd.DataFrame({"categories": ["Fiction", "Juvenile Fiction", "History", "Unknown"]})
    result_df = map_categories(df)
    expected = ["Fiction", "Children's Fiction", "Nonfiction", np.nan]
    
    for res, exp in zip(result_df["simple_categories"].tolist(), expected):
        if pd.isna(exp):
            assert pd.isna(res)
        else:
            assert res == exp

# Dummy classifier that always returns the first candidate as best.
def dummy_classifier(text, candidate_labels):
    return {"labels": candidate_labels, "scores": [0.9] + [0.1] * (len(candidate_labels) - 1)}

def test_generate_prediction():
    candidate_labels = ["Fiction", "Nonfiction"]
    prediction = generate_prediction("Sample text", candidate_labels, dummy_classifier)
    assert prediction == "Fiction"

def test_predict_missing_categories(monkeypatch):
    df = pd.DataFrame({
        "isbn13": [1111111111111, 2222222222222],
        "description": [
            "This is a long enough description " * 2,
            "Another sufficiently long description " * 2,
        ],
        "simple_categories": [pd.NA, pd.NA],
        "categories": ["Fiction", "History"]
    })
    # Override generate_prediction to always return "Fiction"
    monkeypatch.setattr(
        "semantic_book_recommender.text_classification.generate_prediction",
        lambda text, labels, clf: "Fiction"
    )
  
    result_df = predict_missing_categories(df, dummy_classifier)
    assert all(result_df["predicted_categories"] == "Fiction")
