import pandas as pd
import numpy as np
from semantic_book_recommender.text_classification import (
    map_categories,
    predict_missing_categories,
    classify_categories,
)


def test_map_categories():
    df = pd.DataFrame({
        "categories": ["Fiction", "Juvenile Fiction", "History", "Unknown"]
    })
    result = map_categories(df)
    expected = ["Fiction", "Children's Fiction", "Nonfiction", np.nan]

    for res, exp in zip(result["simple_categories"], expected):
        if pd.isna(exp):
            assert pd.isna(res)
        else:
            assert res == exp


class DummyClassifier:
    """Always returns 'Fiction' as the top label."""
    def __call__(self, texts, candidate_labels):
        # batch and single-call both handled
        if isinstance(texts, list):
            return [
                {"labels": ["Fiction", "Nonfiction"], "scores": [0.9, 0.1]}
                for _ in texts
            ]
        else:
            return {"labels": ["Fiction", "Nonfiction"], "scores": [0.9, 0.1]}


def test_predict_missing_categories_success_and_failure():
    df = pd.DataFrame({
        "isbn13": [111, 222],
        "description": ["desc one", "desc two"],
        "simple_categories": [pd.NA, pd.NA],
    })

    # 1) Test normal operation
    preds = predict_missing_categories(df, DummyClassifier())
    assert list(preds["predicted_categories"]) == ["Fiction", "Fiction"]

    # 2) Test classifier failure does not crash and yields None
    class FailingClassifier:
        def __call__(self, texts, candidate_labels):
            raise RuntimeError("model error")

    preds_fail = predict_missing_categories(df, FailingClassifier())
    assert list(preds_fail["predicted_categories"]) == [None, None]


def test_classify_categories_end_to_end(monkeypatch):
    # Prepare input DataFrame with one mapped and one missing category
    df = pd.DataFrame({
        "isbn13": [1, 2],
        "description": ["text a", "text b"],
        "categories": ["Fiction", "History"],
        "simple_categories": ["Fiction", pd.NA]
    })

    # Stub out the classifier initializer (not used directly here)
    monkeypatch.setattr(
        "semantic_book_recommender.text_classification."
        "initialize_zero_shot_classifier",
        lambda: DummyClassifier()
    )
    # Stub out the batch prediction to only fill in isbn13=2
    stub_preds = pd.DataFrame({
        "isbn13": [2],
        "predicted_categories": ["Nonfiction"]
    })
    monkeypatch.setattr(
        "semantic_book_recommender.text_classification."
        "predict_missing_categories",
        lambda books_df, clf: stub_preds
    )

    output = classify_categories(df)

    # Both entries should now have simple_categories filled
    assert set(output["simple_categories"]) == {"Fiction", "Nonfiction"}
    # Original order and isbn13 values preserved
    assert output["isbn13"].tolist() == [1, 2]
