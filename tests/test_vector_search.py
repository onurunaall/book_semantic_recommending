import pandas as pd
import pytest
from semantic_book_recommender.vector_search import retrieve_semantic_recommendations

class DummyDocument:
    def __init__(self, page_content):
        self.page_content = page_content

class DummyVectorStore:
    def similarity_search(self, query, k):
        # Return two dummy documents with known ISBNs in the first token.
        return [
            DummyDocument('"1111111111111 extra text"'),
            DummyDocument('"2222222222222 extra text"')
        ]

def dummy_create_vector_store(*args, **kwargs):
    return DummyVectorStore()

def test_retrieve_semantic_recommendations(monkeypatch):
    df = pd.DataFrame({
        "isbn13": [1111111111111, 2222222222222, 3333333333333],
        "title": ["Book One", "Book Two", "Book Three"],
        "description": ["Desc 1", "Desc 2", "Desc 3"],
        "simple_categories": ["Fiction", "Fiction", "Nonfiction"]
    })
    # Replace the actual create_vector_store with our dummy version.
    monkeypatch.setattr("semantic_book_recommender.vector_search.create_vector_store", dummy_create_vector_store)
    recs = retrieve_semantic_recommendations("dummy query", top_k=10)
    expected_isbns = {1111111111111, 2222222222222}
    result_isbns = set(recs["isbn13"].tolist())
    assert result_isbns == expected_isbns
