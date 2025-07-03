import pandas as pd
import pytest
from semantic_book_recommender.vector_search import retrieve_semantic_recommendations

class DummyDocument:
    def __init__(self, page_content):
        self.page_content = page_content

class DummyVectorStore:
    def similarity_search(self, query, k):
        return [
            DummyDocument('"1111111111111 extra text"'),
            DummyDocument('"2222222222222 extra text"')
        ]

def test_retrieve_semantic_recommendations():
    df = pd.DataFrame({
        "isbn13": [1111111111111, 2222222222222, 3333333333333],
        "title": ["Book One", "Book Two", "Book Three"],
        "description": ["Desc 1", "Desc 2", "Desc 3"],
        "simple_categories": ["Fiction", "Fiction", "Nonfiction"]
    })
    vector_store = DummyVectorStore()
    recs = retrieve_semantic_recommendations(
        "dummy query",
        df,
        vector_store,
        search_k=10
    )
    expected_isbns = {1111111111111, 2222222222222}
    result_isbns = set(recs["isbn13"].tolist())
    assert result_isbns == expected_isbns
