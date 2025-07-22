import unittest
from unittest.mock import Mock, patch
import pandas as pd

from semantic_book_recommender.vector_search import retrieve_semantic_recommendations, create_vector_store


class TestVectorSearch(unittest.TestCase):
    """Test cases for vector search functionality."""

    def setUp(self):
        """Set up test data."""
        self.test_books = pd.DataFrame({
            'isbn13': [1234567890123, 9876543210987, 1111111111111, 2222222222222],
            'title': ['Test Book 1', 'Test Book 2', 'Test Book 3', 'Test Book 4'],
            'simple_categories': ['Fiction', 'Nonfiction', 'Fiction', 'Nonfiction'],
            'joy': [0.8, 0.3, 0.6, 0.9],
            'sadness': [0.2, 0.7, 0.4, 0.1],
            'fear': [0.1, 0.4, 0.8, 0.2],
            'anger': [0.05, 0.15, 0.3, 0.05],
            'surprise': [0.6, 0.5, 0.7, 0.8]
        })
        
        # Mock vector store
        self.mock_vector_store = Mock()
        self.mock_search_results = [
            Mock(page_content='1234567890123 Test description 1'),
            Mock(page_content='9876543210987 Test description 2'),
            Mock(page_content='1111111111111 Test description 3'),
            Mock(page_content='2222222222222 Test description 4'),
        ]

    def test_retrieve_semantic_recommendations_basic(self):
        """Test basic retrieval without filters."""
        self.mock_vector_store.similarity_search.return_value = self.mock_search_results
        
        results = retrieve_semantic_recommendations(
            "test query",
            self.test_books,
            self.mock_vector_store,
            final_k=2
        )
        
        self.assertEqual(len(results), 2)
        self.mock_vector_store.similarity_search.assert_called_once_with("test query", k=50)

    def test_retrieve_semantic_recommendations_category_filter(self):
        """Test retrieval with category filtering."""
        self.mock_vector_store.similarity_search.return_value = self.mock_search_results
        
        results = retrieve_semantic_recommendations(
            "test query",
            self.test_books,
            self.mock_vector_store,
            category="Fiction",
            final_k=10
        )
        
        # Should only return Fiction books
        self.assertTrue(all(results['simple_categories'] == 'Fiction'))
        self.assertEqual(len(results), 2)  # Only 2 Fiction books in test data

    def test_retrieve_semantic_recommendations_tone_sorting(self):
        """Test retrieval with tone-based sorting."""
        self.mock_vector_store.similarity_search.return_value = self.mock_search_results
        
        results = retrieve_semantic_recommendations(
            "test query",
            self.test_books,
            self.mock_vector_store,
            tone="Happy",
            final_k=4
        )
        
        # Check that results are sorted by joy (descending)
        joy_values = results['joy'].tolist()
        self.assertEqual(joy_values, sorted(joy_values, reverse=True))
        self.assertEqual(results.iloc[0]['isbn13'], 2222222222222)  # Highest joy score

    def test_retrieve_semantic_recommendations_combined_filters(self):
        """Test retrieval with both category and tone filters."""
        self.mock_vector_store.similarity_search.return_value = self.mock_search_results
        
        results = retrieve_semantic_recommendations(
            "test query",
            self.test_books,
            self.mock_vector_store,
            category="Fiction",
            tone="Suspenseful",
            final_k=10
        )
        
        # Should only have Fiction books
        self.assertTrue(all(results['simple_categories'] == 'Fiction'))
        # Should be sorted by fear (for Suspenseful)
        fear_values = results['fear'].tolist()
        self.assertEqual(fear_values, sorted(fear_values, reverse=True))
        self.assertEqual(results.iloc[0]['isbn13'], 1111111111111)  # Highest fear in Fiction

    def test_retrieve_semantic_recommendations_malformed_isbn(self):
        """Test handling of malformed ISBNs in search results."""
        malformed_results = [
            Mock(page_content='not_a_number Test description'),
            Mock(page_content='1234567890123 Valid description'),
            Mock(page_content=''),  # Empty content
        ]
        self.mock_vector_store.similarity_search.return_value = malformed_results
        
        results = retrieve_semantic_recommendations(
            "test query",
            self.test_books,
            self.mock_vector_store
        )
        
        # Should only return the valid ISBN
        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]['isbn13'], 1234567890123)


if __name__ == '__main__':
    unittest.main()

# TODO: Add tests for save_tagged_descriptions function
# TODO: Add tests for create_vector_store function with real file operations
# TODO: Test edge cases like empty search results or all results filtered out
# TODO: Test performance with large datasets (integration test)
