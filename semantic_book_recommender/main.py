#!/usr/bin/env python3

import logging
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from semantic_book_recommender.data_exploration import (
    download_dataset,
    load_books,
    clean_books_data,
)
from semantic_book_recommender.text_classification import (
    classify_categories,
    initialize_zero_shot_classifier,
)
from semantic_book_recommender.sentiment_analysis import (
    process_books,
    initialize_classifier as initialize_emotion_classifier,
)
from semantic_book_recommender.vector_search import (
    create_vector_store,
)
from semantic_book_recommender.gradio_dashboard import launch_dashboard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dataset() -> tuple[pd.DataFrame, object]:
    """
    Build the complete dataset through an in-memory pipeline.
    Returns the final DataFrame and Chroma vector store.
    """
    # Download and load raw data
    logger.info("Downloading and loading dataset...")
    dataset_dir = download_dataset()
    raw_books = load_books(dataset_dir)
    
    # Clean the data
    logger.info("Cleaning data...")
    cleaned_books = clean_books_data(raw_books)
    
    # Classify categories
    logger.info("Classifying categories...")
    books_with_categories = classify_categories(cleaned_books)
    
    # Analyze emotions
    logger.info("Analyzing emotions...")
    emotion_classifier = initialize_emotion_classifier()
    emotion_scores = process_books(emotion_classifier, books_with_categories)
    
    # Merge emotion scores with main DataFrame
    final_books = books_with_categories.merge(emotion_scores, on="isbn13")
    
    # Update thumbnails for display
    def format_thumbnail(url):
        if isinstance(url, str) and url:
            return f"{url}&fife=w800"
        return "https://via.placeholder.com/200x300?text=No+Cover"
    
    final_books["large_thumbnail"] = final_books["thumbnail"].apply(format_thumbnail)
    
    # Create vector store in memory
    logger.info("Creating vector store in memory...")
    # Create documents from the DataFrame
    documents = [
        row["tagged_description"]
        for _, row in final_books.iterrows()
    ]
    # The from_texts method is simpler than creating Document objects manually
    vector_store = Chroma.from_texts(texts=documents, embedding=OpenAIEmbeddings())
    
    logger.info("Dataset built successfully!")
    return final_books, vector_store


def main() -> None:
    """Build dataset and launch the Gradio dashboard."""
    books_data, vector_store = build_dataset()
    launch_dashboard(books_data, vector_store)


if __name__ == "__main__":
    main()
