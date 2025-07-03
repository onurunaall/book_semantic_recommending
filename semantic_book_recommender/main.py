#!/usr/bin/env python3

import logging

from data_exploration import (
    clean_books_data,
    download_dataset,
    load_books,
)
from gradio_dashboard import launch_dashboard
from sentiment_analysis import process_books
from text_classification import classify_categories
from vector_search import save_tagged_descriptions


def main() -> None:
    """Run full pipeline and then launch the dashboard."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Downloading and cleaning dataset...")
    path = download_dataset()
    raw = load_books(path)
    cleaned = clean_books_data(raw)

    logger.info("Classifying categories...")
    with_categories = classify_categories(cleaned)

    logger.info("Analyzing emotions...")
    with_emotions = process_books(with_categories)

    logger.info("Saving outputs...")
    with_emotions.to_csv("books_with_emotions.csv", index=False)
    save_tagged_descriptions(cleaned)

    logger.info("Launching dashboard...")
    launch_dashboard()


if __name__ == "__main__":
    main()
