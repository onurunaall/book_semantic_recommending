#!/usr/bin/env python3

from data_exploration import download_dataset, load_books, clean_books_data
from text_classification import classify_categories
from sentiment_analysis import process_books
from vector_search import save_tagged_descriptions
from gradio_dashboard import launch_dashboard

def main() -> None:
    print("Downloading and cleaning the dataset...")
    dataset_path = download_dataset()
    raw_books_df = load_books(dataset_path)
    cleaned_books_df = clean_books_data(raw_books_df)

    print("Classifying book categories...")
    books_with_categories_df = classify_categories(cleaned_books_df)

    print("Computing emotion scores for each book...")
    books_with_emotions_df = process_books(books_with_categories_df)

    print("Saving final processed data and tagged descriptions...")
    books_with_emotions_df.to_csv("books_with_emotions.csv", index=False)
    save_tagged_descriptions(cleaned_books_df)
    print("  → 'books_with_emotions.csv' and 'tagged_description.txt' created.")

    print("Launching the Gradio Dashboard...")
    launch_dashboard()

if __name__ == "__main__":
    main()
