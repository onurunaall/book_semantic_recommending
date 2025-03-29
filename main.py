#!/usr/bin/env python3

import pandas as pd
from data_exploration import download_dataset, load_books, clean_books_data
from text_classification import load_books as load_clean_books, map_categories, initialize_zero_shot_classifier, predict_missing_categories
from sentiment_analysis import load_books as load_books_with_categories, initialize_classifier, process_books
from vector_search import save_tagged_descriptions
from gradio_dashboard import launch_dashboard

def run_pipeline() -> None:
    # Download raw dataset and clean it using data_exploration functions.
    print("Downloading and cleaning the dataset...")
    dataset_path: str = download_dataset()
    raw_books_df: pd.DataFrame = load_books(dataset_path)
    cleaned_books_df: pd.DataFrame = clean_books_data(raw_books_df)
    
    # Save cleaned data to CSV so that other modules can read a consistent file.
    cleaned_books_df.to_csv("books_cleaned.csv", index=False)
    print("  → Cleaned data saved as 'books_cleaned.csv'.")

    # Read the cleaned data from disk to start category mapping.
    print("Mapping and classifying book categories...")
    books_clean_df: pd.DataFrame = pd.read_csv("books_cleaned.csv")
    
    # Map original categories to a simplified set.
    classified_books_df: pd.DataFrame = map_categories(books_clean_df)
    
    # Initialize the zero-shot classifier to predict missing simplified categories.
    zero_shot_classifier = initialize_zero_shot_classifier()
    missing_category_preds: pd.DataFrame = predict_missing_categories(classified_books_df, zero_shot_classifier)
    
    # Merge predictions into the classified data and fill in missing values.
    classified_books_df = pd.merge(classified_books_df, missing_category_preds, on="isbn13", how="left")
    classified_books_df["simple_categories"] = classified_books_df["simple_categories"].fillna(classified_books_df["predicted_categories"])
    classified_books_df.drop(columns=["predicted_categories"], inplace=True)
    
    # Write the updated category data to CSV.
    classified_books_df.to_csv("books_with_categories.csv", index=False)
    print("  → Books with categories saved as 'books_with_categories.csv'.")

    # Load the categorized books to compute emotion scores.
    print("Computing emotion scores for each book...")
    books_with_categories_df: pd.DataFrame = pd.read_csv("books_with_categories.csv")
    
    # Initialize the emotion classifier and process each book for emotion scores.
    emotion_classifier = initialize_classifier()
    emotions_df: pd.DataFrame = process_books(emotion_classifier, books_with_categories_df)
    
    # Merge the computed emotion scores back into the books data.
    books_with_emotions_df: pd.DataFrame = pd.merge(books_with_categories_df, emotions_df, on="isbn13")
    books_with_emotions_df.to_csv("books_with_emotions.csv", index=False)
    print("  → Books with emotion scores saved as 'books_with_emotions.csv'.")

    # Load the cleaned books again to prepare data for the vector search.
    print("Preparing tagged descriptions for vector search...")
    books_for_vector_df: pd.DataFrame = pd.read_csv("books_cleaned.csv")
    
    # Save tagged descriptions to a text file (used later for building the vector store).
    save_tagged_descriptions(books_for_vector_df)
    print("  → Tagged descriptions saved as 'tagged_description.txt'.")

def main() -> None:
    run_pipeline()
    print("All processing done. Launching the Gradio Dashboard...")
    launch_dashboard()

if __name__ == "__main__":
    main()
