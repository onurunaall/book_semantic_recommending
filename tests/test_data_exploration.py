import pandas as pd
from semantic_book_recommender.data_exploration import clean_books_data

def test_clean_books_data():
    sample_data = {
        "isbn13": [1234567890123, 2345678901234],
        "description": [
            "Word " * 30,  # 30 words; should pass threshold.
            "Too short."   # Insufficient words; should be removed.
        ],
        "published_year": [2000, 2010],
        "num_pages": [300, 150],
        "average_rating": [4.2, 3.8],
        "subtitle": ["Subtitle", None],
        "title": ["Book A", "Book B"],
        "categories": ["Fiction", "History"]
    }
  
    raw_df = pd.DataFrame(sample_data)
    cleaned_df = clean_books_data(raw_df)
    
    # Only the first row should remain.
    assert len(cleaned_df) == 1
    
    expected_tagged = str(1234567890123) + " " + sample_data["description"][0]
    assert cleaned_df["tagged_description"].iloc[0] == expected_tagged

    expected_title_and_subtitle = "Book A: Subtitle"
    assert cleaned_df["title_and_subtitle"].iloc[0] == expected_title_and_subtitle
