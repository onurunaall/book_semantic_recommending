#!/usr/bin/env python3
"""
Gradio Dashboard for the Semantic Book Recommender

This module starts a friendly web interface (via Gradio) where you can type in a description of a book,
select a category and emotional tone, and then see a gallery of recommended books complete with thumbnails and captions.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Load any needed environment variables (like API keys)
load_dotenv()

def load_books_data() -> pd.DataFrame:
    """Load the books data enriched with emotion scores from 'books_with_emotions.csv'."""
    books = pd.read_csv("books_with_emotions.csv")
    # Create a URL for a larger thumbnail image.
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    # Use a default image if the thumbnail is missing.
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )
    return books

def load_document_store() -> Chroma:
    """Load the vector store built from tagged descriptions."""
    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(documents, OpenAIEmbeddings())
    return db_books

def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
    db_books: Chroma = None,
    books: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Retrieve book recommendations based on the query, with optional filtering by category and emotional tone.

    Args:
        query (str): The user's book description.
        category (str): The chosen category filter (or "All").
        tone (str): The chosen emotional tone.
        initial_top_k (int): The initial number of similar results to fetch.
        final_top_k (int): The final number of recommendations to return.
        db_books (Chroma): The vector store.
        books (pd.DataFrame): The books DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the recommended books.
    """
    recs = db_books.similarity_search(query, k=initial_top_k)
    # Extract the ISBNs from the returned documents.
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Apply filtering by category if a specific one is chosen.
    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Sort by emotion score if a tone is specified.
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(query: str, category: str, tone: str) -> List[Tuple[str, str]]:
    """
    Generate a list of recommended books with thumbnails and captions.

    Args:
        query (str): The description entered by the user.
        category (str): The selected book category.
        tone (str): The selected emotional tone.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the URL for the large thumbnail and a caption.
    """
    books = load_books_data()
    db_books = load_document_store()
    recommendations = retrieve_semantic_recommendations(query, category, tone, db_books=db_books, books=books)
    results = []
    for _, row in recommendations.iterrows():
        truncated_description = " ".join(row["description"].split()[:30]) + "..."
        # Make the authors string more friendly.
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

def launch_dashboard() -> None:
    """Launch the Gradio dashboard for interactive semantic book recommendations."""
    books = load_books_data()
    categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

    with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
        gr.Markdown("# Semantic Book Recommender")

        with gr.Row():
            user_query = gr.Textbox(
                label="Please enter a description of a book:",
                placeholder="e.g., A story about forgiveness"
            )
            category_dropdown = gr.Dropdown(
                choices=categories, label="Select a category:", value="All"
            )
            tone_dropdown = gr.Dropdown(
                choices=tones, label="Select an emotional tone:", value="All"
            )
            submit_button = gr.Button("Find recommendations")

        gr.Markdown("## Recommendations")
        output = gr.Gallery(label="Recommended books", columns=8, rows=2)

        submit_button.click(
            fn=recommend_books,
            inputs=[user_query, category_dropdown, tone_dropdown],
            outputs=output
        )

    dashboard.launch()

if __name__ == "__main__":
    launch_dashboard()