#!/usr/bin/env python3

import logging
import os

import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY not set; please export it before running."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_books() -> pd.DataFrame:
    """Load cleaned books CSV."""
    return pd.read_csv("books_cleaned.csv")


def save_tagged_descriptions(
    books_df: pd.DataFrame,
    output_file: str = "tagged_description.txt"
) -> None:
    """
    Save 'tagged_description' column to a text file,
    one line per book.
    """
    books_df["tagged_description"].to_csv(
        output_file, sep="\n", index=False, header=False
    )


def create_vector_store(
    docs: list
) -> Chroma:
    """Build Chroma store from a list of documents in memory."""
    return Chroma.from_documents(docs, OpenAIEmbeddings())


def retrieve_semantic_recommendations(
    query: str,
    books_df: pd.DataFrame,
    vector_store: Chroma,
    category: str | None = None,
    tone: str | None = None,
    search_k: int = 50,
    final_k: int = 16,
) -> pd.DataFrame:
    """
    Search the vector store, extract ISBNs safely,
    and return matching rows from books_df with optional filtering.
    """
    # Search vector store
    results = vector_store.similarity_search(query, k=search_k)
    
    # Extract ISBNs
    isbns: list[int] = []
    for doc in results:
        content = doc.page_content.strip('"')
        parts = content.split()
        try:
            isbns.append(int(parts[0]))
        except (IndexError, ValueError):
            logger.warning("Skipping malformed result: %r", content)
    
    # Get matching books
    recs = books_df[books_df["isbn13"].isin(isbns)]
    
    # Apply category filter
    if category and category != "All":
        recs = recs[recs["simple_categories"] == category]
    
    # Apply tone-based sorting FIRST
    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }
    if tone in tone_map:
        emotion_column = tone_map[tone]
        if emotion_column in recs.columns:
            recs = recs.sort_values(by=emotion_column, ascending=False)
    
    # THEN, limit to the final number of recommendations
    return recs.head(final_k)
