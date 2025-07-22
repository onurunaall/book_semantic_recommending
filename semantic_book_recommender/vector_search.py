#!/usr/bin/env python3

import os
import logging
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# === Setup ===
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY not set; please export it before running.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Constants ===
DEFAULT_SEARCH_K = 50
DEFAULT_FINAL_K = 16


# === Vector Store Creation ===
def create_vector_store(docs: List[Document]) -> Chroma:
    """Build Chroma vector store from a list of Document objects."""
    if not docs:
        raise ValueError("Cannot create vector store from empty document list")

    if not isinstance(docs, list):
        raise TypeError("docs must be a list of Document objects")

    for i, doc in enumerate(docs):
        if not isinstance(doc, Document):
            raise TypeError(f"Item {i} is not a Document: {type(doc)}")
        if not doc.page_content or not doc.page_content.strip():
            logger.warning(f"Document {i} has empty content")

    try:
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(docs, embeddings)
        logger.info(f"Created vector store with {len(docs)} documents")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise


# === Semantic Search ===
def retrieve_semantic_recommendations(
    query: str,
    books_df: pd.DataFrame,
    vector_store: Chroma,
    category: Optional[str] = None,
    tone: Optional[str] = None,
    search_k: int = DEFAULT_SEARCH_K,
    final_k: int = DEFAULT_FINAL_K,
) -> pd.DataFrame:
    """Search vector store and return filtered book recommendations."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if books_df is None or books_df.empty:
        raise ValueError("books_df cannot be None or empty")
    if vector_store is None:
        raise ValueError("vector_store cannot be None")
    if search_k <= 0 or final_k <= 0:
        raise ValueError("search_k and final_k must be positive")

    required_cols = ["isbn13", "simple_categories"]
    missing = [col for col in required_cols if col not in books_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in books_df: {missing}")

    query = query.strip()

    try:
        results = vector_store.similarity_search(query, k=search_k)

        if not results:
            logger.warning(f"No results for query: {query}")
            return pd.DataFrame()

        isbns: List[int] = []

        for i, doc in enumerate(results):
            try:
                content = doc.page_content.strip().strip('"').strip("'")
                if not content:
                    logger.warning(f"Result {i} has empty content")
                    continue

                parts = content.split()
                if not parts:
                    logger.warning(f"No parts found in result {i}: {content!r}")
                    continue

                isbn_str = parts[0].strip()
                isbn = int(isbn_str)
                if isbn > 0:
                    isbns.append(isbn)
                else:
                    logger.warning(f"Invalid ISBN {isbn} in result {i}")
            except ValueError:
                logger.warning(f"Could not convert to int in result {i}: {isbn_str}")
            except Exception as e:
                logger.warning(f"Error in result {i}: {e}")
                continue

        if not isbns:
            logger.warning(f"No valid ISBNs extracted for query: {query}")
            return pd.DataFrame()

        recs = books_df[books_df["isbn13"].isin(isbns)].copy()
        if recs.empty:
            logger.warning("No books matched the extracted ISBNs")
            return pd.DataFrame()

        if category and category != "All":
            initial_len = len(recs)
            recs = recs[recs["simple_categories"] == category]
            logger.info(f"Filtered by category '{category}': {initial_len} → {len(recs)}")

        tone_map = {
            "Happy": "joy",
            "Surprising": "surprise",
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness",
        }

        if tone and tone in tone_map:
            emotion_col = tone_map[tone]
            if emotion_col in recs.columns:
                if pd.api.types.is_numeric_dtype(recs[emotion_col]):
                    recs = recs.sort_values(by=emotion_col, ascending=False)
                    logger.info(f"Sorted by emotion: {emotion_col}")
                else:
                    logger.warning(f"Column '{emotion_col}' is not numeric")
            else:
                logger.warning(f"Column '{emotion_col}' not found in data")

        final_recs = recs.head(final_k)
        logger.info(f"Returning {len(final_recs)} recommendations for query: {query}")
        return final_recs

    except Exception as e:
        logger.error(f"Error in semantic retrieval: {e}")
        raise