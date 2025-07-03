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
    text_file: str = "tagged_description.txt"
) -> Chroma:
    """Build Chroma store from a tagged-descriptions file."""
    raw = TextLoader(text_file).load()
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=0, chunk_overlap=0
    )
    docs = splitter.split_documents(raw)
    return Chroma.from_documents(docs, OpenAIEmbeddings())


def retrieve_semantic_recommendations(
    query: str,
    books_df: pd.DataFrame,
    vector_store: Chroma,
    search_k: int = 50
) -> pd.DataFrame:
    """
    Search the vector store, extract ISBNs safely,
    and return matching rows from books_df.
    """
    results = vector_store.similarity_search(query, k=search_k)
    isbns: list[int] = []
    for doc in results:
        content = doc.page_content.strip('"')
        parts = content.split()
        try:
            isbns.append(int(parts[0]))
        except (IndexError, ValueError):
            logger.warning("Skipping malformed result: %r", content)
    return books_df[books_df["isbn13"].isin(isbns)]


def main() -> None:
    """Demo: build store and run one example query."""
    df = load_books()
    save_tagged_descriptions(df)
    store = create_vector_store()

    query = "A book to teach children about nature"
    recs = retrieve_semantic_recommendations(query, df, store)
    logger.info("Example recommendations:")
    for title in recs["title"]:
        logger.info("  - %s", title)


if __name__ == "__main__":
    main()
