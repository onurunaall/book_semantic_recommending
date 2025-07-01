#!/usr/bin/env python3

import os
import logging
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables and check for OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not set; please export it before running.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_books() -> pd.DataFrame:
    """Load the cleaned books CSV into a DataFrame."""
    return pd.read_csv("books_cleaned.csv")

def save_tagged_descriptions(
    books_df: pd.DataFrame,
    output_file: str = "tagged_description.txt"
) -> None:
    """
    Save the 'tagged_description' column to a plain text file,
    one entry per line.
    """
    books_df["tagged_description"].to_csv(
        output_file, sep="\n", index=False, header=False
    )

def create_vector_store(text_file: str = "tagged_description.txt") -> Chroma:
    """
    Create a Chroma vector store from a file of tagged descriptions.
    """
    # Load raw documents from the text file
    raw_docs = TextLoader(text_file).load()
    # Split each document by newline; no overlap, no chunking
    splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
    docs = splitter.split_documents(raw_docs)
    # Build the vector store using OpenAI embeddings
    vector_store = Chroma.from_documents(docs, OpenAIEmbeddings())
    return vector_store

def retrieve_semantic_recommendations(
    query: str,
    books_df: pd.DataFrame,
    vector_store: Chroma,
    search_k: int = 50
) -> pd.DataFrame:
    """
    Given a user query, search the vector store for similar documents,
    extract the ISBNs, and return the matching books from the DataFrame.
    """
    results = vector_store.similarity_search(query, k=search_k)
    isbns = []
    for doc in results:
        # Assume the first token of page_content is the ISBN
        content = doc.page_content.strip('"')
        tokens = content.split()
        try:
            isbn = int(tokens[0])
        except (ValueError, IndexError):
            logger.warning("Skipping malformed result: %r", content)
            continue
        isbns.append(isbn)
    matched_books = books_df[books_df["isbn13"].isin(isbns)]
    return matched_books

def main() -> None:
    logger.info("Loading books and saving tagged descriptions.")
    books_df = load_books()
    save_tagged_descriptions(books_df)

    logger.info("Building vector store.")
    vector_store = create_vector_store()

    example_query = "A book to teach children about nature"
    recs = retrieve_semantic_recommendations(
        example_query, books_df, vector_store
    )

    logger.info("Example recommendations:")
    for title in recs["title"]:
        logger.info("  - %s", title)

if __name__ == "__main__":
    main()
