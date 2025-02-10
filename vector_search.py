#!/usr/bin/env python3
"""
Vector Search for Semantic Book Recommendations

This module saves the tagged descriptions (ISBN + description) from the cleaned data into a text file,
builds a vector store using LangChain’s Chroma (with OpenAI embeddings),
and defines a function to retrieve semantic recommendations based on a user query.
An example demonstration is provided when the module is run directly.
"""

import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

def load_books() -> pd.DataFrame:
    """Load the cleaned books data from 'books_cleaned.csv'."""
    return pd.read_csv("books_cleaned.csv")

def save_tagged_descriptions(books: pd.DataFrame, output_file: str = "tagged_description.txt") -> None:
    """
    Save the 'tagged_description' column from the books DataFrame to a text file.

    Args:
        books (pd.DataFrame): The books DataFrame.
        output_file (str): The output file name.
    """
    books["tagged_description"].to_csv(output_file, sep="\n", index=False, header=False)

def create_vector_store(text_file: str = "tagged_description.txt") -> Chroma:
    """
    Create and return a vector store from the tagged descriptions in a text file.

    Args:
        text_file (str): Path to the text file containing tagged descriptions.

    Returns:
        Chroma: The built vector store.
    """
    load_dotenv()
    raw_documents = TextLoader(text_file).load()
    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_documents)
    vector_store = Chroma.from_documents(documents, OpenAIEmbeddings())
    return vector_store

def retrieve_semantic_recommendations(query: str, top_k: int = 10) -> pd.DataFrame:
    """
    Retrieve book recommendations that are semantically similar to the query.

    Args:
        query (str): The user query.
        top_k (int): The number of top recommendations to return (final result is filtered from a larger set).

    Returns:
        pd.DataFrame: A DataFrame containing the recommended books.
    """
    books = load_books()
    vector_store = create_vector_store()
    # Retrieve more results than needed so we can filter further.
    recs = vector_store.similarity_search(query, k=50)
    books_list = [int(doc.page_content.strip('"').split()[0]) for doc in recs]
    recommended_books = books[books["isbn13"].isin(books_list)]
    return recommended_books

def main() -> None:
    """Demonstrate the vector search by printing out a few recommended book titles for an example query."""
    print("Saving tagged descriptions and building the vector store...")
    books = load_books()
    save_tagged_descriptions(books)
    example_query = "A book to teach children about nature"
    recommendations = retrieve_semantic_recommendations(example_query)
    print("Example Recommendations:")
    for title in recommendations["title"]:
        print("  -", title)

if __name__ == "__main__":
    main()