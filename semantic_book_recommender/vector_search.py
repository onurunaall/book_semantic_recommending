#!/usr/bin/env python3

import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_books() -> pd.DataFrame:
    return pd.read_csv("books_cleaned.csv")

def save_tagged_descriptions(books_df: pd.DataFrame, output_file: str = "tagged_description.txt") -> None:
    # Write the 'tagged_description' column to a text file, one entry per line.
    books_df["tagged_description"].to_csv(output_file, sep="\n", index=False, header=False)

def create_vector_store(text_file: str = "tagged_description.txt") -> Chroma:
    raw_documents = TextLoader(text_file).load()
    
    # Split the text into individual documents using newline as a separator.
    splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
    documents = splitter.split_documents(raw_documents)

    vector_store: Chroma = Chroma.from_documents(documents, OpenAIEmbeddings())
    return vector_store

def retrieve_semantic_recommendations(query: str, top_k: int = 10) -> pd.DataFrame:
    books_df = load_books()
    vector_store = create_vector_store()
    
    # Retrieve more results than needed to allow for sufficient filtering.
    search_results = vector_store.similarity_search(query, k=50)
    
    # Extract ISBNs from the search results.
    isbn_list = []
    for document in search_results:
        content = document.page_content.strip('"')
        tokens = content.split()
        isbn_list.append(int(tokens[0]))
    
    recommended_books = books_df[books_df["isbn13"].isin(isbn_list)]
    return recommended_books

def main() -> None:
    print("Saving tagged descriptions and building the vector store...")
    books_df = load_books()
    save_tagged_descriptions(books_df)
    
    example_query = "A book to teach children about nature"
    recommendations = retrieve_semantic_recommendations(example_query)
    
    print("Example Recommendations:")
    for title in recommendations["title"]:
        print("  -", title)

if __name__ == "__main__":
    main()
