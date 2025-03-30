#!/usr/bin/env python3

from typing import List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()  # Load API keys and other env variables

def load_books_data() -> pd.DataFrame:
    # Load books data with emotion scores and update thumbnail URLs.
    books_df: pd.DataFrame = pd.read_csv("books_with_emotions.csv")
    books_df["large_thumbnail"] = books_df["thumbnail"] + "&fife=w800"
    books_df["large_thumbnail"] = np.where(
        books_df["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books_df["large_thumbnail"]
    )
    return books_df

def load_document_store() -> Chroma:
    # Load tagged descriptions and build the vector store.
    raw_docs = TextLoader("tagged_description.txt").load()
    splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    split_docs = splitter.split_documents(raw_docs)
    vector_store: Chroma = Chroma.from_documents(split_docs, OpenAIEmbeddings())
    return vector_store

def extract_isbn(result: Any) -> int:
    # Extract the ISBN from the document's page_content.
    content: str = result.page_content.strip('"')
    tokens: List[str] = content.split()
    return int(tokens[0])

def retrieve_semantic_recommendations(
    query: str,
    category: Optional[str] = None,
    tone: Optional[str] = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
    vector_store: Optional[Chroma] = None,
    books_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    # Run similarity search on the vector store.
    search_results = vector_store.similarity_search(query, k=initial_top_k)
    
    # Extract ISBN numbers using a helper function.
    isbn_list: List[int] = [extract_isbn(result) for result in search_results]
    
    # Filter books that match the extracted ISBNs.
    recommended_books: pd.DataFrame = books_df[books_df["isbn13"].isin(isbn_list)].head(initial_top_k)
    
    # Apply category filter if a specific category is chosen.
    if category and category != "All":
        recommended_books = recommended_books[recommended_books["simple_categories"] == category].head(final_top_k)
    else:
        recommended_books = recommended_books.head(final_top_k)
    
    # Sort recommendations based on emotional tone if specified.
    if tone == "Happy":
        recommended_books.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        recommended_books.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        recommended_books.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        recommended_books.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        recommended_books.sort_values(by="sadness", ascending=False, inplace=True)
    
    return recommended_books

def recommend_books(query: str, category: str, tone: str) -> List[Tuple[str, str]]:
    # Load the books data and vector store, then get recommendations.
    books_data: pd.DataFrame = load_books_data()
    vector_db: Chroma = load_document_store()
    recs: pd.DataFrame = retrieve_semantic_recommendations(
        query=query,
        category=category,
        tone=tone,
        vector_store=vector_db,
        books_df=books_data
    )
    
    recommended_list: List[Tuple[str, str]] = []
    for _, row in recs.iterrows():
        short_description: str = " ".join(row["description"].split()[:30]) + "..."
        authors: List[str] = row["authors"].split(";")
        
        if len(authors) == 2:
            formatted_authors: str = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            formatted_authors = f"{', '.join(authors[:-1])}, and {authors[-1]}"
        else:
            formatted_authors = row["authors"]
        
        caption: str = f"{row['title']} by {formatted_authors}: {short_description}"
        
        recommended_list.append((row["large_thumbnail"], caption))
    
    return recommended_list

def launch_dashboard() -> None:
    # Prepare dropdown options.
    books_data: pd.DataFrame = load_books_data()
    
    category_options: List[str] = ["All"] + sorted(books_data["simple_categories"].dropna().unique())
    tone_options: List[str] = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

    with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
        gr.Markdown("# Semantic Book Recommender")
        
        with gr.Row():
            query_input = gr.Textbox(
                label="Please enter a description of a book:",
                placeholder="e.g., A story about forgiveness"
            )
            
            category_dropdown = gr.Dropdown(choices=category_options, label="Select a category:", value="All")
            tone_dropdown = gr.Dropdown(choices=tone_options, label="Select an emotional tone:", value="All")
            search_button = gr.Button("Find recommendations")
        
        gr.Markdown("## Recommendations")
        
        results_gallery = gr.Gallery(label="Recommended books", columns=8, rows=2)
        
        search_button.click(
            fn=recommend_books,
            inputs=[query_input, category_dropdown, tone_dropdown],
            outputs=results_gallery
        )
        
    dashboard.launch()

if __name__ == "__main__":
    launch_dashboard()
