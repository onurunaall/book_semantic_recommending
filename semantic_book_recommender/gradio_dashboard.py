#!/usr/bin/env python3

from typing import Any, List, Optional, Tuple

import logging
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

from semantic_book_recommender.vector_search import (
    retrieve_semantic_recommendations,
    create_vector_store,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FALLBACK_IMAGE = (
    "https://via.placeholder.com/200x300?text=No+Cover"
)


def load_books_data() -> pd.DataFrame:
    """Load books with emotions and update thumbnails."""
    df = pd.read_csv("books_with_emotions.csv")

    def fmt(url: Any) -> str:
        if isinstance(url, str) and url:
            return f"{url}&fife=w800"
        return FALLBACK_IMAGE

    df["large_thumbnail"] = df["thumbnail"].apply(fmt)
    return df


def load_document_store() -> Chroma:
    """Load tagged descriptions and build a Chroma store."""
    return create_vector_store("tagged_description.txt")


def launch_dashboard(
    books_df: pd.DataFrame = None,
    vector_store_obj = None
) -> None:
    """Set up and launch the Gradio interface."""
    
    # Load data if not provided
    if books_df is None:
        books_df = load_books_data()
    
    if vector_store_obj is None:
        vector_store_obj = load_document_store()
    
    def recommend_books(
        query: str, category: str, tone: str
    ) -> List[Tuple[str, str]]:
        """Build the gallery output for Gradio."""
        recs = retrieve_semantic_recommendations(
            query, books_df, vector_store_obj, category, tone
        )
        
        out: List[Tuple[str, str]] = []
        for _, row in recs.iterrows():
            words = row["description"].split()
            short = " ".join(words[:30]) + "..."
            authors = [
                a.strip() for a in row["authors"].split(";")
                if a.strip()
            ]
            if not authors:
                auth = ""
            elif len(authors) == 1:
                auth = authors[0]
            elif len(authors) == 2:
                auth = " and ".join(authors)
            else:
                auth = (
                    f"{', '.join(authors[:-1])} and {authors[-1]}"
                )
            caption = f"{row['title']} by {auth}: {short}"
            out.append((row["large_thumbnail"], caption))
        return out
    
    cat_opts = ["All"] + sorted(
        books_df["simple_categories"].dropna().unique()
    )
    tone_opts = [
        "All", "Happy", "Surprising",
        "Angry", "Suspenseful", "Sad",
    ]

    with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
        gr.Markdown("# Semantic Book Recommender")
        with gr.Row():
            q = gr.Textbox(
                label="Description of a book:",
                placeholder="e.g. A story about forgiveness"
            )
            c = gr.Dropdown(
                choices=cat_opts,
                label="Category",
                value="All"
            )
            t = gr.Dropdown(
                choices=tone_opts,
                label="Emotional tone",
                value="All"
            )
            btn = gr.Button("Find recommendations")
        gr.Markdown("## Recommendations")
        gallery = gr.Gallery(
            label="Books", columns=8, rows=2
        )
        btn.click(
            fn=recommend_books,
            inputs=[q, c, t],
            outputs=gallery
        )
    dashboard.launch()


if __name__ == "__main__":
    launch_dashboard()
