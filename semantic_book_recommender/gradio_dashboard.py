#!/usr/bin/env python3

import logging
from typing import List, Tuple

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma

from semantic_book_recommender.vector_search import retrieve_semantic_recommendations

# === Setup ===
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FALLBACK_IMAGE = "https://via.placeholder.com/200x300?text=No+Cover"
DESCRIPTION_TRUNCATE_WORDS = 30


# === Launch UI ===
def launch_dashboard(books_df: pd.DataFrame, vector_store_obj: Chroma) -> None:
    """Set up and launch the Gradio interface."""
    if books_df is None or vector_store_obj is None:
        raise ValueError("Both books_df and vector_store_obj must be provided")

    required_cols = [
        "title", "authors", "description", "large_thumbnail", "simple_categories"
    ]
    missing = [col for col in required_cols if col not in books_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if books_df.empty:
        raise ValueError("books_df cannot be empty")

    # === Recommendation function ===
    def recommend_books(query: str, category: str, tone: str) -> List[Tuple[str, str]]:
        if not query.strip():
            return []

        try:
            recs = retrieve_semantic_recommendations(
                query=query.strip(),
                books_df=books_df,
                vector_store=vector_store_obj,
                category=category,
                tone=tone
            )

            if recs.empty:
                return [(FALLBACK_IMAGE, "No recommendations found.")]

            results = []
            for _, row in recs.iterrows():
                try:
                    desc = str(row.get("description", "No description"))
                    words = desc.split()
                    summary = " ".join(words[:DESCRIPTION_TRUNCATE_WORDS])
                    if len(words) > DESCRIPTION_TRUNCATE_WORDS:
                        summary += "..."

                    authors = str(row.get("authors", "Unknown Author")).split(";")
                    authors = [a.strip() for a in authors if a.strip()]

                    if not authors:
                        author_str = "Unknown Author"
                    elif len(authors) == 1:
                        author_str = authors[0]
                    elif len(authors) == 2:
                        author_str = " and ".join(authors)
                    else:
                        author_str = f"{', '.join(authors[:-1])} and {authors[-1]}"

                    title = str(row.get("title", "Unknown Title"))
                    thumbnail = str(row.get("large_thumbnail", FALLBACK_IMAGE))

                    if not thumbnail.startswith(("http://", "https://")):
                        thumbnail = FALLBACK_IMAGE

                    caption = f"{title} by {author_str}: {summary}"
                    results.append((thumbnail, caption))

                except Exception as e:
                    logger.warning(f"Error processing book row: {e}")
                    continue

            return results or [(FALLBACK_IMAGE, "Error processing recommendations")]

        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return [(FALLBACK_IMAGE, f"Search failed: {e}")]

    # === UI ===
    try:
        categories = books_df["simple_categories"].dropna().unique()
        category_options = ["All"] + sorted(str(cat) for cat in categories if cat)
    except Exception as e:
        logger.warning(f"Could not extract categories: {e}")
        category_options = ["All"]

    tone_options = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

    with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
        gr.Markdown("# Semantic Book Recommender")

        with gr.Row():
            query_input = gr.Textbox(
                label="Description of a book:",
                placeholder="e.g. A story about forgiveness",
                max_lines=3
            )
            category_dropdown = gr.Dropdown(
                choices=category_options, label="Category", value="All"
            )
            tone_dropdown = gr.Dropdown(
                choices=tone_options, label="Emotional tone", value="All"
            )
            search_button = gr.Button("Find recommendations", variant="primary")

        gr.Markdown("## Recommendations")
        gallery = gr.Gallery(label="Books", columns=4, rows=4, height="auto")

        search_button.click(
            fn=recommend_books,
            inputs=[query_input, category_dropdown, tone_dropdown],
            outputs=gallery,
            show_progress=True
        )

    try:
        dashboard.launch()
    except Exception as e:
        logger.error(f"Failed to launch dashboard: {e}")
        raise