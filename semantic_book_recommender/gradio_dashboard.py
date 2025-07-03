#!/usr/bin/env python3

from typing import Any, List, Optional, Tuple

import logging
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

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
    raw = TextLoader("tagged_description.txt").load()
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=0, chunk_overlap=0
    )
    docs = splitter.split_documents(raw)
    return Chroma.from_documents(docs, OpenAIEmbeddings())


def extract_isbn(result: Any) -> Optional[int]:
    """Extract ISBN from document content."""
    content = result.page_content.strip('"')
    parts = content.split()
    try:
        return int(parts[0])
    except (IndexError, ValueError):
        logger.warning("Malformed ISBN in: %r", content)
        return None


def retrieve_semantic_recommendations(
    query: str,
    category: Optional[str],
    tone: Optional[str],
    books_df: pd.DataFrame,
    vector_store: Chroma,
    initial_k: int = 50,
    final_k: int = 16
) -> pd.DataFrame:
    """Retrieve and filter recommendations."""
    results = vector_store.similarity_search(query, k=initial_k)
    isbns = [
        isbn for isbn in (extract_isbn(r) for r in results)
        if isbn is not None
    ]
    recs = books_df[books_df["isbn13"].isin(isbns)]

    if category and category != "All":
        recs = recs[recs["simple_categories"] == category]

    recs = recs.head(final_k)

    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }
    if tone in tone_map:
        recs = recs.sort_values(
            by=tone_map[tone], ascending=False
        )
    return recs


# load once
books_data = load_books_data()
vector_store = load_document_store()


def recommend_books(
    query: str, category: str, tone: str
) -> List[Tuple[str, str]]:
    """Build the gallery output for Gradio."""
    recs = retrieve_semantic_recommendations(
        query, category, tone, books_data, vector_store
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


def launch_dashboard() -> None:
    """Set up and launch the Gradio interface."""
    cat_opts = ["All"] + sorted(
        books_data["simple_categories"].dropna().unique()
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
