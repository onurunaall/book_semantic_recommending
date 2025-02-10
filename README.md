# Semantic Book Recommender

Welcome to the Semantic Book Recommender project! This project takes a raw dataset of books, cleans and enriches the data, computes emotion scores and category labels using transformer models, builds a vector store for semantic search, and finally launches an interactive dashboard so that users can search for books by describing what they’re looking for. All these steps are tied together in a unified application—you only need to run one file (main.py) to get the full experience.

## Overview

The application performs the following steps:
1. **Data Download and Cleaning:**  
   The system downloads the latest 7k books dataset from Kaggle, cleans the data by filtering out incomplete records, computes extra features (such as word counts and book age), and creates a “tagged description” (by combining each book’s ISBN with its description).

2. **Text Classification:**  
   Using a zero‑shot classification model (BART fine‑tuned on MNLI), the project maps the original categories into simpler labels (e.g. “Fiction” vs. “Nonfiction”) and automatically predicts any missing category labels from the book descriptions.

3. **Sentiment (Emotion) Analysis:**  
   A transformer-based emotion classifier is applied to each book’s description. The model reads the text and assigns scores to different emotion labels (like joy, anger, fear, etc.). The highest scores per emotion are then stored alongside the book data.

4. **Semantic Search via Vector Embeddings:**  
   Each tagged description is converted into a high‑dimensional vector embedding that captures its semantic meaning. These vectors are stored in a vector store so that, when you later enter a query, the system retrieves books whose descriptions are semantically similar to your input.

5. **Interactive Dashboard:**  
   Finally, an interactive Gradio dashboard is launched. You can type a description (for example, “a heartwarming story about forgiveness”), choose a category and even select an emotional tone, and the system will show you book recommendations complete with cover thumbnails and short summaries—all in one web interface.
