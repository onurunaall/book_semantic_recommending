# Semantic Book Recommender

This is a semantic book recommendation project I built to learn how to work with data, machine learning models, and build a simple user interface. The idea is to let users describe a book they’re interested in, and the app tries to recommend books that match that description in meaning and mood.

It uses book metadata and descriptions, applies some filtering and cleanup, then classifies books, scores their emotions, and finally allows the user to search semantically through a simple Gradio dashboard.

---

## Features

- **Dataset download and cleanup**  
  Removes books with missing fields or short descriptions.

- **Category mapping**  
  Simplifies book categories and fills in missing ones using a language model.

- **Sentiment analysis**  
  Scores book descriptions with emotions like joy, sadness, anger, etc.

- **Vector search**  
  Uses OpenAI embeddings to make a searchable vector database from book descriptions.

- **Gradio app**  
  Lets you input a book idea, choose a category and mood, and shows book covers with short summaries.
