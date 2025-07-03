#!/usr/bin/env python3
import setuptools

setuptools.setup(
    name="semantic-book-recommender",
    version="0.1.0",
    author="Onur Ünal",
    author_email="upklw@student.kit.edu",
    description="A semantic book recommender system with data cleaning, classification, sentiment analysis, and a Gradio dashboard.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/onurunaall/book_semantic_recommending",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "kagglehub",
        "gradio",
        "python-dotenv",
        "nltk",
        "tqdm",
        "langchain>=0.2.0",
        "langchain-community>=0.2.0",
        "langchain-chroma>=0.1.1",
        "langchain-openai>=0.1.0",
        "langchain-text-splitters>=0.2.0",
        "chromadb>=0.5.0",
        "transformers>=4.40.0",
        "torch",
    ],
    python_requires='>=3.10',
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "coverage",
            "codecov"
        ]
    },
    entry_points={
        "console_scripts": [
            "semantic-book-recommender=semantic_book_recommender.main:main",
        ],
    },
)
