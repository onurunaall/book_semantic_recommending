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
        "numpy==1.26.4",
        "pandas==2.2.3",
        "kagglehub==0.3.5",
        "gradio>=5.0.0",
        "python-dotenv==1.0.1",
        "langchain==0.3.12",
        "langchain-chroma>=0.1.5",
        "langchain-community==0.3.12",
        "langchain-openai==0.2.12",
        "langchain-text-splitters==0.3.3",
        "chromadb==0.5.4",
        "transformers==4.47.1",
        "torch==2.5.1",
        "tqdm==4.67.1",
        "nltk"
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
