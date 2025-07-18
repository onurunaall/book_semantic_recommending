from setuptools import setup, find_packages

setup(
    name="semantic-book-recommender",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas~=2.0.0",
        "numpy~=1.24.0",
        "transformers~=4.36.0",
        "torch~=2.1.0",
        "nltk~=3.8.0",
        "gradio~=4.0.0",
        "langchain~=0.2.0",
        "langchain-community~=0.2.0",
        "langchain-openai~=0.1.0",
        "langchain-chroma~=0.1.0",
        "chromadb~=0.4.0",
        "python-dotenv~=1.0.0",
        "kagglehub~=0.2.0",
    ],
    python_requires=">=3.10",
)
