from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="semantic-book-recommender",
    version="0.1.0",
    author="Onur Unal",
    author_email="onur.unak492@gmail.com",
    description="A semantic book recommendation system using ML and vector search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/semantic-book-recommender",
    
    packages=find_packages(),
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    python_requires=">=3.10",
    
    install_requires=[
        "pandas>=2.0.0,<2.3.0",
        "numpy>=1.24.0,<2.0.0",
        "transformers>=4.36.0,<4.40.0",
        "torch>=1.13.0,<3.0.0",
        "nltk>=3.8.0,<4.0.0",
        "gradio>=4.0.0,<5.0.0",
        "langchain>=0.2.0,<0.3.0",
        "langchain-community>=0.2.0,<0.3.0",
        "langchain-openai>=0.1.0,<0.2.0",
        "langchain-chroma>=0.1.0,<0.2.0",
        "chromadb>=0.4.0,<0.5.0",
        "python-dotenv>=1.0.0,<2.0.0",
        "kagglehub>=0.2.0,<1.0.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "mypy>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "semantic-book-recommender=semantic_book_recommender.main:main",
        ],
    },
    
    include_package_data=True,
    zip_safe=False,  # Required if package ships with non-Python files
)