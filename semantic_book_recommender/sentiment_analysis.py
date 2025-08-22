#!/usr/bin/env python3

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import nltk
from transformers import Pipeline, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected emotion labels from j-hartmann/emotion-english-distilroberta-base
EXPECTED_EMOTION_LABELS = {"anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"}
DEFAULT_EMOTION_SCORES = {
    "anger": 0.0,
    "disgust": 0.0, 
    "fear": 0.0,
    "joy": 0.0,
    "neutral": 1.0,  # Default to neutral
    "sadness": 0.0,
    "surprise": 0.0
}


def ensure_nltk_data() -> None:
    """Ensure required NLTK data is available with proper error handling."""
    try:
        # First check if punkt is already available
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK punkt tokenizer already available")
        return
    except LookupError:
        pass
    
    try:
        # Check if punkt_tab is available (newer NLTK versions)
        nltk.data.find('tokenizers/punkt_tab')
        logger.info("NLTK punkt_tab tokenizer already available")
        return
    except LookupError:
        pass
    
    # Try to download punkt data
    try:
        logger.info("Downloading NLTK punkt tokenizer...")
        success = nltk.download("punkt", quiet=False)
        
        if not success:
            # Try the new punkt_tab for newer NLTK versions (3.8.2+)
            logger.warning("punkt download failed, trying punkt_tab...")
            success = nltk.download("punkt_tab", quiet=False)
        
        if not success:
            raise RuntimeError("Failed to download NLTK punkt tokenizer")
            
        logger.info("NLTK punkt tokenizer downloaded successfully")
        
    except Exception as e:
        error_msg = (
            f"NLTK data download failed: {e}. Please manually install with: python -m nltk.downloader punkt")
        raise RuntimeError(error_msg) from e


def get_optimal_device() -> str:
    """Determine the best available device with comprehensive error handling."""
    if torch.cuda.is_available():
        try:
            test_tensor = torch.ones(1).cuda()
            _ = test_tensor * 2  # Simple operation test
            logger.info("Using CUDA device")
            return "cuda"
        except Exception as e:
            logger.warning(f"CUDA available but failed test: {e}")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            # Test MPS device actually works
            test_tensor = torch.ones(1, device="mps")
            test_result = test_tensor * 2  # Simple operation test
            del test_tensor, test_result  # Clean up
            logger.info("Using MPS device")
            return "mps"
        except Exception as e:
            logger.warning(f"MPS available but failed test: {e}. Falling back to CPU.")
            # Set fallback environment variable for future operations
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    logger.info("Using CPU device")
    return "cpu"


def validate_classifier_output(test_result: List[List[Dict[str, Any]]]) -> None:
    """Validate that classifier output has expected structure and labels."""
    if not isinstance(test_result, list) or not test_result:
        raise ValueError("Classifier returned unexpected format: expected non-empty list")
    
    first_result = test_result[0]
    if not isinstance(first_result, list) or not first_result:
        raise ValueError("Classifier returned unexpected format: expected list of predictions")
    
    # Validate emotion labels
    returned_labels = {item["label"] for item in first_result if isinstance(item, dict) and "label" in item}
    
    if not EXPECTED_EMOTION_LABELS.issubset(returned_labels):
        missing = EXPECTED_EMOTION_LABELS - returned_labels
        raise ValueError(f"Classifier missing expected emotion labels: {missing}")
    
    # Validate score structure
    for item in first_result:
        if not isinstance(item, dict) or "score" not in item:
            raise ValueError("Classifier output missing required 'score' field")
        if not isinstance(item["score"], (int, float)):
            raise ValueError("Classifier score must be numeric")


def initialize_classifier() -> Pipeline:
    """Initialize emotion classifier with comprehensive validation."""
    device = get_optimal_device()
    
    try:
        logger.info("Initializing emotion classifier...")
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,  # Return all emotion scores
            device=device,
            return_all_scores=True  # Ensure all scores are returned
        )
        
        # Test the classifier with a sample input
        test_input = "This is a test sentence."
        test_result = classifier(test_input)
        
        # Validate the output structure
        validate_classifier_output(test_result)
        
        returned_labels = {item["label"] for item in test_result[0]}
        logger.info(f"Emotion classifier initialized successfully on {device}")
        logger.info(f"Available emotion labels: {sorted(returned_labels)}")
        
        return classifier
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize emotion classifier: {e}") from e


def calculate_max_emotion_scores(sentence_predictions: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """Return max score per label across all sentence predictions."""
    max_scores: Dict[str, float] = DEFAULT_EMOTION_SCORES.copy()
    
    for preds in sentence_predictions:
        if not preds:  # Skip empty predictions
            continue
            
        for entry in preds:
            if not isinstance(entry, dict) or "label" not in entry or "score" not in entry:
                logger.warning(f"Invalid prediction entry: {entry}")
                continue
                
            label = entry["label"]
            score = entry["score"]
            
            if not isinstance(score, (int, float)):
                logger.warning(f"Invalid score type for label {label}: {type(score)}")
                continue
                
            if label in max_scores:
                if score > max_scores[label]:
                    max_scores[label] = score
            else:
                logger.warning(f"Unexpected emotion label: {label}")
    
    return max_scores


def create_empty_prediction(sentence: str) -> List[Dict[str, Any]]:
    """Create a properly structured empty prediction for failed classifications."""
    return [
        {"label": label, "score": score}
        for label, score in DEFAULT_EMOTION_SCORES.items()
    ]


def validate_input(books_df: pd.DataFrame) -> None:
    """Validate input DataFrame structure and content."""
    if books_df.empty:
        raise ValueError("Input DataFrame is empty")
    
    required_columns = ["isbn13", "description"]
    missing_cols = [col for col in required_columns if col not in books_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for null descriptions
    null_descriptions = books_df["description"].isna().sum()
    if null_descriptions > 0:
        logger.warning(f"Found {null_descriptions} books with null descriptions")
    
    # Check for duplicate ISBNs
    duplicate_isbns = books_df["isbn13"].duplicated().sum()
    if duplicate_isbns > 0:
        raise ValueError(f"Found {duplicate_isbns} duplicate ISBN values")
    
    logger.info(f"Input validation passed for {len(books_df)} books")


def safe_tokenize(text: str) -> List[str]:
    """Safely tokenize text with error handling."""
    if pd.isna(text) or not isinstance(text, str):
        return [""]  # Return single empty sentence for invalid input
    
    try:
        sentences = nltk.sent_tokenize(text.strip())
        return sentences if sentences else [""]
    except Exception as e:
        logger.warning(f"Tokenization failed for text: {e}")
        return [str(text)]  # Fallback: treat entire text as one sentence


def process_books(classifier: Pipeline, books_df: pd.DataFrame, batch_size: int = 128) -> pd.DataFrame:
    """
    Process books for emotion analysis with comprehensive error handling.
    
    Args:
        classifier: Initialized emotion classification pipeline
        books_df: DataFrame with 'isbn13' and 'description' columns
        batch_size: Number of sentences to process in each batch
        
    Returns:
        DataFrame with ISBN and emotion scores
    """
    # Validate inputs
    validate_input(books_df)
    
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    logger.info(f"Processing {len(books_df)} books for emotion analysis")
    
    records = []
    total_sentences = 0
    failed_books = 0
    
    # Process books one by one for better error isolation
    for idx, (isbn, description) in enumerate(zip(books_df["isbn13"], books_df["description"])):
        try:
            # Tokenize description into sentences
            sentences = safe_tokenize(description)
            total_sentences += len(sentences)
            
            # Process sentences in batches
            sentence_preds: List[List[Dict[str, Any]]] = []
            
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                
                try:
                    batch_preds = classifier(batch)
                    
                    # Validate batch predictions structure
                    if not isinstance(batch_preds, list) or len(batch_preds) != len(batch):
                        raise ValueError(f"Classifier returned unexpected batch size: {len(batch_preds)} vs {len(batch)}")
                    
                    sentence_preds.extend(batch_preds)
                    
                except Exception as e:
                    logger.warning(f"Failed to process batch for book {isbn}: {e}")
                    
                    # Add properly structured empty predictions for failed sentences
                    for sentence in batch:
                        sentence_preds.append(create_empty_prediction(sentence))
                    
                    # Fail fast on critical errors
                    if "CUDA out of memory" in str(e) or "MPS out of memory" in str(e):
                        raise RuntimeError(f"GPU memory exhausted during emotion classification: {e}") from e
            
            # Calculate max scores for this book
            max_scores = calculate_max_emotion_scores(sentence_preds)
            max_scores["isbn13"] = isbn
            records.append(max_scores)
            
            # Progress logging
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(books_df)} books")
                
        except Exception as e:
            logger.error(f"Failed to process book {isbn}: {e}")
            failed_books += 1
            
            # Add default scores for completely failed book
            default_record = DEFAULT_EMOTION_SCORES.copy()
            default_record["isbn13"] = isbn
            records.append(default_record)
    
    # Final logging
    logger.info(f"Emotion analysis complete:")
    logger.info(f"- Total books processed: {len(books_df)}")
    logger.info(f"- Total sentences analyzed: {total_sentences}")
    logger.info(f"- Books with processing failures: {failed_books}")
    
    if failed_books > len(books_df) * 0.1:  # More than 10% failures
        logger.warning(f"High failure rate: {failed_books}/{len(books_df)} books failed processing")
    
    result_df = pd.DataFrame.from_records(records)
    
    if len(result_df) != len(books_df):
        raise RuntimeError(f"Output size mismatch: {len(result_df)} vs {len(books_df)}")
    
    return result_df


# Initialize NLTK data when module is imported
try:
    ensure_nltk_data()
except RuntimeError as e:
    logger.error(f"Failed to initialize NLTK data: {e}")
    logger.error("Emotion analysis functionality will not work properly")