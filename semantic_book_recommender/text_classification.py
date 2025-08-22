#!/usr/bin/env python3

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from transformers import Pipeline, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default category mapping - can be overridden
DEFAULT_CATEGORY_MAPPING = {
    "Fiction": "Fiction",
    "Juvenile Fiction": "Children's Fiction",
    "Biography & Autobiography": "Nonfiction",
    "History": "Nonfiction",
    "Literary Criticism": "Nonfiction",
    "Philosophy": "Nonfiction",
    "Religion": "Nonfiction",
    "Comics & Graphic Novels": "Fiction",
    "Drama": "Fiction",
    "Juvenile Nonfiction": "Children's Nonfiction",
    "Science": "Nonfiction",
    "Poetry": "Fiction",
}

DEFAULT_CANDIDATE_LABELS = ["Fiction", "Nonfiction"]


def get_optimal_device() -> str:
    """
    Determine the best available device for PyTorch operations.
    
    Returns:
        str: Device name ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        try:
            # Test MPS device actually works
            test_tensor = torch.ones(1, device="mps")
            return "mps"
        except Exception as e:
            logger.warning(f"MPS device failed test: {e}. Falling back to CPU.")
            return "cpu"
    else:
        return "cpu"


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Raises:
        ValueError: If required columns are missing
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def map_categories(books_df: pd.DataFrame, category_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Map detailed categories to simplified ones.
    
    Args:
        books_df: DataFrame with 'categories' column
        category_mapping: Custom mapping dictionary
        
    Returns:
        DataFrame with 'simple_categories' column added
        
    Raises:
        ValueError: If required columns are missing
    """
    validate_dataframe_columns(books_df, ["categories"])
    
    if category_mapping is None:
        category_mapping = DEFAULT_CATEGORY_MAPPING
    
    result_df = books_df.copy()
    result_df["simple_categories"] = result_df["categories"].map(category_mapping)
    
    # Log unmapped categories
    unmapped = result_df[
        result_df["categories"].notna() & 
        result_df["simple_categories"].isna()
    ]["categories"].unique()
    
    if len(unmapped) > 0:
        logger.info(f"Unmapped categories found: {list(unmapped)}")
    
    return result_df


def initialize_zero_shot_classifier(model_name: str = "facebook/bart-large-mnli", device: Optional[str] = None) -> Pipeline:
    """
    Initialize zero-shot classifier with robust device selection.
    
    Args:
        model_name: HuggingFace model name
        device: Specific device to use (auto-detected if None)
        
    Returns:
        Initialized pipeline
        
    Raises:
        RuntimeError: If model initialization fails
    """
    if device is None:
        device = get_optimal_device()
    
    logger.info(f"Initializing zero-shot classifier on device: {device}")
    
    try:
        return pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )
    except Exception as e:
        logger.error(f"Failed to initialize classifier on {device}: {e}")
        if device != "cpu":
            logger.info("Falling back to CPU")
            return pipeline(
                "zero-shot-classification",
                model=model_name,
                device="cpu",
            )
        raise RuntimeError(f"Failed to initialize classifier: {e}")


def extract_prediction_safely(prediction: dict, index: int) -> Optional[str]:
    """
    Safely extract the top prediction from classifier output.
    
    Args:
        prediction: Single prediction dictionary
        index: Index for logging purposes
        
    Returns:
        Top predicted label or None if invalid
    """
    labels = prediction.get("labels", [])
    scores = prediction.get("scores", [])
    
    if not labels or not scores:
        logger.warning(f"Empty labels or scores in prediction {index}")
        return None
    
    if len(labels) != len(scores):
        logger.warning(f"Mismatched labels/scores length in prediction {index}")
        return None
    
    # Check for all-NaN scores (indicates classifier failure)
    scores_array = np.array(scores)
    if np.all(np.isnan(scores_array)):
        logger.warning(f"All scores are NaN in prediction {index}")
        return None
    
    try:
        # Use regular argmax after NaN check
        max_idx = np.argmax(scores_array)
        return labels[max_idx]
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to extract prediction {index}: {e}")
        return None


def predict_missing_categories(books_df: pd.DataFrame, classifier: Pipeline, candidate_labels: Optional[List[str]] = None, batch_size: int = 32) -> pd.DataFrame:
    """
    Predict missing categories using zero-shot classification.
    
    Args:
        books_df: DataFrame with books data
        classifier: Initialized classifier pipeline
        candidate_labels: Labels for classification
        batch_size: Batch size for processing
        
    Returns:
        DataFrame with isbn13 and predicted_categories columns
        
    Raises:
        ValueError: If required columns are missing
    """
    validate_dataframe_columns(books_df, ["isbn13", "description", "simple_categories"])
    
    if candidate_labels is None:
        candidate_labels = DEFAULT_CANDIDATE_LABELS
    
    missing = books_df[books_df["simple_categories"].isna()]
    if missing.empty:
        return pd.DataFrame(columns=["isbn13", "predicted_categories"])

    missing = missing[["isbn13", "description"]].reset_index(drop=True)
    
    # Filter out books with invalid descriptions
    valid_mask = (missing["description"].notna() & (missing["description"].astype(str).str.len() > 0))
    
    if not valid_mask.any():
        logger.warning("No valid descriptions found for classification")
        return pd.DataFrame({
            "isbn13": missing["isbn13"],
            "predicted_categories": [None] * len(missing),
        })
    
    valid_missing = missing[valid_mask].copy()
    descs = valid_missing["description"].astype(str).tolist()
    
    # Process in batches to handle memory constraints
    all_predictions = []
    
    for i in range(0, len(descs), batch_size):
        batch_descs = descs[i:i + batch_size]
        
        try:
            batch_preds = classifier(batch_descs, candidate_labels=candidate_labels)
            all_predictions.extend(batch_preds)
        except Exception as e:
            logger.error(f"Batch classification failed for batch {i//batch_size}: {e}")
            # Add None predictions for failed batch
            all_predictions.extend([{"labels": [], "scores": []}] * len(batch_descs))
    
    # Extract predictions safely
    predicted_labels = [
        extract_prediction_safely(pred, i) 
        for i, pred in enumerate(all_predictions)
    ]
    
    # Create result DataFrame for valid entries
    result_valid = pd.DataFrame({
        "isbn13": valid_missing["isbn13"],
        "predicted_categories": predicted_labels,
    })
    
    # Add None predictions for invalid entries
    invalid_missing = missing[~valid_mask]
    if not invalid_missing.empty:
        result_invalid = pd.DataFrame({
            "isbn13": invalid_missing["isbn13"],
            "predicted_categories": [None] * len(invalid_missing),
        })
        result = pd.concat([result_valid, result_invalid], ignore_index=True)
    else:
        result = result_valid
    
    return result


def classify_categories(
    books_df: pd.DataFrame,
    category_mapping: Optional[Dict[str, str]] = None,
    candidate_labels: Optional[List[str]] = None,
    classifier: Optional[Pipeline] = None,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Map categories and predict missing ones using zero-shot classification.
    
    Args:
        books_df: Input DataFrame with books data
        category_mapping: Custom category mapping
        candidate_labels: Labels for zero-shot classification
        classifier: Pre-initialized classifier (creates new one if None)
        batch_size: Batch size for processing
        
    Returns:
        DataFrame with simple_categories filled
        
    Raises:
        ValueError: If input validation fails
    """
    validate_dataframe_columns(books_df, ["isbn13", "categories", "description"])
    
    # Map known categories
    df = map_categories(books_df, category_mapping)
    
    # Count missing categories
    missing_count = df["simple_categories"].isna().sum()
    if missing_count == 0:
        logger.info("No missing categories to predict")
        return df
    
    logger.info(f"Predicting {missing_count} missing categories")
    
    # Initialize classifier if not provided
    if classifier is None:
        classifier = initialize_zero_shot_classifier()
    
    # Predict missing categories
    preds = predict_missing_categories(
        df, 
        classifier, 
        candidate_labels, 
        batch_size
    )
    
    # Merge predictions
    df = df.merge(preds, on="isbn13", how="left")
    
    # Fill missing categories with predictions
    df["simple_categories"] = df["simple_categories"].fillna(
        df["predicted_categories"]
    )
    
    # Log results
    filled_count = preds["predicted_categories"].notna().sum()
    logger.info(f"Successfully predicted {filled_count} out of {missing_count} categories")
    
    return df.drop(columns=["predicted_categories"])