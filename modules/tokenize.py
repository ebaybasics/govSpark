"""
Module for breaking documents into sentence-level tokens.
Uses NLTK's sentence tokenizer for accurate sentence boundary detection.
"""

import nltk
from typing import List
import re
from config import SENTENCE_MIN_LENGTH

# Download required NLTK data if not already present
def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded."""
    required_data = ['punkt', 'punkt_tab']
    
    for data_name in required_data:
        try:
            nltk.data.find(f'tokenizers/{data_name}')
        except LookupError:
            print(f"Downloading NLTK data: {data_name}")
            nltk.download(data_name, quiet=True)

# Download required data
ensure_nltk_data()


def tokenize_into_sentences(text: str) -> List[str]:
    """
    Break a document into individual sentences using NLTK's sentence tokenizer.
    
    This function uses NLTK's Punkt sentence tokenizer, which is trained
    to recognize sentence boundaries in various languages and handles
    abbreviations, decimal numbers, and other edge cases.
    
    Args:
        text (str): The input text to tokenize into sentences
        
    Returns:
        List[str]: A list of individual sentences
    """
    # Use NLTK's sent_tokenize for accurate sentence boundary detection
    # This is more reliable than simply splitting on periods
    sentences = nltk.sent_tokenize(text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    
    for sentence in sentences:
        # Remove extra whitespace and normalize
        cleaned_sentence = re.sub(r'\s+', ' ', sentence.strip())
        
        # Filter out very short sentences (likely not meaningful)
        if len(cleaned_sentence) >= SENTENCE_MIN_LENGTH:
            # Remove sentences that are mostly numbers or special characters
            if _is_meaningful_sentence(cleaned_sentence):
                cleaned_sentences.append(cleaned_sentence)
    
    return cleaned_sentences


def _is_meaningful_sentence(sentence: str) -> bool:
    """
    Check if a sentence contains meaningful content.
    
    Helper function to filter out sentences that are mostly
    numbers, special characters, or very short fragments.
    
    Args:
        sentence (str): The sentence to evaluate
        
    Returns:
        bool: True if sentence appears meaningful, False otherwise
    """
    # Count alphabetic characters
    alpha_chars = sum(1 for c in sentence if c.isalpha())
    total_chars = len(sentence)
    
    # Require at least 40% alphabetic characters
    if total_chars == 0 or (alpha_chars / total_chars) < 0.4:
        return False
    
    # Require at least 3 words
    words = sentence.split()
    if len(words) < 3:
        return False
    
    return True


def get_tokenization_stats(sentences: List[str]) -> dict:
    """
    Get statistics about the tokenization results.
    
    Args:
        sentences (List[str]): List of tokenized sentences
        
    Returns:
        dict: Statistics about the tokenization
    """
    if not sentences:
        return {"total_sentences": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
    
    lengths = [len(sentence) for sentence in sentences]
    
    return {
        "total_sentences": len(sentences),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths)
    }