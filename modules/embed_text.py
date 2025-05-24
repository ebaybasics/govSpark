"""
Module for creating text embeddings using SBERT (Sentence-BERT).
Handles sentence encoding and embedding generation for semantic analysis.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import torch
from tqdm import tqdm
from config import EMBEDDING_MODEL, BATCH_SIZE


def load_embedding_model(model_name: str = None) -> SentenceTransformer:
    """
    Load and initialize the SBERT model for text embedding.
    
    This function loads a pre-trained Sentence-BERT model that can convert
    text sentences into high-dimensional vectors (embeddings) that capture
    semantic meaning. Similar sentences will have similar embeddings.
    
    Args:
        model_name (str, optional): Name of the SBERT model to load.
                                   If None, uses the default from config.
        
    Returns:
        SentenceTransformer: Loaded SBERT model ready for encoding
        
    Raises:
        Exception: If model loading fails
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL
    
    try:
        print(f"🔄 Loading SBERT model: {model_name}")
        
        # Load the pre-trained model
        # This will download the model if it's not already cached locally
        model = SentenceTransformer(model_name)
        
        # Check if CUDA is available and use GPU if possible
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        print(f"✅ Model loaded successfully on {device.upper()}")
        print(f"   • Model dimensions: {model.get_sentence_embedding_dimension()}")
        
        return model
        
    except Exception as e:
        raise Exception(f"Failed to load embedding model {model_name}: {str(e)}")


def create_embeddings(sentences: List[str], model: SentenceTransformer = None, 
                     batch_size: int = None, show_progress: bool = True) -> np.ndarray:
    """
    Convert a list of sentences into numerical embeddings using SBERT.
    
    This function takes text sentences and converts them into high-dimensional
    vectors that capture semantic meaning. The resulting embeddings can be used
    for clustering, similarity analysis, and other NLP tasks.
    
    Args:
        sentences (List[str]): List of sentences to convert to embeddings
        model (SentenceTransformer, optional): Pre-loaded SBERT model.
                                              If None, loads default model.
        batch_size (int, optional): Number of sentences to process at once.
                                   If None, uses default from config.
        show_progress (bool): Whether to show progress bar during processing
        
    Returns:
        np.ndarray: Array of embeddings with shape (n_sentences, embedding_dim)
        
    Raises:
        ValueError: If sentences list is empty
        Exception: If embedding creation fails
    """
    if not sentences:
        raise ValueError("Cannot create embeddings for empty sentence list")
    
    if model is None:
        model = load_embedding_model()
    
    if batch_size is None:
        batch_size = BATCH_SIZE
    
    try:
        print(f"🔄 Creating embeddings for {len(sentences):,} sentences...")
        
        # Create embeddings with progress tracking
        # The encode method handles batching internally for efficiency
        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better clustering results
        )
        
        print(f"✅ Embeddings created successfully")
        print(f"   • Shape: {embeddings.shape}")
        print(f"   • Data type: {embeddings.dtype}")
        print(f"   • Memory usage: {embeddings.nbytes / 1024 / 1024:.1f} MB")
        
        return embeddings
        
    except Exception as e:
        raise Exception(f"Failed to create embeddings: {str(e)}")


def get_embedding_stats(embeddings: np.ndarray) -> dict:
    """
    Calculate statistics about the generated embeddings.
    
    Args:
        embeddings (np.ndarray): Array of embeddings
        
    Returns:
        dict: Statistics about the embeddings
    """
    if embeddings.size == 0:
        return {
            "num_embeddings": 0,
            "embedding_dim": 0,
            "mean_norm": 0,
            "std_norm": 0,
            "memory_mb": 0
        }
    
    # Calculate L2 norms for each embedding
    norms = np.linalg.norm(embeddings, axis=1)
    
    return {
        "num_embeddings": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "memory_mb": embeddings.nbytes / 1024 / 1024
    }


def save_embeddings(embeddings: np.ndarray, filepath: str) -> bool:
    """
    Save embeddings to disk for later use.
    
    Args:
        embeddings (np.ndarray): Embeddings to save
        filepath (str): Path where to save the embeddings
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        np.save(filepath, embeddings)
        print(f"💾 Embeddings saved to: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Failed to save embeddings: {e}")
        return False


def load_embeddings(filepath: str) -> np.ndarray:
    """
    Load previously saved embeddings from disk.
    
    Args:
        filepath (str): Path to the saved embeddings file
        
    Returns:
        np.ndarray: Loaded embeddings
        
    Raises:
        Exception: If loading fails
    """
    try:
        embeddings = np.load(filepath)
        print(f"📁 Embeddings loaded from: {filepath}")
        print(f"   • Shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        raise Exception(f"Failed to load embeddings from {filepath}: {str(e)}")


def calculate_sentence_similarity(embeddings: np.ndarray, 
                                sentence_idx1: int, 
                                sentence_idx2: int) -> float:
    """
    Calculate cosine similarity between two sentence embeddings.
    
    Args:
        embeddings (np.ndarray): Array of all embeddings
        sentence_idx1 (int): Index of first sentence
        sentence_idx2 (int): Index of second sentence
        
    Returns:
        float: Cosine similarity between the two sentences (0-1)
    """
    if (sentence_idx1 >= len(embeddings) or sentence_idx2 >= len(embeddings) or
        sentence_idx1 < 0 or sentence_idx2 < 0):
        raise ValueError("Invalid sentence indices")
    
    # Calculate cosine similarity
    embedding1 = embeddings[sentence_idx1]
    embedding2 = embeddings[sentence_idx2]
    
    # Since embeddings are normalized, dot product gives cosine similarity
    similarity = np.dot(embedding1, embedding2)
    
    return float(similarity)