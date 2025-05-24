"""
Configuration file for govSpark project.
Stores all tunable parameters and settings in one place.
"""

# Document processing settings
DOCUMENT_PATH = "BILLS.html"
SENTENCE_MIN_LENGTH = 10  # Minimum characters for a valid sentence

# Embedding model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SBERT model for text embeddings
BATCH_SIZE = 32  # Number of sentences to process at once

# UMAP dimension reduction settings
UMAP_N_NEIGHBORS = 15  # Number of neighbors for UMAP
UMAP_MIN_DIST = 0.1   # Minimum distance between points in low-dimensional space
UMAP_N_COMPONENTS = 2  # Number of dimensions to reduce to (2D for plotting)
UMAP_RANDOM_STATE = 42  # For reproducible results

# HDBSCAN clustering settings
HDBSCAN_MIN_CLUSTER_SIZE = 5  # Minimum number of points in a cluster
HDBSCAN_MIN_SAMPLES = 3       # Minimum samples for core points

# Visualization settings
PLOT_FIGURE_SIZE = (12, 8)    # Width, height in inches
PLOT_DPI = 300                # Resolution for saved plots
PLOT_SAVE_PATH = "cluster_visualization.png"

# Summary settings
EXAMPLES_PER_CLUSTER = 3      # Number of example sentences to show per cluster
MAX_SENTENCE_LENGTH = 200     # Maximum characters to display per sentence