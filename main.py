"""
Main execution file for govSpark document analysis.
Orchestrates the entire pipeline: loading → tokenizing → embedding → clustering → visualization
"""

from modules.load_document import load_html_document, validate_document_content
from modules.tokenize import tokenize_into_sentences, get_tokenization_stats
from modules.embed_text import create_embeddings, get_embedding_stats, save_embeddings
from config import DOCUMENT_PATH
import sys


def main():
    """
    Main function that orchestrates the entire document analysis pipeline.
    
    This function calls each module in sequence:
    1. Load the HTML document
    2. Tokenize into sentences
    3. Create embeddings using SBERT
    4. Reduce dimensions (to be implemented)
    5. Cluster sentences (to be implemented)
    6. Visualize clusters (to be implemented)
    7. Summarize results (to be implemented)
    """
    
    print("🚀 Starting govSpark Document Analysis Pipeline")
    print("=" * 60)
    
    # Step 1: Load the HTML document
    print(f"📄 Loading document: {DOCUMENT_PATH}")
    try:
        document_text = load_html_document(DOCUMENT_PATH)
        
        # Validate the loaded content
        if not validate_document_content(document_text):
            print("❌ Error: Document content appears to be invalid or too short")
            sys.exit(1)
            
        print(f"✅ Successfully loaded document ({len(document_text):,} characters)")
        
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        sys.exit(1)
    
    # Step 2: Tokenize the document into sentences
    print("\n🔤 Tokenizing document into sentences...")
    try:
        sentences = tokenize_into_sentences(document_text)
        
        # Get and display tokenization statistics
        stats = get_tokenization_stats(sentences)
        print(f"✅ Tokenization complete:")
        print(f"   • Total sentences: {stats['total_sentences']:,}")
        print(f"   • Average length: {stats['avg_length']:.1f} characters")
        print(f"   • Length range: {stats['min_length']}-{stats['max_length']} characters")
        
        # Show a few example sentences
        print(f"\n📝 Example sentences:")
        for i, sentence in enumerate(sentences[:3]):
            print(f"   {i+1}. {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
            
    except Exception as e:
        print(f"❌ Error during tokenization: {e}")
        sys.exit(1)
    
    # Step 3: Create embeddings using SBERT
    print("\n🧠 Creating sentence embeddings using SBERT...")
    try:
        embeddings = create_embeddings(sentences, show_progress=True)
        
        # Get and display embedding statistics
        embed_stats = get_embedding_stats(embeddings)
        print(f"✅ Embeddings created successfully:")
        print(f"   • Number of embeddings: {embed_stats['num_embeddings']:,}")
        print(f"   • Embedding dimensions: {embed_stats['embedding_dim']}")
        print(f"   • Mean norm: {embed_stats['mean_norm']:.3f}")
        print(f"   • Memory usage: {embed_stats['memory_mb']:.1f} MB")
        
        # Save embeddings for potential reuse
        save_embeddings(embeddings, "embeddings.npy")
        
    except Exception as e:
        print(f"❌ Error during embedding creation: {e}")
        sys.exit(1)
    
    # Placeholder for remaining steps
    print(f"\n⏳ Next steps (to be implemented):")
    print(f"   • Reduce dimensions with UMAP")
    print(f"   • Cluster using HDBSCAN")
    print(f"   • Visualize clusters")
    print(f"   • Generate cluster summaries")
    
    print(f"\n🎉 Pipeline execution complete!")


if __name__ == "__main__":
    main()