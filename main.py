# main.py

from source_files.src.loader import load_data
from source_files.src.chunker import chunk_data, process_chunks
from source_files.src.summarizer import summarize_data
from source_files.src.aggregator import combine_results

def main():
    # Load data
    data = load_data('data/source.txt')
    
    # Chunk data - adding required chunk_size parameter
    # Using process_chunks which returns a list of chunks
    chunk_size = 1000  # Adjust this value based on your needs
    chunks = process_chunks(data, chunk_size)
    
    # Summarize each chunk using the correct function name
    summaries = summarize_data(chunks)
    
    # Convert summaries to a dictionary format for aggregation
    summary_dict = {f"chunk_{i}": summary for i, summary in enumerate(summaries)}
    results = [summary_dict]
    
    # Aggregate summaries using the available function
    final_summary = combine_results(results)
    
    # Print the final summary
    print(final_summary)

if __name__ == "__main__":
    main()