def chunk_data(data, chunk_size):
    """Splits the input data into chunks of specified size."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def process_chunks(data, chunk_size):
    """Processes the data in chunks and returns a list of processed chunks."""
    chunks = list(chunk_data(data, chunk_size))
    # Here you can add additional processing logic for each chunk if needed
    return chunks

# Example usage
if __name__ == "__main__":
    sample_data = "This is a sample text that will be chunked into smaller pieces."
    chunk_size = 10
    chunks = process_chunks(sample_data, chunk_size)
    print(chunks)