# main.py

from source_files.src.loader import load_data
from source_files.src.chunker import chunk_data, process_chunks
from source_files.src.summarizer import summarize_data
from source_files.src.aggregator import combine_results
from html_parser import HTMLParser

def main():
    """Main function to demonstrate HTML parser usage."""
    
    # Create an instance of our HTML parser
    parser = HTMLParser()
    
    # Example HTML source - replace with your actual HTML file or URL
    html_source = "BILLS.html"  # Could be a file path or URL
    
    # Load the HTML document
    if parser.load_html(html_source):
        print(f"Successfully loaded HTML from: {html_source}")
        
        # Example of extracting information
        title = parser.get_title()
        print(f"Document title: {title}")
        

        # The parsed document is now in memory and ready to be used
        print("HTML document loaded and parsed successfully!")
    else:
        print(f"Failed to load HTML from: {html_source}")

if __name__ == "__main__":
    main()