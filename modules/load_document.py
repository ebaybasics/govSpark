"""
Module for loading and extracting text from HTML documents.
Handles file reading and basic text preprocessing.
"""

from bs4 import BeautifulSoup
import os
# from typing import str


def load_html_document(file_path: str) -> str:
    """
    Load an HTML document and extract clean text content.
    
    This function reads an HTML file, parses it using BeautifulSoup,
    and extracts only the text content while removing HTML tags.
    
    Args:
        file_path (str): Path to the HTML file to load
        
    Returns:
        str: Clean text content extracted from the HTML document
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        Exception: If there's an error reading or parsing the file
    """
    # Check if file exists before attempting to read
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HTML file not found: {file_path}")
    
    try:
        # Read the HTML file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Parse HTML content using BeautifulSoup
        # 'html.parser' is a built-in parser that doesn't require external dependencies
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements completely
        # These contain code/styling that isn't useful for text analysis
        for script in soup(["script", "style"]):
            script.extract()
        
        # Extract text content from the parsed HTML
        # get_text() method extracts all text while removing HTML tags
        text = soup.get_text()
        
        # Clean up the extracted text
        # Split into lines and remove extra whitespace
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each and remove empty chunks
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Join all text chunks with single spaces
        clean_text = ' '.join(chunk for chunk in chunks if chunk)
        
        return clean_text
        
    except Exception as e:
        raise Exception(f"Error processing HTML file {file_path}: {str(e)}")


def validate_document_content(text: str) -> bool:
    """
    Validate that the loaded document contains meaningful content.
    
    Args:
        text (str): The text content to validate
        
    Returns:
        bool: True if content is valid, False otherwise
    """
    # Check if text is not empty and has minimum length
    if not text or len(text.strip()) < 100:
        return False
    
    # Check if text contains actual words (not just numbers/symbols)
    words = text.split()
    if len(words) < 10:
        return False
    
    return True