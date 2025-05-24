from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import os

class HTMLParser:
    """Class for parsing and working with HTML documents."""
    
    def __init__(self):
        self.soup = None
        self.html_content = None
    
    def load_from_file(self, file_path):
        """Load HTML from a local file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.html_content = file.read()
            self.soup = BeautifulSoup(self.html_content, 'html.parser')
            return True
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return False
    
    def load_from_url(self, url):
        """Load HTML from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.html_content = response.text
            self.soup = BeautifulSoup(self.html_content, 'html.parser')
            return True
        except Exception as e:
            print(f"Error loading URL {url}: {e}")
            return False
    
    def load_html(self, source):
        """Load HTML from file path, URL, or raw HTML string."""
        if os.path.exists(source):
            return self.load_from_file(source)
        elif urlparse(source).scheme:
            return self.load_from_url(source)
        else:
            # Treat as raw HTML content
            self.html_content = source
            self.soup = BeautifulSoup(source, 'html.parser')
            return True
    
    def get_title(self):
        """Get the title of the HTML document."""
        if self.soup and self.soup.title:
            return self.soup.title.string
        return None
    
    def get_text(self):
        """Extract all text from the HTML document."""
        if self.soup:
            return self.soup.get_text()
        return None
    
    def find_elements(self, tag, attributes=None):
        """Find elements by tag name and optional attributes."""
        if self.soup:
            if attributes:
                return self.soup.find_all(tag, attributes)
            else:
                return self.soup.find_all(tag)
        return []
    
    def get_links(self):
        """Extract all links from the HTML document."""
        if self.soup:
            return [a.get('href') for a in self.soup.find_all('a') if a.get('href')]
        return []
    
    def get_raw_html(self):
        """Return the raw HTML content."""
        return self.html_content