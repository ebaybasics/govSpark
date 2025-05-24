def load_data(file_path):
    """Load data from a specified file path."""
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def load_json(file_path):
    """Load JSON data from a specified file path."""
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_csv(file_path):
    """Load CSV data from a specified file path."""
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

# Additional loading functions can be added as needed.