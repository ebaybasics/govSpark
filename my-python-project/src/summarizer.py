def summarize_text(text):
    # This function takes a block of text and returns a summarized version of it.
    # For simplicity, we will return the first sentence as a summary.
    return text.split('.')[0] + '.' if '.' in text else text

def summarize_data(data):
    # This function takes a list of texts and returns their summaries.
    return [summarize_text(text) for text in data]