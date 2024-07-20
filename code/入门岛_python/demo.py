import string
from collections import Counter
import re

def wordcount(text):
    """
    Count the occurrences of each word in an English string.
    
    Args:
    - text: A string containing English words.
    
    Returns:
    - A dictionary where keys are words and values are the number of occurrences.
    """
    # Remove punctuation and convert to lowercase
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = text.translate(translator).lower()
    
    # Find all words using regex to handle boundaries properly
    words = re.findall(r'\b\w+\b', cleaned_text)
    
    # Count the occurrences of each word
    word_counts = Counter(words)
    
    # Convert Counter object to a dictionary
    return dict(word_counts)

# Example usage
if __name__ == '__main__':
    text = """
    Got this panda plush toy for my daughter's birthday,
    who loves it and takes it everywhere. It's soft and
    super cute, and its face has a friendly look. It's
    a bit small for what I paid though. I think there
    might be other options that are bigger for the
    same price. It arrived a day earlier than expected,
    so I got to play with it myself before I gave it
    to her.
    """
    print(wordcount(text))