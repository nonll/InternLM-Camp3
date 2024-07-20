import string
from collections import Counter
import re

def preprocess_text(text):
    """Remove punctuation and convert to lowercase."""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).lower()

def wordcount_translate(text):
    """Count words using translate method."""
    counter = Counter(preprocess_text(text).split())
    return dict(counter)

def wordcount_re_findall(text):
    """Count words using regular expression."""
    counter = Counter(re.findall(r'\b\w+\b', preprocess_text(text)))
    return dict(counter)

def wordcount_splitlines(text):
    """Count words by splitting lines and then words after preprocessing."""
    processed_text = preprocess_text(text)
    counter = Counter(word for line in processed_text.splitlines() for word in line.split())
    return dict(counter)

def wordcount_strip(text):
    """Count words using strip and split on each line after preprocessing."""
    processed_text = preprocess_text(text)
    counter = Counter(word for line in processed_text.splitlines() for word in line.strip().split())
    return dict(counter)

if __name__ == '__main__':
    text = """
    Hello world!
    This is an example.
    Word count is fun.
    Is it fun to count words?
    Yes, it is fun!
    """
    
    print(wordcount_translate(text))
    print(wordcount_re_findall(text))
    print(wordcount_splitlines(text))
    print(wordcount_strip(text))