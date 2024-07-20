import string
from collections import Counter
import re

def wordcount_translate(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()
    words = text.split()
    return Counter(words)

def wordcount_re_findall(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return Counter(words)

def wordcount_splitlines(text):
    words = []
    for line in text.lower().splitlines():
        words.extend(line.split())
    return Counter(words)

def wordcount_strip(text):
    words = [line.strip().lower().split() for line in text.splitlines()]
    return Counter(sum(words, []))

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
