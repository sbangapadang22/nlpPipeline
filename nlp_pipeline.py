import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def tokenize_text(text):
    return word_tokenize(text)

# token function example - splits into list
if __name__ == "__main__":
    text = "Sample text for tokenization."
    tokens = tokenize_text(text)
    print(tokens)
