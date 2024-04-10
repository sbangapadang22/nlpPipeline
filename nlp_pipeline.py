import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk
from nltk.classify import NaiveBayesClassifier

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('movie_reviews')

def tokenize_text(text):
    return word_tokenize(text)

def pos_tag_text(tokens):
    return pos_tag(tokens)

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

def named_entity_recognition(tokens):
    tagged_tokens = pos_tag(tokens)
    chunks = ne_chunk(tagged_tokens)
    named_entities = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            named_entities.append((' '.join(c[0] for c in chunk), chunk.label()))
    return named_entities

# TODO: replace with own training data + own classification algorithm, using below for learning
def text_classification(text):
    positive_reviews = nltk.corpus.movie_reviews.fileids('pos')
    negative_reviews = nltk.corpus.movie_reviews.fileids('neg')

    feature_set = []
    for review in positive_reviews:
        words = nltk.word_tokenize(nltk.corpus.movie_reviews.raw(review))
        feature_set.append((dict([(word.lower(), True) for word in words]), 'positive'))

    for review in negative_reviews:
        words = nltk.word_tokenize(nltk.corpus.movie_reviews.raw(review))
        feature_set.append((dict([(word.lower(), True) for word in words]), 'negative'))

    classifier = NaiveBayesClassifier.train(feature_set)
    words = nltk.word_tokenize(text.lower())
    features = dict([(word, True) for word in words])
    classification = classifier.classify(features)
    return classification

def process_text(text):
    tokens = tokenize_text(text)
    pos_tags = pos_tag_text(tokens)
    lemmas = lemmatize_text(tokens)
    sentiment = analyze_sentiment(text)
    named_entities = named_entity_recognition(tokens)
    classification = text_classification(text)

    return {
        'tokens': tokens,
        'pos_tags': pos_tags,
        'lemmas': lemmas,
        'sentiment': sentiment,
        'named_entities': named_entities,
        'classification': classification
    }

if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog. It was a fantastic experience!"
    result = process_text(text)
    print(result)
