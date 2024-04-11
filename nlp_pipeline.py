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

class NLPTask:
    def process(self, text):
        raise NotImplementedError("Subclasses must implement the process method.")

class Tokenizer(NLPTask):
    def process(self, text):
        return word_tokenize(text)

class POSTagger(NLPTask):
    def process(self, tokens):
        return pos_tag(tokens)

class Lemmatizer(NLPTask):
    def process(self, tokens):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]

class SentimentAnalyzer(NLPTask):
    def process(self, text):
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)
        return sentiment_scores

class NamedEntityRecognizer(NLPTask):
    def process(self, tokens):
        tagged_tokens = pos_tag(tokens)
        chunks = ne_chunk(tagged_tokens)
        named_entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                named_entities.append((' '.join(c[0] for c in chunk), chunk.label()))
        return named_entities

class TextClassifier(NLPTask):
    def process(self, text):
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

class NLPPipeline:
    def __init__(self, tasks):
        self.tasks = tasks

    def process(self, text):
        result = {}
        for task_name, task in self.tasks.items():
            if task_name == 'Tokenizer':
                tokens = task.process(text)
                result['tokens'] = tokens
            elif task_name == 'POSTagger':
                pos_tags = task.process(result['tokens'])
                result['pos_tags'] = pos_tags
            elif task_name == 'Lemmatizer':
                lemmas = task.process(result['tokens'])
                result['lemmas'] = lemmas
            elif task_name == 'SentimentAnalyzer':
                sentiment = task.process(text)
                result['sentiment'] = sentiment
            elif task_name == 'NamedEntityRecognizer':
                named_entities = task.process(result['tokens'])
                result['named_entities'] = named_entities
            elif task_name == 'TextClassifier':
                classification = task.process(text)
                result['classification'] = classification
        return result

if __name__ == "__main__":
    text = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. The movie was fantastic and had great acting!"

    pipeline_tasks = {
        'Tokenizer': Tokenizer(),
        'POSTagger': POSTagger(),
        'Lemmatizer': Lemmatizer(),
        'SentimentAnalyzer': SentimentAnalyzer(),
        'NamedEntityRecognizer': NamedEntityRecognizer(),
        'TextClassifier': TextClassifier()
    }

    pipeline = NLPPipeline(pipeline_tasks)
    result = pipeline.process(text)
    print(result)