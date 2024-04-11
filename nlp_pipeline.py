import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk
from nltk.classify import NaiveBayesClassifier
import os
import PyPDF2
import docx
import csv
import json

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

    def process(self, input_data, input_format):
        if input_format == 'text':
            text = input_data
        elif input_format == 'file':
            text = self.read_file(input_data)
        elif input_format == 'csv':
            text = self.read_csv(input_data)
        elif input_format == 'json':
            text = self.read_json(input_data)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        text = self.preprocess_text(text)
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
    
    def read_file(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        elif file_extension == '.docx':
            doc = docx.Document(file_path)
            return ' '.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
    def read_csv(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            text = ' '.join([' '.join(row) for row in csv_reader])
            return text
        
    def read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            text = ' '.join([str(value) for value in data.values()])
            return text
        
    def preprocess_text(self, text):
        # TODO: Process text differently? Just lowercasing here
        return text.lower()

if __name__ == "__main__":
    pipeline_tasks = {
        'Tokenizer': Tokenizer(),
        'POSTagger': POSTagger(),
        'Lemmatizer': Lemmatizer(),
        'SentimentAnalyzer': SentimentAnalyzer(),
        'NamedEntityRecognizer': NamedEntityRecognizer(),
        'TextClassifier': TextClassifier()
    }

    pipeline = NLPPipeline(pipeline_tasks)

    # Process plain text
    text_input = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. The movie was fantastic and had great acting!"
    text_result = pipeline.process(text_input, input_format='text')
    print("Text Input Result:")
    print(text_result)

    # Process file input (txt, pdf, docx)
    file_path = 'placeholder.pdf'  # Replace 
    file_result = pipeline.process(file_path, input_format='file')
    print("\nFile Input Result:")
    print(file_result)

    # Process CSV input
    csv_path = 'placeholder.csv'  # Replace
    csv_result = pipeline.process(csv_path, input_format='csv')
    print("\nCSV Input Result:")
    print(csv_result)

    # Process JSON input
    json_path = 'placeholder.json'  # Replace
    json_result = pipeline.process(json_path, input_format='json')
    print("\nJSON Input Result:")
    print(json_result)