import hazm
import re


class Preprocessor:

    def __init__(self):
        self.Normalizer = hazm.Normalizer()
        self.stopwords_list = hazm.stopwords_list()
        self.Stemmer = hazm.Stemmer()

    def normalize(self, text):
        text = self.Normalizer.normalize(text)
        return text

    def clear_emojis(self, text):
        from emojis import clear_emojis as clear
        return clear(text)

    def clear_punctuation(self, text):
        text = re.sub('-', ' ', text)
        # text = re.sub('_', ' ', text)
        text = re.sub('\u061F', '', text)  # ?
        text = re.sub('\u060C', '', text)  # comma
        text = re.sub('\u061B', '', text)  # semicolon
        text = re.sub('[?!@#$\'",.;:()|]', '', text)
        return text

    def clear_stopwords(self, words):
        clean_words = list()
        for word in words:
            if word not in self.stopwords_list:
                clean_words.append(word)
        return clean_words

    def tokenize(self, text):
        return hazm.word_tokenize(text)

    def stem(self, words):
        stemmed = list()
        for word in words:
            stemmed.append(self.Stemmer.stem(word))
        return stemmed

    def preprocess_pipeline(self, input, steps=[]):
        for step in steps:
            input = getattr(self, step)(input)
        return input
