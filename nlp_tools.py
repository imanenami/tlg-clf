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
        text = re.sub('[?!@#$\'",.;:()|/]', '', text)
        return text

    def clear_links(self, text):
        url_pattern = re.compile('(\w+:\/{2}[\d\w-]+(?:\.[\d\w-]+)*(?:(?:\/[^\s/]*))*)', flags=re.UNICODE)
        tme_pattern = re.compile('t\.me\/[^\s]+')
        text = url_pattern.sub(r'', text)
        text = tme_pattern.sub(r'', text)
        return text

    def clear_stopwords(self, words):
        clean_words = list()
        for word in words:
            if word not in self.stopwords_list:
                clean_words.append(word)
        return clean_words

    def tokenize(self, text):
        return hazm.word_tokenize(text)

    def filtered_tokenize(self, text, min_length=3):
        func = lambda x: len(x) >= min_length
        return list(filter(func, self.tokenize(text)))

    def stem(self, words):
        stemmed = list()
        for word in words:
            stemmed.append(self.Stemmer.stem(word))
        return stemmed

    def preprocess_pipeline(self, input, steps=[]):
        VALID_STEPS = ['normalize', 'clear_emojis', 'clear_punctuation', 'clear_links',
                       'clear_stopwords', 'tokenize', 'filtered_tokenize', 'stem']
        for step in steps:
            assert step in VALID_STEPS
            input = getattr(self, step)(input)
        return input
