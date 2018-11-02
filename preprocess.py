import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.sparse import hstack
from nlp_tools import Preprocessor
from emojis import find_emojis

# Initialize the NLP preprocessor
pp = Preprocessor()

# Labels omitted from the data
LABELS_WITH_NOT_ENOUGH_DATA = [5, 8, 9]
NLP_PREPROCESSING_STEPS = ['normalize', 'clear_emojis', 'clear_links', 'clear_punctuation']


# Load Data
def load_data(path):
    df = pd.read_excel(path, encoding='utf-8')
    df.dropna(inplace=True)

    # Drop labels for which we don't have enough data
    for label in LABELS_WITH_NOT_ENOUGH_DATA:
        df.drop(df[df.label == label].index, axis=0, inplace=True)

    df.reset_index(inplace=True)

    return df


def holdout_split(df, test_size=0.2, random_state=666):
    # Take away the hold-out set
    train_indices, test_indices, _, _ = \
        train_test_split(df.index.values, df.label.values, test_size=test_size, random_state=random_state,
                         stratify=df.label.values)

    df_train = df.iloc[train_indices, :]
    df_test = df.iloc[test_indices, :]

    return df_train, df_test


def fit_vectorizers(df):
    assert 'text' in df.columns.tolist()
    # NLP Preprocessing
    df['pp_text'] = df.text.map(lambda x: pp.preprocess_pipeline(str(x), NLP_PREPROCESSING_STEPS))
    cvec_text = CountVectorizer(encoding='utf-8', tokenizer=pp.tokenize, stop_words=pp.stopwords_list,
                                max_features=4000, binary=True)
    cvec_text.fit(df.pp_text)

    cvec_emoji = CountVectorizer(encoding='utf-8', tokenizer=find_emojis, max_features=100)
    cvec_emoji.fit(df.text)

    return cvec_text, cvec_emoji


def extract_features(df, cvec_text, cvec_emoji):
    assert 'text' in df.columns.tolist()

    df['pp_text'] = df.text.map(lambda x: pp.preprocess_pipeline(str(x), NLP_PREPROCESSING_STEPS))

    text_features = cvec_text.transform(df.pp_text)
    emoji_features = cvec_emoji.transform(df.text)
    features = hstack([text_features, emoji_features])

    return features
