import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from scipy.sparse import hstack
from nlp_tools import Preprocessor
from emojis import find_emojis

# Initialize the NLP preprocessor
pp = Preprocessor()

# Load Data
df = pd.read_excel('C:\\Users\\mjzifan\\Documents\\tlg_data.xlsx', encoding='utf-8')
df.dropna(inplace=True)

# Drop labels for which we don't have enough data
LABELS_WITH_NOT_ENOUGH_DATA = [5, 8, 9]
for label in LABELS_WITH_NOT_ENOUGH_DATA:
    df.drop(df[df.label == label].index, axis=0, inplace=True)

df.reset_index(inplace=True)

# Take away the hold-out set
train_indices, test_indices, _, _ =\
    train_test_split(df.index.values, df.label.values, test_size=0.2, random_state=666, stratify=df.label.values)

df_train = df.iloc[train_indices, :]
df_test = df.iloc[test_indices, :]

# Our NLP pre-processing pipeline
steps = ['normalize', 'clear_emojis', 'clear_punctuation']
df_train['pp_text'] = df_train.text.map(lambda x: pp.preprocess_pipeline(str(x), steps))

cvec_text = CountVectorizer(encoding='utf-8', tokenizer=pp.tokenize, max_features=2000)
cvec_text.fit(df_train.pp_text)

cvec_emoji = CountVectorizer(encoding='utf-8', tokenizer=find_emojis, max_features=50)
cvec_emoji.fit(df_train.text)

text_features = cvec_text.transform(df_train.pp_text)
emoji_features = cvec_emoji.transform(df_train.text)
features = hstack([text_features, emoji_features])

nmf = NMF(n_components=100)
nmf.fit(text_features)
reduced_features = nmf.transform(text_features)
