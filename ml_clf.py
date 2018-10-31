import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from nlp_tools import Preprocessor

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
df['pp_text'] = df.text.map(lambda x: pp.preprocess_pipeline(str(x), steps))

cvec = CountVectorizer(encoding='utf-8', tokenizer=pp.tokenize, max_features=2000)
cvec.fit(df.pp_text)

text_features = cvec.transform(df.pp_text)

nmf = NMF(n_components=100)
nmf.fit(text_features)
reduced_features = nmf.transform(text_features)
