import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
from nlp_tools import Preprocessor

pp = Preprocessor()

df = pd.read_excel('/Users/iman/tlg_data.xlsx', encoding='utf-8')
df.dropna(inplace=True)

steps = ['normalize', 'clear_emojis', 'clear_punctuation']
df['pp_text'] = df.text.map(lambda x: pp.preprocess_pipeline(str(x), steps))

cvec = CountVectorizer(encoding='utf-8', tokenizer=pp.tokenize, max_features=2000)
cvec.fit(df.pp_text)

text_features = cvec.transform(df.pp_text)

nmf = NMF(n_components=100)
nmf.fit(text_features)
reduced_features = nmf.transform(text_features)
