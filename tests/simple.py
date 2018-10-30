import pandas as pd
from nlp_tools import Preprocessor

pp = Preprocessor()
df = pd.read_excel('/Users/iman/tlg_data.xlsx', encoding='utf-8')

steps = ['normalize', 'clear_emojis', 'clear_punctuation', 'tokenize', 'clear_stopwords']
pp_pipeline = pp.preprocess_pipeline(df.text.iloc[3], steps)
print (pp_pipeline)