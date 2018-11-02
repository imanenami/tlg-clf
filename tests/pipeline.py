from preprocess import *
from nlp_tools import Preprocessor

df = load_data('/Users/iman/tlg_data.xlsx')
pp = Preprocessor()
NLP_PREPROCESSING_STEPS = ['clear_links', 'normalize', 'clear_emojis', 'clear_punctuation']

df['pp_text'] = df.text.map(lambda x: pp.preprocess_pipeline(str(x), NLP_PREPROCESSING_STEPS))

df.pp_text.map(lambda x: print(pp.filtered_tokenize(str(x))))

txt_tokens = cvec_text.get_feature_names()
emj_tokens = cvec_emoji.get_feature_names()
for label in range(6):
    print('label == {}'.format(label))
    edf = pd.DataFrame(X[y[:,label] == 1].sum(axis=0).transpose())
    for i in edf.nlargest(n=25, columns=0).index:
        if i < 2000:
            print(txt_tokens[i])
        else:
            print(emj_tokens[i-2000])


