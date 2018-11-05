import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
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
train_indices, test_indices, _, _ = \
    train_test_split(df.index.values, df.label.values, test_size=0.2, random_state=666, stratify=df.label.values)

df_train = df.iloc[train_indices, :]
df_test = df.iloc[test_indices, :]

# Battle between news / non-news
df_train_copy = df_train.copy()
df_train.label[df_train.label != 1] = 2

# Our NLP pre-processing pipeline
NLP_PREPROCESSING_STEPS = ['clear_links', 'clear_emojis', 'clear_punctuation']
preprocess_func = lambda x: pp.preprocess_pipeline(str(x), NLP_PREPROCESSING_STEPS)

text_pipeline = Pipeline(
    [
        ('text_preprocess', FunctionTransformer(lambda x: list(map(preprocess_func, x)), validate=False)),
        ('text_features', TfidfVectorizer(encoding='utf-8', tokenizer=pp.filtered_tokenize,
                                          stop_words=pp.stopwords_list, max_features=4000))
    ])

emoji_pipeline = Pipeline(
    [
        ('emoji_features', CountVectorizer(encoding='utf-8', tokenizer=find_emojis, max_features=200))
    ]
)

main_pipeline = Pipeline(
    [
        ('union', FeatureUnion(transformer_list=[('text', text_pipeline), ('emoji', emoji_pipeline)])),
        ('scaler', MaxAbsScaler()),
        ('k_best', SelectKBest(chi2, k=1000)),
        ('Classifier', RandomForestClassifier(n_estimators=256, min_samples_split=5))
    ]
)

X = df_train.text
y = pd.get_dummies(df_train.label).values

scores = cross_val_score(main_pipeline, X, y, cv=5)
print(scores, scores.mean())
