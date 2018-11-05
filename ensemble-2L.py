import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import NMF
from sklearn.preprocessing import MaxAbsScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
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

# Battle between news / non-news begins!
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

news_pipeline = Pipeline(
    [
        ('union', FeatureUnion(transformer_list=[('text', text_pipeline), ('emoji', emoji_pipeline)])),
        ('scaler', MaxAbsScaler()),
        ('kbest', SelectKBest(chi2, k=1000)),
        ('clf', RandomForestClassifier())
    ]
)

param_grid = {
    'clf__n_estimators': [256, 512, 1024],
    'clf__min_samples_split': [3, 4, 5, 6],
    'kbest__k': [500, 1000, 2000]
}

X = df_train.text
y = pd.get_dummies(df_train.label).values

# Hyper-parameter tuning
grid_cv = GridSearchCV(news_pipeline, param_grid, cv=5, verbose=1)
# grid_cv.fit(X, y)
# print(grid_cv.best_params_)

# Set tuned parameters
news_pipeline.set_params(clf__min_samples_split=6, clf__n_estimators=256, kbest__k=2000)

# scores = cross_val_score(news_pipeline, X, y, cv=5)
# print(scores, scores.mean())

# news_pipeline.fit(X, y)
# df_test.label[df_test.label != 1] = 2
#
# X_test = df_test.text
# y_test = pd.get_dummies(df_test.label).values
#
# test_score = news_pipeline.score(X_test, y_test)
# print(test_score)

df_train = df_train_copy.copy()
df_train.drop(df_train[df_train.label == 1].index, axis=0, inplace=True)

others_pipeline = news_pipeline

others_pipeline.set_params(clf__min_samples_split=5, clf__n_estimators=256, kbest__k=2000)
X = df_train.text
y = pd.get_dummies(df_train.label).values

brk = int(0.8 * len(X))
others_pipeline.fit(X[:brk], y[:brk])
print (confusion_matrix(pd.DataFrame(y[brk:]).idxmax(axis=1), pd.DataFrame(others_pipeline.predict(X[brk:])).idxmax(axis=1)))

# scores = cross_val_score(others_pipeline, X, y, cv=5)
# print(scores, scores.mean())
