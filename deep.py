from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from preprocess import *

df = load_data('/Users/iman/tlg_data.xlsx')
df_train, df_test = holdout_split(df)
cvec_text, cvec_emoji = fit_vectorizers(df_train)

features_train = extract_features(df_train, cvec_emoji, cvec_text).toarray()
features_test = extract_features(df_test, cvec_text, cvec_emoji).toarray()

y_train = pd.get_dummies(df_train.label).values
y_test = pd.get_dummies(df_test.label).values

model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(features_train.shape[-1],), kernel_regularizer=l2(0.05)))
model.add(Dense(20, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(features_train, y_train, epochs=10, validation_data=(features_test, y_test), verbose=2)