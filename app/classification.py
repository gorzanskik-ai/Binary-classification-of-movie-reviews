import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

df = pd.read_csv(r'data/clean_transform.csv')
df.head()
X = df['review'].copy()
y = df['sentiment'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

model = pickle.load(open('data/tfidf_model.pickle', "rb"))


def classify(review):
    review = pd.Series(review)
    review_tfidf = vectorizer.transform(review)
    review_pred = model.predict(review_tfidf)
    review_prob = model.predict_proba(review_tfidf)

    prob = max(max(review_prob)) * 100
    pred = 'negative' if review_pred == 0 else 'positive'
    return f'{pred} -> {prob:.2f} %'
