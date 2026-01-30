import pandas as pd
import joblib  # <--- Added this at the top!
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
print("Loading preprocessed data...")
df = pd.read_csv('preprocessed_imdb_data.csv')
df.dropna(inplace=True)

# 2. Vectorization
print("Vectorizing text (Bigrams enabled)...")
tfidf = TfidfVectorizer(
    ngram_range=(1, 2), 
    max_features=50000, 
    sublinear_tf=True
)
X = tfidf.fit_transform(df['review'])
y = df['sentiment']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train
print("Training Logistic Regression (C=10)...")
model = LogisticRegression(C=10, solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# 5. Check Accuracy
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# 6. SAVE EVERYTHING (The important part!)
print("\nSaving model and vectorizer...")
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("--- SUCCESS! ---")
print("Files 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl' are now in your folder.")