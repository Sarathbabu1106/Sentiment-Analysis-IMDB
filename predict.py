import joblib
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# 1. Setup (Fast)
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# 2. Load the "Saved Brain" (Instant)
print("Loading model... please wait a second.")
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def get_sentiment(review):
    # Preprocess the input exactly like the training data
    text = re.sub(r'[^a-zA-Z]', ' ', review).lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = " ".join(text)
    
    # Transform and Predict
    vector = tfidf.transform([text])
    prediction = model.predict(vector)
    return "POSITIVE 😊" if prediction[0] == 1 else "NEGATIVE 😡"

# 3. Interactive Loop
print("\n--- Sentiment Analysis Ready ---")
while True:
    user_input = input("\nEnter a movie review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    
    result = get_sentiment(user_input)
    print(f"Analysis: {result}")