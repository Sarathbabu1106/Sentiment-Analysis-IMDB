# 🎬 IMDB Sentiment Analysis (90.6% Accuracy)

A high-performance sentiment classifier built with Python that predicts whether a movie review is **Positive** or **Negative**. 

## 📊 Project Highlights
- **Accuracy:** 90.60% 🚀
- **Algorithm:** Logistic Regression (Optimized with C=10)
- **Vectorization:** TF-IDF with Bigrams (captures context like "not good")
- **Efficiency:** Model is saved as a `.pkl` file for instant inference without retraining.
- **Dataset:** 50,000 IMDB Reviews



## 🛠️ Tech Stack
- **Python 3.11**
- **Scikit-learn** (Model training & TF-IDF)
- **NLTK** (Stemming & Stopword removal)
- **Pandas** (Data handling)
- **Joblib** (Model persistence)

## 📁 File Structure
- `TrainLogisRegre.py`: The training engine. Cleans the data, trains the model, and saves the "brain."
- `predict.py`: A lightweight script for real-time user testing.
- `requirements.txt`: List of necessary libraries.

## 🚀 How to Use
1. **Installation:**
   ```bash
   pip install pandas scikit-learn joblib nltk