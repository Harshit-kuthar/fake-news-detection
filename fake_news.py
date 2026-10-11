"""
Project Name: Fake News Detection System
Author: Harshit 
Description: A Machine Learning model using TF-IDF Vectorizer and PassiveAggressiveClassifier 
to classify news articles as Real or Fake.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    try:
        df = pd.read_csv('news.csv')
        print("Dataset successfully loaded from 'news.csv'.")
        return df
    except FileNotFoundError:
        print("Notice: 'news.csv' not found. Loading built-in training samples for demonstration...")
        data = {
            'text': [
                "Government announces a brand new financial relief package for small business owners and farmers.",
                "Breaking: Scientists confirm that aliens have officially taken over the New York city mayor office.",
                "Stock markets hit an all-time high this week following positive economic growth reports and inflation control.",
                "Drink this miraculous liquid daily to reverse aging completely within two hours guaranteed.",
                "New tech policy approved by parliament to boost semiconductor manufacturing and AI research across colleges.",
                "Miracle plant discovered in deep jungle cures all known human diseases instantly without side effects."
            ],
            'label': ['REAL', 'FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE']
        }
        return pd.DataFrame(data)

def main():
    print("--- Starting Fake News Detection Pipeline ---\n")
    df = load_data()
    print(f"Dataset Shape: {df.shape}")
    print(df.head(2), "\n")
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)
    
    pac = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    pac.fit(tfidf_train, y_train)
    
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    
    print(f"Model Training Completed Successfully!")
    print(f"Test Accuracy: {score * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("--- Testing with Custom Headlines ---")
    test_headlines = [
        "Government passes new legislation to promote renewable green energy projects.",
        "Secret portal found under your bed leading directly to Mars."
    ]
    
    vectorized_test_headlines = tfidf_vectorizer.transform(test_headlines)
    predictions = pac.predict(vectorized_test_headlines)
    
    for headline, pred in zip(test_headlines, predictions):
        print(f"Headline: '{headline}'\nPrediction: --> **{pred}**\n")

if __name__ == "__main__":
    main()