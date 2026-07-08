import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import re

VECTORIZER_FILE = 'ticket_vectorizer.joblib'
MODEL_FILE = 'ticket_model.joblib'

def clean_text(text):
    """Removes special characters, numbers, and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def load_or_train_model(csv_path='tickets.csv'):
    """Loads the saved model or trains a new one using a full Data Science pipeline."""
    
    if os.path.exists(VECTORIZER_FILE) and os.path.exists(MODEL_FILE):
        print("✅ Pre-trained ML models loaded successfully.")
        vectorizer = joblib.load(VECTORIZER_FILE)
        model = joblib.load(MODEL_FILE)
        return vectorizer, model
        
    print("⚙️ No saved models found. Starting training pipeline...")
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Ticket Description', 'Ticket Type'])
        
        df['Cleaned_Description'] = df['Ticket Description'].apply(clean_text)
        
        X_raw = df['Cleaned_Description']
        y = df['Ticket Type']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.2, random_state=42, stratify=y
        )
        
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english', 
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            dual="auto", 
            random_state=42,
            max_iter=2000
        )
        model.fit(X_train_vec, y_train)
        
        y_pred = model.predict(X_test_vec)
        print("\n📊 --- ML Model Evaluation Report ---")
        print(classification_report(y_test, y_pred))
        print("--------------------------------------\n")
        
        X_full_vec = vectorizer.fit_transform(X_raw)
        model.fit(X_full_vec, y)
        
        joblib.dump(vectorizer, VECTORIZER_FILE)
        joblib.dump(model, MODEL_FILE)
        
        print("New models trained and saved to disk.")
        return vectorizer, model
        
    except FileNotFoundError:
        print(f"Error: Dataset '{csv_path}' not found.")
        return None, None

def predict_category(text, vectorizer, model):
    """Cleans the input text and predicts its category."""
    cleaned_text = clean_text(text)
    vec_input = vectorizer.transform([cleaned_text])
    return model.predict(vec_input)[0]