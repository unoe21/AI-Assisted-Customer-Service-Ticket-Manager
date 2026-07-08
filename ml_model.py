import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
import os

# Filenames for model persistence
VECTORIZER_FILE = 'ticket_vectorizer.joblib'
MODEL_FILE = 'ticket_model.joblib'

def load_or_train_model(csv_path='tickets.csv'):
    """Loads the saved model or trains a new one if it doesn't exist."""
    
    # 1. IF ALREADY TRAINED: Load models
    if os.path.exists(VECTORIZER_FILE) and os.path.exists(MODEL_FILE):
        vectorizer = joblib.load(VECTORIZER_FILE)
        model = joblib.load(MODEL_FILE)
        return vectorizer, model
        
    # 2. IF NOT TRAINED: Load data and train the model
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Ticket Description', 'Ticket Type'])
        
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english', 
            ngram_range=(1, 2)
        )
        
        X = vectorizer.fit_transform(df['Ticket Description'])
        y = df['Ticket Type']
        
        # Balanced model to handle minority classes
        model = LinearSVC(dual="auto", class_weight='balanced')
        model.fit(X, y)
        
        # Save
        joblib.dump(vectorizer, VECTORIZER_FILE)
        joblib.dump(model, MODEL_FILE)
        
        return vectorizer, model
    except FileNotFoundError:
        return None, None

def predict_category(text, vectorizer, model):
    """Predicts the category of a new text input."""
    vec_input = vectorizer.transform([text])
    return model.predict(vec_input)[0]