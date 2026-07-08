import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
import os

# Fájlnevek, ahová a betanított modellt mentjük
VECTORIZER_FILE = 'ticket_vectorizer.joblib'
MODEL_FILE = 'ticket_model.joblib'

def load_or_train_model(csv_path='tickets.csv'):
    """Betölti a lementett modellt, vagy ha nincs, betanítja és lementi."""
    
    # 1. HA MÁR BE VAN TANÍTVA: Csak betöltjük (Villámgyors)
    if os.path.exists(VECTORIZER_FILE) and os.path.exists(MODEL_FILE):
        vectorizer = joblib.load(VECTORIZER_FILE)
        model = joblib.load(MODEL_FILE)
        return vectorizer, model
        
    # 2. HA MÉG NINCS BETANÍTVA: Beolvasás és tanulás
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
        
        # Okosabb, kiegyensúlyozott modell
        model = LinearSVC(dual="auto", class_weight='balanced')
        
        model.fit(X, y)
        
        # A betanított "agy" lementése a mappába
        joblib.dump(vectorizer, VECTORIZER_FILE)
        joblib.dump(model, MODEL_FILE)
        
        return vectorizer, model
    except FileNotFoundError:
        return None, None

def predict_category(text, vectorizer, model):
    """Megjósolja egy új szöveg kategóriáját."""
    vec_input = vectorizer.transform([text])
    return model.predict(vec_input)[0]