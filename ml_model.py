import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def train_ticket_model(csv_path='tickets.csv'):
    """Betölti az adatokat és betanítja az ML modellt finomhangolt paraméterekkel."""
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Ticket Description', 'Ticket Type'])
        
        # 1. Finomhangolt TF-IDF (Zajszűrés)
        # Alapértelmezett, biztonságos TF-IDF (Nincs agresszív kidobálás)
        vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english', 
            ngram_range=(1, 2)
        )
        
        X = vectorizer.fit_transform(df['Ticket Description'])
        y = df['Ticket Type']
        
        # Alapértelmezett modell, súlyozás nélkül
        model = LinearSVC(dual="auto")
        
        model.fit(X, y)
        
        return vectorizer, model
    except FileNotFoundError:
        return None, None

def predict_category(text, vectorizer, model):
    """Megjósolja egy új szöveg kategóriáját."""
    vec_input = vectorizer.transform([text])
    return model.predict(vec_input)[0]