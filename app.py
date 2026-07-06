import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import google.generativeai as genai

# ==========================================
# 1. OLDAL BEÁLLÍTÁSAI
# ==========================================
st.set_page_config(page_title="AI Ticket Asszisztens", layout="wide", page_icon="🤖")
st.title("🤖 AI-Asszisztált Ügyfélszolgálati Rendszer")
st.markdown("Ez az alkalmazás automatikusan kategorizálja a beérkező ügyfélpanaszokat (Scikit-Learn), majd a Google Gemini segítségével személyre szabott választervezetet generál.")
st.divider()

# ==========================================
# 2. ADAT ÉS MODELL BETÖLTÉSE (CACHING)
# ==========================================
# A @st.cache_resource biztosítja, hogy a betanítás csak egyszer, az app elindulásakor fusson le
@st.cache_resource
def load_and_train_model():
    try:
        # CSV betöltése (ellenőrizd, hogy a tickets.csv a mappában van!)
        df = pd.read_csv('tickets.csv')
        
        # Üres sorok eldobása a kritikus oszlopokból
        df = df.dropna(subset=['Ticket Description', 'Ticket Type'])
        
        # Szöveg számmá alakítása (max 5000 szó a gyorsaság érdekében)
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['Ticket Description'])
        y = df['Ticket Type']
        
        # Scikit-Learn modell (LinearSVC) betanítása
        model = LinearSVC(dual="auto")
        model.fit(X, y)
        
        return vectorizer, model
    except FileNotFoundError:
        return None, None

# Töltőképernyő amíg a modell tanul
with st.spinner("Modell betanítása a historikus adatokon..."):
    vectorizer, model = load_and_train_model()

if vectorizer is None:
    st.error("Hiba: A 'tickets.csv' fájl nem található! Kérlek, tedd az app.py mellé a VS Code-ban.")
    st.stop() # Megállítjuk a program futását, amíg nincs adat

# ==========================================
# 3. OLDALSÁV (SIDEBAR) - API BEÁLLÍTÁSOK
# ==========================================
st.sidebar.header("⚙️ Beállítások")
st.sidebar.markdown("Kérlek, add meg a [Google AI Studio](https://aistudio.google.com/)-ban generált ingyenes API kulcsodat.")

api_key = st.sidebar.text_input("Gemini API kulcs:", type="password")

if api_key:
    genai.configure(api_key=api_key)
    # A leggyorsabb és költséghatékonyabb modellt használjuk
    llm_model = genai.GenerativeModel('gemini-1.5-flash') 
else:
    st.sidebar.warning("A kategória predikció működik, de a válaszgeneráláshoz meg kell adnod a kulcsot!")

# ==========================================
# 4. FELHASZNÁLÓI FELÜLET ÉS LOGIKA
# ==========================================
st.subheader("📩 Új beérkező jegy feldolgozása")

# Szövegdoboz a panasznak
new_complaint = st.text_area("Másold ide az ügyfél e-mailjének tartalmát (angolul):", height=150, placeholder="Például: My internet connection keeps dropping every 10 minutes...")

# Futtatás gomb
if st.button("Jegy Elemzése és Válasz Generálása", type="primary"):
    if not new_complaint:
        st.warning("Kérlek, írj be vagy másolj be egy panaszt az elemzéshez!")
    else:
        # A) Kategória Predikció (Klasszikus ML)
        # Fontos: Csak transform()-ot használunk az új adatnál!
        vec_input = vectorizer.transform([new_complaint])
        predicted_category = model.predict(vec_input)[0]
        
        # Vizuális visszajelzés
        st.success(f"**Predikált Jegy Típus (Scikit-Learn):** {predicted_category.upper()}")
        
        # B) Válasz Generálása (GenAI)
        if not api_key:
            st.info("A választ csak akkor tudom legenerálni, ha az oldalsávon megadod a Gemini API kulcsot.")
        else:
            with st.spinner("AI Választervezet fogalmazása..."):
                prompt = f"""
                You are a professional and empathetic customer support agent. 
                A customer sent the following message: "{new_complaint}"
                
                Our internal Machine Learning system categorized this ticket as: {predicted_category}.
                
                Please write a polite, professional, and concise response email to the customer. 
                Acknowledge their specific problem based on the category, and assure them we are looking into it.
                Do not invent specific names or order numbers. Use [Placeholder] if needed.
                """
                
                try:
                    response = llm_model.generate_content(prompt)
                    
                    st.markdown("### 📝 Javasolt Válaszlevél (Gemini)")
                    st.info(response.text)
                    
                except Exception as e:
                    st.error(f"Hiba történt a Gemini hívása során: {e}")