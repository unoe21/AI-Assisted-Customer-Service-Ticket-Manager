import streamlit as st
import pandas as pd
import ml_model
import helper

# 1. OLDAL BEÁLLÍTÁSAI
st.set_page_config(page_title="AI Ticket Asszisztens", layout="wide", page_icon="🤖")
st.title("🤖 AI-Asszisztált Ügyfélszolgálati Rendszer")
st.divider()

# 2. MODELL BETÖLTÉSE (Most már a gyors joblib fájlokból)
@st.cache_resource
def load_ai_models():
    return ml_model.load_or_train_model('tickets.csv')

with st.spinner("Intelligens mag betöltése..."):
    vectorizer, model = load_ai_models()

if vectorizer is None:
    st.error("Hiba: A 'tickets.csv' fájl nem található!")
    st.stop()

# 3. OLDALSÁV (Beállítások és Hangnem)
st.sidebar.header("⚙️ Beállítások")
api_key = st.sidebar.text_input("Gemini API kulcs:", type="password")

st.sidebar.markdown("---")
selected_tone = st.sidebar.selectbox(
    "📝 Válaszadás stílusa (Hangnem):",
    [
        "Professional and Polite (Hivatalos)", 
        "Friendly and Empathetic (Barátságos)", 
        "Extremely Apologetic (Sűrű elnézést kérő)", 
        "Short and Direct (Lényegretörő)"
    ]
)
# Kicsípjük a zárójeles magyar magyarázatot, csak az angol utasítást küldjük a Gemininek
tone_english = selected_tone.split(" (")[0]

# 4. FŐ LOGIKA (Fülek létrehozása)
tab1, tab2 = st.tabs(["📩 Új Jegy Kezelése", "📊 Vezetői Statisztikák"])

# --- ELSŐ FÜL: A munkaállomás ---
with tab1:
    st.subheader("Ügyfél panasz feldolgozása")
    new_complaint = st.text_area("Másold ide az ügyfél e-mailjének tartalmát:", height=150)

    if st.button("Jegy Elemzése és Válasz Generálása", type="primary"):
        if not new_complaint:
            st.warning("Kérlek, írj be egy panaszt!")
        else:
            # ML Predikció
            category = ml_model.predict_category(new_complaint, vectorizer, model)
            st.success(f"**Predikált Kategória:** {category.upper()}")
            
            # Gemini Hívás
            if not api_key:
                st.info("A válaszgeneráláshoz add meg az API kulcsot az oldalsávban!")
            else:
                with st.spinner("AI Választervezet fogalmazása..."):
                    try:
                        response_text = helper.generate_email_response(api_key, new_complaint, category, tone_english)
                        st.markdown(f"### 📝 Javasolt Válaszlevél ({tone_english})")
                        st.info(response_text)
                    except Exception as e:
                        st.error(f"Hiba a Gemini hívásakor: {e}")

# --- MÁSODIK FÜL: Analitika ---
with tab2:
    st.subheader("Historikus panaszok eloszlása")
    try:
        # Beolvassuk a CSV-t a grafikonhoz
        df = pd.read_csv('tickets.csv').dropna(subset=['Ticket Type'])
        category_counts = df['Ticket Type'].value_counts()
        
        # Oszlopdiagram rajzolása
        st.bar_chart(category_counts)
        st.caption("A betanító adatbázisunkban található panaszok megoszlása kategóriánként.")
    except:
        st.warning("Nem sikerült betölteni a statisztikai adatokat.")