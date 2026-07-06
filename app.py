import streamlit as st
import ml_model
import helper

# 1. OLDAL BEÁLLÍTÁSAI
st.set_page_config(page_title="AI Ticket Asszisztens", layout="wide", page_icon="🤖")
st.title("🤖 AI-Asszisztált Ügyfélszolgálati Rendszer")
st.divider()

# 2. MODELL BETÖLTÉSE A KÜLÖN FÁJLBÓL
@st.cache_resource
def load_ai_models():
    return ml_model.train_ticket_model('tickets.csv')

with st.spinner("Modell betanítása a historikus adatokon..."):
    vectorizer, model = load_ai_models()

if vectorizer is None:
    st.error("Hiba: A 'tickets.csv' fájl nem található!")
    st.stop()

# 3. OLDALSÁV
st.sidebar.header("⚙️ Beállítások")
api_key = st.sidebar.text_input("Gemini API kulcs:", type="password")

# 4. FŐ LOGIKA
st.subheader("📩 Új beérkező jegy feldolgozása")
new_complaint = st.text_area("Másold ide az ügyfél e-mailjének tartalmát:", height=150)

if st.button("Jegy Elemzése és Válasz Generálása", type="primary"):
    if not new_complaint:
        st.warning("Kérlek, írj be egy panaszt!")
    else:
        # A) Kategória predikció meghívása
        category = ml_model.predict_category(new_complaint, vectorizer, model)
        st.success(f"**Predikált Kategória (Scikit-Learn):** {category.upper()}")
        
        # B) Válaszgenerálás meghívása
        if not api_key:
            st.info("A válaszgeneráláshoz add meg az API kulcsot az oldalsávban!")
        else:
            with st.spinner("AI Választervezet fogalmazása..."):
                try:
                    response_text = helper.generate_email_response(api_key, new_complaint, category)
                    st.markdown("### 📝 Javasolt Válaszlevél")
                    st.info(response_text)
                except Exception as e:
                    st.error(f"Hiba a Gemini hívásakor: {e}")