import streamlit as st
import pandas as pd
import ml_model
import response
import os
from dotenv import load_dotenv

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="AI Ticket Assistant", layout="wide", page_icon="🤖")
st.title("🤖 AI-Assisted Customer Service System")
st.divider()

# Load environment variables automatically from the .env file
load_dotenv()

# 2. LOAD MODELS
@st.cache_resource
def load_ai_models():
    return ml_model.load_or_train_model('tickets.csv')

with st.spinner("Loading intelligent core..."):
    vectorizer, model = load_ai_models()

if vectorizer is None:
    st.error("Error: The 'tickets.csv' file was not found!")
    st.stop()

# 3. SIDEBAR (Settings and Tone)
st.sidebar.header("⚙️ Settings")

# Check if the API key is securely loaded from the .env file
env_api_key = os.getenv("GEMINI_API_KEY")

if env_api_key:
    # 1. ESET: Sikerült kiolvasni a .env fájlból
    api_key = env_api_key
    st.sidebar.success("✅ API Key securely loaded from .env")
else:
    # 2. ESET: Nincs .env fájl, ezért kérjük be a felületen
    st.sidebar.warning("⚠️ No .env file found. Manual input required.")
    api_key = st.sidebar.text_input("Please enter your Gemini API Key:", type="password")

st.sidebar.markdown("---")
selected_tone = st.sidebar.selectbox(
    "📝 Response Tone:",
    [
        "Professional and Polite", 
        "Friendly and Empathetic", 
        "Extremely Apologetic", 
        "Short and Direct"
    ]
)

# 4. MAIN LOGIC (Tabs)
tab1, tab2 = st.tabs(["📩 Process New Ticket", "📊 Dashboard & Analytics"])

# --- TAB 1: WORKSPACE ---
with tab1:
    st.subheader("Process Customer Complaint")
    new_complaint = st.text_area("Paste the customer's email content here:", height=150)

    if st.button("Analyze Ticket & Generate Response", type="primary"):
        if not new_complaint:
            st.warning("Please enter a complaint!")
        else:
            # ML Prediction
            category = ml_model.predict_category(new_complaint, vectorizer, model)
            st.success(f"**Predicted Category:** {category.upper()}")
            
            # Gemini Generation
            if not api_key:
                st.info("Please enter your API key in the sidebar to generate a response!")
            else:
                with st.spinner("Drafting AI Response..."):
                    try:
                        response_text = response.generate_email_response(api_key, new_complaint, category, selected_tone)
                        st.markdown(f"### 📝 Suggested Email Response ({selected_tone})")
                        st.info(response_text)
                    except Exception as e:
                        st.error(f"Error communicating with Gemini: {e}")

# --- TAB 2: ANALYTICS ---
with tab2:
    st.subheader("Historical Complaint Distribution")
    try:
        # Load CSV for the chart
        df = pd.read_csv('tickets.csv').dropna(subset=['Ticket Type'])
        category_counts = df['Ticket Type'].value_counts()
        
        # Draw Bar Chart
        st.bar_chart(category_counts)
        st.caption("Distribution of complaints by category in our training database.")
    except:
        st.warning("Could not load statistical data.")