import ml_model
import helper

# ==========================================
# BEÁLLÍTÁSOK
# ==========================================
# Ide másold be a működő AQ-s kulcsodat!
API_KEY = "AQ.Ab8RN6Kx_tfFZ63BH2GvelB78IoGbt5BrBJYAecCpYzFh6FsjA"

COMPLEX_TEST_CASES = [
    {
        "id": 1,
        "title": "A Dühös Kombó (Fiókhiba + Szállítás + Visszatérítés)",
        "text": "I have been trying to access my profile for two days, but the system keeps throwing a 404 error. Because I couldn't log in, I couldn't cancel my subscription in time. Now my credit card has been charged $99 for a service I don't even want, and the router you shipped to me arrived completely broken. I demand an immediate refund for the charge, a replacement for the hardware, and someone to fix my account login right now!"
    },
    {
        "id": 2,
        "title": "A Félrevezető Dicséret (Sok pozitív zaj + 1 kritikus szoftverhiba)",
        "text": "I really love your product! The design is absolutely great and the shipping was super fast, the package arrived two days earlier than expected. The delivery guy was also very polite. However, I have a major issue: whenever I try to export my monthly financial report to a PDF file, the entire application crashes to the desktop and deletes all my saved data from the last hour. Please look into this software bug immediately."
    }
]

# ==========================================
# TESZTELÉSI LOGIKA
# ==========================================
print("🚀 KOMPLEX STRESSZTESZT INDÍTÁSA...\n")

print("1. LÉPÉS: Scikit-Learn modell betanítása...")
vectorizer, model = ml_model.train_ticket_model('tickets.csv')

if vectorizer is None or model is None:
    print("❌ HIBA: A tickets.csv nem található vagy sérült!")
    exit()
print("✅ Modell sikeresen betanítva a háttérben.\n")

print("2. LÉPÉS: Extrém esetek elemzése és AI válaszgenerálás...")
print("=" * 70)

for case in COMPLEX_TEST_CASES:
    print(f"\n[TESZT #{case['id']}] - {case['title']}")
    print(f"Bemeneti panasz:\n\"{case['text']}\"\n")
    
    # 1. Scikit-Learn Predikció
    predicted_cat = ml_model.predict_category(case['text'], vectorizer, model)
    print(f"🤖 Scikit-Learn által választott fő kategória: >> {predicted_cat.upper()} <<")
    
    # 2. Gemini Válaszgenerálás
    print("✨ Gemini 2.5 válasz fogalmazása (figyeld, hogyan kezeli a több szálat!)...")
    try:
        response_email = helper.generate_email_response(API_KEY, case['text'], predicted_cat)
        print("-" * 50)
        print(response_email.strip())
        print("-" * 50)
        print(f"✅ Komplex Teszt #{case['id']} sikeresen teljesítve!")
    except Exception as e:
        print(f"❌ HIBA a Gemini hívásánál: {e}")
        
    print("=" * 70)

print("\n🎉 Stresszteszt véget ért!")