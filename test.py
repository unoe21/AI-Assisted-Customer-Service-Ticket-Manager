import pandas as pd
import ml_model
import helper

# Ide másold be a működő AQ-s kulcsodat!
API_KEY = "AQ...IDE_MÁSOLD_A_KULCSOT..."

print("🚀 AUTOMATIZÁLT TESZTELÉS VALÓS CSV ADATOKKAL...\n")
vectorizer, model = ml_model.train_ticket_model('tickets.csv')

if vectorizer is None:
    print("❌ HIBA: A tickets.csv nem található!")
    exit()

print("✅ Modell kész. Adatok kinyerése a CSV-ből...\n")
print("=" * 60)

# Kiveszünk 3 véletlenszerű sort a saját adataidból
df = pd.read_csv('tickets.csv').dropna(subset=['Ticket Description', 'Ticket Type'])
sample_cases = df.sample(3, random_state=42) # A random_state miatt mindig ugyanazt a 3-at veszi ki tesztelésre

passed_tests = 0
test_id = 1

for index, row in sample_cases.iterrows():
    real_text = row['Ticket Description']
    expected_cat = row['Ticket Type']
    
    print(f"TESZT #{test_id}")
    # Csak az első 100 karaktert írjuk ki, hogy ne spammelje tele a képernyőt
    print(f"Valós bemenet: '{real_text[:100]}...'") 
    
    # Elvégzi a predikciót
    actual_prediction = ml_model.predict_category(real_text, vectorizer, model)
    
    print(f"Várt kategória (CSV-ből):  [{expected_cat}]")
    print(f"Gép tippje (Predikció):    [{actual_prediction}]")
    
    # Összehasonlítás
    if actual_prediction == expected_cat:
        print("Eredmény: ✅ SIKERES TESZT")
        passed_tests += 1
    else:
        print("Eredmény: ❌ SIKERTELEN TESZT")
        
    print("-" * 60)
    test_id += 1

# Ha legalább egy teszt sikeres volt, teszteljük a Geminit is vele
if passed_tests > 0:
    print("\n✨ Gemini API tesztelése az első sikeres mintával...")
    first_success_text = sample_cases.iloc[0]['Ticket Description']
    first_success_cat = sample_cases.iloc[0]['Ticket Type']
    try:
        response = helper.generate_email_response(API_KEY, first_success_text, first_success_cat)
        print("✅ Gemini API: SIKERES (Levél legenerálva)")
    except Exception as e:
        print(f"❌ Gemini API HIBA: {e}")

print(f"\n📊 TESZTEREDMÉNYEK: 3 / {passed_tests} teszt sikeres.")