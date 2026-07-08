import google.generativeai as genai

def generate_email_response(api_key, complaint, category, tone="Professional"):
    """Létrehozza a válaszlevelet a Gemini 2.5 modell segítségével a megadott hangnemben."""
    
    # Biztonsági tisztítás (levágjuk a láthatatlan szóközöket)
    clean_key = api_key.strip()
    genai.configure(api_key=clean_key)
    
    # A legújabb tesztelt modellünk
    llm_model = genai.GenerativeModel('gemini-2.5-flash')
    
    # A prompt kiegészült a tone (hangnem) utasítással
    prompt = f"""
    You are a professional and empathetic customer support agent. 
    A customer sent the following message: "{complaint}"
    
    Our internal Machine Learning system categorized this ticket as: {category}.
    
    Please write a polite, professional, and concise response email to the customer. 
    Crucial instruction: The tone of your email MUST be strictly: {tone}.
    
    Acknowledge their specific problem based on the category, and assure them we are looking into it.
    Do not invent specific names or order numbers. Use [Placeholder] if needed.
    """
    
    response = llm_model.generate_content(prompt)
    return response.text