<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>README Preview</title>
    <style>
        /* GitHub stílus szimulációja */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            line-height: 1.6;
            color: #24292e;
            max-width: 850px;
            margin: 0 auto;
            padding: 40px;
        }
        h1, h2 {
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
        }
        p, ul, ol {
            margin-top: 0;
            margin-bottom: 16px;
        }
        code {
            background-color: rgba(27,31,35,0.05);
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
            font-size: 85%;
        }
        pre {
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow: auto;
            line-height: 1.45;
        }
        pre code {
            background-color: transparent;
            padding: 0;
            font-size: 85%;
            border-radius: 0;
        }
        ul, ol {
            padding-left: 2em;
        }
        
        /* Reszponzív képtartó rács */
        .image-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 16px;
        }
        .image-box {
            flex: 1 1 calc(50% - 20px);
            min-width: 280px; /* Mobilon ez alá nem megy, inkább egymás alá ugrik */
        }
        .image-box img {
            width: 100%;
            height: auto;
            border: 1px solid #eaecef;
            border-radius: 6px;
            margin-bottom: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05); /* Pici árnyék, hogy kiemelkedjen */
        }
        .image-box em {
            display: block;
            font-size: 0.9em;
            color: #586069;
            text-align: center;
        }
    </style>
</head>
<body>

    <h1>AI-Assisted Customer Service Assistant</h1>

    <p>An AI-powered first-line customer support assistant that automates ticket handling. It uses Machine Learning to instantly categorize incoming emails and Generative AI to draft context-aware, personalized responses. The interactive web interface allows support agents to review predictions, dynamically adjust response tones, and analyze historical ticket trends.</p>

    <h2>Screenshots</h2>
    
    <div class="image-container">
        <div class="image-box">
            <img src="images/screenshot_main.png" alt="Main Interface - Ticket Processing">
        </div>
        <div class="image-box">
            <img src="images/screenshot_dashboard.png" alt="Analytics Dashboard">
        </div>
    </div>

    <h2>Technologies Used</h2>
    <ul>
        <li><strong>Python:</strong> Core programming language.</li>
        <li><strong>Streamlit:</strong> Interactive web interface and analytics dashboard.</li>
        <li><strong>Scikit-Learn:</strong> Machine Learning classification (<code>LinearSVC</code>) and text vectorization (TF-IDF).</li>
        <li><strong>Google Gemini 2.5 API:</strong> Generative LLM for drafting tailored responses.</li>
        <li><strong>Pandas &amp; Joblib:</strong> Data manipulation and model serialization.</li>
    </ul>

    <h2>How It Works</h2>
    <ol>
        <li><strong>Classification:</strong> A trained ML model reads the incoming ticket and classifies it into a predefined category.</li>
        <li><strong>Response Generation:</strong> The predicted category and customer's text are passed to the Gemini API to craft a tailored reply.</li>
        <li><strong>Control &amp; Customization:</strong> Agents can use the UI to adjust the tone (e.g., Professional, Apologetic) before finalizing the email.</li>
    </ol>

    <h2>Project Structure</h2>
    <ul>
        <li><code>app.py</code>: Streamlit web interface and main application flow.</li>
        <li><code>ml_model.py</code>: Text vectorization, model training, persistence, and prediction.</li>
        <li><code>response.py</code>: Google Gemini API connection and prompt.</li>
    </ul>

    <h2>Quick Start</h2>
    
    <ol>
        <li><strong>Install Dependencies:</strong>
<pre><code>pip install -r requirements.txt</code></pre>
        </li>
        
        <li><strong>Configure API Key:</strong>
            <p>Create a <code>.env</code> file in the root directory:</p>
<pre><code>GEMINI_API_KEY=your_actual_gemini_api_key</code></pre>
        </li>
        
        <li><strong>Launch the App:</strong>
<pre><code>python -m streamlit run app.py</code></pre>
        </li>
    </ol>

</body>
</html>