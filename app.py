import streamlit as st
import joblib
import re
import string
import os
from pathlib import Path
import uuid

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Sarcasm Detector 🎭",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# PREMIUM CSS STYLING
# ============================================================================

def load_css():
    """
    Applies premium custom CSS for a dark-themed, high-contrast interface.
    """
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
            color: #ffffff;
        }
        
        /* Main container styling with dark theme */
        .main {
            background: #1a1a1a;
            padding: 2rem;
            min-height: 100vh;
        }
        
        /* Hero Section */
        .hero-section {
            text-align: center;
            padding: 3rem 0;
            margin-bottom: 3rem;
        }
        
        .title {
            font-size: 4.5rem;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 4px 20px rgba(255,255,255,0.2);
            margin-bottom: 1rem;
            animation: fadeInDown 0.8s ease;
        }
        
        .subtitle {
            color: #d1d1d1;
            font-size: 1.4rem;
            font-weight: 300;
            text-shadow: 0 2px 10px rgba(255,255,255,0.1);
            animation: fadeInUp 0.8s ease;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Glass Card styling */
        .glass-card {
            background: rgba(40, 40, 40, 0.9);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 2rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        }
        
        .solid-card {
            background: #2a2a2a;
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 2rem;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-shadow: 0 2px 10px rgba(255,255,255,0.1);
        }
        
        /* Text area styling */
        .stTextArea textarea {
            border-radius: 15px;
            border: 2px solid rgba(255, 255, 255, 0.2);
            background: #333333;
            color: #ffffff;
            font-size: 1.1rem;
            padding: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextArea textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            border: none;
            border-radius: 15px;
            padding: 1rem 3rem;
            font-size: 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
        }
        
        /* Result boxes */
        .result-box {
            padding: 2.5rem;
            border-radius: 15px;
            text-align: center;
            margin: 2rem 0;
            font-size: 2rem;
            font-weight: 700;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            animation: scaleIn 0.5s ease;
        }
        
        @keyframes scaleIn {
            from {
                opacity: 0;
                transform: scale(0.8);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        .sarcastic {
            background: linear-gradient(135deg, #d63384 0%, #e35050 100%);
            color: #ffffff;
        }
        
        .not-sarcastic {
            background: linear-gradient(135deg, #1e90ff 0%, #00b7eb 100%);
            color: #ffffff;
        }
        
        /* Confidence bar */
        .confidence-container {
            margin: 2rem 0;
        }
        
        .confidence-bar {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            height: 40px;
            overflow: hidden;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #56ab2f 0%, #a8e063 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-weight: 700;
            font-size: 1.1rem;
            transition: width 1s ease;
            box-shadow: 0 0 20px rgba(86, 171, 47, 0.5);
        }
        
        /* Metric styling */
        .metric-card {
            background: rgba(40, 40, 40, 0.9);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            background: rgba(60, 60, 60, 0.9);
            transform: translateY(-5px);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: 0 2px 10px rgba(255,255,255,0.2);
        }
        
        .metric-label {
            font-size: 1rem;
            color: #d1d1d1;
            margin-top: 0.5rem;
        }
        
        /* Example buttons */
        .example-btn {
            background: rgba(40, 40, 40, 0.9);
            backdrop-filter: blur(10px);
            color: #ffffff;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin: 0.5rem;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            cursor: pointer;
            text-align: left;
        }
        
        .example-btn:hover {
            background: rgba(60, 60, 60, 0.9);
            transform: translateX(5px);
        }
        
        /* Info sections */
        .info-item {
            background: rgba(40, 40, 40, 0.9);
            padding: 1rem;
            border-radius: 10px;
            margin: 0.8rem 0;
            border-left: 4px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }
        
        .info-item:hover {
            background: rgba(60, 60, 60, 0.9);
            transform: translateX(5px);
        }
        
        /* Feature badges */
        .feature-badge {
            display: inline-block;
            background: rgba(40, 40, 40, 0.9);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            margin: 0.3rem;
            font-size: 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        
        /* Spinner styling */
        .stSpinner > div {
            border-top-color: #ffffff !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.4);
        }

        /* Enhanced example buttons for Streamlit */
        .stButton > button[aria-label*="sarc"] {
            background: linear-gradient(135deg, #d63384 0%, #e35050 100%);
            box-shadow: 0 4px 15px rgba(214, 51, 132, 0.4);
        }

        .stButton > button[aria-label*="sarc"]:hover {
            box-shadow: 0 6px 25px rgba(214, 51, 132, 0.6);
        }

        .stButton > button[aria-label*="non_sarc"] {
            background: linear-gradient(135deg, #1e90ff 0%, #00b7eb 100%);
            box-shadow: 0 4px 15px rgba(30, 144, 255, 0.4);
        }

        .stButton > button[aria-label*="non_sarc"]:hover {
            box-shadow: 0 6px 25px rgba(30, 144, 255, 0.6);
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def clean_text(text):
    """
    Cleans and preprocesses input text (same as training script).
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """
    Loads the trained model and vectorizer.
    Uses caching to avoid reloading on every interaction.
    """
    try:
        model = joblib.load('sarcasm_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run 'train_model.py' first to train the model.")
        st.stop()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_sarcasm(text, model, vectorizer):
    """
    Predicts whether the input text is sarcastic or not.
    
    Returns:
        is_sarcastic (bool): True if sarcastic, False otherwise
        confidence (float): Confidence score (0-100)
        probabilities (tuple): (non_sarcastic_prob, sarcastic_prob)
    """
    cleaned = clean_text(text)
    
    if not cleaned:
        return None, None, None
    
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    
    is_sarcastic = bool(prediction)
    confidence = probabilities[prediction] * 100
    
    return is_sarcastic, confidence, probabilities

# ============================================================================
# SAMPLE EXAMPLES
# ============================================================================

SARCASTIC_EXAMPLES = [
    "Oh great, another meeting that could have been an email",
    "I love it when my code works on the first try. Said no programmer ever",
    "Wow, traffic is so light today. Said nobody in this city ever",
    "Nothing says 'I love you' like forgetting your anniversary",
    "Study finds that people who wake up early are morning people",
    "Tech company announces groundbreaking new feature: it actually works"
]

NON_SARCASTIC_EXAMPLES = [
    "Scientists discover new treatment for rare disease",
    "Local school wins national science competition",
    "New restaurant opens downtown with diverse menu",
    "Company reports record profits in quarterly earnings",
    "City announces plans for new public park",
    "Research shows benefits of daily exercise"
]

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application function.
    """
    load_css()
    model, vectorizer = load_model()
    
    # Hero Section
    st.markdown('''
        <div class="hero-section">
            <div class="title">🎭 Sarcasm Detector</div>
            <div class="subtitle">AI-Powered Sarcasm Detection • Real-Time Analysis • Instant Results</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Handle example selection from session state
    if 'example_text' in st.session_state:
        example = st.session_state.pop('example_text')
        st.session_state['main_input'] = example
    
    # Main Layout
    col1, col2 = st.columns([1.5, 1], gap="large")
    
    with col1:
        # Input Section
       
        st.markdown('<h2 style="color: #ffffff; margin-bottom: 1.5rem;">📝 Enter Your Text</h2>', unsafe_allow_html=True)
        
        default_value = st.session_state.get('main_input', '')
        user_input = st.text_area(
            "",
            value=default_value,
            height=180,
            placeholder="Type or paste your text here... Our AI will analyze it for sarcasm! 🤖",
            label_visibility="collapsed",
            key="main_input"
        )
        
        analyze_btn = st.button("🔍 Analyze for Sarcasm", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Results Section
        if analyze_btn and user_input.strip():
            with st.spinner("🧠 AI is thinking..."):
                is_sarcastic, confidence, probabilities = predict_sarcasm(user_input, model, vectorizer)
                
                if is_sarcastic is None:
                    st.error("⚠️ Please enter valid text!")
                else:
                    st.markdown('<div class="solid-card">', unsafe_allow_html=True)
                    
                    # Main Result
                    if is_sarcastic:
                        st.markdown('''
                            <div class="result-box sarcastic">
                                😏 SARCASM DETECTED!
                            </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown('''
                            <div class="result-box not-sarcastic">
                                😊 NO SARCASM DETECTED
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    # Confidence Bar
                    
                    # Detailed Metrics
                    
                    
        elif analyze_btn and not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze!")

    with col2:
        # About Section
        st.markdown('<div class="section-header">ℹ️ How It Works</div>', unsafe_allow_html=True)
        
        st.markdown('''
            <div class="info-item">
                <strong>1. Text Processing</strong><br>
                Your text is cleaned and preprocessed
            </div>
            <div class="info-item">
                <strong>2. Feature Extraction</strong><br>
                AI extracts linguistic patterns
            </div>
            <div class="info-item">
                <strong>3. ML Analysis</strong><br>
                Model predicts sarcasm probability
            </div>
            <div class="info-item">
                <strong>4. Results</strong><br>
                Get instant confidence scores
            </div>
        ''', unsafe_allow_html=True)
        
        
        # Model Info
        
        st.markdown('<div class="section-header">🤖 AI Model</div>', unsafe_allow_html=True)
        
        st.markdown('''
            <span class="feature-badge">🎯 Logistic Regression</span>
            <span class="feature-badge">📊 TF-IDF Features</span>
            <span class="feature-badge">🔬 N-gram Analysis</span>
            <span class="feature-badge">⚡ Real-time</span>
        ''', unsafe_allow_html=True)
        
        
    
    # Examples Section
    st.markdown('<div class="section-header" style="text-align: center;">💡 Try These Examples</div>', unsafe_allow_html=True)
    
    ex_col1, ex_col2 = st.columns(2)
    
    with ex_col1:
        st.markdown('<p style="color: #ffffff; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">😏 Sarcastic Examples</p>', unsafe_allow_html=True)
        for i, example in enumerate(SARCASTIC_EXAMPLES):
            if st.button(example, key=f"sarc_{i}", use_container_width=True, help="Click to load this sarcastic example"):
                st.session_state['example_text'] = example
                st.rerun()
    
    with ex_col2:
        st.markdown('<p style="color: #ffffff; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">😊 Genuine Examples</p>', unsafe_allow_html=True)
        for i, example in enumerate(NON_SARCASTIC_EXAMPLES):
            if st.button(example, key=f"non_sarc_{i}", use_container_width=True, help="Click to load this genuine example"):
                st.session_state['example_text'] = example
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown('''
        <div style="text-align: center; color: rgba(255,255,255,0.8); font-size: 0.9rem;">
            Built with ❤️ using Streamlit & Scikit-learn | Powered by AI & Machine Learning
        </div>
    ''', unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()