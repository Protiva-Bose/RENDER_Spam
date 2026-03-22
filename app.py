import os
import pandas as pd
import numpy as np
import re
import joblib
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Create Flask app - MUST be named 'app'
app = Flask(__name__)
CORS(app)

print("="*50)
print("🚀 Spam Detector API Starting...")
print("="*50)

# Load model files
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')
df = pd.read_csv('spam_dataset.csv')

print(f"✅ Model loaded! Dataset: {len(df)} messages")

# ============================================
# PREPROCESS FUNCTION
# ============================================

def preprocess_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' PHONE ', text)
    text = re.sub(r'[^a-zA-Z\s!?$]', ' ', text)
    text = ' '.join(text.split())
    
    return text

# ============================================
# NEWSAPI FACT CHECKER
# ============================================

NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '5f91ed780eef4bd6b8ed3c788d715ee1')

def check_news_fact(query):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 3
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('status') == 'ok':
            articles = data.get('articles', [])
            trusted_sources = ['bbc.com', 'reuters.com', 'apnews.com', 'cnn.com']
            results = []
            for article in articles[:2]:
                url = article.get('url', '')
                is_trusted = any(source in url.lower() for source in trusted_sources)
                results.append({
                    'title': article.get('title', '')[:100],
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'trusted': is_trusted,
                    'url': url
                })
            return results
    except:
        pass
    return []

# ============================================
# ANALYZE FUNCTION
# ============================================

def analyze_message(text):
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    
    proba = model.predict_proba(vectorized)[0]
    prediction = model.predict(vectorized)[0]
    probability = float(proba[1]) if prediction == 1 else float(proba[0])
    
    analysis = {
        'is_spam': bool(prediction),
        'probability': probability,
        'risk_level': 'High' if probability > 0.7 else ('Medium' if probability > 0.4 else 'Low'),
        'explanation': [],
        'spam_indicators': [],
        'suggestions': [],
        'news_sources': []
    }
    
    text_lower = text.lower()
    
    # Spam patterns
    if any(w in text_lower for w in ['won', 'prize', 'lottery', 'million']):
        analysis['spam_indicators'].append('Financial scam pattern')
        analysis['explanation'].append('Contains lottery/prize scam keywords')
    
    if any(w in text_lower for w in ['urgent', 'immediately', 'asap']):
        analysis['spam_indicators'].append('Urgency pressure tactic')
        analysis['explanation'].append('Uses urgent language to pressure you')
    
    if any(w in text_lower for w in ['verify', 'account', 'password', 'login']):
        analysis['spam_indicators'].append('Phishing attempt')
        analysis['explanation'].append('Asks for account verification (phishing)')
    
    if text.count('!') > 2:
        analysis['spam_indicators'].append('Excessive exclamation marks')
    
    # Fake news detection
    fake_news = None
    if any(c in text_lower for c in ['bangladesh', 'india', 'usa', 'uk']):
        if any(w in text_lower for w in ['dead', 'collapsed', 'destroyed', 'crisis']):
            news = check_news_fact(text_lower.split()[0])
            if news and any('growth' in n['title'].lower() or 'success' in n['title'].lower() for n in news):
                analysis['explanation'].append("⚠️ FAKE NEWS! Contradicts real news")
                analysis['is_spam'] = True
                analysis['probability'] = min(analysis['probability'] + 0.3, 0.99)
                analysis['risk_level'] = 'High'
                analysis['news_sources'] = news
    
    # Suggestions
    if analysis['is_spam']:
        analysis['suggestions'] = [
            "🚫 DO NOT click any links",
            "🚫 DO NOT reply or share personal info",
            "📱 Report as spam to your provider"
        ]
    else:
        analysis['suggestions'] = [
            "✓ This appears legitimate",
            "🔍 Still verify sender if unsure"
        ]
    
    return analysis

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'service': 'Spam Detector API',
        'version': '3.0',
        'dataset_size': len(df),
        'model_loaded': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '')
        if not message:
            return jsonify({'success': False, 'error': 'No message'}), 400
        
        analysis = analyze_message(message)
        return jsonify({
            'success': True,
            'is_spam': analysis['is_spam'],
            'probability': analysis['probability'],
            'risk_level': analysis['risk_level'],
            'message': message,
            'explanation': analysis['explanation'][:5],
            'suggestions': analysis['suggestions'][:3],
            'spam_indicators': analysis['spam_indicators'][:5],
            'news_sources': analysis['news_sources']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({
        'success': True,
        'dataset_size': len(df),
        'spam_count': int(df[df['label']==1].shape[0]),
        'ham_count': int(df[df['label']==0].shape[0]),
        'best_model': 'Ensemble',
        'accuracy': 0.98
    })

@app.route('/report', methods=['POST'])
def report():
    try:
        global df
        data = request.get_json()
        message = data.get('message', '')
        is_spam = data.get('is_spam', False)
        
        new_row = pd.DataFrame({
            'label': [1 if is_spam else 0],
            'message': [message],
            'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
        
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv('spam_dataset.csv', index=False)
        
        return jsonify({
            'success': True,
            'message': 'Report saved',
            'dataset_size': len(df)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'dataset_size': len(df)
    })

# ============================================
# RUN SERVER
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)