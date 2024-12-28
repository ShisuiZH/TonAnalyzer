from flask import Flask, request, jsonify, render_template
import pickle
import re

app = Flask(__name__)

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^а-яёәіңғүұқөһa-z\s]', '', text)
    return text

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    text = request.json.get('text', '')
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0][1]
    
    if prediction == 1 and probability > 0.6:
        result = "Позитивный 😊"
    elif 0.5 <= probability <= 0.6:
        result = "Нейтральный 😐"
    else:
        result = "Негативный 😞"
    
    return jsonify({'sentiment': result})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)