from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import os

app = Flask(__name__)
DATA_PATH = 'language.csv'
MODEL_PATH = 'model_pipeline.pkl'

# Train or load model
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
    print('Loaded pipeline from', MODEL_PATH)
else:
    print('Training pipeline from', DATA_PATH)
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=['Text', 'language'])
    texts = data['Text'].astype(str).values
    labels = data['language'].astype(str).values
    
    pipeline = make_pipeline(CountVectorizer(), MultinomialNB())
    pipeline.fit(texts, labels)
    
    joblib.dump(pipeline, MODEL_PATH)
    print('Saved pipeline to', MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    text = payload.get('text', '')

    if not isinstance(text, str) or text.strip() == '':
        return jsonify({'error': 'Please provide non-empty text.'}), 400

    pred = pipeline.predict([text])[0]

    # Confidence score
    confidence = None
    try:
        probs = pipeline.predict_proba([text])[0]
        class_index = list(pipeline.classes_).index(pred)
        confidence = float(probs[class_index])
    except:
        pass

    return jsonify({'prediction': pred, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
