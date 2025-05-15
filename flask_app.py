import pandas as pd
from flask import Flask, request, jsonify
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)

def load_model():
    with open('hate_speech_model.joblib', 'rb') as f:
        return joblib.load(f)

def load_vectorizer():
    with open('vectorizer.joblib', 'rb') as f:
        return joblib.load(f)

loaded_model = load_model()
loaded_vectorizer = load_vectorizer()

@app.route('/predict', methods=['POST'])
def predict_hateful_content():
    data = request.get_json()
    text = [data.get("text", "")]  

    if not text[0].strip():
        return jsonify({"error caused by empty text"}), 400

    input_vector = loaded_vectorizer.transform(text)
    prediction = loaded_model.predict(input_vector)

    result = "This is not harmful" if prediction[0] ==1 else "This is harmful"

    return jsonify({
        "input": text[0],
        "prediction": result,
        "name": "Naing Lin Thu",
        "student-id": "PIUS20220032"
    })
@app.route("/")
def home():
    return "Welcome!"
if __name__ == "__main__":
    app.run(debug=True)