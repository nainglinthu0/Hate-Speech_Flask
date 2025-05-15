import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

def load_data(file):
    df = pd.read_csv(file)
    return df

def split_data(df):
    X = df["Content"]  
    y = df["Label"]  
    return train_test_split(X, y, test_size=0.1, random_state=42)

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    print("\nFinal Evaluation Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("See the Classification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["This is not Harmful", "This is harmful"]
    ))

if __name__ == "__main__":
    model = joblib.load("hate_speech_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    df = load_data("HateSpeechDataset.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    evaluate_model(model, vectorizer, X_test, y_test)