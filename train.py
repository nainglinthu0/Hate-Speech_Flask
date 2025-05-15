import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data(file):
    df = pd.read_csv(file)
    return df

def split_data(df):
    X = df["Content"] #Feature
    y = df["Label"]    #Target
    return train_test_split(X, y, test_size=0.1, random_state=42)

def train_model(X_train, y_train):
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    print("\nFinal Evaluation Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("See the Classification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["This is not harmful", "This is harmful"]
    ))


def save_model(model, vectorizer):
    joblib.dump(model, "hate_speech_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    print("Model and vectorizer are saved now.")


if __name__ == "__main__":
    df = load_data("HateSpeechDataset.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    model, vectorizer = train_model(X_train, y_train)

    evaluate_model(model, vectorizer, X_test, y_test)
    save_model(model, vectorizer)