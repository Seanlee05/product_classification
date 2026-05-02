import pandas as pd
import re
import joblib
import time
import nltk
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    X = df.iloc[:, 0].astype(str)
    y = df.iloc[:, 1].astype(str)
    print("Preprocessing text...")
    X_cleaned = X.apply(preprocess_text)
    return X_cleaned, y

def save_confusion_matrix(y_true, y_pred, filename, title):
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

def train_and_evaluate():
    X, y = load_data('data/Cleaned_Training_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Linear SVM (SGD)": SGDClassifier(class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    best_model = None
    best_f1 = 0
    best_name = ""
    
    print("\n--- Evaluating 3 Classification Approaches ---")
    for name, clf in models.items():
        start_time = time.time()
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"{name}: Macro F1={macro_f1:.4f} ({time.time()-start_time:.2f}s)")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = pipeline
            best_name = name

    print(f"\nWINNER: {best_name}")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best))
    
    # GENERATE MATRIX
    save_confusion_matrix(y_test, y_pred_best, 'data/train_confusion_matrix.png', f'Confusion Matrix: Training - {best_name}')
    
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(best_model, 'models/product_classifier.pkl')
    print("Model saved to models/product_classifier.pkl")

if __name__ == "__main__":
    train_and_evaluate()