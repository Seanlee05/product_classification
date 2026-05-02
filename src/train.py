import pandas as pd
import re
import joblib
import time
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier # <--- FIXED IMPORT
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

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
    # Since we cleaned the data into 2 columns already, we load it simply
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Ensure the columns match our cleaned format
    # column 0: product_description, column 1: product_category
    X = df.iloc[:, 0].astype(str)
    y = df.iloc[:, 1].astype(str)
    
    print("Preprocessing text...")
    X_cleaned = X.apply(preprocess_text)
    return X_cleaned, y

def train_and_evaluate():
    # 1. Load Data (Make sure this path points to your CLEANED file)
    X, y = load_data('data/Cleaned_Training_data.csv')
    
    # 2. Stratified Split (Crucial for the imbalance we found earlier)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Define Models to evaluate
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
        
        # Build Pipeline: TF-IDF -> Classifier
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('clf', clf)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate using Macro F1 (Best for imbalanced datasets)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        train_time = time.time() - start_time
        
        print(f"\n{name}:")
        print(f"Training Time: {train_time:.2f} seconds")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model = pipeline
            best_name = name

    print("\n" + "="*50)
    print(f"Selecting {best_name} (Macro F1: {best_f1:.4f})")
    print("="*50 + "\n")
    
    # Full report for the best model
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best))
    
    # 4. Save the best model
    model_path = 'models/product_classifier.pkl'
    # Ensure directory exists
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()