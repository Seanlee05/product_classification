# src/predict.py
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report

# Import the preprocess function from your training script
# Ensure train.py is in the same folder or Python path
from train import preprocess_text

def run_predictions():
    # 1. Load the winning model
    model_path = 'models/product_classifier.pkl'
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Please run train.py first.")
        return

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # TASK 1: VALIDATION (Testing Accuracy)
    val_path = 'data/Cleaned_Validation.csv'
    if os.path.exists(val_path):
        print(f"\n--- Official Validation Report ({val_path}) ---")
        df_val = pd.read_csv(val_path)
        
        # Preprocess and Predict
        df_val['cleaned_text'] = df_val['product_description'].apply(preprocess_text)
        y_pred = model.predict(df_val['cleaned_text'])
        
        # Generate official report comparing human_label vs predicted
        print(classification_report(df_val['human_label'], y_pred))
    else:
        print(f"Warning: {val_path} not found. Skipping validation report.")

    # TASK 2: QUERY (Inference for Submission)
    query_path = 'data/Cleaned_Query.csv'
    output_path = 'data/query_predictions.csv'
    
    if os.path.exists(query_path):
        print(f"\nGenerating predictions for Query data: {query_path}...")
        df_query = pd.read_csv(query_path)
        
        # Preprocess
        df_query['cleaned_text'] = df_query['product_description'].apply(preprocess_text)
        
        # Run the model
        df_query['predicted_category'] = model.predict(df_query['cleaned_text'])
        
        # Save only the required columns for the artifact submission
        # We drop 'cleaned_text' to keep the output file neat
        df_query[['product_description', 'predicted_category']].to_csv(output_path, index=False)
        
        print(f"SUCCESS: Saved {len(df_query)} predictions to {output_path}")
    else:
        print(f"Warning: {query_path} not found. Skipping query predictions.")

if __name__ == "__main__":
    run_predictions()