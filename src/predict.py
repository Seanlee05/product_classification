# src/predict.py
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Import the preprocess function from your training script
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
    df_val = pd.DataFrame()
    if os.path.exists(val_path):
        print(f"\n--- Official Validation Report ({val_path}) ---")
        df_val = pd.read_csv(val_path)
        
        # Preprocess and Predict
        df_val['cleaned_text'] = df_val['product_description'].apply(preprocess_text)
        y_pred = model.predict(df_val['cleaned_text'])
        df_val['predicted_category'] = y_pred
        
        # Rename column to match your screenshot requirement
        df_val = df_val.rename(columns={'human_label': 'HUMAN_VERIFIED_Category'})
        
        # Generate official report
        print(classification_report(df_val['HUMAN_VERIFIED_Category'], y_pred))

        # --- ADDON: CONFUSION MATRIX GENERATION (BLUE) ---
        labels = sorted(df_val['HUMAN_VERIFIED_Category'].unique())
        cm = confusion_matrix(df_val['HUMAN_VERIFIED_Category'], y_pred, labels=labels)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix: Cleaned Validation')
        plt.ylabel('Actual Category')
        plt.xlabel('Predicted Category')
        plt.tight_layout()
        plt.savefig('data/valid_confusion_matrix.png')
        plt.close()
        print("Confusion Matrix saved to data/valid_confusion_matrix.png")
        # ------------------------------------------------
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
        
        # Create empty column to match screenshot format
        df_query['HUMAN_VERIFIED_Category'] = ""
        
        # Combine Query and Validation for the final CSV output shown in screenshot
        final_output = pd.concat([df_query, df_val], ignore_index=True)
        
        # Save to CSV with the specific columns shown in your screenshot
        final_output[['product_description', 'HUMAN_VERIFIED_Category', 'predicted_category']].to_csv(output_path, index=False)
        
        print(f"SUCCESS: Saved predictions to {output_path}")
    else:
        print(f"Warning: {query_path} not found. Skipping query predictions.")

if __name__ == "__main__":
    run_predictions()