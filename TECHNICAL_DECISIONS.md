# Technical Decisions & Data Analysis

## 1. Data Analysis & Preprocessing
The primary challenge of this project was the structural integrity and quality of the raw data.

### Challenges Identified:
* **Train csv:** The raw training file contained extra columns and trailing semicolons (`;;;`) in the class names. I implemented a custom **"Anchor Logic"** cleaning step to remove these, strip unnecessary quotes, and re-align the data into two clean columns: `product_description` and `product_category`.
* **Malformed Validation Data:** The `Query_and_Validation_data.csv` was malformed due to unquoted commas in product descriptions. This caused a standard CSV parser to shift text into "Unnamed" columns.
* **Label Noise:** Several entries in the human-verified column contained descriptive text (e.g., "Chocolate Truffle") rather than the 5 target categories.
* **Class Imbalance:** The dataset showed a significant skew. The largest class (`Dry Goods`) was nearly 6x larger than the smallest class (`Specialty & Miscellaneous`).

### Preprocessing Decisions:
* **Structural Repair:** Instead of skipping "bad lines," which will cause lost data. I used a line-by-line reading strategy to find the category anchor (semicolon) and merge any description text that had been incorrectly split across commas.
* **Target Filtering:** For validation, I strictly filtered the labels to the 5 valid target classes, isolating high-quality verified records for final evaluation.
* **Text Cleaning Pipeline:** 
    * Lowercasing.
    * Regex-based removal of special characters and digits.
    * Tokenization and removal of English stop-words to reduce feature noise.

## 2. Model Development & Selection
I systematically compared three distinct approaches using a **TF-IDF Vectorizer** (10,000 features, 1-2 grams) to convert text into numerical vectors.

### Evaluation Results:
| Model | Training Time | Macro F1-Score |
| :--- | :--- | :--- |
| Logistic Regression | 2.42s | 0.8780 |
| **Linear SVM (via SGD)** | **0.55s** | **0.8958** |
| Random Forest | 13.58s | 0.8789 |
    
### Why Linear SVM (SGD) Won:
The **Linear SVM** was selected for production because it achieved the highest **Macro F1-Score (0.8958)** with the lowest training time. SVMs are mathematically robust for high-dimensional text data where the relationship between words and categories is often linearly separable.

### Handling Class Imbalance:
To prevent the model from ignoring the minority "Specialty" class, I implemented:
1. **Stratified Splitting:** Ensuring the train/test sets maintained the same class proportions.
2. **Class Weighting:** Applied `class_weight='balanced'` during training, which automatically adjusts weights inversely proportional to class frequencies.

## 3. Error Analysis
The model achieved an overall accuracy of **93%** on the validation set.

**Observation on "Specialty & Miscellaneous":**
While classes like "Beverages" reached F1-scores of 0.95, "Specialty" scored significantly lower (**~0.70**). 
* **Reasoning:** This is a "catch-all" category. Its vocabulary overlaps significantly with other classes, making it semantically ambiguous. 
* **Recommendation:** In a production environment, this class should trigger a "low confidence" flag for manual human review.

## 4. Production Service & Architecture
* **FastAPI:** Chosen for the inference service due to its high performance, asynchronous support, and automatic documentation (Swagger UI).
* **Docker:** The entire environment is containerized using a `python:3.9-slim` base image to ensure the model runs identically in development and production.
* **Endpoint Design:**
    * `GET /health`: Verifies the service is alive and the `.pkl` model is correctly loaded into memory.
    * `POST /predict`: Accepts JSON input and returns the predicted category along with the original description.

## 5. Deployment & Scaling
* **Containerization:** The `Dockerfile` packages the model and code.
* **Scaling:** The service is stateless. In a cloud environment, it can be deployed on **AWS ECS Fargate** or **Kubernetes**, scaling horizontally based on CPU/Request count.
* **Monitoring Strategy:** I recommend tracking **Prediction Latency** and **Label Distribution**. If the predicted labels start drifting significantly from the training distribution, it indicates a shift in the product catalog that requires a model retrain.

## 6. Results Analysis
![Alt Text](https://github.com/Seanlee05/product_classification/blob/main/data/valid_result.png)
* **Overall Accuracy:** 93% on verified validation data.
* **Top Performing Class:** `Household & Personal Care` (F1-Score: 0.96). The vocabulary for this class (e.g., "shampoo", "detergent", "diapers") is very distinct.
* **Challenging Class:** `Specialty & Miscellaneous` (F1-Score: 0.68). 
  * *Root Cause:* As a "catch-all" category, it contains terms that overlap with `Dry Goods`. For example, "Organic Rice Vinegar" could semantically fit in both.
  * *Recommendation:* This class would benefit from a "Human-in-the-loop" review if the model's prediction confidence is below 70%.

## 7. System Architecture Diagram
This diagram shows the end-to-end data flow of the production system:

```text
[ Raw Data (.csv) ] 
       |
       v
[ src/train.py ] --> Evaluates Models --> [ models/product_classifier.pkl ]
       |                                
       v
[ src/predict.py ] --> Internal Validation & Inference --> [ data/Query_and_Validation_predictions.csv ]
                                                       --> [ data/valid_confusion_matrix.png ]
                                                       --> [ data/train_confusion_matrix.png ]
       |
       v
[ src/api.py (FastAPI) ] <--- Model Loading
       |
       v
[ Docker Container ] <--- Port 8000
       |
       v
[ Production Cloud (AWS/Azure/GCP) ]
