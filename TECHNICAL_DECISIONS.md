---

### **2. Final `TECHNICAL_DECISIONS.md`**

This document combines your data cleaning steps, the model evaluation results, and the justification for your final architecture.

```markdown
# Technical Decisions & Data Analysis

## 1. Data Analysis & Preprocessing
The primary challenge of this project was the quality of the raw data.

### Challenges Identified:
* **Messy CSV Structure:** The raw training file contained extra columns and trailing semicolons (`;;;`) in the class names. I implemented a cleaning step to remove these, strip unnecessary quotes, and re-align the data into two clean columns: `product_description` and `Category`.
* **Malformed Validation Data:** The `Query_and_Validation_data.csv` was malformed due to unquoted commas in product descriptions, causing the CSV parser to misread entries.
* **Class Imbalance:** The dataset was highly imbalanced. The largest class (`Dry Goods`) was nearly 6x larger than the smallest class (`Specialty & Miscellaneous`).

### Preprocessing Decisions:
* **Schema Validation:** For the validation set, I filtered the `HUMAN_VERIFIED_Category` to only include the 5 valid target classes, successfully isolating **881 verified records** from the noise.
* **Text Cleaning:** I utilized a pipeline that included lowercasing, removal of non-alphabetical characters, and English stop-word removal to create a clean vocabulary for the vectorizer.

## 2. Model Development & Selection
I systematically compared three distinct approaches using a TF-IDF Vectorizer with a maximum of 10,000 features.

### Evaluation Results:
| Model | Training Time | Macro F1-Score |
| :--- | :--- | :--- |
| Logistic Regression | 3.10s | 0.8780 |
| **Linear SVM (SGD)** | **0.61s** | **0.8958** |
| Random Forest | 13.58s | 0.8789 |

### Why Linear SVM Won:
The Linear SVM (via `SGDClassifier`) was selected for production because it outperformed more complex models in both accuracy and efficiency. It is highly optimized for sparse, high-dimensional text data and provides the lowest inference latency for a production API.

### Handling Imbalance:
To prevent the model from biasing toward the majority class, I used **Stratified Train/Test Splits** and applied **Class Weighting** (`class_weight='balanced'`). This penalized the model more heavily for mistakes on minority classes.

## 3. Error Analysis
The model achieved an overall accuracy of **93%** on the validation set.

**Observation on "Specialty & Miscellaneous":**
While other categories reached F1-scores of 0.93-0.96, the Miscellaneous class scored lower at **0.68**. 
* **Reasoning:** This is a "catch-all" category. Its vocabulary overlaps significantly with the other 4 classes, making it semantically ambiguous. In a production setting, this indicates that this specific class would benefit from more unique training examples or a manual "human-in-the-loop" review for low-confidence predictions.

## 4. Production Service & Architecture
* **FastAPI:** I implemented a service with `/predict` and `/health` endpoints. FastAPI was chosen for its high performance and built-in request validation.
* **Containerization:** The solution is fully containerized using Docker to ensure reproducibility across different environments.
* **Scaling Strategy:** The service is stateless, allowing for horizontal scaling behind a Load Balancer (e.g., AWS ALB with ECS Fargate).
* **Monitoring:** I recommend monitoring "Prediction Confidence" distributions. A downward trend over time would signify Data Drift, requiring model retraining.