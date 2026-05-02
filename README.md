# Product Classification System

A production-ready machine learning service to automatically classify product descriptions into 5 predefined categories.

## Project Structure
* `src/`: Core logic (Training, Prediction, API).
* `models/`: Trained model artifacts (`.pkl`).
* `data/`: Prediction outputs (raw datasets excluded per instructions).
* `tests/`: Unit tests for the API.
* `Dockerfile`: Container configuration.
* `TECHNICAL_DECISIONS.md`: Design justifications and data analysis findings.

# Product Classifier Deployment & Usage Guide

## Method 1: Deployment with Docker (Recommended)
Use this method to build and run the entire production service in an isolated container.

### Build the Image
Clone repo
```bash
git clone https://github.com/Seanlee05/product_classification.git
```
Open your terminal in the project root directory and run:

```bash
docker build -t product-classifier .
docker run -p 8000:8000 product-classifier
```

### Once the container is running, Can do testing at here:

Interactive API Documentation (Swagger UI): http://localhost:8000/docs

API Health Check: http://localhost:8000/health

## Method 2: Manual Setup with Conda

Use this method if you wish to run the training and prediction scripts locally.

```bash
Create and Activate Environment
conda create -n my_env python=3.10 -y
conda activate my_env
pip install -r requirements.txt
```

# Start the API Server
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```
## Testing

To verify the API logic and health endpoints, run the test suite:
```bash
python -m pytest
```
# Project Structure and Artifacts

The project is organized into modular components to satisfy production engineering standards:

```bash
├── src/
│   ├── __init__.py
│   ├── train.ipynb          # model comparison, and serialization
│   ├── predict.ipynb        # Inference logic, result generation, and validation visualization
│   └── api.py               # FastAPI implementation with Pydantic validation
│
├── models/
│   ├── product_classifier.pkl    # Trained Linear SVM model
│   └── label_encoder.pkl
│
├── tests/
│   ├── __init__.py
│   └── test_api.py    # unit test for API
│
├── notebooks/
│   └── 01_EDA_and_Preprocessing.ipynb    # data cleaning, data preprocessing scripts
│
├── data/
│   ├── Query_and_Validation_prediction.csv   # Final predictions
│   └── valid_confusion_matrix.png            # Model performance visualization
|   └── train_confusion_matric.png            # Model performance visualization
│
├── Dockerfile                 # Containerization instructions
├── requirements.txt           # Python dependencies (production optimized)
├── TECHNICAL_DECISIONS.md     # Analysis, architecture, design decisions
```
# Important Notes
Per instructions, the raw input datasets (Training_data.csv and Query_and_Validation_data.csv) are excluded from this repository via .gitignore.
To retrain the model, place these files in the data/ directory.
