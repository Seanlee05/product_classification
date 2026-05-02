# Product Classification System

A production-ready machine learning service to automatically classify product descriptions into 5 predefined categories.

## Project Structure
* `src/`: Core logic (Training, Prediction, API).
* `models/`: Trained model artifacts (`.pkl`).
* `data/`: Prediction outputs (raw datasets excluded per instructions).
* `tests/`: Unit tests for the API.
* `Dockerfile`: Container configuration.
* `TECHNICAL_DECISIONS.md`: Design justifications and data analysis findings.

## 1. Setup Environment (Conda)
To set up the environment using Conda:
```bash
conda create -n product-cls python=3.10
conda activate product-cls
pip install -r requirements.txt