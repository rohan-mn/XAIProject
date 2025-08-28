# XAIProject — Counterfactual Explanations on UCI Heart Disease (Streamlit)

Interactive Explainable AI demo for tabular medical data.  
Train a classifier on the **UCI Heart Disease** dataset and explore its predictions using:
- **Global explainability** (feature importances / SHAP-like summaries)
- **Local explainability** (per-instance breakdowns)
- **Counterfactual explanations**: “What minimal changes flip the prediction?”

> Dataset included: `heart_disease_uci.csv` (for quick local runs).

## ✨ Features
- Simple Streamlit UI (`app.py`)
- Load/train a model on the included dataset
- Inspect predictions and per-feature contributions
- Generate counterfactuals for any selected record
- Export explanations for reports

## 🗂️ Repository Structure
XAIProject/      
├─ app.py # Streamlit app entry point  
├─ heart_disease_uci.csv # Sample dataset  
└─ requirements.txt # Python dependencies   

## How to run:

pip install -r requirements.txt  
streamlit run app.py

