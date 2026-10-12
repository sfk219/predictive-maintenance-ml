# Predictive Maintenance ML (Synthetic Data)

A **runnable, educational** predictive-maintenance classifier built with Python, pandas and scikit-learn. Generates synthetic equipment readings and evaluates failure prediction with a stratified held-out test set.

## Run
```bash
pip install -r requirements.txt
python -m maintenance_ml.train
```

Outputs `artifacts/model.joblib` and `artifacts/metrics.json`. The model uses standardization and logistic regression; accuracy is only for synthetic constructed data.

## Test
```bash
pip install pytest
pytest -q
```

**Tech:** Python, pandas, numpy, scikit-learn, machine learning, model evaluation. No company data or real equipment deployments; not industrially validated.