"""Create a metadata-wrapped model file `real_estate_model.pkl` for the Streamlit app.

This script will:
- load an existing model from `models/best_model_RandomForest.pkl` (or another file if provided)
- attempt to infer feature names from `X_train.csv`
- attempt to infer target name from `y_train.csv` (fallback to 'price')
- save `real_estate_model.pkl` as a dict with keys: 'model', 'feature_names', 'target_name'

Run from project root:
    python scripts/create_model_metadata.py

"""
from pathlib import Path
import joblib
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
DEFAULT_MODEL = MODEL_DIR / 'best_model_RandomForest.pkl'
OUT_FILE = ROOT / 'real_estate_model.pkl'
X_TRAIN = ROOT / 'X_train.csv'
Y_TRAIN = ROOT / 'y_train.csv'

model_path = DEFAULT_MODEL if DEFAULT_MODEL.exists() else None
if len(sys.argv) > 1:
    cand = Path(sys.argv[1])
    if cand.exists():
        model_path = cand

if model_path is None:
    print("No saved model found. Place a model file in 'models/' or pass a path as the first argument.")
    sys.exit(1)

print(f"Loading model from {model_path}")
model = joblib.load(model_path)

# Infer feature names from X_train.csv if available
feature_names = None
if X_TRAIN.exists():
    try:
        X = pd.read_csv(X_TRAIN)
        feature_names = X.columns.tolist()
        print(f"Inferred {len(feature_names)} feature names from {X_TRAIN}")
    except Exception as e:
        print(f"Failed to read X_train.csv: {e}")

# Infer target name from y_train.csv if available
target_name = None
if Y_TRAIN.exists():
    try:
        y = pd.read_csv(Y_TRAIN)
        # if single column, use its name; else fallback
        if y.shape[1] == 1:
            target_name = y.columns[0]
        else:
            target_name = 'price'
        print(f"Inferred target name '{target_name}' from {Y_TRAIN}")
    except Exception as e:
        print(f"Failed to read y_train.csv: {e}")

if target_name is None:
    target_name = 'price'

metadata = {'model': model, 'feature_names': feature_names, 'target_name': target_name}

print(f"Saving metadata to {OUT_FILE}")
joblib.dump(metadata, OUT_FILE)
print("Done.")
