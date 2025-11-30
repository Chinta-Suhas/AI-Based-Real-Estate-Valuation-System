"""Fix nested metadata in real_estate_model.pkl if the 'model' key contains another metadata dict.

This can happen when a saved model file already contains metadata and we wrapped it again.
The script normalizes so that top-level dict has keys: 'model', 'feature_names', 'target_name'.
"""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'real_estate_model.pkl'
X_TRAIN = ROOT / 'X_train.csv'

if not META.exists():
    print('No metadata file at', META)
    raise SystemExit(1)

m = joblib.load(META)
print('Loaded top-level keys:', list(m.keys()) if isinstance(m, dict) else type(m))

# If top-level 'model' is a dict that itself has 'model', unwrap
if isinstance(m, dict) and 'model' in m and isinstance(m['model'], dict) and 'model' in m['model']:
    inner = m['model']
    model_obj = inner.get('model')
    # prefer explicit feature_names/target_name from outer metadata, else inner, else infer
    feature_names = m.get('feature_names') or inner.get('feature_names')
    target_name = m.get('target_name') or inner.get('target_name')
    print('Found nested metadata - extracting inner model')
else:
    # already fine
    print('No nested metadata detected. Nothing to do.')
    raise SystemExit(0)

# If still missing feature_names, try to infer from X_train
if not feature_names and X_TRAIN.exists():
    try:
        X = pd.read_csv(X_TRAIN)
        feature_names = X.columns.tolist()
        print(f'Inferred {len(feature_names)} feature names from {X_TRAIN}')
    except Exception as e:
        print('Failed to read X_train.csv for feature names:', e)

if not target_name:
    target_name = 'price'

new_meta = {'model': model_obj, 'feature_names': feature_names, 'target_name': target_name}
joblib.dump(new_meta, META)
print('Saved normalized metadata to', META)
