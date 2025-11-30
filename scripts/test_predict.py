from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'real_estate_model.pkl'
X_TEST = ROOT / 'X_test.csv'

if not META.exists():
    print('Metadata file not found:', META)
    raise SystemExit(1)

m = joblib.load(META)
print('Loaded metadata keys:', list(m.keys()))
model = m.get('model')
fn = m.get('feature_names')
print('Feature count from metadata:', len(fn) if fn else 0)

if not X_TEST.exists():
    print('X_test.csv not found at', X_TEST)
    raise SystemExit(1)

X = pd.read_csv(X_TEST)
print('X_test shape:', X.shape)

if fn:
    missing = [c for c in fn if c not in X.columns]
    print('Missing features in X_test:', missing)
    Xr = X.reindex(columns=fn).fillna(0)
else:
    # fallback to numeric
    Xr = X.select_dtypes(include=['number']).fillna(0)

print('Prepared X shape for prediction:', Xr.shape)

try:
    pred = model.predict(Xr.iloc[[0]])
    print('Sample prediction:', float(pred[0]))
except Exception as e:
    print('Prediction failed:', e)
    raise
