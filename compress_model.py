import joblib
import os
from pathlib import Path

ROOT = Path(__file__).parent
orig = ROOT / 'real_estate_model.pkl'
backup = ROOT / 'real_estate_model_backup.pkl'
compressed = ROOT / 'real_estate_model_compressed.pkl'

print('orig exists:', orig.exists())
if not orig.exists():
    raise SystemExit('real_estate_model.pkl not found')

# backup
if not backup.exists():
    print('Creating backup...')
    os.rename(orig, backup)
else:
    print('Backup already exists')

# load from backup and dump compressed
print('Loading model from backup...')
model = joblib.load(backup)
print('Dumping compressed model (compress=9)...')
joblib.dump(model, compressed, compress=9)

# show sizes
def size_mb(p):
    return p.stat().st_size/1024/1024

print(f'Backup size: {size_mb(backup):.2f} MB')
print(f'Compressed size: {size_mb(compressed):.2f} MB')

# replace original path
print('Replacing original with compressed file...')
if orig.exists():
    orig.unlink()
os.rename(compressed, orig)
print('Replacement done. New file:', orig)
print(f'New file size: {size_mb(orig):.2f} MB')
