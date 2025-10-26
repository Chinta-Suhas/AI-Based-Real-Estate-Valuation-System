1. Create a new GitHub repository and push these files to the repository root. Also push your `project.ipynb` and `india_housing_prices.csv` and any trained model like `real_estate_model.pkl`.


2. Ensure the repository contains at least:
- `streamlit_app.py`
- `requirements.txt`
- `india_housing_prices.csv` (optional if you plan to upload at runtime)


3. Go to https://share.streamlit.io and sign in with your GitHub account.


4. Click **New app** → choose the repo, branch (e.g., `main`) and the file path `streamlit_app.py`.


5. Click **Deploy**. Streamlit will install dependencies from `requirements.txt` and start your app.


Notes:
- If you want automatic training on deploy, include `india_housing_prices.csv` in the repo root. The app trains a quick RandomForest and saves `model.pkl` on first run.
- If you already have a trained model, upload it as `real_estate_model.pkl` (or `model.pkl`) to the repo to skip training.
- To view logs, open the app on Streamlit Cloud and click Logs. If training times out, consider training locally and uploading the saved model file.