# Real-State-Price-Prediction-
First Assignment of Machine Learning Course - 2024 
## Real Estate Price Prediction

A lightweight project for predicting real estate prices using a single Jupyter notebook.

### Project Structure
- `Real_State_price_prediction.ipynb`: main workflow covering data loading, exploration, feature engineering, model training, and evaluation.

### Requirements
You can run the notebook with either conda or pip. Python 3.9+ is recommended.

Common packages used in this type of workflow:
- numpy, pandas
- scikit-learn
- matplotlib, seaborn
- jupyter or jupyterlab

If your environment is missing any of these, see the setup instructions below.

### Quick Start
1) Clone or copy this folder to your machine.
2) Start Jupyter and open the notebook.

```bash
# Option A: conda (recommended)
conda create -n realestate-ml python=3.10 -y
conda activate realestate-ml
conda install -y numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook

# Option B: pip + venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Open `Real_State_price_prediction.ipynb` and run the cells top-to-bottom.

### Data
- Place your dataset file(s) in this folder or update the notebook's data path accordingly.
- Expected format is typically a CSV with columns such as features (e.g., bedrooms, bathrooms, sqft, location encodings) and a target column (e.g., price). Adjust column names in the notebook as needed.

### Reproducibility
- Set a random seed in the notebook where models are initialized or where train/test splits occur (e.g., `random_state=42`).
- Use the same environment each time (conda env or pip venv as above).

### Typical Workflow in the Notebook
1) Exploratory Data Analysis (EDA)
2) Data cleaning and preprocessing (handling missing values, encoding categoricals, scaling)
3) Feature selection/engineering
4) Train/validation split
5) Model training (e.g., Linear Regression, Random Forest, Gradient Boosting)
6) Evaluation (e.g., RMSE, MAE, R^2)
7) Basic model comparison and selection

### Saving Models (optional)
If you add persistence, you can export the trained model with joblib:
```python
import joblib
joblib.dump(model, "model.joblib")
```
Then load it later with:
```python
model = joblib.load("model.joblib")
```

### Troubleshooting
- If plots don’t render, ensure the notebook uses an inline backend (e.g., `%matplotlib inline`).
- If a CSV isn’t found, check out https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction.
- If imports fail, confirm the environment is active and packages are installed.

### License
This project is provided as-is for educational purposes. Add your preferred license if needed.


