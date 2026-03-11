"""
train_model.py  —  PriceMyRide Model Training Pipeline
Loads all 4 datasets, benchmarks multiple algorithms, picks the best by R²,
saves model.pkl and meta.pkl.
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge

# ── 1. Helper ────────────────────────────────────────────────────────────────
def extract_num(series):
    return pd.to_numeric(series.astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')

# ── 2. Load all 4 datasets ────────────────────────────────────────────────────
print("Loading datasets...")

KEEP = ['name','brand','year','selling_price','km_driven','fuel','transmission',
        'owner','seller_type','mileage_kmpl','engine_cc','max_power_bhp','seats']

# Dataset 1: car data.csv
d1 = pd.read_csv('archive/car data.csv')
d1 = d1.rename(columns={'Car_Name':'name','Year':'year','Selling_Price':'selling_price_lakh',
    'Kms_Driven':'km_driven','Fuel_Type':'fuel','Seller_Type':'seller_type',
    'Transmission':'transmission','Owner':'owner'})
d1['selling_price'] = d1['selling_price_lakh'] * 1e5
d1['brand'] = d1['name'].str.split().str[0]
d1['mileage_kmpl'] = np.nan; d1['engine_cc'] = np.nan; d1['max_power_bhp'] = np.nan; d1['seats'] = 5.0
d1['owner'] = d1['owner'].map({0:'First Owner',1:'Second Owner',2:'Third Owner',
    3:'Fourth & Above Owner'}).fillna('First Owner')

# Dataset 2: CAR DETAILS FROM CAR DEKHO.csv
d2 = pd.read_csv('archive/CAR DETAILS FROM CAR DEKHO.csv')
d2['brand'] = d2['name'].str.split().str[0]
d2['mileage_kmpl'] = np.nan; d2['engine_cc'] = np.nan; d2['max_power_bhp'] = np.nan; d2['seats'] = 5.0

# Dataset 3: Car details v3.csv
d3 = pd.read_csv('archive/Car details v3.csv')
d3['brand'] = d3['name'].str.split().str[0]
d3['mileage_kmpl'] = extract_num(d3['mileage'])
d3['engine_cc']    = extract_num(d3['engine'])
d3['max_power_bhp'] = extract_num(d3['max_power'])
d3['seats'] = d3['seats'].fillna(5.0)

# Dataset 4: car details v4.csv
d4 = pd.read_csv('archive/car details v4.csv')
d4 = d4.rename(columns={'Make':'brand','Price':'selling_price','Year':'year',
    'Kilometer':'km_driven','Fuel Type':'fuel','Transmission':'transmission',
    'Owner':'owner','Seller Type':'seller_type','Seating Capacity':'seats'})
d4['name'] = d4['brand'] + ' ' + d4['Model'].astype(str)
d4['engine_cc']     = extract_num(d4['Engine'])
d4['max_power_bhp'] = extract_num(d4['Max Power'])
d4['mileage_kmpl']  = np.nan
d4['owner'] = d4['owner'].astype(str).replace({
    'First':'First Owner','Second':'Second Owner','Third':'Third Owner',
    'Fourth':'Fourth & Above Owner','Fourth & Above':'Fourth & Above Owner'})

combined = pd.concat([d1[KEEP], d2[KEEP], d3[KEEP], d4[KEEP]], ignore_index=True)
print(f"  Combined raw rows: {len(combined):,}")

# ── 3. Clean ──────────────────────────────────────────────────────────────────
for c in ['fuel','transmission','owner','seller_type']:
    combined[c] = combined[c].astype(str).str.strip()

combined['fuel'] = combined['fuel'].replace({
    'Cng':'CNG','Lpg':'LPG','Cng + Cng':'CNG','Petrol + Cng':'CNG','Petrol + Lpg':'LPG'})
combined = combined[combined['fuel'].isin(['Petrol','Diesel','CNG','LPG','Electric'])]
combined = combined[combined['owner'].isin([
    'First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car'])]
combined = combined.dropna(subset=['selling_price','year','km_driven','fuel','transmission','owner'])
combined = combined[combined['selling_price'] > 5000]
combined = combined[combined['year'].between(1990, 2024)]
mu, sd = combined['selling_price'].mean(), combined['selling_price'].std()
combined = combined[combined['selling_price'].between(mu - 3*sd, mu + 3*sd)]
combined['car_age'] = 2024 - combined['year']
combined['seats']   = combined['seats'].fillna(5.0)
combined['log_km']  = np.log1p(combined['km_driven'])   # log transform helps tree models
print(f"  Cleaned rows: {len(combined):,}")

# ── 4. Encode ─────────────────────────────────────────────────────────────────
cat_cols = ['fuel','transmission','owner','seller_type','brand']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined[col+'_enc'] = le.fit_transform(combined[col].astype(str))
    encoders[col] = le

FEATURES = ['car_age','km_driven','log_km','fuel_enc','transmission_enc',
            'owner_enc','seller_type_enc','brand_enc',
            'mileage_kmpl','engine_cc','max_power_bhp','seats']

df_m = combined[FEATURES + ['selling_price']].copy()
for col in ['mileage_kmpl','engine_cc','max_power_bhp']:
    df_m[col] = df_m[col].fillna(df_m[col].median())
df_m = df_m.dropna()
X, y = df_m[FEATURES], df_m['selling_price']
print(f"  Final training rows: {len(X):,}  |  Features: {len(FEATURES)}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 5. Benchmark models ───────────────────────────────────────────────────────
print("\nBenchmarking models...")

candidates = {
    "GradientBoosting (tuned)": GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.85, min_samples_leaf=5, random_state=42),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=400, max_features=0.8,
        min_samples_leaf=3, n_jobs=-1, random_state=42),

    "RandomForest (tuned)": RandomForestRegressor(
        n_estimators=400, max_features=0.7, max_depth=None,
        min_samples_leaf=3, n_jobs=-1, random_state=42),
}

results = {}
for name, mdl in candidates.items():
    mdl.fit(X_tr, y_tr)
    preds = mdl.predict(X_te)
    r2   = r2_score(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    results[name] = {'model': mdl, 'r2': r2, 'rmse': rmse}
    print(f"  {name:<30}  R²={r2*100:.2f}%   RMSE=₹{rmse/1e5:.2f}L")

# ── 6. Pick best ──────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['r2'])
best      = results[best_name]
model     = best['model']
r2        = best['r2']
rmse      = best['rmse']
print(f"\n✅ Best model: {best_name}  (R²={r2*100:.2f}%  RMSE=₹{rmse/1e5:.2f}L)")

# Retrain best model on the FULL dataset for maximum coverage
print("Retraining best model on full dataset...")
model.fit(X, y)

# ── 7. Save artifacts ─────────────────────────────────────────────────────────
META = {
    'brands':           sorted(combined['brand'].unique().tolist()),
    'fuels':            sorted(combined['fuel'].unique().tolist()),
    'transmissions':    sorted(combined['transmission'].unique().tolist()),
    'owners':           sorted(combined['owner'].unique().tolist()),
    'seller_types':     sorted(combined['seller_type'].unique().tolist()),
    'year_min':         int(combined['year'].min()),
    'year_max':         int(combined['year'].max()),
    'mileage_median':   float(df_m['mileage_kmpl'].median()),
    'engine_median':    float(df_m['engine_cc'].median()),
    'power_median':     float(df_m['max_power_bhp'].median()),
    'r2_score':         float(r2),
    'rmse':             float(rmse),
    'model_name':       best_name,
    'n_estimators':     getattr(model, 'n_estimators', 'N/A'),
    'training_rows':    int(len(X)),
}

with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'encoders': encoders, 'features': FEATURES}, f)
with open('meta.pkl', 'wb') as f:
    pickle.dump(META, f)

print(f"\n🎉 Saved model.pkl and meta.pkl")
print(f"   Model : {best_name}")
print(f"   R²    : {r2*100:.2f}%")
print(f"   RMSE  : ₹{rmse/1e5:.2f}L")
print(f"   Rows  : {len(X):,}")
print(f"   Brands: {len(META['brands'])}")
