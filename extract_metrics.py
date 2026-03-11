import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

print("Loading model and metadata...")
with open('model.pkl', 'rb') as f:
    obj = pickle.load(f)
    model = obj['model']
    FEATURES = obj['features']
    encoders = obj['encoders']

with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)

print("Preparing to load dataset to calculate metrics...")
# To compute R2 and RMSE we need test labels again, so we just run predictions on the test set.

def extract_num(series):
    return pd.to_numeric(series.astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')

# Dataset 1
d1 = pd.read_csv('data/car data.csv')
d1 = d1.rename(columns={'Car_Name':'name','Year':'year','Selling_Price':'selling_price_lakh','Kms_Driven':'km_driven','Fuel_Type':'fuel','Seller_Type':'seller_type','Transmission':'transmission','Owner':'owner'})
d1['selling_price'] = d1['selling_price_lakh'] * 1e5
d1['brand'] = d1['name'].str.split().str[0]
d1['mileage_kmpl'] = np.nan; d1['engine_cc'] = np.nan; d1['max_power_bhp'] = np.nan; d1['seats'] = 5.0
d1['owner'] = d1['owner'].map({0:'First Owner',1:'Second Owner',2:'Third Owner',3:'Fourth & Above Owner'}).fillna('First Owner')

# Dataset 2
d2 = pd.read_csv('data/CAR DETAILS FROM CAR DEKHO.csv')
d2['brand'] = d2['name'].str.split().str[0]
d2['mileage_kmpl'] = np.nan; d2['engine_cc'] = np.nan; d2['max_power_bhp'] = np.nan; d2['seats'] = 5.0

# Dataset 3
d3 = pd.read_csv('data/Car details v3.csv')
d3['brand'] = d3['name'].str.split().str[0]
d3['mileage_kmpl'] = extract_num(d3['mileage']); d3['engine_cc'] = extract_num(d3['engine']); d3['max_power_bhp'] = extract_num(d3['max_power'])
d3['seats'] = d3['seats'].fillna(5.0)

# Dataset 4
d4 = pd.read_csv('data/car details v4.csv')
d4 = d4.rename(columns={'Make':'brand','Price':'selling_price','Year':'year','Kilometer':'km_driven','Fuel Type':'fuel','Transmission':'transmission','Owner':'owner','Seller Type':'seller_type','Seating Capacity':'seats'})
d4['name'] = d4['brand'] + ' ' + d4['Model'].astype(str)
d4['engine_cc'] = extract_num(d4['Engine']); d4['max_power_bhp'] = extract_num(d4['Max Power']); d4['mileage_kmpl'] = np.nan
d4['owner'] = d4['owner'].astype(str).replace({'First':'First Owner','Second':'Second Owner','Third':'Third Owner','Fourth':'Fourth & Above Owner','Fourth & Above':'Fourth & Above Owner'})

KEEP = ['name','brand','year','selling_price','km_driven','fuel','transmission','owner','seller_type','mileage_kmpl','engine_cc','max_power_bhp','seats']
combined = pd.concat([d1[KEEP], d2[KEEP], d3[KEEP], d4[KEEP]], ignore_index=True)

# Clean & Feature Engineering
for c in ['fuel','transmission','owner','seller_type']:
    combined[c] = combined[c].astype(str).str.strip()
combined['fuel'] = combined['fuel'].replace({'Cng':'CNG','Lpg':'LPG','Cng + Cng':'CNG','Petrol + Cng':'CNG','Petrol + Lpg':'LPG'})
combined = combined[combined['fuel'].isin(['Petrol','Diesel','CNG','LPG','Electric'])]
combined = combined[combined['owner'].isin(['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car'])]
combined = combined.dropna(subset=['selling_price','year','km_driven','fuel','transmission','owner'])
combined = combined[combined['selling_price'] > 5000]
combined = combined[combined['year'].between(1990, 2024)]
mu, sd = combined['selling_price'].mean(), combined['selling_price'].std()
combined = combined[combined['selling_price'].between(mu - 3*sd, mu + 3*sd)]
combined['car_age'] = 2024 - combined['year']
combined['seats'] = combined['seats'].fillna(5.0)

# Encode — filter to only rows whose category values exist in each encoder's classes
for col in ['fuel','transmission','owner','seller_type','brand']:
    known = set(encoders[col].classes_)
    combined = combined[combined[col].astype(str).isin(known)]
    combined[col+'_enc'] = encoders[col].transform(combined[col].astype(str))

df_m = combined[FEATURES + ['selling_price']].copy()
for col in ['mileage_kmpl','engine_cc','max_power_bhp']:
    df_m[col] = df_m[col].fillna(df_m[col].median())
df_m = df_m.dropna()
X, y = df_m[FEATURES], df_m['selling_price']

# We calculate the train_test_split as in the training script
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict on test set
preds = model.predict(X_te)
r2 = r2_score(y_te, preds)
rmse = np.sqrt(mean_squared_error(y_te, preds))

print(f"R2: {r2:.4f} ({r2*100:.2f}%)  |  RMSE: Rs {rmse:,.0f} ({rmse/1e5:.2f} L)")

# Update meta
meta['r2_score']      = float(r2)
meta['rmse']          = float(rmse)
meta['model_name']    = 'Gradient Boosting Regressor'
meta['n_estimators']  = 300
meta['training_rows'] = int(len(X))

with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
print("meta.pkl updated successfully with metrics!")
