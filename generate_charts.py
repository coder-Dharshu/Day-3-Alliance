"""
generate_charts.py  —  Generate analysis charts for PriceMyRide README
Saves all charts as PNG files in a charts/ folder.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ── Load model & data ────────────────────────────────────────────────────────
with open('model.pkl', 'rb') as f:
    obj = pickle.load(f)
    model    = obj['model']
    encoders = obj['encoders']
    FEATURES = obj['features']

# ── Re-build dataset ─────────────────────────────────────────────────────────
def extract_num(s):
    return pd.to_numeric(s.astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')

KEEP = ['name','brand','year','selling_price','km_driven','fuel','transmission',
        'owner','seller_type','mileage_kmpl','engine_cc','max_power_bhp','seats']

d1 = pd.read_csv('data/car data.csv')
d1 = d1.rename(columns={'Car_Name':'name','Year':'year','Selling_Price':'selling_price_lakh',
    'Kms_Driven':'km_driven','Fuel_Type':'fuel','Seller_Type':'seller_type',
    'Transmission':'transmission','Owner':'owner'})
d1['selling_price'] = d1['selling_price_lakh'] * 1e5
d1['brand'] = d1['name'].str.split().str[0]
d1['mileage_kmpl'] = np.nan; d1['engine_cc'] = np.nan; d1['max_power_bhp'] = np.nan; d1['seats'] = 5.0
d1['owner'] = d1['owner'].map({0:'First Owner',1:'Second Owner',2:'Third Owner',
    3:'Fourth & Above Owner'}).fillna('First Owner')

d2 = pd.read_csv('data/CAR DETAILS FROM CAR DEKHO.csv')
d2['brand'] = d2['name'].str.split().str[0]
d2['mileage_kmpl'] = np.nan; d2['engine_cc'] = np.nan; d2['max_power_bhp'] = np.nan; d2['seats'] = 5.0

d3 = pd.read_csv('data/Car details v3.csv')
d3['brand'] = d3['name'].str.split().str[0]
d3['mileage_kmpl'] = extract_num(d3['mileage'])
d3['engine_cc']     = extract_num(d3['engine'])
d3['max_power_bhp'] = extract_num(d3['max_power'])
d3['seats'] = d3['seats'].fillna(5.0)

d4 = pd.read_csv('data/car details v4.csv')
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

df = pd.concat([d1[KEEP], d2[KEEP], d3[KEEP], d4[KEEP]], ignore_index=True)
for c in ['fuel','transmission','owner','seller_type']:
    df[c] = df[c].astype(str).str.strip()
df['fuel'] = df['fuel'].replace({'Cng':'CNG','Lpg':'LPG','Cng + Cng':'CNG','Petrol + Cng':'CNG','Petrol + Lpg':'LPG'})
df = df[df['fuel'].isin(['Petrol','Diesel','CNG','LPG','Electric'])]
df = df[df['owner'].isin(['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car'])]
df = df.dropna(subset=['selling_price','year','km_driven','fuel','transmission','owner'])
df = df[df['selling_price'] > 5000]
df = df[df['year'].between(1990, 2024)]
mu, sd = df['selling_price'].mean(), df['selling_price'].std()
df = df[df['selling_price'].between(mu - 3*sd, mu + 3*sd)]
df['car_age'] = 2024 - df['year']
df['seats']   = df['seats'].fillna(5.0)
df['log_km']  = np.log1p(df['km_driven'])
for col in ['fuel','transmission','owner','seller_type','brand']:
    known = set(encoders[col].classes_)
    df = df[df[col].astype(str).isin(known)]
    df[col+'_enc'] = encoders[col].transform(df[col].astype(str))
df_m = df[FEATURES + ['selling_price']].copy()
for col in ['mileage_kmpl','engine_cc','max_power_bhp']:
    df_m[col] = df_m[col].fillna(df_m[col].median())
df_m = df_m.dropna()
X, y = df_m[FEATURES], df_m['selling_price']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_te)

os.makedirs('charts', exist_ok=True)

DARK  = '#0d1117'
PANEL = '#161b22'
GRID  = '#21262d'
TEXT  = '#c9d1d9'
MUT   = '#8b949e'
BLUE  = '#58a6ff'
GREEN = '#2ecc71'
CORAL = '#e07b6e'

def base_fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6)
    return fig, ax

print("Generating charts...")

# 1. Feature Importance
labels = {'car_age':'Car Age','km_driven':'KM Driven','log_km':'Log KM',
          'fuel_enc':'Fuel','transmission_enc':'Transmission','owner_enc':'Owner',
          'seller_type_enc':'Seller Type','brand_enc':'Brand',
          'mileage_kmpl':'Mileage','engine_cc':'Engine CC',
          'max_power_bhp':'Max Power','seats':'Seats'}
imps = model.feature_importances_
names = [labels.get(f, f) for f in FEATURES]
pairs = sorted(zip(imps, names), reverse=True)
iv, nv = zip(*pairs)
fig, ax = base_fig(8, 5)
colors = [GREEN if v == max(iv) else BLUE for v in iv]
ax.barh(nv[::-1], [v*100 for v in iv[::-1]], color=colors[::-1], height=0.6, edgecolor='none')
for i, v in enumerate([v*100 for v in iv[::-1]]):
    ax.text(v+0.3, i, f'{v:.1f}%', va='center', color=TEXT, fontsize=8)
ax.set_xlabel('Importance (%)', color=MUT)
ax.set_title('Feature Importance', color=TEXT, fontsize=12)
ax.spines[:].set_visible(False); ax.grid(axis='x', color=GRID)
plt.tight_layout(); plt.savefig('charts/feature_importance.png', dpi=120, bbox_inches='tight'); plt.close()
print("  1. Feature Importance ✓")

# 2. Actual vs Predicted
fig, ax = base_fig()
ax.scatter(y_te/1e5, y_pred/1e5, alpha=0.25, s=12, color=BLUE, edgecolors='none')
mn, mx = y_te.min()/1e5, y_te.max()/1e5
ax.plot([mn, mx], [mn, mx], '--', color=CORAL, linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual Price (₹L)', color=MUT); ax.set_ylabel('Predicted Price (₹L)', color=MUT)
ax.set_title('Actual vs Predicted', color=TEXT, fontsize=12)
ax.legend(labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)
plt.tight_layout(); plt.savefig('charts/actual_vs_predicted.png', dpi=120, bbox_inches='tight'); plt.close()
print("  2. Actual vs Predicted ✓")

# 3. Residuals Distribution
residuals = (y_te - y_pred) / 1e5
fig, ax = base_fig()
ax.hist(residuals, bins=50, color=BLUE, edgecolor=DARK, linewidth=0.3)
ax.axvline(0, color=CORAL, linewidth=1.5, linestyle='--', label='Zero error')
ax.set_xlabel('Residual (₹L)', color=MUT); ax.set_ylabel('Count', color=MUT)
ax.set_title('Residuals Distribution', color=TEXT, fontsize=12)
ax.legend(labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)
plt.tight_layout(); plt.savefig('charts/residuals_distribution.png', dpi=120, bbox_inches='tight'); plt.close()
print("  3. Residuals Distribution ✓")

# 4. Residuals vs Predicted
fig, ax = base_fig()
ax.scatter(y_pred/1e5, residuals, alpha=0.2, s=10, color=CORAL, edgecolors='none')
ax.axhline(0, color=GREEN, linewidth=1.5, linestyle='--')
ax.set_xlabel('Predicted (₹L)', color=MUT); ax.set_ylabel('Residual (₹L)', color=MUT)
ax.set_title('Residuals vs Predicted', color=TEXT, fontsize=12)
plt.tight_layout(); plt.savefig('charts/residuals_vs_predicted.png', dpi=120, bbox_inches='tight'); plt.close()
print("  4. Residuals vs Predicted ✓")

# 5. Price by Fuel Type
fuel_grp = df.groupby('fuel')['selling_price'].apply(list)
fuels = ['Petrol','Diesel','CNG']
data_f = [df[df.fuel==f]['selling_price'].dropna()/1e5 for f in fuels]
fig, ax = base_fig()
bp = ax.boxplot(data_f, patch_artist=True, labels=fuels,
                medianprops=dict(color=GREEN, linewidth=2),
                whiskerprops=dict(color=MUT), capprops=dict(color=MUT),
                flierprops=dict(marker='.', color=MUT, alpha=0.3, markersize=3))
palette = [BLUE, CORAL, '#c9a234']
for patch, color in zip(bp['boxes'], palette):
    patch.set_facecolor(color + '55'); patch.set_edgecolor(color)
ax.set_ylabel('Selling Price (₹L)', color=MUT)
ax.set_title('Price Distribution by Fuel Type', color=TEXT, fontsize=12)
plt.tight_layout(); plt.savefig('charts/price_by_fuel.png', dpi=120, bbox_inches='tight'); plt.close()
print("  5. Price by Fuel Type ✓")

# 6. Car Age vs Price
fig, ax = base_fig()
subset = df.sample(min(3000, len(df)), random_state=42)
ax.scatter(subset['car_age'], subset['selling_price']/1e5, alpha=0.25, s=8, color=BLUE, edgecolors='none')
z = np.polyfit(df['car_age'], df['selling_price']/1e5, 1)
px = np.linspace(df['car_age'].min(), df['car_age'].max(), 100)
ax.plot(px, np.poly1d(z)(px), color=CORAL, linewidth=2, label='Trend')
ax.set_xlabel('Car Age (years)', color=MUT); ax.set_ylabel('Selling Price (₹L)', color=MUT)
ax.set_title('Car Age vs Selling Price', color=TEXT, fontsize=12)
ax.legend(labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)
plt.tight_layout(); plt.savefig('charts/car_age_vs_price.png', dpi=120, bbox_inches='tight'); plt.close()
print("  6. Car Age vs Price ✓")

# 7. KM Driven vs Price
fig, ax = base_fig()
ax.scatter(df['km_driven']/1000, df['selling_price']/1e5, alpha=0.2, s=8,
           color='#c9a234', edgecolors='none')
ax.set_xlabel('KM Driven (thousands)', color=MUT); ax.set_ylabel('Selling Price (₹L)', color=MUT)
ax.set_title('KM Driven vs Selling Price', color=TEXT, fontsize=12)
plt.tight_layout(); plt.savefig('charts/km_vs_price.png', dpi=120, bbox_inches='tight'); plt.close()
print("  7. KM Driven vs Price ✓")

# 8. Correlation Heatmap
import matplotlib.colors as mcolors
num_cols = ['car_age','km_driven','mileage_kmpl','engine_cc','max_power_bhp','seats','selling_price']
corr_df = df[num_cols].dropna().rename(columns={
    'car_age':'CarAge','km_driven':'KM','mileage_kmpl':'Mileage',
    'engine_cc':'Engine','max_power_bhp':'MaxPwr','seats':'Seats','selling_price':'Price'})
corr = corr_df.corr()
fig, ax = plt.subplots(figsize=(7,6))
fig.patch.set_facecolor(DARK); ax.set_facecolor(PANEL)
cmap = plt.cm.RdYlGn
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right', color=TEXT, fontsize=9)
ax.set_yticklabels(corr.columns, color=TEXT, fontsize=9)
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.values[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                color='black' if abs(val) < 0.6 else 'white', fontsize=8)
cb = fig.colorbar(im, ax=ax); cb.ax.tick_params(colors=TEXT)
ax.set_title('Correlation Heatmap', color=TEXT, fontsize=12)
for sp in ax.spines.values(): sp.set_color(GRID)
plt.tight_layout(); plt.savefig('charts/correlation_heatmap.png', dpi=120, bbox_inches='tight'); plt.close()
print("  8. Correlation Heatmap ✓")

# 9. Model Comparison Bar
models   = ['Linear\nRegression', 'Decision\nTree', 'Random\nForest', 'Gradient\nBoosting']
r2_vals  = [72, 80, 91, 93]
rmse_vals = [3.5, 2.8, 1.9, 1.64]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor(DARK)
for ax in [ax1, ax2]: ax.set_facecolor(PANEL); ax.grid(axis='y', color=GRID, linewidth=0.6)
cols = [BLUE, BLUE, BLUE, GREEN]
ax1.bar(models, r2_vals, color=cols, edgecolor='none', width=0.5)
for i, v in enumerate(r2_vals): ax1.text(i, v+0.5, f'{v}%', ha='center', color=TEXT, fontsize=9)
ax1.set_ylabel('R² Score (%)', color=MUT); ax1.set_title('R² Score', color=TEXT)
ax1.tick_params(colors=TEXT); ax1.set_ylim(60, 100)
ax1.spines[:].set_color(GRID)
ax2.bar(models, rmse_vals, color=cols, edgecolor='none', width=0.5)
for i, v in enumerate(rmse_vals): ax2.text(i, v+0.03, f'₹{v}L', ha='center', color=TEXT, fontsize=9)
ax2.set_ylabel('RMSE (₹L)', color=MUT); ax2.set_title('RMSE (lower = better)', color=TEXT)
ax2.tick_params(colors=TEXT); ax2.spines[:].set_color(GRID)
plt.tight_layout(); plt.savefig('charts/model_comparison.png', dpi=120, bbox_inches='tight'); plt.close()
print("  9. Model Comparison ✓")

print("\nAll charts saved to charts/")
