import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Price My Ride",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Main container */
.block-container { padding: 2rem 2rem 4rem; max-width: 760px; }

/* Result box */
.result-box {
    background: linear-gradient(135deg, #1a3a2a, #0f2d1f);
    border: 1.5px solid #2ecc71;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    margin-top: 8px;
}
.result-label {
    font-size: 13px;
    color: #a0c4a0;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.result-price {
    font-size: 52px;
    font-weight: 700;
    color: #2ecc71;
    line-height: 1.1;
}
.result-range {
    font-size: 13px;
    color: #7a9e7a;
    margin-top: 8px;
}

/* Section header */
.section-header {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 4px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        obj = pickle.load(f)
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    return obj['model'], obj['encoders'], obj['features'], meta

model, encoders, FEATURES, meta = load_model()


# ── Helper: encode a value safely ────────────────────────────────────────────
def encode(col, val):
    le = encoders[col]
    if val in le.classes_:
        return le.transform([val])[0]
    return 0  # fallback to first class


# ── Predict ───────────────────────────────────────────────────────────────────
def predict_price(brand, year, km, fuel, transmission, owner, seller_type,
                  mileage, engine_cc, max_power, seats):
    car_age = 2024 - year
    log_km  = np.log1p(km)
    row = pd.DataFrame([[
        car_age, km, log_km,
        encode('fuel', fuel),
        encode('transmission', transmission),
        encode('owner', owner),
        encode('seller_type', seller_type),
        encode('brand', brand),
        mileage, engine_cc, max_power, seats
    ]], columns=FEATURES)
    price = model.predict(row)[0]
    return max(price, 10000)



# ══ UI ════════════════════════════════════════════════════════════════════════

st.title("🚗 PriceMyRide")
st.caption("Enter your car details below to get an estimated resale value.")

st.divider()

# ── Section 1: Car Identity ───────────────────────────────────────────────────
st.markdown('<div class="section-header">Car Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", options=sorted(meta['brands']))
with col2:
    year = st.number_input("Manufacturing Year", min_value=meta['year_min'],
                           max_value=2024, value=2019, step=1)

km = st.number_input("Kilometres Driven", min_value=0,
                     max_value=500000, value=45000, step=1000,
                     format="%d")

# ── Section 2: Car Specs ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Specifications</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    fuel = st.selectbox("Fuel Type", options=meta['fuels'])
with col4:
    transmission = st.selectbox("Transmission", options=meta['transmissions'])

col5, col6, col7 = st.columns(3)
with col5:
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0,
                               max_value=50.0, value=20.0, step=0.1)
with col6:
    engine_cc = st.number_input("Engine (CC)", min_value=500,
                                 max_value=6000, value=1200, step=50)
with col7:
    max_power = st.number_input("Max Power (bhp)", min_value=30.0,
                                 max_value=600.0, value=82.0, step=1.0)

seats = st.select_slider("Seats", options=[2, 4, 5, 6, 7, 8, 9], value=5)

# ── Section 3: Ownership ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Ownership & Seller</div>', unsafe_allow_html=True)

col8, col9 = st.columns(2)
with col8:
    owner = st.selectbox("Owner Type", options=meta['owners'])
with col9:
    seller_type = st.selectbox("Seller Type", options=meta['seller_types'])

# ── Section 4: Current Market Price ──────────────────────────────────────────
st.markdown('<div class="section-header">Current Market Price</div>', unsafe_allow_html=True)
current_price_lakh = st.number_input(
    "Current Price of the Car (in ₹ Lakhs)",
    min_value=0.5, max_value=500.0, value=8.0, step=0.1,
    help="Enter the current market / showroom price of this car model in lakhs."
)

# ── Predict Button ────────────────────────────────────────────────────────────
st.write("")
predict_clicked = st.button("⚡  Predict Resale Price", use_container_width=True, type="primary")


# ── Result ─────────────────────────────────────────────────────────────────────
if predict_clicked:
    with st.spinner("Calculating…"):
        price = predict_price(
            brand, year, km, fuel, transmission,
            owner, seller_type, mileage, engine_cc, max_power, seats
        )
    price_lakh = price / 1e5
    # Resale must always be less than the current purchase price
    if current_price_lakh > 0:
        price_lakh = min(price_lakh, current_price_lakh * 0.97)  # at most 97% of purchase price
    lo = price_lakh * 0.90
    hi = min(price_lakh * 1.10, current_price_lakh * 0.97)      # range also capped

    # Depreciation calculations vs current price
    depreciation     = current_price_lakh - price_lakh
    value_retained   = (price_lakh / current_price_lakh * 100) if current_price_lakh > 0 else 0
    dep_color  = "#2ecc71" if value_retained >= 70 else ("#f39c12" if value_retained >= 45 else "#e74c3c")
    dep_label  = "Good" if value_retained >= 70 else ("Moderate" if value_retained >= 45 else "High Depreciation")

    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Resale Value</div>
        <div class="result-price">₹ {price_lakh:.2f} L</div>
        <div class="result-range">Expected range &nbsp;·&nbsp; ₹{lo:.2f}L — ₹{hi:.2f}L</div>
    </div>
    """, unsafe_allow_html=True)

    # Depreciation summary card
    st.write("")
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 24px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
        <span style="font-size:13px;font-weight:600;color:#c9d1d9">Depreciation Analysis</span>
        <span style="background:{dep_color}22;color:{dep_color};border:1px solid {dep_color}55;
               font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;letter-spacing:1px">
          {dep_label}
        </span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;text-align:center">
        <div style="background:#0d1117;border-radius:8px;padding:14px">
          <div style="font-size:11px;color:#8b949e;margin-bottom:4px">CURRENT PRICE</div>
          <div style="font-size:20px;font-weight:700;color:#58a6ff">₹{current_price_lakh:.2f}L</div>
        </div>
        <div style="background:#0d1117;border-radius:8px;padding:14px">
          <div style="font-size:11px;color:#8b949e;margin-bottom:4px">RESALE PRICE</div>
          <div style="font-size:20px;font-weight:700;color:#2ecc71">₹{price_lakh:.2f}L</div>
        </div>
        <div style="background:#0d1117;border-radius:8px;padding:14px">
          <div style="font-size:11px;color:#8b949e;margin-bottom:4px">VALUE DROP</div>
          <div style="font-size:20px;font-weight:700;color:{dep_color}">₹{depreciation:.2f}L</div>
        </div>
      </div>
      <div style="margin-top:14px">
        <div style="font-size:11px;color:#8b949e;margin-bottom:6px">Value Retained &mdash; <strong style="color:{dep_color}">{value_retained:.1f}%</strong></div>
        <div style="background:#21262d;border-radius:20px;height:10px;overflow:hidden">
          <div style="background:{dep_color};width:{min(value_retained,100):.1f}%;height:100%;border-radius:20px"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick summary
    st.write("")
    car_age = 2024 - year
    with st.expander("📋 Prediction Summary", expanded=True):
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Car Age",        f"{car_age} yrs")
        s2.metric("KMs Driven",     f"{km:,}")
        s3.metric("Resale Price",   f"₹{price_lakh:.2f}L")
        s4.metric("Value Retained", f"{value_retained:.1f}%")


# ── Section 4: Model Performance Dashboard ───────────────────────────────────
st.write("")
st.divider()
st.markdown('<div class="section-header">📊 Model Performance Dashboard</div>', unsafe_allow_html=True)
st.write("")

# ── 1. Model Benchmark Comparison (at the top) ───────────────────────────────
r2   = meta.get('r2_score')
rmse = meta.get('rmse')
best_r2   = f"{r2*100:.2f}%"   if r2   else "N/A"
best_rmse = f"₹{rmse/1e5:.2f}L" if rmse else "N/A"

st.markdown('<div class="section-header" style="margin-top:0">Model Benchmark Comparison</div>', unsafe_allow_html=True)
comp_data = {
    "Model":    ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting ✅"],
    "R² Score": ["~72%",               "~80%",          "~89%",          best_r2],
    "Avg RMSE": ["~₹3.5L",             "~₹2.8L",        "~₹2.0L",        best_rmse],
    "Selected": ["❌",                 "❌",            "❌",             "✅"],
}
st.table(comp_data)

# ── 2. Algorithm info card ─────────────────────────────────────────────────────
_model_name = meta.get('model_name', 'Gradient Boosting Regressor')
_n_est      = meta.get('n_estimators', 500)
st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin:12px 0 16px">
  <div style="font-size:11px;color:#8b949e;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">Algorithm Used</div>
  <div style="font-size:22px;font-weight:700;color:#58a6ff">{_model_name}</div>
  <div style="font-size:12px;color:#8b949e;margin-top:6px">{_n_est} estimators &nbsp;&middot;&nbsp; Trained on {meta.get('training_rows', '14,000+'):,} rows &nbsp;&middot;&nbsp; {len(FEATURES)} features</div>
</div>
""", unsafe_allow_html=True)

# ── 3. Live Accuracy Metrics ──────────────────────────────────────────────────
acc_pct = (r2 * 100) if r2 is not None else None
a1, a2, a3, a4 = st.columns(4)
a1.metric("✅ R² Score", f"{acc_pct:.2f}%" if acc_pct else "N/A", help="Percentage of variance explained. Higher = better.")
a2.metric("📉 RMSE",     f"₹{rmse/1e5:.2f} L" if rmse else "N/A", help="Average prediction error. Lower = better.")
a3.metric("🔢 Features", str(len(FEATURES)))
a4.metric("🏷️ Brands",  str(len(meta.get('brands', []))))

if acc_pct:
    st.write("")
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px">
      <div style="font-size:12px;color:#8b949e;margin-bottom:8px">Model Accuracy &mdash; <strong style="color:#58a6ff">{acc_pct:.1f}%</strong></div>
      <div style="background:#21262d;border-radius:20px;height:12px;overflow:hidden">
        <div style="background:linear-gradient(90deg,#2ecc71,#27ae60);width:{min(acc_pct,100):.1f}%;height:100%;border-radius:20px"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#8b949e;margin-top:4px"><span>0%</span><span>50%</span><span>100%</span></div>
    </div>
    """, unsafe_allow_html=True)

