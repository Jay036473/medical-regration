import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
# Naya import errors calculate karva mate (New import for calculating errors)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# =========================
# CUSTOM CSS WITH BACKGROUND IMAGE
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.75)), 
                url("https://images.unsplash.com/photo-1555215695-3004980ad54e?q=80&w=1920&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.main-title{
    text-align:center;
    font-size:40px;
    font-weight:700;
    color:#ff4b4b;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:#cccccc;
    margin-bottom:25px;
}

.stButton>button{
    width:100%;
    background-color:#ff4b4b;
    color:white;
    font-size:18px;
    border-radius:10px;
    height:50px;
    border: none;
}

.stButton>button:hover{
    background-color:#e63946;
    color:white;
}

.result-box{
    background-color: rgba(17, 17, 17, 0.85); 
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:24px;
    color:#00ff9d;
    font-weight:bold;
    border: 1px solid #00ff9d;
}

label {
    color: white !important;
    font-weight: 600;
}

/* Customizing metric text color */
[data-testid="stMetricValue"] {
    color: #00ff9d;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚗 Car Price Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning Model using Random Forest</p>', unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(r"D:\PythonProject\csv\CRAS.csv")
    df["brand"] = df["name"].apply(lambda x: x.split()[0])
    df["car_age"] = 2026 - df["year"]
    df = df.drop(["name", "year"], axis=1)
    return df

# =========================
# TRAIN MODEL (UPDATED TO CALCULATE MULTIPLE SCORES)
# =========================
@st.cache_resource
def train_model(df):
    X = df.drop("selling_price", axis=1)
    y = np.log1p(df["selling_price"])

    cat_cols = ["fuel", "seller_type", "transmission", "owner", "brand"]

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
    model.fit(X_train, y_train)
    
    # Calculate predictions on the test set
    y_pred_log = model.predict(X_test)
    
    # Convert predictions and actual values back from log scale
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred_log)
    
    # Calculate metrics
    r2 = r2_score(y_test_actual, y_pred_actual)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    
    # Convert INR errors to Euros (Assuming 1 Euro = 90 INR)
    eur_rate = 90
    metrics = {
        "r2": r2,
        "mae_euro": mae / eur_rate,
        "rmse_euro": rmse / eur_rate
    }
    
    return model, metrics

df = load_data()
model, metrics = train_model(df)

# =========================
# DISPLAY MODEL PERFORMANCE SCORES (NEW SECTION)
# =========================
st.markdown('<p class="subtitle" style="font-size:22px; color:#ffffff;">🎯 Model Performance Metrics</p>', unsafe_allow_html=True)

# Using st.columns to display scores side-by-side
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.metric(label="R² Score (Accuracy)", value=f"{metrics['r2']:.2%}")
with col_m2:
    st.metric(label="Mean Absolute Error", value=f"€ {metrics['mae_euro']:,.2f}")
with col_m3:
    st.metric(label="Root Mean Squared Error", value=f"€ {metrics['rmse_euro']:,.2f}")

st.write("---")

# =========================
# DATA PREVIEW
# =========================
with st.expander("🔍 Preview Dataset"):
    st.dataframe(df.head(50), use_container_width=True)
    st.write("Total Rows:", df.shape[0])

st.write("---")

# =========================
# INPUT SECTION
# =========================
st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Car Year", 2000, 2026, value=2015)
    fuel = st.selectbox("Fuel Type", df["fuel"].unique())
    transmission = st.selectbox("Transmission", df["transmission"].unique())

with col2:
    km_driven = st.number_input("Kilometers Driven", 0, 500000, step=5000)
    seller = st.selectbox("Seller Type", df["seller_type"].unique())
    owner = st.selectbox("Owner Type", df["owner"].unique())

brand = st.selectbox("Car Brand", sorted(df["brand"].unique()))

# =========================
# PREDICTION
# =========================
if st.button("Predict Car Price"):
    car_age = 2026 - year
    input_data = pd.DataFrame({
        "km_driven": [km_driven], "fuel": [fuel], "seller_type": [seller],
        "transmission": [transmission], "owner": [owner], "brand": [brand], "car_age": [car_age]
    })

    log_prediction = model.predict(input_data)
    price_inr = max(0, int(np.expm1(log_prediction[0])))
    eur_rate = 90
    price_euro = price_inr / eur_rate

    st.markdown(
        f'<div class="result-box">Estimated Car Price: € {price_euro:,.2f}</div>',
        unsafe_allow_html=True
    )

# =========================
# 6 DATA CHARTS
# =========================
st.write("---")
st.markdown('<p class="main-title" style="font-size:32px; color:#00ff9d;">📊 Market Insights</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Explore the dataset through visualizations</p>', unsafe_allow_html=True)

# Using columns to put charts side-by-side
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig1 = px.histogram(df, x="selling_price", nbins=40, title="1. Distribution of Selling Prices",
                        template="plotly_dark", color_discrete_sequence=['#ff4b4b'], height=550)
    st.plotly_chart(fig1, use_container_width=True)

    fig3 = px.box(df, x="fuel", y="selling_price", title="3. Price by Fuel Type",
                  template="plotly_dark", color="fuel", height=550)
    st.plotly_chart(fig3, use_container_width=True)

    top_brands = df.groupby('brand')['selling_price'].mean().reset_index().sort_values(by='selling_price', ascending=False).head(10)
    fig5 = px.bar(top_brands, x='brand', y='selling_price', title="5. Top 10 Most Expensive Brands (Avg)",
                  template="plotly_dark", color='brand', height=550)
    st.plotly_chart(fig5, use_container_width=True)

with chart_col2:
    fig2 = px.scatter(df, x="car_age", y="selling_price", title="2. Car Age vs. Selling Price",
                      template="plotly_dark", color="transmission", opacity=0.7, height=550)
    st.plotly_chart(fig2, use_container_width=True)

    fig4 = px.pie(df, names="transmission", title="4. Market Share by Transmission",
                  template="plotly_dark", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu, height=550)
    st.plotly_chart(fig4, use_container_width=True)

    fig6 = px.scatter(df, x="km_driven", y="selling_price", title="6. KM Driven vs. Selling Price",
                      template="plotly_dark", color="fuel", opacity=0.6, height=550)
    st.plotly_chart(fig6, use_container_width=True)
