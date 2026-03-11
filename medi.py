import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Medical Cost Predictor", page_icon="⚕️", layout="wide")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.stApp {
background: linear-gradient(rgba(0,0,0,0.80), rgba(0,0,0,0.80)),
url("https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7");
background-size: cover;
background-position: center;
background-attachment: fixed;
}

.main-title{
text-align:center;
font-size:40px;
font-weight:700;
color:#00e5ff;
}

.subtitle{
text-align:center;
font-size:18px;
color:#cccccc;
margin-bottom:25px;
}

.result-box{
background-color: rgba(17,17,17,0.85);
padding:20px;
border-radius:10px;
text-align:center;
font-size:24px;
color:#00ff9d;
font-weight:bold;
border:1px solid #00ff9d;
}

label{
color:white !important;
font-weight:600;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚕️ Medical Insurance Cost Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning Model using Random Forest</p>', unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():

    df = pd.read_csv(r"D:\PythonProject\csv\insurance.csv")

    eur_rate = 0.92
    df["charges"] = df["charges"] * eur_rate

    return df


# =========================
# TRAIN MODEL
# =========================
@st.cache_resource
def train_model(df):

    X = df.drop("charges", axis=1)
    y = np.log1p(df["charges"])

    cat_cols = ["sex","smoker","region"]

    preprocessor = ColumnTransformer(
        transformers=[("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)],
        remainder="passthrough"
    )

    model = Pipeline([
        ("preprocessor",preprocessor),
        ("regressor",RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42))
    ])

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    model.fit(X_train,y_train)

    return model


df = load_data()
model = train_model(df)

# =========================
# DATA PREVIEW
# =========================
with st.expander("🔍 Preview Dataset"):
    st.dataframe(df.head(50),use_container_width=True)
    st.write("Total Records:",df.shape[0])

# =========================
# STATISTICAL SUMMARY
# =========================
st.write("---")
st.subheader("📊 Dataset Statistical Summary")

summary = df.describe().T
summary = summary[["count","mean","std","min","25%","50%","75%","max"]]

st.dataframe(summary,use_container_width=True)

# =========================
# INPUT SECTION
# =========================
st.write("---")
st.subheader("Enter Patient Details")

col1,col2,col3 = st.columns(3)

with col1:
    age = st.number_input("Age",18,100,30)
    sex = st.selectbox("Gender",df["sex"].unique())

with col2:
    bmi = st.number_input("BMI",15.0,60.0,25.5)
    smoker = st.selectbox("Smoker",df["smoker"].unique())

with col3:
    children = st.number_input("Children",0,10,0)
    region = st.selectbox("Region",df["region"].unique())

# =========================
# PREDICTION
# =========================
if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame({
        "age":[age],
        "sex":[sex],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker],
        "region":[region]
    })

    log_pred = model.predict(input_data)

    cost = max(0,int(np.expm1(log_pred[0])))

    st.markdown(
    f'<div class="result-box">Estimated Medical Insurance Bill: € {cost:,.2f}</div>',
    unsafe_allow_html=True
    )

# =========================
# DATA CHARTS
# =========================
st.write("---")
st.markdown('<p class="main-title" style="font-size:32px;color:#00ff9d;">📊 Health Data Insights</p>',unsafe_allow_html=True)

chart_col1,chart_col2 = st.columns(2)

with chart_col1:

    fig1 = px.histogram(df,x="charges",nbins=40,
                        title="1. Distribution of Medical Charges (€)",
                        template="plotly_dark")
    st.plotly_chart(fig1,use_container_width=True)

    fig3 = px.box(df,x="smoker",y="charges",
                  title="3. Insurance Costs by Smoker Status",
                  template="plotly_dark",
                  color="smoker")
    st.plotly_chart(fig3,use_container_width=True)

    fig5 = px.scatter(df,x="bmi",y="charges",
                      title="5. BMI vs Medical Charges",
                      template="plotly_dark",
                      color="smoker")
    st.plotly_chart(fig5,use_container_width=True)

with chart_col2:

    fig2 = px.scatter(df,x="age",y="charges",
                      title="2. Age vs Insurance Charges",
                      template="plotly_dark",
                      color="smoker")
    st.plotly_chart(fig2,use_container_width=True)

    region_cost = df.groupby("region")["charges"].mean().reset_index()

    fig4 = px.bar(region_cost,x="region",y="charges",
                  title="4. Average Charges by Region",
                  template="plotly_dark",
                  color="region")
    st.plotly_chart(fig4,use_container_width=True)

    fig6 = px.box(df,x="children",y="charges",
                  title="6. Charges by Number of Children",
                  template="plotly_dark",
                  color="children")
    st.plotly_chart(fig6,use_container_width=True)

# =========================
# GENDER CHART
# =========================
st.write("---")

gender_cost = df.groupby("sex")["charges"].mean().reset_index()

fig7 = px.bar(gender_cost,
              x="sex",
              y="charges",
              title="7. Average Medical Charges: Male vs Female",
              template="plotly_dark",
              color="sex",
              text_auto=True)

st.plotly_chart(fig7,use_container_width=True)