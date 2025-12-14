import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Load Model & Columns-
@st.cache_resource
def load_artifacts():
    with open("notebooks/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("notebooks/X_train_columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, columns

model, model_columns = load_artifacts()

# Centered Title & Subtitle
st.markdown(
    "<h1 style='text-align:center;color:#1E90FF;'>🏠 House Price Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; font-size:16px; color:white;'>"
    "Enter the house details below to estimate the selling price"
    "</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<hr style='border: 1px solid #1E90FF;'>",  
    unsafe_allow_html=True
)

# Input Section
st.markdown(
    "<h3 style='color:#1E90FF;'>🏢Property Details</h3>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    OverallQual = st.number_input(
        "Overall Quality (1–10)", 1, 10, 6
    )

    GrLivArea = st.number_input(
        "Above Ground Living Area (sq ft)", 300, 4000, 1500
    )

    TotalBsmtSF = st.number_input(
        "Total Basement Area (sq ft)", 0, 3000, 800
    )

    FirstFlrSF = st.number_input(
        "1st Floor Area (sq ft)", 300, 3000, 900
    )

    SecondFlrSF = st.number_input(
        "2nd Floor Area (sq ft)", 0, 3000, 500
    )

with col2:
    GarageCars = st.number_input(
        "Garage Capacity (max 5 cars)", 0, 5, 2
    )

    YearBuilt = st.number_input(
        "Year Built", 1870, 2010, 2000
    )

    FullBath = st.number_input(
        "Full Bathrooms", 0, 4, 2
    )

    TotRmsAbvGrd = st.number_input(
        "Total Rooms Above Ground", 2, 12, 6
    )

    Fireplaces = st.number_input(
        "Number of Fireplaces", 0, 5, 1
    )

BsmtFinSF1 = st.number_input(
    "Basement Finished Area (sq ft)", 0, 2000, 500
)

# Input DataFrame
input_data = {
    "OverallQual": OverallQual,
    "GrLivArea": GrLivArea,
    "TotalBsmtSF": TotalBsmtSF,
    "1stFlrSF": FirstFlrSF,
    "2ndFlrSF": SecondFlrSF,
    "GarageCars": GarageCars,
    "YearBuilt": YearBuilt,
    "FullBath": FullBath,
    "TotRmsAbvGrd": TotRmsAbvGrd,
    "Fireplaces": Fireplaces,
    "BsmtFinSF1": BsmtFinSF1
}

input_df = pd.DataFrame(input_data, index=[0])

# Match Training Columns
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_columns]

# Prediction
st.markdown(
    "<hr style='border: 1px solid #1E90FF;'>",
    unsafe_allow_html=True
)


if st.button("Predict House Price"):
    try:
        log_price = model.predict(input_df)[0]
        price = np.exp(log_price)

        st.success(f"Estimated House Price: $ {price:,.0f}")


    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; color:var(--muted); font-size:13px; padding:20px; 
                border-top:1px solid rgba(255,255,255,0.05); margin-top:30px;'>
        Developed by Mrunal • House Price Prediction
    </div>
    """,
    unsafe_allow_html=True
)