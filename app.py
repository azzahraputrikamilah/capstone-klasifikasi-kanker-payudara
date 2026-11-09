import os
import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

st.set_page_config(page_title="Halaman Modelling", layout="wide")
st.write("""
# Welcome to my machine learning dashboard

This dashboard created by : Azzahra Putri Kamilah, Malika Azzahra, Aliya Amara""")

add_selectitem = st.sidebar.selectbox("Want to open about?", ("Breast Cancer!",))

MODEL_PATH = "generate_breast_cancer.pkl"
DATA_PATH = "data.csv"
IMAGE_PATH = "breast-cancer.jpg"

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

@st.cache_data
def load_reference(path):
    return pd.read_csv(path, index_col=0)

def breast_cancer():
    st.write("""
    This app predicts the **Breast Cancer** diagnosis (Malignant or Benign)

    Data obtained from the [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) by UCIML.
    """)
    loaded_model = load_model(MODEL_PATH)
    fitur = list(loaded_model.feature_names_in_)   # fitur & urutannya sama dengan saat training
    ref = load_reference(DATA_PATH)

    st.sidebar.header('User Input Features:')
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        kurang = [f for f in fitur if f not in input_df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ada di file CSV: {kurang}")
            st.stop()
        input_df = input_df[fitur]
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            data = {}
            for f in fitur:
                lo, hi, med = float(ref[f].min()), float(ref[f].max()), float(ref[f].median())
                data[f] = st.sidebar.slider(f, lo, hi, med, step=(hi - lo) / 200)
            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_features()

    if os.path.exists(IMAGE_PATH):
        img = Image.open(IMAGE_PATH)
        st.image(img, width=500)

    if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        prediction = loaded_model.predict(df)
        proba = loaded_model.predict_proba(df)[:, 1]
        st.subheader('Prediction: ')
        hasil = ['Benign (Jinak)' if p == 0 else 'Malignant (Ganas)' for p in prediction]
        with st.spinner('Wait for it...'):
            time.sleep(2)
            for h, p in zip(hasil, proba):
                st.success(f"Prediction of this app is {h} (probabilitas ganas: {p:.2%})")

def main():
    if add_selectitem == "Breast Cancer!":
        breast_cancer()

if __name__ == "__main__":
    main()
