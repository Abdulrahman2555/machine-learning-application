import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from utils.preprocessing import handle_outliers, select_target, encode_and_filter, clean_features
from utils.models import build_supervised_model, build_unsupervised_model
from utils.evaluation import evaluate_supervised_split, evaluate_supervised_cv, evaluate_unsupervised
from utils.prediction import predict_form


st.title("🔴 ML App _ Supervised, Unsupervised & ANN  ")


df = None
st.header("🔷 upload file")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
btd = st.button("notes")
if btd:
    st.write("⚪ You must upload a CSV file first.")
    st.write("⚪ Then, perform data cleaning.")
    st.write("⚪ Next, choose a model based on your dataset.")
    st.write("⚪ After that, you can adjust the parameters and models freely until you reach the highest accuracy.")
    st.write("⚪ Finally, enter new values and the system will generate predictions for you.")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    original_df = df.copy()
    st.header("🔷 Preview of Data:")
    btn = st.button("show data")
    if btn:
        st.table(df.head(12))
        st.write("⚪ note that weak correlation with target will removed ")
        df_corr = df.copy()
        cat_cols = df_corr.select_dtypes(include=['object', 'string']).columns
        for col in cat_cols:
            le = LabelEncoder()
            df_corr[col] = le.fit_transform(df_corr[col].astype(str))
        f, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(f)

    df.drop_duplicates()
    df = df.dropna(thresh=len(df) // 2 + 1, axis=1)

    # ── Outliers ──
    df = handle_outliers(df)

    # ── Target ──
    X_raw, y_raw, y, y_le, y_is_encoded, target_col = select_target(df)

    # ── Encode & filter ──
    x, preprocessor, new_columns, useful_features = encode_and_filter(X_raw, y, target_col)

    # ── Clean features ──
    x, feature_selector = clean_features(x)

    # ── Validation ──
    st.header("🔷  Validation")
    val_method = st.radio("Validation method", ("Train/Test Split", "Cross-Validation"), key="val_method")
    num = st.number_input("Random state", 42, key="main_random")
    if val_method == "Train/Test Split":
        testsize = st.slider("Test size (for supervised)", 0.1, 0.5, 0.25, key="main_testsize")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, random_state=num)
    if val_method == "Cross-Validation":
        cv = st.slider("Number of CV folds", 2, 10, 5, key="main_cv")

    # ── Learning type & model ──
    st.sidebar.header("1) learning type")
    choice = st.sidebar.radio("Choose learning type", ("Supervised", "Unsupervised"))

    model = None
    sup_type = None
    transformed = None

    if choice == "Supervised":
        sup_type = st.sidebar.radio("task ", ("Regression", "Classification"))
        model = build_supervised_model(sup_type, num)

    elif choice == "Unsupervised":
        model, transformed = build_unsupervised_model(x)

    model = model

    # ── Evaluation ──
    st.divider()
    st.subheader("🔷 Model Evaluation")

    if choice == "Supervised" and val_method == "Train/Test Split":
        evaluate_supervised_split(model, sup_type, x_train, x_test, y_train, y_test)

    if choice == "Supervised" and val_method == "Cross-Validation":
        evaluate_supervised_cv(model, sup_type, x, y, cv)

    if choice == "Unsupervised":
        evaluate_unsupervised(model, x, transformed)

    # ── Prediction ──
    if choice == "Supervised":
        cv_val = cv if val_method == "Cross-Validation" else 5
        predict_form(
            original_df=original_df,
            target_col=target_col,
            preprocessor=preprocessor,
            new_columns=new_columns,
            useful_features=useful_features,
            model=model,
            y_le=y_le,
            y_is_encoded=y_is_encoded,
            val_method=val_method,
            sup_type=sup_type,
            choice=choice,
            x=x,
            y=y,
            cv=cv_val,
        )



