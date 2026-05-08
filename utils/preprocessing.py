import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Binarizer, Normalizer
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectKBest
from sklearn.compose import ColumnTransformer


def handle_outliers(df):
    numerical = [col for col in df.columns if df[col].dtype != "O"]

    st.header("🔷 Outliers Handling")

    if st.button("Show Outliers "):
        fig, axes = plt.subplots(len(numerical), 1, figsize=(8, 4 * len(numerical)))
        if len(numerical) == 1:
            axes = [axes]
        for i, var in enumerate(numerical):
            sns.boxplot(x=df[var], ax=axes[i])
            axes[i].set_title(f"{var} (Before Removal)")
        st.pyplot(fig)

    if st.button("Remove Outliers "):
        df_clean = df.copy()
        for var in numerical:
            Q1 = df_clean[var].quantile(0.25)
            Q3 = df_clean[var].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df_clean[
                (df_clean[var] < (Q1 - 1.5 * IQR)) | (df_clean[var] > (Q3 + 1.5 * IQR))
            ]
            df_clean.drop(outliers.index, inplace=True)

        fig, axes = plt.subplots(len(numerical), 1, figsize=(8, 4 * len(numerical)))
        if len(numerical) == 1:
            axes = [axes]
        for i, var in enumerate(numerical):
            sns.boxplot(x=df_clean[var], ax=axes[i])
            axes[i].set_title(f"{var} (After Removal)")
        st.pyplot(fig)

        st.success("Outliers removed successfully!")
        df = df_clean

    return df


def select_target(df):
    st.header("🔷 target column :")
    target_col = st.selectbox("⚪ enter Target Column:", df.columns)
    st.success(f"✅ Target column set to: {target_col}")

    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 10:
        st.warning("⟵ Please use a Regression model")
    else:
        st.warning("⟵ Please use a Classification model")

    y_le = None
    y_is_encoded = False

    if y_raw.dtype == "object" or y_raw.dtype.name == "category":
        y_le = LabelEncoder()
        y = y_le.fit_transform(y_raw)
        y_is_encoded = True
    else:
        y = y_raw.values

    return X_raw, y_raw, y, y_le, y_is_encoded, target_col


def encode_and_filter(X_raw, y, target_col):
    text_columns = X_raw.select_dtypes(include=['object', 'string']).columns
    numeric_columns = X_raw.select_dtypes(exclude=['object', 'string']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(text_columns)),
            ("num", "passthrough", list(numeric_columns))
        ]
    )

    preprocessor.fit(X_raw)
    X_transformed = preprocessor.transform(X_raw)

    try:
        new_columns = preprocessor.get_feature_names_out()
        new_columns = list(new_columns)
    except Exception:
        if len(text_columns) > 0 and hasattr(preprocessor.named_transformers_["cat"], "get_feature_names_out"):
            ohe = preprocessor.named_transformers_["cat"]
            if hasattr(ohe, "categories_") and len(ohe.categories_) > 0:
                ohe_features = ohe.get_feature_names_out(list(text_columns))
                new_columns = list(ohe_features) + list(numeric_columns)
            else:
                new_columns = list(numeric_columns)
        else:
            new_columns = list(numeric_columns)

    X = pd.DataFrame(X_transformed, columns=new_columns, index=X_raw.index)

    corr_matrix = X.join(pd.Series(y, name=target_col)).corr()
    target_corr = corr_matrix[target_col].abs()
    useful_features = target_corr[target_corr >= 0.05].index
    useful_features = useful_features.drop(target_col, errors="ignore")

    x = X[useful_features]

    return x, preprocessor, new_columns, useful_features


def clean_features(x):
    st.header("🔷 cleaning data before choosing model")
    st.write(" ⚪ Keep in mind that cleaning the data is optional, not required ")

    col1, col2, col3 = st.columns(3)

    num_imputer = None
    with col1:
        st.markdown("1)) replacing missing values ")
        numeric_imp = st.selectbox("Numeric imputation", ["mean", "median", "most frequent"], key="main_num_imp")
        if numeric_imp == "mean":
            num_imputer = SimpleImputer(strategy="mean")
        elif numeric_imp == "median":
            num_imputer = SimpleImputer(strategy="median")
        else:
            num_imputer = SimpleImputer(strategy="most_frequent")
        x = num_imputer.fit_transform(x)

    scall_data = None
    with col2:
        st.markdown("2)) scalling data")
        scaler_choice = st.selectbox(
            "Scaling method",
            ["StandardScaler", "MinMaxScaler", "MaxAbsScaler", "Binarizer", "Normalizer"],
            key="scaler_choice"
        )
        if scaler_choice == "StandardScaler":
            scall_data = StandardScaler()
        elif scaler_choice == "MinMaxScaler":
            scall_data = MinMaxScaler(feature_range=(0, 1))
        elif scaler_choice == "MaxAbsScaler":
            normmz = st.selectbox("normalizer with", ["l1", "l2", "max"])
            scall_data = MaxAbsScaler(norm=normmz)
        elif scaler_choice == "Binarizer":
            binarizerr = st.number_input("threshold", min_value=0, max_value=1000, value=1, step=0.1)
            scall_data = Binarizer(threshold=binarizerr)
        elif scaler_choice == "Normalizer":
            normm = st.selectbox("normalizer with", ["l1", "l2", "max"])
            scall_data = Normalizer(norm=normm)
        x = scall_data.fit_transform(x)

    feature_selector = None
    with col3:
        st.markdown("3)) reduce features")

        rfeature = st.selectbox(
            "Choose feature selection method",
            ["SelectPercentile", "SelectKBest"],
            key="feature_method"
        )

        score_func_choice = st.selectbox(
            "Select score function",
            ["f_classif", "chi2"],
            key="score_func"
        )

        if score_func_choice == "f_classif":
            score_func = f_classif
        elif score_func_choice == "chi2":
            score_func = chi2

        if rfeature == "SelectPercentile":
            percentile_value = st.number_input("Select percentile", min_value=1, max_value=100, value=10, step=1, key="percentile_val")
            feature_selector = SelectPercentile(score_func=score_func, percentile=percentile_value)
        elif rfeature == "SelectKBest":
            k_value = st.number_input("Select number of features (k)", min_value=1, max_value=100, value=5, step=1, key="k_val")
            feature_selector = SelectKBest(score_func=score_func, k=k_value)
        feature_selector = feature_selector

    return x, feature_selector
