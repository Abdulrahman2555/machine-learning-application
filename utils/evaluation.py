import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    mean_absolute_error, r2_score, median_absolute_error,
    mean_squared_error, mean_absolute_percentage_error, max_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, make_scorer,
)
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def evaluate_supervised_split(model, sup_type, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if sup_type == "Classification":
        cm = confusion_matrix(y_test, y_pred)
        fg, ax = plt.subplots(figsize=(15, 15))
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("actual")
        ax.set_xticks(range(len(np.unique(y_test))))
        ax.set_yticks(range(len(np.unique(y_test))))
        sns.heatmap(cm, annot=True, ax=ax)
        st.pyplot(fg)

        st.text("Classification report:")
        st.code(classification_report(y_test, y_pred))

    else:
        mse   = mean_squared_error(y_test, y_pred)
        mae   = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2    = r2_score(y_test, y_pred)
        mape  = mean_absolute_percentage_error(y_test, y_pred) * 100
        maxer = max_error(y_test, y_pred)

        st.metric("R2 (test accuracy)", f"{r2:.4f}")
        st.metric("MSE",  f"{mse:.4f}")
        st.metric("MAE",  f"{mae:.4f}")
        st.metric("MedAE", f"{medae:.4f}")
        st.metric("MAPE (%)", f"{mape:.2f}")
        st.metric("Max Error", f"{maxer:.4f}")

        fig_fit, ax_fit = plt.subplots()
        ax_fit.scatter(y_test, y_pred, alpha=0.7, c="g", label="Data points")
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax_fit.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect fit (y = x)")
        ax_fit.set_xlabel("Actual")
        ax_fit.set_ylabel("Predicted")
        ax_fit.set_title("Actual vs Predicted")
        ax_fit.legend()
        st.pyplot(fig_fit)

        resid = y_test - y_pred
        fig, ax = plt.subplots()
        ax.scatter(y_pred, resid)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Predicted")
        st.pyplot(fig)

        if isinstance(model, MLPRegressor) and hasattr(model, "loss_curve_"):
            fig2, ax2 = plt.subplots()
            ax2.plot(model.loss_curve_)
            ax2.set_title("Training Loss (MLP)")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Loss")
            st.pyplot(fig2)


def evaluate_supervised_cv(model, sup_type, x, y, cv):
    if sup_type == "Classification":
        scorers = {
            "accuracy":  make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average="weighted", zero_division=0),
            "recall":    make_scorer(recall_score,    average="weighted", zero_division=0),
            "f1":        make_scorer(f1_score,        average="weighted", zero_division=0),
        }
        if len(np.unique(y)) == 2:
            scorers["roc_auc"] = make_scorer(roc_auc_score, needs_proba=True)

        scores = cross_validate(model, x, y, cv=cv, scoring=scorers)
        st.metric("Accuracy (test accuracy)", f"{scores['test_accuracy'].mean():.4f}")
        st.metric("Precision", f"{scores['test_precision'].mean():.4f}")
        st.metric("Recall",    f"{scores['test_recall'].mean():.4f}")
        st.metric("F1 Score",  f"{scores['test_f1'].mean():.4f}")
        if "test_roc_auc" in scores:
            st.metric("ROC AUC", f"{scores['test_roc_auc'].mean():.4f}")

    else:
        scorers = {
            "r2 (test accuracy)": make_scorer(r2_score),
            "mse":                make_scorer(mean_squared_error),
            "mae":                make_scorer(mean_absolute_error),
            "medae":              make_scorer(median_absolute_error),
            "mape":               make_scorer(mean_absolute_percentage_error),
            "max_error":          make_scorer(max_error),
        }
        scores = cross_validate(model, x, y, cv=cv, scoring=scorers)
        st.metric("MSE",       f"{scores['test_mse'].mean():.4f}")
        st.metric("MAE",       f"{scores['test_mae'].mean():.4f}")
        st.metric("MedAE",     f"{scores['test_medae'].mean():.4f}")
        st.metric("R2",        f"{scores['test_r2 (test accuracy)'].mean():.4f}")
        st.metric("MAPE (%)",  f"{scores['test_mape'].mean() * 100:.2f}")
        st.metric("Max Error", f"{scores['test_max_error'].mean():.4f}")


def evaluate_unsupervised(model, x, transformed):
    with st.spinner("Fitting clustering model..."):
        model.fit(x)
        labels = model.labels_

    st.success("Clustering done ✅")
    st.write("Cluster label counts:")
    st.write(pd.Series(labels).value_counts().sort_index())

    try:
        transformed = model.transform(x)
        if len(np.unique(labels)) > 1 and transformed.shape[0] > len(np.unique(labels)):
            sil = silhouette_score(transformed, labels)
            st.metric("Silhouette Score", f"{sil:.4f}")
    except Exception as e:
        st.warning(f"Silhouette not available: {e}")

    try:
        pca_vis = PCA(n_components=2)
        reduced = pca_vis.fit_transform(transformed)
        fig, ax = plt.subplots()
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10")
        ax.set_title("PCA (2D) of Clusters")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)
    except Exception:
        st.info("PCA plot not available.")
