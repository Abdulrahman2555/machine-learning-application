import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPRegressor


def predict_form(original_df, target_col, preprocessor, new_columns, useful_features,
                 model, y_le, y_is_encoded, val_method, sup_type, choice, x, y, cv):

    st.divider()
    st.subheader("🔷 Predict on Custom Input")
    st.caption("📝 Provide feature values as in the original dataset.")

    with st.form("predict_form"):
        input_data = {}

        for col in original_df.drop(columns=[target_col]).columns:
            if pd.api.types.is_numeric_dtype(original_df[col]):
                default_val = float(original_df[col].median()) if pd.notnull(original_df[col].median()) else 0.0
                val = st.number_input(f"{col}", value=default_val)
            else:
                opts = sorted(original_df[col].astype(str).dropna().unique().tolist())
                opts = opts if opts else ["missing"]
                val = st.selectbox(f"{col}", opts)

            input_data[col] = val

        submitted = st.form_submit_button("🚀 Predict")

        if submitted:
            input_df = pd.DataFrame([input_data])

            if val_method == "Cross-Validation":
                cv_preds = cross_val_predict(model, x, y, cv=cv)
                last_pred = cv_preds[-1]

                if y_is_encoded and y_le is not None:
                    last_pred_decoded = y_le.inverse_transform([int(last_pred)])[0]
                else:
                    last_pred_decoded = last_pred

                st.success(f"🎯 Cross-Validation Prediction: {last_pred_decoded}")

            else:
                df_temp = preprocessor.transform(input_df)
                df_temp = pd.DataFrame(df_temp, columns=new_columns)
                df_temp = df_temp[useful_features.drop(target_col, errors="ignore")]

                pred_encoded = model.predict(df_temp)

                if choice == "Supervised" and sup_type == "Classification" and y_is_encoded and y_le is not None:
                    try:
                        pred_decoded = y_le.inverse_transform(pred_encoded.astype(int))
                    except Exception:
                        pred_decoded = y_le.inverse_transform(pred_encoded)

                    st.success(f"🎉 Prediction: {pred_decoded[0]}")

                    if hasattr(model["model"], "predict_proba"):
                        proba = model.predict_proba(df_temp)
                        class_labels = model["model"].classes_

                        if y_is_encoded:
                            try:
                                class_labels = y_le.inverse_transform(class_labels.astype(int))
                            except Exception:
                                class_labels = y_le.inverse_transform(class_labels)

                        st.write("🔎 Class probabilities:")
                        st.write(pd.DataFrame(proba, columns=[f"class_{c}" for c in class_labels]))

                elif choice == "Supervised" and sup_type == "Classification":
                    st.success(f"🎉 Prediction: {pred_encoded[0]}")

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(df_temp)
                        class_labels = model.classes_
                        st.write("🔎 Class probabilities:")
                        st.write(pd.DataFrame(proba, columns=[f"class_{c}" for c in class_labels]))

                else:
                    pred_value = pred_encoded[0]
                    st.success(f"🎉 Prediction: {pred_value}")

                    if isinstance(model, MLPRegressor) and hasattr(model, "loss_curve_"):
                        fig2, ax2 = plt.subplots()
                        ax2.plot(model.loss_curve_)
                        ax2.set_title("Training Loss (MLP)")
                        ax2.set_xlabel("Iteration")
                        ax2.set_ylabel("Loss")
                        st.pyplot(fig2)
