#Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, matthews_corrcoef, precision_score, recall_score, f1_score

st.set_page_config(page_title='BITS Assignment-2 (2025AA05653) of Machine Learning ', layout='wide')
st.title('Wisconsin Breast Cancer Classsification App')

@st.cache_resource
def load_models():
    models= {}
    models["Logistic Regression"]= joblib.load("models/Logistic-Regression.pkl")
    models["Decision Tree Classifier"]= joblib.load("models/Decision-Tree-Classifier.pkl")
    models["k Nearest Neighbor Classifier"]= joblib.load("models/k-Nearest-Neighbor-Classifier.pkl")
    models["Naive Bayes Classifier"]= joblib.load("models/Naive-Bayes-Classifier.pkl")
    models["Random Forrest"]= joblib.load("models/Random-Forrest.pkl")
    models["XGBoost"]= joblib.load("models/XGBoost.pkl")

    return models

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")

models= load_models()
scaler= load_scaler()

st.sidebar.header("Model Selection")
selected_model_name= st.sidebar.selectbox(
    "Choose Classification Model", options=list(models.keys())
)

model= models[selected_model_name]

st.sidebar.header("Upload Test Data")
uploaded_file= st.sidebar.file_uploader("Choose TEST data (only CSV file)", type='csv')

if uploaded_file is not None:
    df= pd.read_csv(uploaded_file)
    st.subheader("Dataset sample")
    st.write(df.head())

    if "id" in df.columns:
        df= df.drop(columns=["id"])
    
    if "diagnosis" not in df.columns:
        st.error("Dataset must contain 'diagnosis' column.")
    else:
        if df["diagnosis"].dtype== object:
            y= df["diagnosis"].map({"M": 1, "B": 0})
        else:
            y= df["diagnosis"]

        X= df.drop(columns=["diagnosis"])

        # mean= np.abs(X.mean()).mean()
        # std= np.abs(X.std()-1).mean()
        # if mean < 0.1 and std<0.1:
        #     st.info("Test data is scaled")
        #     X_scaled= X.values
        # else:
        #     st.info("Test Data scaling using scaler")
        #     X_scaled= scaler.transform(X)

        y_pred= model.predict(X)
        y_prob= model.predict_proba(X)[:,1]

        st.subheader(f"Evaluation Metrics- {selected_model_name}")

        accuracy= accuracy_score(y, y_pred)
        precision= precision_score(y, y_pred)
        recall= recall_score(y, y_pred)
        f1= f1_score(y, y_pred)
        auc= roc_auc_score(y, y_prob)
        mcc= matthews_corrcoef(y, y_pred)

        col1, col2, col3, col4, col5, col6= st.columns(6)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("AUC: ", f"{auc:.4f}")
        col6.metric("MCC:", f"{mcc:.4f}")

        st.markdown("--")
        left_space, center_col, right_space= st.columns([1,3,1])

        with center_col:

            st.subheader("Confusion Matrix")
            cm= confusion_matrix(y, y_pred)
            fig, ax= plt.subplots()
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Reds", xticklabels=["Benign(0)", "Malignant(1)"],
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.markdown("Classification Report")

            report= classification_report(y, y_pred, target_names=["Benign", "Malignant"], output_dict=True)

            report_df= pd.DataFrame(report).transpose()

            st.subheader("Classification Report")
            st.dataframe(report_df, use_container_width= True)
        
        st.markdown("--")
        st.subheader("Comparison Table: All models")

        def all_metrics(m):
            preds= m.predict(X)
            prob= m.predict_proba(X)[:,1] if hasattr(m, "predict_proba") else m.decision_function(X)
            return[accuracy_score(y,preds), roc_auc_score(y, prob), precision_score(y, preds), recall_score(y,preds), f1_score(y,preds), matthews_corrcoef(y,preds)]
        
        all_results=[]
        for model_name, loaded_model in models.items():
            scores= all_metrics(loaded_model)
            all_results.append([model_name]+scores)

        comp_df= pd.DataFrame(all_results, columns=["Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])
        comp_df= comp_df.round(4)

        st.dataframe(comp_df, use_container_width=True)

else:
    st.warning("Please upload the Winsconsin Breast Cancer CSV dataset.")




