import numpy as np
import pandas as pd
import matplotlib as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, matthews_corrcoef, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

df= pd.read_csv("data.csv")

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

df=df.drop(columns=["Unnamed: 32"])

df.dropna(axis=1, how='all', inplace=True)

df["diagnosis"]= df["diagnosis"].map({"M": 1, "B": 0})
X= df.drop("diagnosis", axis=1)
y= df["diagnosis"]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler= StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
X_test_scaled_df= pd.DataFrame(X_test_scaled, columns=X.columns)
X_test_scaled_df["diagnosis"]= y_test.values
X_test_scaled_df.to_csv("test_data.csv", index=False)

print(X_test_scaled_df.head())

models= {}

models={
    "Logistic Regression": LogisticRegression(max_iter=1000), "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "k Nearest Neighbor Classifier": KNeighborsClassifier(n_neighbors=5), "Naive Bayes Classifier": GaussianNB(),
    "Random Forrest": RandomForestClassifier(random_state=42), "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results= []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred= model.predict(X_test_scaled)
    y_prob= model.predict_proba(X_test_scaled)[:,1]

    acc= accuracy_score(y_test, y_pred)
    auc= roc_auc_score(y_test, y_prob)
    mcc= matthews_corrcoef(y_test, y_pred)
    precision= precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1= f1_score(y_test, y_pred)

    results.append([name, acc, auc, mcc, precision, recall, f1])

    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}") 
    print(f"MCC: {mcc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, f"{name.replace(' ','-')}.pkl")

results_df= pd.DataFrame(results, columns=["Model", "Accuracy", "AUC", "MCC", "Precision", "Recall", "F1-Score"])

print("\nModel Comparison")
print(results_df.sort_values(by="AUC", ascending=False))