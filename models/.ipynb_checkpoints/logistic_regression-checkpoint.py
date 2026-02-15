import numpy as np
import pandas as pd
import matplotlib as plt
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, matthews_corrcoef, precision_score, recall_score, f1_score

def train_logistic(X, y):
    model= LogisticRegression(max_iter=1000)
    model.fit(X,y)
    joblib.dump(model, "saved_models/logistic_model.pkl")
    return model

model_path= os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "saved_models", "logistic_model.py")

def predict_logistic(X):
    model= joblib.load(model_path)
    prediction= model.predict(X)
    return prediction