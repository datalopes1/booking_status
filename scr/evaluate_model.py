import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from predict import load_model
from utils import load_data
from train_model import split_data
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model with common metrics

    Args:
        model (Pipeline): trained model.
        X_test (pd.DataFrame): test set features.
        y_test (pd.Series): test set target. 

    Returns:
        tuple: 
            - y_proba: positive class probability.
            - y_pred: model predictions.
    """
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    print(f"Validation Metrics - Results\n{'-' * 25}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    print(f"{'='*15} Classification Report {'='*15}\n")
    print(classification_report(y_test, y_pred))

    return y_proba, y_pred

def cross_validate_model(model, X_val, y_val, cv_splits=5, average_f1='weighted'):
    """
    Cross validate the model for performance evaluation

    Args:
        model (Pipeline): trained model.
        X_train (pd.DataFrame): train set features.
        y_train (pd.Series): train set targets.
        cv_splits(int): total cross validation splits.
        average_f1 (str): Type of F1 score average. 

    Returns:
        tuple:
            - mean_f1: mean for the scores.
            - std_f1: standard deviation for the scores.
            - all_scores: each fold F1 score.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = make_scorer(f1_score, average=average_f1)

    scores = cross_val_score(model, X_val, y_val, cv=cv, scoring=scoring)
    
    print(f"Average F1 ({average_f1}): {scores.mean():.4f}")
    print(f"Std Dev: {scores.std():.4f}")
    
    return scores.mean(), scores.std(), scores


if __name__ == "__main__":
    
    data = pd.read_csv("data/processed/train_data.csv")
    model = load_model("models/classifier.pkl")

    X = data.drop(columns = 'booking_status', axis = 1)
    y = data['booking_status']
    
    X_train, X_test, y_train, y_test = split_data(X, y)

    mean_f1, std_f1, all_scores = cross_validate_model(model, X_train, y_train)
    y_proba, y_pred = evaluate_model(model, X_test, y_test)

