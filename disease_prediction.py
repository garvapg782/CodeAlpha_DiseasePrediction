"""
CodeAlpha Machine Learning Internship - Task 4
Disease Prediction from Medical Data (Breast Cancer Wisconsin dataset)

Trains and compares SVM, Logistic Regression, Random Forest and XGBoost,
evaluates them, tunes the best candidate and saves plots + the final model.
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    """Load the dataset as a DataFrame. Target: 1 = malignant, 0 = benign."""
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    # sklearn encodes 0 = malignant, 1 = benign. Flip so 1 means "disease present".
    df["target"] = 1 - df["target"]
    return df


def explore(df):
    print("Shape:", df.shape)
    print("Missing values:", int(df.isnull().sum().sum()))
    print("\nClass distribution (1 = malignant, 0 = benign):")
    print(df["target"].value_counts())

    plt.figure(figsize=(4, 4))
    sns.countplot(x="target", data=df, palette=["#4C9F70", "#D1495B"])
    plt.xticks([0, 1], ["Benign", "Malignant"])
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/class_distribution.png", dpi=150)
    plt.close()

    corr = df.corr()["target"].drop("target").abs().sort_values(ascending=False).head(10)
    plt.figure(figsize=(7, 4))
    sns.barplot(x=corr.values, y=corr.index, color="#3A7CA5")
    plt.title("Top 10 Features Correlated with Diagnosis")
    plt.xlabel("|Correlation|")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/top_correlations.png", dpi=150)
    plt.close()


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            eval_metric="logloss", random_state=RANDOM_STATE,
        ),
    }


def evaluate(models, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, probas = [], {}
    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        probas[name] = proba
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1-Score": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
            "CV ROC-AUC (5-fold)": cv_auc,
        })
    results = pd.DataFrame(rows).set_index("Model").round(4)
    return results, probas


def plot_roc(probas, y_test):
    plt.figure(figsize=(6, 5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150)
    plt.close()


def tune_best(X_train, y_train):
    """Grid search over the SVM, a strong performer on this dataset."""
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=RANDOM_STATE))])
    grid = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.001],
        "clf__kernel": ["rbf", "linear"],
    }
    search = GridSearchCV(
        pipe, grid, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("\nBest SVM params:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def final_report(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print("\n=== Tuned SVM: Test Set Report ===")
    print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Tuned SVM)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()


def feature_importance(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE).fit(X_train, y_train)
    imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
    plt.figure(figsize=(7, 4))
    sns.barplot(x=imp.values, y=imp.index, color="#D1495B")
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()
    print("\nTop 5 important features:\n", imp.head(5).round(4))


def main():
    df = load_data()
    explore(df)

    X, y = df.drop(columns="target"), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = build_models()
    results, probas = evaluate(models, X_train, X_test, y_train, y_test)
    print("\n=== Model Comparison (Test Set) ===")
    print(results.to_string())
    results.to_csv(f"{OUT_DIR}/model_comparison.csv")
    plot_roc(probas, y_test)

    best = tune_best(X_train, y_train)
    final_report(best, X_test, y_test)
    feature_importance(X_train, y_train)

    joblib.dump(best, f"{OUT_DIR}/best_model.joblib")
    print(f"\nSaved model and plots to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
