"""
Hospital Payment Status Prediction — Model Training
=====================================================
Target   : payment_status  →  Paid | Pending | Failed  (3-class)

Features
  • amount             – bill amount (numeric)
  • payment_method     – Cash | Credit Card | Insurance | UPI (categorical)
  • treatment_type     – Chemotherapy | ECG | MRI | … (categorical, joined from treatments)
  • insurance_provider – HealthIndia | MedCare Plus | PulseSecure | WellnessCorp
                         (categorical, joined from patients)

Models compared
  1. Logistic Regression   – linear baseline
  2. Random Forest         – ensemble, handles non-linearity
  3. Gradient Boosting     – boosted ensemble, typically best on tabular data

Run:  python payment_model.py
Saves:
  payment_model.pkl      – best pipeline (highest weighted-avg F1 on test set)
  payment_meta.pkl       – unique values per categorical + amount range + model report
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

BASE = Path(__file__).parent

CATEGORICAL_FEATURES = ["payment_method", "treatment_type", "insurance_provider"]
NUMERIC_FEATURES     = ["amount"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET               = "payment_status"
CLASSES              = ["Paid", "Pending", "Failed"]


# ─── 1. Load & merge ──────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    billing    = pd.read_csv(BASE / "billing.csv")
    treatments = pd.read_csv(BASE / "treatments.csv")
    patients   = pd.read_csv(BASE / "patients.csv")

    # Clean
    billing["amount"]         = pd.to_numeric(billing["amount"], errors="coerce")
    billing["payment_status"] = billing["payment_status"].str.strip()
    billing["payment_method"] = billing["payment_method"].str.strip()
    treatments["treatment_type"] = treatments["treatment_type"].str.strip()
    patients["insurance_provider"] = patients["insurance_provider"].str.strip()

    # Join treatments → billing (via treatment_id)
    df = billing.merge(
        treatments[["treatment_id", "treatment_type"]],
        on="treatment_id", how="inner"
    )
    # Join patients → billing (via patient_id)
    df = df.merge(
        patients[["patient_id", "insurance_provider"]],
        on="patient_id", how="inner"
    )

    df = df.dropna(subset=ALL_FEATURES + [TARGET])
    return df


# ─── 2. Build preprocessing ───────────────────────────────────────────────────

def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ), CATEGORICAL_FEATURES),
    ])


# ─── 3. Train & compare ───────────────────────────────────────────────────────

def train_and_save():
    df = build_dataset()
    print(f"Training rows  : {len(df)}")
    print(f"Class balance  :\n{df[TARGET].value_counts().to_string()}\n")

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # Meta for UI
    meta = {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES}
    meta["amount_range"] = [float(df["amount"].min()), float(df["amount"].max())]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pre = make_preprocessor()

    candidates = {
        "Logistic Regression": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced",
                solver="lbfgs", random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("pre", make_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=2,
                class_weight="balanced", random_state=42
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("pre", make_preprocessor()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=5, subsample=0.8, random_state=42
            )),
        ]),
    }

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    report = {}   # name → dict of metrics
    best_name, best_f1, best_pipe = None, -1.0, None

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test) if hasattr(pipe["clf"], "predict_proba") else None

        test_f1 = f1_score(y_test, y_pred, average="weighted")
        cv_f1   = cross_val_score(pipe, X, y, cv=cv,
                                  scoring="f1_weighted", n_jobs=-1).mean()
        cr      = classification_report(y_test, y_pred, output_dict=True)

        acc = cr["accuracy"]
        report[name] = {
            "test_f1_weighted": round(test_f1, 4),
            "cv_f1_weighted":   round(cv_f1,   4),
            "accuracy":         round(acc,      4),
            "per_class": {
                cls: {
                    "precision": round(cr.get(cls, {}).get("precision", 0), 3),
                    "recall":    round(cr.get(cls, {}).get("recall",    0), 3),
                    "f1":        round(cr.get(cls, {}).get("f1-score",  0), 3),
                    "support":   int(cr.get(cls, {}).get("support",     0)),
                }
                for cls in CLASSES
            },
        }

        print(f"[{name}]")
        print(f"  Test Accuracy      : {acc:.4f}")
        print(f"  Test F1 (weighted) : {test_f1:.4f}")
        print(f"  CV F1  (weighted)  : {cv_f1:.4f}")
        print()

        if test_f1 > best_f1:
            best_f1   = test_f1
            best_name = name
            best_pipe = pipe

    print(f"Best model: {best_name}  (weighted F1 = {best_f1:.4f})")

    meta["model_report"] = report
    meta["best_model"]   = best_name

    with open(BASE / "payment_model.pkl", "wb") as f:
        pickle.dump(best_pipe, f)
    with open(BASE / "payment_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # Also save all three pipelines for optional display
    with open(BASE / "payment_all_models.pkl", "wb") as f:
        pickle.dump(candidates, f)

    print("\nSaved → payment_model.pkl, payment_meta.pkl, payment_all_models.pkl")
    return best_pipe, meta


if __name__ == "__main__":
    train_and_save()
