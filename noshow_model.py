"""
Hospital Appointment No-Show Prediction — Model Training
=========================================================
Features
  • patient_age         – derived from date_of_birth
  • gender              – M / F
  • insurance_provider  – categorical
  • specialization      – doctor specialization (categorical)
  • reason_for_visit    – categorical
  • appointment_weekday – Mon-Sun (derived from appointment_date)

Target  : no_show  (1 = No-show, 0 = Completed / Cancelled)
          Training uses only resolved appointments (Completed + No-show).
          Cancelled rows are excluded — the outcome is unknown.
          Scheduled rows are excluded from training but their feature
          values are stored in model_meta so the live form works.

Run:  python noshow_model.py
Saves:  noshow_model.pkl  – trained sklearn Pipeline
        noshow_meta.pkl   – dict of unique category values for UI dropdowns
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

BASE = Path(__file__).parent

CATEGORICAL_FEATURES = [
    "gender",
    "insurance_provider",
    "specialization",
    "reason_for_visit",
    "appointment_weekday",
]
NUMERIC_FEATURES = ["patient_age"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]


# ─── 1. Load & merge ──────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    patients     = pd.read_csv(BASE / "patients.csv")
    doctors      = pd.read_csv(BASE / "doctors.csv")
    appointments = pd.read_csv(BASE / "appointments.csv")

    # ── Clean patients ──
    patients["date_of_birth"] = pd.to_datetime(patients["date_of_birth"], errors="coerce")
    today = pd.Timestamp("2024-01-01")
    patients["patient_age"] = (
        (today - patients["date_of_birth"]).dt.days / 365.25
    ).fillna(0).astype(int)
    patients["gender"] = patients["gender"].str.strip().str.upper()

    # ── Clean appointments ──
    appointments["appointment_date"] = pd.to_datetime(
        appointments["appointment_date"], errors="coerce"
    )
    appointments["appointment_weekday"] = appointments["appointment_date"].dt.day_name()
    appointments["status"] = appointments["status"].str.strip()
    appointments["reason_for_visit"] = appointments["reason_for_visit"].str.strip()

    # ── Join ──
    df = appointments.merge(
        patients[["patient_id", "patient_age", "gender", "insurance_provider"]],
        on="patient_id", how="inner"
    ).merge(
        doctors[["doctor_id", "specialization"]],
        on="doctor_id", how="inner"
    )

    # ── Build label — keep only resolved rows ──
    resolved = df[df["status"].isin(["Completed", "No-show"])].copy()
    resolved["no_show"] = (resolved["status"] == "No-show").astype(int)

    return resolved, df   # (training df, full df for meta)


# ─── 2. Train ─────────────────────────────────────────────────────────────────

def train_and_save():
    resolved, full_df = build_dataset()
    print(f"Resolved rows  : {len(resolved)}  "
          f"(No-show={resolved['no_show'].sum()}, "
          f"Completed={(resolved['no_show']==0).sum()})")

    X = resolved[ALL_FEATURES]
    y = resolved["no_show"]

    # ── Collect meta from FULL dataset so UI dropdowns cover all values ──
    meta = {
        col: sorted(full_df[col].dropna().unique().tolist())
        for col in CATEGORICAL_FEATURES
    }
    # Guarantee weekday order
    meta["appointment_weekday"] = [
        d for d in WEEKDAY_ORDER if d in meta["appointment_weekday"]
    ]
    meta["patient_age_range"] = [
        int(full_df["patient_age"].min()),
        int(full_df["patient_age"].max()),
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ), CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    cv  = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

    print(f"\nTest ROC-AUC   : {auc:.4f}")
    print(f"5-fold CV AUC  : {cv.mean():.4f} ± {cv.std():.4f}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=["Show", "No-show"]))

    # ── Save artefacts ──
    with open(BASE / "noshow_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(BASE / "noshow_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("\nSaved → noshow_model.pkl, noshow_meta.pkl")
    return model, meta


if __name__ == "__main__":
    train_and_save()
