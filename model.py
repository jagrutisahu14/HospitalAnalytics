"""
Hospital Bill Amount Prediction — Model Training
=================================================
Features used
  • patient_age          – derived from date_of_birth
  • gender               – M / F
  • insurance_provider   – categorical
  • specialization       – doctor's specialization (categorical)
  • treatment_type       – categorical
  • reason_for_visit     – categorical

Target  : amount  (bill amount from billing.csv)

Run:  python model.py
Saves:  bill_model.pkl   – trained pipeline (preprocessing + regressor)
        model_meta.pkl   – dict of unique category values for the UI form
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

BASE = Path(__file__).parent


# ─── 1. Load & merge ──────────────────────────────────────────────────────────

def build_training_data() -> pd.DataFrame:
    patients     = pd.read_csv(BASE / "patients.csv")
    doctors      = pd.read_csv(BASE / "doctors.csv")
    appointments = pd.read_csv(BASE / "appointments.csv")
    treatments   = pd.read_csv(BASE / "treatments.csv")
    billing      = pd.read_csv(BASE / "billing.csv")

    # Derive patient age
    patients["date_of_birth"] = pd.to_datetime(patients["date_of_birth"], errors="coerce")
    today = pd.Timestamp("2024-01-01")
    patients["patient_age"] = ((today - patients["date_of_birth"]).dt.days / 365.25).fillna(0).astype(int)
    patients["gender"] = patients["gender"].str.strip().str.upper()

    # Join: appointments ← patients
    df = appointments.merge(
        patients[["patient_id", "patient_age", "gender", "insurance_provider"]],
        on="patient_id", how="inner"
    )
    # ← doctors
    df = df.merge(
        doctors[["doctor_id", "specialization"]],
        on="doctor_id", how="inner"
    )
    # ← treatments
    df = df.merge(
        treatments[["appointment_id", "treatment_id", "treatment_type"]],
        on="appointment_id", how="inner"
    )
    # ← billing  (target: amount)
    df = df.merge(
        billing[["treatment_id", "amount"]],
        on="treatment_id", how="inner"
    )

    # Drop rows with missing target or key features
    feature_cols = ["patient_age", "gender", "insurance_provider",
                    "specialization", "treatment_type", "reason_for_visit"]
    df = df.dropna(subset=feature_cols + ["amount"])
    return df


# ─── 2. Build & train pipeline ────────────────────────────────────────────────

CATEGORICAL_FEATURES = [
    "gender",
    "insurance_provider",
    "specialization",
    "treatment_type",
    "reason_for_visit",
]
NUMERIC_FEATURES = ["patient_age"]


def train_and_save():
    df = build_training_data()
    print(f"Training rows : {len(df)}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["amount"]

    # Collect unique values per category for the UI dropdowns
    meta = {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES}
    meta["patient_age_range"] = [int(df["patient_age"].min()), int(df["patient_age"].max())]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline([
        ("pre", preprocessor),
        ("reg", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"Test MAE      : ₹{mae:,.2f}")
    print(f"Test R²       : {r2:.4f}")

    # ── Save ──
    with open(BASE / "bill_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(BASE / "model_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("Saved  → bill_model.pkl, model_meta.pkl")
    return model, meta


if __name__ == "__main__":
    train_and_save()
