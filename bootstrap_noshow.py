"""
bootstrap_noshow.py
====================
Trains the no-show model and writes noshow_model.pkl + noshow_meta.pkl
without relying on any shell/subprocess. Reads the CSV files directly.
Run once:  python bootstrap_noshow.py
"""
import pickle, sys
from pathlib import Path
from io import StringIO

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

BASE = Path(__file__).parent

CATEGORICAL_FEATURES = [
    "gender", "insurance_provider", "specialization",
    "reason_for_visit", "appointment_weekday",
]
NUMERIC_FEATURES = ["patient_age"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday",
                 "Friday","Saturday","Sunday"]

# ── Load ──────────────────────────────────────────────────────────────────────
patients     = pd.read_csv(BASE / "patients.csv")
doctors      = pd.read_csv(BASE / "doctors.csv")
appointments = pd.read_csv(BASE / "appointments.csv")

# ── Feature engineering ───────────────────────────────────────────────────────
patients["date_of_birth"] = pd.to_datetime(patients["date_of_birth"], errors="coerce")
today = pd.Timestamp("2024-01-01")
patients["patient_age"] = (
    (today - patients["date_of_birth"]).dt.days / 365.25
).fillna(0).astype(int)
patients["gender"] = patients["gender"].str.strip().str.upper()

appointments["appointment_date"] = pd.to_datetime(
    appointments["appointment_date"], errors="coerce"
)
appointments["appointment_weekday"] = appointments["appointment_date"].dt.day_name()
appointments["status"]           = appointments["status"].str.strip()
appointments["reason_for_visit"] = appointments["reason_for_visit"].str.strip()

# ── Merge ─────────────────────────────────────────────────────────────────────
full_df = appointments.merge(
    patients[["patient_id","patient_age","gender","insurance_provider"]],
    on="patient_id", how="inner"
).merge(
    doctors[["doctor_id","specialization"]],
    on="doctor_id", how="inner"
)

# ── Build label (resolved rows only) ─────────────────────────────────────────
resolved = full_df[full_df["status"].isin(["Completed","No-show"])].copy()
resolved["no_show"] = (resolved["status"] == "No-show").astype(int)

print(f"Resolved rows  : {len(resolved)}  "
      f"(No-show={resolved['no_show'].sum()}, "
      f"Completed={(resolved['no_show']==0).sum()})")

# ── Meta (use full_df so dropdowns cover ALL status values) ───────────────────
meta = {col: sorted(full_df[col].dropna().unique().tolist())
        for col in CATEGORICAL_FEATURES}
meta["appointment_weekday"] = [
    d for d in WEEKDAY_ORDER if d in meta["appointment_weekday"]
]
meta["patient_age_range"] = [
    int(full_df["patient_age"].min()),
    int(full_df["patient_age"].max()),
]

# ── Train ─────────────────────────────────────────────────────────────────────
X = resolved[ALL_FEATURES]
y = resolved["no_show"]

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

print(f"Test ROC-AUC   : {auc:.4f}")
print("\nClassification report (test set):")
print(classification_report(y_test, y_pred, target_names=["Show","No-show"]))

# ── Save ──────────────────────────────────────────────────────────────────────
with open(BASE / "noshow_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(BASE / "noshow_meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("Saved → noshow_model.pkl, noshow_meta.pkl")
