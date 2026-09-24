"""
Hospital Analytics — Backend Data Pipeline
Loads, cleans, merges all 5 tables and exposes pre-computed
DataFrames + KPI dictionaries consumed by the dashboard.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent


# ── 1. LOAD ───────────────────────────────────────────────────────────────────

def load_raw():
    patients     = pd.read_csv(BASE / "patients.csv")
    doctors      = pd.read_csv(BASE / "doctors.csv")
    appointments = pd.read_csv(BASE / "appointments.csv")
    treatments   = pd.read_csv(BASE / "treatments.csv")
    billing      = pd.read_csv(BASE / "billing.csv")
    return patients, doctors, appointments, treatments, billing


# ── 2. CLEAN ──────────────────────────────────────────────────────────────────

def clean_patients(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardise gender — dataset mixes F/M labels with actual name genders; keep as-is
    df["date_of_birth"]    = pd.to_datetime(df["date_of_birth"],    errors="coerce")
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["gender"] = df["gender"].str.strip().str.upper()
    # Derive age
    today = pd.Timestamp("2024-01-01")
    df["age"] = ((today - df["date_of_birth"]).dt.days / 365.25).fillna(0).astype(int)
    # Age bands
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 35, 50, 65, 120],
        labels=["0-18", "19-35", "36-50", "51-65", "65+"],
        right=True,
    )
    df.drop_duplicates(subset="patient_id", inplace=True)
    return df


def clean_doctors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["years_experience"] = pd.to_numeric(df["years_experience"], errors="coerce")
    df.drop_duplicates(subset="doctor_id", inplace=True)
    return df


def clean_appointments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["appointment_date"] = pd.to_datetime(df["appointment_date"], errors="coerce")
    df["status"] = df["status"].str.strip()
    df["reason_for_visit"] = df["reason_for_visit"].str.strip()
    df["month"] = df["appointment_date"].dt.to_period("M").astype(str)
    df.drop_duplicates(subset="appointment_id", inplace=True)
    return df


def clean_treatments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["treatment_date"] = pd.to_datetime(df["treatment_date"], errors="coerce")
    df.drop_duplicates(subset="treatment_id", inplace=True)
    return df


def clean_billing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["bill_date"] = pd.to_datetime(df["bill_date"], errors="coerce")
    df["payment_status"] = df["payment_status"].str.strip()
    df["payment_method"] = df["payment_method"].str.strip()
    df.drop_duplicates(subset="bill_id", inplace=True)
    return df


# ── 3. MERGE (master fact table) ─────────────────────────────────────────────

def build_master(patients, doctors, appointments, treatments, billing) -> pd.DataFrame:
    # appointments ← patients
    df = appointments.merge(
        patients[["patient_id", "first_name", "last_name", "gender", "age", "age_group",
                  "insurance_provider", "registration_date"]],
        on="patient_id", how="left"
    )
    # ← doctors
    df = df.merge(
        doctors[["doctor_id", "first_name", "last_name", "specialization",
                 "years_experience", "hospital_branch"]].rename(
            columns={"first_name": "doc_first", "last_name": "doc_last"}
        ),
        on="doctor_id", how="left"
    )
    # ← treatments
    df = df.merge(
        treatments[["appointment_id", "treatment_id", "treatment_type", "cost"]],
        on="appointment_id", how="left"
    )
    # ← billing
    df = df.merge(
        billing[["treatment_id", "bill_id", "amount", "payment_method", "payment_status"]],
        on="treatment_id", how="left"
    )
    df["doctor_name"]  = df["doc_first"].fillna("") + " " + df["doc_last"].fillna("")
    df["patient_name"] = df["first_name"].fillna("") + " " + df["last_name"].fillna("")
    return df


# ── 4. KPIs ───────────────────────────────────────────────────────────────────

def compute_kpis(master: pd.DataFrame, billing_clean: pd.DataFrame) -> dict:
    total_patients   = master["patient_id"].nunique()
    total_doctors    = master["doctor_id"].nunique()
    total_appts      = master["appointment_id"].nunique()
    completed_appts  = (master["status"] == "Completed").sum()
    completion_rate  = round(completed_appts / total_appts * 100, 1) if total_appts else 0
    no_show_rate     = round((master["status"] == "No-show").sum() / total_appts * 100, 1)
    total_revenue    = billing_clean.loc[billing_clean["payment_status"] == "Paid", "amount"].sum()
    pending_revenue  = billing_clean.loc[billing_clean["payment_status"] == "Pending", "amount"].sum()
    avg_treatment_cost = master["cost"].mean()
    return {
        "total_patients":      total_patients,
        "total_doctors":       total_doctors,
        "total_appointments":  total_appts,
        "completion_rate":     completion_rate,
        "no_show_rate":        no_show_rate,
        "total_revenue":       round(total_revenue, 2),
        "pending_revenue":     round(pending_revenue, 2),
        "avg_treatment_cost":  round(avg_treatment_cost, 2),
    }


# ── 5. AGGREGATIONS ───────────────────────────────────────────────────────────

def appt_by_status(master):
    return master.groupby("status")["appointment_id"].nunique().reset_index(name="count")

def appt_by_month(master):
    return master.groupby("month")["appointment_id"].nunique().reset_index(name="count")

def appt_by_reason(master):
    return master.groupby("reason_for_visit")["appointment_id"].nunique().reset_index(name="count")

def revenue_by_method(billing):
    paid = billing[billing["payment_status"] == "Paid"]
    return paid.groupby("payment_method")["amount"].sum().reset_index(name="revenue")

def billing_status_dist(billing):
    return billing.groupby("payment_status")["amount"].sum().reset_index(name="amount")

def treatment_cost_by_type(master):
    return master.groupby("treatment_type")["cost"].mean().reset_index(name="avg_cost")

def appts_per_doctor(master):
    df = master.groupby(["doctor_name", "specialization"])["appointment_id"].nunique().reset_index(name="appointments")
    return df.sort_values("appointments", ascending=False)

def patients_by_insurance(master):
    return master.drop_duplicates("patient_id").groupby("insurance_provider")["patient_id"].count().reset_index(name="patients")

def patients_by_age_group(master):
    return master.drop_duplicates("patient_id").groupby("age_group", observed=True)["patient_id"].count().reset_index(name="patients")

def monthly_revenue(master):
    paid = master[master["payment_status"] == "Paid"].copy()
    paid["month"] = pd.to_datetime(paid["appointment_date"]).dt.to_period("M").astype(str)
    return paid.groupby("month")["amount"].sum().reset_index(name="revenue")

def top_treatment_types(master):
    return master.groupby("treatment_type")["appointment_id"].nunique().reset_index(name="count").sort_values("count", ascending=False)


# ── 6. DEEP ANALYSIS AGGREGATIONS ─────────────────────────────────────────────

def noshow_by_weekday(master):
    """No-show rate (%) per weekday, ordered Mon–Sun."""
    df = master.copy()
    df["weekday"] = pd.to_datetime(df["appointment_date"], errors="coerce").dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    grp = df.groupby("weekday")["status"].agg(
        total="count",
        noshow=lambda s: (s == "No-show").sum()
    ).reset_index()
    grp["noshow_rate"] = (grp["noshow"] / grp["total"] * 100).round(1)
    grp["weekday"] = pd.Categorical(grp["weekday"], categories=order, ordered=True)
    return grp.sort_values("weekday").reset_index(drop=True)


def revenue_by_spec_method(master):
    """Paid revenue grouped by specialization × payment method."""
    paid = master[master["payment_status"] == "Paid"].copy()
    return (
        paid.groupby(["specialization", "payment_method"])["amount"]
        .sum().reset_index(name="revenue")
    )


def doctor_efficiency(master):
    """Per-doctor: appointment count, avg treatment cost, paid revenue generated."""
    df = master.copy()
    df["_paid_amount"] = df["amount"].where(df["payment_status"] == "Paid", 0)
    agg = df.groupby(["doctor_name", "specialization"]).agg(
        appointments=("appointment_id", "nunique"),
        avg_cost=("cost", "mean"),
        paid_revenue=("_paid_amount", "sum"),
    ).reset_index()
    agg["avg_cost"] = agg["avg_cost"].round(2)
    agg["paid_revenue"] = agg["paid_revenue"].round(2)
    return agg.sort_values("appointments", ascending=False).reset_index(drop=True)


def pipeline_funnel(master, billing):
    """Appointment → Treatment → Billed → Paid funnel counts."""
    total_appts      = master["appointment_id"].nunique()
    treated          = master["treatment_id"].notna().sum()
    billed           = billing["bill_id"].nunique()
    paid             = (billing["payment_status"] == "Paid").sum()
    return pd.DataFrame({
        "stage": ["Appointments", "Treated", "Billed", "Paid"],
        "count": [total_appts, treated, billed, paid],
    })


def bill_heatmap(master):
    """Avg bill amount: insurance_provider (row) × treatment_type (col)."""
    paid = master[master["payment_status"].notna()].copy()
    pivot = (
        paid.groupby(["insurance_provider", "treatment_type"])["amount"]
        .mean().round(0).reset_index(name="avg_amount")
    )
    return pivot


def appt_by_hour(master):
    """Appointment count by hour of day (0–23)."""
    df = master.copy()
    df["hour"] = pd.to_datetime(df["appointment_time"], format="%H:%M:%S", errors="coerce").dt.hour
    return df.groupby("hour")["appointment_id"].nunique().reset_index(name="count")


def patient_retention(master):
    """Count of patients with 1 visit vs 2+ visits."""
    visit_counts = master.groupby("patient_id")["appointment_id"].nunique()
    single  = (visit_counts == 1).sum()
    repeat  = (visit_counts > 1).sum()
    return pd.DataFrame({
        "category": ["Single Visit", "Repeat Visitor"],
        "patients": [int(single), int(repeat)],
    })


def payment_failure_rate(billing):
    """Failure rate (%) per payment method."""
    grp = billing.groupby("payment_method")["payment_status"].agg(
        total="count",
        failed=lambda s: (s == "Failed").sum()
    ).reset_index()
    grp["failure_rate"] = (grp["failed"] / grp["total"] * 100).round(1)
    return grp.sort_values("failure_rate", ascending=False).reset_index(drop=True)


def cost_vs_billed(master):
    """Treatment cost vs bill amount — one row per billed appointment."""
    df = master[master["amount"].notna() & master["cost"].notna()].copy()
    return df[["appointment_id", "treatment_type", "specialization",
               "cost", "amount", "payment_status"]].drop_duplicates("appointment_id")


def status_month_heatmap(master):
    """Count of each appointment status per month (for calendar heatmap)."""
    df = master.copy()
    df["month"] = pd.to_datetime(df["appointment_date"], errors="coerce").dt.to_period("M").astype(str)
    pivot = (
        df.groupby(["month", "status"])["appointment_id"]
        .nunique().reset_index(name="count")
    )
    return pivot


# ── 7. MAIN ENTRY POINT ───────────────────────────────────────────────────────

def get_all_data():
    patients_raw, doctors_raw, appts_raw, treats_raw, billing_raw = load_raw()

    patients   = clean_patients(patients_raw)
    doctors    = clean_doctors(doctors_raw)
    appts      = clean_appointments(appts_raw)
    treats     = clean_treatments(treats_raw)
    billing    = clean_billing(billing_raw)

    master = build_master(patients, doctors, appts, treats, billing)
    kpis   = compute_kpis(master, billing)

    return {
        "master":              master,
        "kpis":                kpis,
        "appt_status":         appt_by_status(master),
        "appt_month":          appt_by_month(master),
        "appt_reason":         appt_by_reason(master),
        "revenue_method":      revenue_by_method(billing),
        "billing_status":      billing_status_dist(billing),
        "treatment_cost":      treatment_cost_by_type(master),
        "doctor_appts":        appts_per_doctor(master),
        "insurance_dist":      patients_by_insurance(master),
        "age_group_dist":      patients_by_age_group(master),
        "monthly_revenue":     monthly_revenue(master),
        "treatment_type_dist": top_treatment_types(master),
        "patients":            patients,
        "doctors":             doctors,
        # ── deep analysis ──
        "noshow_weekday":      noshow_by_weekday(master),
        "rev_spec_method":     revenue_by_spec_method(master),
        "doctor_efficiency":   doctor_efficiency(master),
        "pipeline_funnel":     pipeline_funnel(master, billing),
        "bill_heatmap":        bill_heatmap(master),
        "appt_hour":           appt_by_hour(master),
        "patient_retention":   patient_retention(master),
        "payment_failure":     payment_failure_rate(billing),
        "cost_vs_billed":      cost_vs_billed(master),
        "status_month_hm":     status_month_heatmap(master),
    }


if __name__ == "__main__":
    data = get_all_data()
    print("✅ Pipeline OK")
    print(f"   Master shape : {data['master'].shape}")
    print(f"   KPIs         : {data['kpis']}")
