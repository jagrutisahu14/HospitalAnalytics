"""
Hospital Revenue Forecasting — Model Training
==============================================
Approach
  • Aggregate all-source monthly revenue from billing (all statuses included as billed;
    a "paid-only" series is also built for collected revenue).
  • Engineer calendar features: month_num, quarter, is_q_end, lag_1, lag_2,
    rolling_mean_3, trend (row index).
  • Compare 3 forecasting strategies:
      1. Linear Regression on trend features    – interpretable baseline
      2. Holt-Winters Exponential Smoothing     – classic time-series method
      3. Random Forest Regressor on lag features– captures non-linear patterns
  • All models evaluated on a held-out 3-month window (last 3 months of 2023).
  • 6-month forward forecast (Jan–Jun 2024) produced by all three models.
  • Best model (lowest MAE on hold-out) is flagged in saved meta.

Run:  python revenue_model.py
Saves:
  revenue_meta.pkl  – full dict: historical series, forecasts, metrics, model details
  (No .pkl model file needed — forecasts are pre-computed and stored in meta.)
"""

from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
FORECAST_MONTHS = 6   # how many months ahead to forecast
TEST_MONTHS     = 3   # hold-out window for evaluation


# ─── 1. Build monthly revenue series ─────────────────────────────────────────

def build_series() -> pd.DataFrame:
    billing    = pd.read_csv(BASE / "billing.csv")
    treatments = pd.read_csv(BASE / "treatments.csv")

    billing["bill_date"] = pd.to_datetime(billing["bill_date"], errors="coerce")
    billing["amount"]    = pd.to_numeric(billing["amount"],   errors="coerce")
    billing["payment_status"] = billing["payment_status"].str.strip()
    treatments["treatment_type"] = treatments["treatment_type"].str.strip()
    treatments["cost"]   = pd.to_numeric(treatments["cost"],  errors="coerce")

    # Join treatment_type and cost onto billing
    df = billing.merge(
        treatments[["treatment_id", "treatment_type", "cost"]],
        on="treatment_id", how="left"
    )

    df["month"] = df["bill_date"].dt.to_period("M")

    # Total billed revenue per month (all statuses)
    total_monthly = (
        df.groupby("month")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "total_billed"})
    )
    # Collected (Paid only)
    paid_monthly = (
        df[df["payment_status"] == "Paid"]
        .groupby("month")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "paid_revenue"})
    )
    # Bill count
    count_monthly = (
        df.groupby("month")["bill_id"]
        .count()
        .reset_index()
        .rename(columns={"bill_id": "bill_count"})
    )

    series = (
        total_monthly
        .merge(paid_monthly,  on="month", how="left")
        .merge(count_monthly, on="month", how="left")
    )
    series = series.sort_values("month").reset_index(drop=True)
    series["paid_revenue"] = series["paid_revenue"].fillna(0)
    series["month_str"]    = series["month"].astype(str)
    series["month_dt"]     = series["month"].dt.to_timestamp()
    series["month_num"]    = range(1, len(series) + 1)
    return series


# ─── 2. Feature engineering (lag-based, for Random Forest) ───────────────────

def add_lag_features(df: pd.DataFrame, target: str = "total_billed") -> pd.DataFrame:
    """Add lag and rolling features; requires month_num, month_dt already present."""
    df = df.copy()
    df["lag_1"]         = df[target].shift(1)
    df["lag_2"]         = df[target].shift(2)
    df["rolling_3"]     = df[target].shift(1).rolling(3).mean()
    df["month_of_year"] = df["month_dt"].dt.month
    df["quarter"]       = df["month_dt"].dt.quarter
    df["is_q_end"]      = (df["month_dt"].dt.month % 3 == 0).astype(int)
    df["trend"]         = df["month_num"]
    return df


# ─── 3. Model 1 — Linear Regression (trend-only) ─────────────────────────────
# With only 12 months of data, restricting to a single 'trend' (integer index)
# avoids overfitting from month_of_year / quarter / is_q_end dummies.

def forecast_linear(train: pd.DataFrame, n_future: int, target: str):
    """
    Fit a simple linear trend on month_num and return:
      (fitted_values, future_forecasts, fitted_model)
    Both arrays are clipped to >= 0.
    """
    # Build trend feature directly from month_num — no external columns needed
    X_train = train["month_num"].values.reshape(-1, 1)
    y_train = train[target].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_fit = model.predict(X_train)

    last_month_num = int(train["month_num"].iloc[-1])
    X_future = np.arange(last_month_num + 1,
                         last_month_num + 1 + n_future).reshape(-1, 1)
    y_future = model.predict(X_future)

    return np.maximum(y_fit, 0), np.maximum(y_future, 0), model


# ─── 4. Model 2 — Holt-Winters Exponential Smoothing ─────────────────────────

def forecast_holtwinters(train_vals: np.ndarray, n_future: int) -> np.ndarray:
    """
    Manual double exponential smoothing (Holt's linear trend method).
    Avoids statsmodels dependency — pure numpy.
    """
    alpha, beta = 0.4, 0.3   # smoothing weights tuned heuristically

    n = len(train_vals)
    level  = np.zeros(n)
    trend_ = np.zeros(n)

    level[0]  = train_vals[0]
    trend_[0] = train_vals[1] - train_vals[0] if n > 1 else 0.0

    for t in range(1, n):
        prev_level = level[t - 1]
        prev_trend = trend_[t - 1]
        level[t]   = alpha * train_vals[t] + (1 - alpha) * (prev_level + prev_trend)
        trend_[t]  = beta  * (level[t] - prev_level) + (1 - beta) * prev_trend

    # Fitted values (in-sample)
    fitted = level + trend_

    # Forecast
    last_level = level[-1]
    last_trend = trend_[-1]
    forecast = np.array([
        max(last_level + h * last_trend, 0)
        for h in range(1, n_future + 1)
    ])
    return fitted, forecast


# ─── 5. Model 3 — Random Forest on lag features ───────────────────────────────

def forecast_rf(full_series: pd.DataFrame, train: pd.DataFrame,
                n_future: int, target: str):
    feat_cols = ["lag_1", "lag_2", "rolling_3", "month_of_year", "quarter", "is_q_end", "trend"]

    # Build lag features on the full history slice, then restrict to train window
    df_feat = add_lag_features(full_series, target).dropna(subset=feat_cols)
    train_feat = df_feat[df_feat["month_num"] <= train["month_num"].max()]

    X_train = train_feat[feat_cols].values
    y_train = train_feat[target].values

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=5,
        min_samples_leaf=1, random_state=42
    )
    rf.fit(X_train, y_train)
    y_fit = rf.predict(X_train)

    # Iterative forecast: each step feeds back as lag for the next
    history = full_series[target].tolist()
    future  = []
    last_month_dt  = full_series["month_dt"].iloc[-1]
    last_trend     = full_series["month_num"].iloc[-1]

    for i in range(1, n_future + 1):
        lag1       = history[-1]
        lag2       = history[-2] if len(history) >= 2 else lag1
        roll3      = np.mean(history[-3:]) if len(history) >= 3 else lag1
        m          = last_month_dt + pd.DateOffset(months=i)
        month_of_y = m.month
        quarter    = m.quarter
        is_q_end   = int(m.month % 3 == 0)
        trend_val  = last_trend + i

        row = np.array([[lag1, lag2, roll3, month_of_y, quarter, is_q_end, trend_val]])
        pred = float(rf.predict(row)[0])
        pred = max(pred, 0)
        future.append(pred)
        history.append(pred)

    return y_fit, np.array(future), rf


# ─── 6. Evaluate on hold-out ─────────────────────────────────────────────────

def evaluate(actuals: np.ndarray, preds: np.ndarray, name: str) -> dict:
    mae  = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2   = r2_score(actuals, preds) if len(actuals) > 1 else float("nan")
    mape = np.mean(np.abs((actuals - preds) / np.maximum(actuals, 1))) * 100
    print(f"  [{name}]  MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.1f}%  R²={r2:.3f}")
    return {"mae": round(mae, 2), "rmse": round(rmse, 2),
            "mape": round(mape, 2), "r2": round(r2, 4)}


# ─── 7. Main ─────────────────────────────────────────────────────────────────

def train_and_save():
    series = build_series()
    print(f"Historical months : {len(series)}")
    print(series[["month_str", "total_billed", "paid_revenue", "bill_count"]].to_string(index=False))
    print()

    TARGET = "total_billed"
    n      = len(series)
    n_test = TEST_MONTHS
    n_tr   = n - n_test

    train_s = series.iloc[:n_tr].copy()
    test_s  = series.iloc[n_tr:].copy()

    # Generate future month labels (Jan–Jun 2024)
    last_dt = series["month_dt"].iloc[-1]
    future_labels = [
        (last_dt + pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(1, FORECAST_MONTHS + 1)
    ]

    # ── Linear Regression ──
    print("Training Linear Regression…")
    # Train on train_s, predict n_test steps ahead to get hold-out preds
    lr_fit_tr, lr_test_and_fwd, _ = forecast_linear(train_s, n_test + FORECAST_MONTHS, TARGET)
    lr_test_pred = lr_test_and_fwd[:n_test]
    lr_fcast_fwd = lr_test_and_fwd[n_test:]

    # Fitted values on the full 12-month history (for in-sample overlay)
    lr_fit_full, _, _ = forecast_linear(series, FORECAST_MONTHS, TARGET)
    # Forward forecast from the full history (Jan–Jun 2024)
    _, lr_fcast_full, _ = forecast_linear(series, FORECAST_MONTHS, TARGET)

    # ── Holt-Winters ──
    print("Training Holt-Winters…")
    hw_fit_tr, hw_fcast_all = forecast_holtwinters(train_s[TARGET].values,
                                                   FORECAST_MONTHS + n_test)
    hw_test_pred = hw_fcast_all[:n_test]
    hw_fcast_fwd = hw_fcast_all[n_test:]

    # Fitted on full for display
    hw_fit_full, _ = forecast_holtwinters(series[TARGET].values, FORECAST_MONTHS)

    # ── Random Forest ──
    print("Training Random Forest…")
    rf_fit_tr, rf_fcast_all, _ = forecast_rf(train_s, train_s, FORECAST_MONTHS + n_test, TARGET)
    rf_test_pred = rf_fcast_all[:n_test]
    rf_fcast_fwd = rf_fcast_all[n_test:]

    rf_fit_full, _, _ = forecast_rf(series, series, FORECAST_MONTHS, TARGET)

    # ── Evaluate on hold-out ──
    print("\nHold-out evaluation (last 3 months of 2023):")
    actuals = test_s[TARGET].values
    lr_metrics = evaluate(actuals, lr_test_pred,  "Linear Regression")
    hw_metrics = evaluate(actuals, hw_test_pred,  "Holt-Winters")
    rf_metrics = evaluate(actuals, rf_test_pred,  "Random Forest")

    model_metrics = {
        "Linear Regression": lr_metrics,
        "Holt-Winters":      hw_metrics,
        "Random Forest":     rf_metrics,
    }
    best_model = min(model_metrics, key=lambda m: model_metrics[m]["mae"])
    print(f"\nBest model (lowest MAE): {best_model}")

    # Choose best forward forecast
    best_fcast = {
        "Linear Regression": lr_fcast_fwd,
        "Holt-Winters":      hw_fcast_fwd,
        "Random Forest":     rf_fcast_fwd,
    }[best_model]

    # ── Pack meta ──
    hist_months = series["month_str"].tolist()
    meta = {
        # Historical series
        "hist_months":      hist_months,
        "hist_total_billed": series["total_billed"].tolist(),
        "hist_paid_revenue": series["paid_revenue"].tolist(),
        "hist_bill_count":   series["bill_count"].tolist(),

        # Fitted in-sample curves (for overlay).
        # LR and HW cover all 12 months; RF lacks the first 2 rows (lag NaNs),
        # so pad with None so indices align with hist_months.
        "fit_lr": [round(float(v), 2) for v in lr_fit_full],
        "fit_hw": [round(float(v), 2) for v in hw_fit_full],
        "fit_rf": [None, None] + [round(float(v), 2) for v in rf_fit_full],

        # Forward forecast (all 3 models)
        "future_months":    future_labels,
        "fcast_lr":  [round(float(v), 2) for v in lr_fcast_full],
        "fcast_hw":  [round(float(v), 2) for v in hw_fcast_fwd],
        "fcast_rf":  [round(float(v), 2) for v in rf_fcast_fwd],
        "fcast_best":  [round(float(v), 2) for v in best_fcast],
        "best_model":  best_model,

        # Metrics
        "model_metrics": model_metrics,

        # Hold-out actuals for chart
        "test_months":   test_s["month_str"].tolist(),
        "test_actuals":  test_s["total_billed"].tolist(),

        # Summary stats for KPIs
        "avg_monthly_revenue": round(float(series["total_billed"].mean()), 2),
        "max_monthly_revenue": round(float(series["total_billed"].max()), 2),
        "min_monthly_revenue": round(float(series["total_billed"].min()), 2),
        "total_annual_billed": round(float(series["total_billed"].sum()), 2),
        "total_annual_paid":   round(float(series["paid_revenue"].sum()), 2),
        "avg_next_6m":         round(float(np.mean(best_fcast)), 2),
        "peak_month":          hist_months[int(np.argmax(series["total_billed"].values))],
        "trough_month":        hist_months[int(np.argmin(series["total_billed"].values))],
    }

    with open(BASE / "revenue_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("\nSaved → revenue_meta.pkl")
    return meta


if __name__ == "__main__":
    train_and_save()
