"""
Hospital Bill Prediction — Dash Prediction Page
================================================
Exports:
  prediction_layout  – Dash layout component
  register_predict_callbacks(app) – registers the predict callback
"""

import pickle
from pathlib import Path

import pandas as pd
from dash import dcc, html, Input, Output, State

BASE = Path(__file__).parent

# ── Load model artefacts ──────────────────────────────────────────────────────
def _load_artefacts():
    model_path = BASE / "bill_model.pkl"
    meta_path  = BASE / "model_meta.pkl"
    if not model_path.exists() or not meta_path.exists():
        return None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return model, meta


_MODEL, _META = _load_artefacts()

# ── Colour palette (mirrors app.py) ──────────────────────────────────────────
ACCENT   = "#3b82f6"
SUCCESS  = "#22c55e"
DANGER   = "#ef4444"
TEXT     = "#1e293b"
MUTED    = "#64748b"
CARD_BG  = "#ffffff"
GREY_BG  = "#f1f5f9"

# ── Helpers ──────────────────────────────────────────────────────────────────
def _opts(values):
    return [{"label": v, "value": v} for v in values]


def _field(label, child):
    return html.Div(
        style={"display": "flex", "flexDirection": "column", "gap": "6px"},
        children=[
            html.Label(label, style={"fontSize": "13px", "fontWeight": "600", "color": TEXT}),
            child,
        ],
    )


_dropdown_style = {
    "fontSize": "14px",
    "borderRadius": "8px",
}

_input_style = {
    "padding": "8px 12px",
    "borderRadius": "8px",
    "border": "1px solid #cbd5e1",
    "fontSize": "14px",
    "width": "100%",
    "boxSizing": "border-box",
    "color": TEXT,
    "outline": "none",
}


# ── Layout ────────────────────────────────────────────────────────────────────

def prediction_layout():
    if _META is None:
        return html.Div(
            "⚠️  Model not trained yet. Run  python model.py  first.",
            style={"padding": "40px", "color": DANGER, "fontWeight": "600"},
        )

    age_min, age_max = _META.get("patient_age_range", [0, 100])

    form = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "20px",
        },
        children=[
            _field("Patient Age",
                html.Div([
                    dcc.Slider(
                        id="pred-age",
                        min=age_min, max=age_max, step=1,
                        value=35,
                        marks={age_min: str(age_min), age_max: str(age_max)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    )
                ], style={"paddingTop": "10px"})
            ),
            _field("Gender",
                dcc.Dropdown(
                    id="pred-gender",
                    options=_opts(_META["gender"]),
                    placeholder="Select gender…",
                    style=_dropdown_style,
                    clearable=False,
                )
            ),
            _field("Insurance Provider",
                dcc.Dropdown(
                    id="pred-insurance",
                    options=_opts(_META["insurance_provider"]),
                    placeholder="Select insurance…",
                    style=_dropdown_style,
                    clearable=False,
                )
            ),
            _field("Doctor Specialization",
                dcc.Dropdown(
                    id="pred-specialization",
                    options=_opts(_META["specialization"]),
                    placeholder="Select specialization…",
                    style=_dropdown_style,
                    clearable=False,
                )
            ),
            _field("Treatment Type",
                dcc.Dropdown(
                    id="pred-treatment",
                    options=_opts(_META["treatment_type"]),
                    placeholder="Select treatment…",
                    style=_dropdown_style,
                    clearable=False,
                )
            ),
            _field("Reason for Visit",
                dcc.Dropdown(
                    id="pred-reason",
                    options=_opts(_META["reason_for_visit"]),
                    placeholder="Select reason…",
                    style=_dropdown_style,
                    clearable=False,
                )
            ),
        ],
    )

    result_box = html.Div(
        id="pred-result",
        style={
            "marginTop": "28px",
            "padding": "20px 28px",
            "borderRadius": "12px",
            "background": GREY_BG,
            "fontSize": "15px",
            "color": MUTED,
            "textAlign": "center",
            "border": "1px dashed #cbd5e1",
        },
        children="Fill in all fields and click Predict to see the estimated bill amount.",
    )

    return html.Div(
        style={"padding": "0"},
        children=[
            html.P(
                "Fill in the patient and visit details below. The model will estimate "
                "the expected bill amount using a Gradient Boosting Regressor trained on "
                "historical hospital data.",
                style={"color": MUTED, "fontSize": "14px", "marginTop": "0", "marginBottom": "24px"},
            ),
            html.Div(
                style={
                    "background": CARD_BG,
                    "borderRadius": "12px",
                    "padding": "28px 32px",
                    "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
                },
                children=[
                    form,
                    html.Div(
                        style={"marginTop": "28px", "textAlign": "right"},
                        children=[
                            html.Button(
                                "Predict Bill Amount",
                                id="pred-btn",
                                n_clicks=0,
                                style={
                                    "background": ACCENT,
                                    "color": "#fff",
                                    "border": "none",
                                    "borderRadius": "8px",
                                    "padding": "10px 28px",
                                    "fontSize": "15px",
                                    "fontWeight": "700",
                                    "cursor": "pointer",
                                    "letterSpacing": ".3px",
                                },
                            )
                        ],
                    ),
                    result_box,
                ],
            ),
        ],
    )


# ── Callback ──────────────────────────────────────────────────────────────────

def register_predict_callbacks(app):
    if _MODEL is None:
        return  # nothing to register if model not trained

    @app.callback(
        Output("pred-result", "children"),
        Output("pred-result", "style"),
        Input("pred-btn", "n_clicks"),
        State("pred-age",           "value"),
        State("pred-gender",        "value"),
        State("pred-insurance",     "value"),
        State("pred-specialization","value"),
        State("pred-treatment",     "value"),
        State("pred-reason",        "value"),
        prevent_initial_call=True,
    )
    def predict(n_clicks, age, gender, insurance, specialization, treatment, reason):
        base_style = {
            "marginTop": "28px",
            "padding": "20px 28px",
            "borderRadius": "12px",
            "fontSize": "15px",
            "textAlign": "center",
            "border": "none",
        }

        missing = [
            label for label, val in [
                ("Gender", gender),
                ("Insurance Provider", insurance),
                ("Doctor Specialization", specialization),
                ("Treatment Type", treatment),
                ("Reason for Visit", reason),
            ] if not val
        ]
        if missing:
            return (
                f"⚠️  Please fill in: {', '.join(missing)}.",
                {**base_style, "background": "#fff7ed", "color": "#92400e"},
            )

        row = pd.DataFrame([{
            "patient_age":       int(age) if age is not None else 35,
            "gender":            gender,
            "insurance_provider": insurance,
            "specialization":    specialization,
            "treatment_type":    treatment,
            "reason_for_visit":  reason,
        }])

        predicted = float(_MODEL.predict(row)[0])

        result_children = html.Div([
            html.P("Estimated Bill Amount", style={"margin": "0 0 8px", "fontSize": "13px", "fontWeight": "600", "color": MUTED, "letterSpacing": ".4px"}),
            html.H2(f"₹ {predicted:,.2f}", style={"margin": "0", "fontSize": "36px", "fontWeight": "800", "color": SUCCESS}),
            html.P(
                f"Age {age} · {gender} · {insurance} · {specialization} · {treatment} · {reason}",
                style={"margin": "10px 0 0", "fontSize": "12px", "color": MUTED},
            ),
        ])

        return (
            result_children,
            {**base_style, "background": "#f0fdf4", "border": f"1px solid {SUCCESS}"},
        )
