"""
Hospital No-Show Prediction — Dash Page
========================================
Exports:
  noshow_layout()                  – Dash layout for the predictor tab
  register_noshow_callbacks(app)   – registers the predict callback
"""

import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

BASE = Path(__file__).parent

# ── Load artefacts ────────────────────────────────────────────────────────────
def _load():
    mp = BASE / "noshow_model.pkl"
    ep = BASE / "noshow_meta.pkl"
    if not mp.exists() or not ep.exists():
        return None, None
    with open(mp, "rb") as f:
        model = pickle.load(f)
    with open(ep, "rb") as f:
        meta = pickle.load(f)
    return model, meta


_MODEL, _META = _load()

# ── Palette ───────────────────────────────────────────────────────────────────
ACCENT  = "#3b82f6"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER  = "#ef4444"
PURPLE  = "#8b5cf6"
TEXT    = "#1e293b"
MUTED   = "#64748b"
CARD_BG = "#ffffff"
GREY_BG = "#f1f5f9"

# ── Helpers ───────────────────────────────────────────────────────────────────
def _opts(values):
    return [{"label": v, "value": v} for v in values]


def _field(label, child, col_span=1):
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "6px",
            "gridColumn": f"span {col_span}",
        },
        children=[
            html.Label(
                label,
                style={"fontSize": "13px", "fontWeight": "600", "color": TEXT},
            ),
            child,
        ],
    )


_DD = {"fontSize": "14px", "borderRadius": "8px"}


def _gauge(prob: float) -> go.Figure:
    """Build a semi-circular gauge showing no-show probability."""
    pct = round(prob * 100, 1)
    if prob < 0.35:
        bar_color = SUCCESS
        risk_label = "Low Risk"
    elif prob < 0.60:
        bar_color = WARNING
        risk_label = "Moderate Risk"
    else:
        bar_color = DANGER
        risk_label = "High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": bar_color}},
        title={"text": f"No-Show Probability — <b>{risk_label}</b>",
               "font": {"size": 14, "color": TEXT}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": MUTED,
                     "tickfont": {"size": 11}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "#dcfce7"},
                {"range": [35, 60], "color": "#fef9c3"},
                {"range": [60, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": TEXT, "width": 3},
                "thickness": 0.75,
                "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="white",
        font_color=TEXT,
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def noshow_layout():
    if _META is None:
        return html.Div(
            "⚠️  Model not trained yet. Run  python noshow_model.py  first.",
            style={"padding": "40px", "color": DANGER, "fontWeight": "600"},
        )

    age_min, age_max = _META.get("patient_age_range", [0, 100])

    form_grid = html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr 1fr",
            "gap": "20px",
        },
        children=[
            _field("Patient Age",
                html.Div([
                    dcc.Slider(
                        id="ns-age",
                        min=age_min, max=age_max, step=1,
                        value=40,
                        marks={age_min: str(age_min), age_max: str(age_max)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    )
                ], style={"paddingTop": "10px"}),
            ),
            _field("Gender",
                dcc.Dropdown(id="ns-gender", options=_opts(_META["gender"]),
                             placeholder="Select…", style=_DD, clearable=False),
            ),
            _field("Insurance Provider",
                dcc.Dropdown(id="ns-insurance",
                             options=_opts(_META["insurance_provider"]),
                             placeholder="Select…", style=_DD, clearable=False),
            ),
            _field("Doctor Specialization",
                dcc.Dropdown(id="ns-specialization",
                             options=_opts(_META["specialization"]),
                             placeholder="Select…", style=_DD, clearable=False),
            ),
            _field("Reason for Visit",
                dcc.Dropdown(id="ns-reason",
                             options=_opts(_META["reason_for_visit"]),
                             placeholder="Select…", style=_DD, clearable=False),
            ),
            _field("Appointment Day",
                dcc.Dropdown(id="ns-weekday",
                             options=_opts(_META["appointment_weekday"]),
                             placeholder="Select…", style=_DD, clearable=False),
            ),
        ],
    )

    return html.Div(children=[
        html.P(
            "Select the patient and appointment details below. "
            "The model predicts the probability that this appointment will be a no-show, "
            "trained on historical data using a Random Forest classifier.",
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
                form_grid,
                html.Div(
                    style={"marginTop": "28px", "textAlign": "right"},
                    children=[
                        html.Button(
                            "Predict No-Show Risk",
                            id="ns-btn",
                            n_clicks=0,
                            style={
                                "background": PURPLE,
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
            ],
        ),

        # Result area — gauge + detail strip
        html.Div(id="ns-result", style={"marginTop": "24px"}),
    ])


# ── Callback ──────────────────────────────────────────────────────────────────

def register_noshow_callbacks(app):
    if _MODEL is None:
        return

    @app.callback(
        Output("ns-result", "children"),
        Input("ns-btn", "n_clicks"),
        State("ns-age",           "value"),
        State("ns-gender",        "value"),
        State("ns-insurance",     "value"),
        State("ns-specialization","value"),
        State("ns-reason",        "value"),
        State("ns-weekday",       "value"),
        prevent_initial_call=True,
    )
    def predict(n_clicks, age, gender, insurance, specialization, reason, weekday):
        missing = [
            lbl for lbl, v in [
                ("Gender", gender),
                ("Insurance Provider", insurance),
                ("Doctor Specialization", specialization),
                ("Reason for Visit", reason),
                ("Appointment Day", weekday),
            ] if not v
        ]
        if missing:
            return html.Div(
                f"⚠️  Please fill in: {', '.join(missing)}.",
                style={
                    "padding": "16px 20px",
                    "background": "#fff7ed",
                    "borderRadius": "10px",
                    "color": "#92400e",
                    "fontSize": "14px",
                    "fontWeight": "600",
                },
            )

        row = pd.DataFrame([{
            "patient_age":        int(age) if age is not None else 40,
            "gender":             gender,
            "insurance_provider": insurance,
            "specialization":     specialization,
            "reason_for_visit":   reason,
            "appointment_weekday": weekday,
        }])

        prob      = float(_MODEL.predict_proba(row)[0][1])
        prob_show = 1.0 - prob

        if prob < 0.35:
            risk_color  = SUCCESS
            risk_label  = "Low Risk"
            risk_advice = "Patient is likely to attend. Standard scheduling applies."
        elif prob < 0.60:
            risk_color  = WARNING
            risk_label  = "Moderate Risk"
            risk_advice = "Consider an SMS/call reminder 24 h before the appointment."
        else:
            risk_color  = DANGER
            risk_label  = "High Risk"
            risk_advice = "High no-show likelihood — recommend confirming the slot or overbooking."

        gauge_fig = _gauge(prob)

        detail_strip = html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr",
                "gap": "16px",
                "marginTop": "16px",
            },
            children=[
                _stat_card("No-Show Probability", f"{prob*100:.1f}%",  risk_color),
                _stat_card("Show-Up Probability",  f"{prob_show*100:.1f}%", SUCCESS),
                _stat_card("Risk Level",            risk_label,        risk_color),
            ],
        )

        advice_box = html.Div(
            style={
                "marginTop": "16px",
                "padding": "14px 18px",
                "background": "#f8fafc",
                "borderLeft": f"4px solid {risk_color}",
                "borderRadius": "6px",
                "fontSize": "14px",
                "color": TEXT,
            },
            children=[
                html.Strong("Recommendation: "),
                risk_advice,
            ],
        )

        input_summary = html.P(
            f"Age {age} · {gender} · {insurance} · {specialization} · {reason} · {weekday}",
            style={"fontSize": "12px", "color": MUTED, "marginTop": "12px", "marginBottom": "0"},
        )

        return html.Div(
            style={
                "background": CARD_BG,
                "borderRadius": "12px",
                "padding": "24px 28px",
                "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
            },
            children=[
                dcc.Graph(figure=gauge_fig, config={"displayModeBar": False}),
                detail_strip,
                advice_box,
                input_summary,
            ],
        )


def _stat_card(label, value, color):
    return html.Div(
        style={
            "background": GREY_BG,
            "borderRadius": "10px",
            "padding": "16px 20px",
            "borderTop": f"4px solid {color}",
        },
        children=[
            html.P(label, style={"margin": "0", "fontSize": "12px",
                                 "color": MUTED, "fontWeight": "600"}),
            html.H3(value, style={"margin": "6px 0 0", "fontSize": "24px",
                                  "fontWeight": "800", "color": color}),
        ],
    )
