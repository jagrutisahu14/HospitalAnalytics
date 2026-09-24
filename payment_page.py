"""
Hospital Payment Status Prediction — Dash Page
================================================
Exports:
  payment_layout()                  – Dash layout for the Payment Status Predictor tab
  register_payment_callbacks(app)   – registers predict callback
"""

import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

BASE = Path(__file__).parent

# ── Load artefacts ────────────────────────────────────────────────────────────
def _load():
    mp = BASE / "payment_model.pkl"
    ep = BASE / "payment_meta.pkl"
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
TEAL    = "#14b8a6"
TEXT    = "#1e293b"
MUTED   = "#64748b"
CARD_BG = "#ffffff"
GREY_BG = "#f1f5f9"

STATUS_COLOR = {"Paid": SUCCESS, "Pending": WARNING, "Failed": DANGER}
STATUS_ICON  = {"Paid": "✅", "Pending": "⏳", "Failed": "❌"}

MODEL_COLORS = ["#3b82f6", "#8b5cf6", "#14b8a6"]  # blue, purple, teal

# ── Helpers ───────────────────────────────────────────────────────────────────
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


_DD = {"fontSize": "14px", "borderRadius": "8px"}


def _stat_card(label, value, color=ACCENT):
    return html.Div(
        style={
            "background": GREY_BG,
            "borderRadius": "10px",
            "padding": "14px 18px",
            "borderTop": f"4px solid {color}",
            "textAlign": "center",
        },
        children=[
            html.P(label, style={"margin": "0", "fontSize": "11px",
                                 "color": MUTED, "fontWeight": "600", "letterSpacing": ".3px"}),
            html.H3(value, style={"margin": "4px 0 0", "fontSize": "20px",
                                  "fontWeight": "800", "color": color}),
        ],
    )


# ── Model comparison chart ────────────────────────────────────────────────────

def _build_comparison_chart(report: dict) -> go.Figure:
    models   = list(report.keys())
    accuracy = [report[m]["accuracy"]         * 100 for m in models]
    f1_test  = [report[m]["test_f1_weighted"] * 100 for m in models]
    f1_cv    = [report[m]["cv_f1_weighted"]   * 100 for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy (%)", x=models, y=accuracy,
        marker_color=MODEL_COLORS[0],
        text=[f"{v:.1f}%" for v in accuracy], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Test F1 Weighted (%)", x=models, y=f1_test,
        marker_color=MODEL_COLORS[1],
        text=[f"{v:.1f}%" for v in f1_test], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="CV F1 Weighted (%)", x=models, y=f1_cv,
        marker_color=MODEL_COLORS[2],
        text=[f"{v:.1f}%" for v in f1_cv], textposition="outside",
    ))
    fig.update_layout(
        title="Model Comparison — Accuracy & F1 Score",
        barmode="group",
        yaxis=dict(range=[0, 115], title="Score (%)", ticksuffix="%"),
        xaxis_title="",
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color=TEXT,
        margin=dict(l=20, r=20, t=50, b=60),
        height=340,
    )
    return fig


def _build_per_class_chart(report: dict, model_name: str) -> go.Figure:
    classes  = ["Paid", "Pending", "Failed"]
    pc       = report[model_name]["per_class"]
    precision = [pc[c]["precision"] for c in classes]
    recall    = [pc[c]["recall"]    for c in classes]
    f1        = [pc[c]["f1"]        for c in classes]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=classes, y=precision,
                         marker_color=MODEL_COLORS[0],
                         text=[f"{v:.2f}" for v in precision], textposition="outside"))
    fig.add_trace(go.Bar(name="Recall",    x=classes, y=recall,
                         marker_color=MODEL_COLORS[1],
                         text=[f"{v:.2f}" for v in recall], textposition="outside"))
    fig.add_trace(go.Bar(name="F1-Score",  x=classes, y=f1,
                         marker_color=MODEL_COLORS[2],
                         text=[f"{v:.2f}" for v in f1], textposition="outside"))
    fig.update_layout(
        title=f"Per-Class Metrics — {model_name}",
        barmode="group",
        yaxis=dict(range=[0, 1.25], title="Score"),
        xaxis_title="Payment Status",
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color=TEXT,
        margin=dict(l=20, r=20, t=50, b=60),
        height=320,
    )
    return fig


def _metrics_table(report: dict) -> html.Div:
    """HTML table showing all three models side-by-side."""
    models = list(report.keys())
    rows = []

    # Header
    rows.append(html.Tr([
        html.Th("Metric", style=_th()),
        *[html.Th(m, style=_th()) for m in models],
    ]))
    for metric, label in [
        ("accuracy",         "Accuracy"),
        ("test_f1_weighted", "Test F1 (Weighted)"),
        ("cv_f1_weighted",   "CV F1 (Weighted)"),
    ]:
        vals = [report[m][metric] for m in models]
        best = max(range(len(vals)), key=lambda i: vals[i])
        rows.append(html.Tr([
            html.Td(label, style=_td(bold=True)),
            *[
                html.Td(
                    f"{v:.4f}",
                    style=_td(highlight=(i == best)),
                )
                for i, v in enumerate(vals)
            ],
        ]))

    # Per-class rows
    for cls in ["Paid", "Pending", "Failed"]:
        color = STATUS_COLOR[cls]
        for sub, sub_label in [("precision","Precision"),("recall","Recall"),("f1","F1")]:
            vals = [report[m]["per_class"][cls][sub] for m in models]
            best = max(range(len(vals)), key=lambda i: vals[i])
            rows.append(html.Tr([
                html.Td(f"{cls} — {sub_label}", style=_td(bold=True, color=color)),
                *[
                    html.Td(f"{v:.3f}", style=_td(highlight=(i == best)))
                    for i, v in enumerate(vals)
                ],
            ]))

    return html.Div(
        style={"overflowX": "auto", "marginTop": "16px"},
        children=[
            html.Table(
                style={
                    "width": "100%", "borderCollapse": "collapse",
                    "fontSize": "13px", "color": TEXT,
                },
                children=rows,
            )
        ],
    )


def _th():
    return {
        "padding": "10px 14px",
        "background": "#1e293b",
        "color": "#fff",
        "fontWeight": "700",
        "textAlign": "left",
        "fontSize": "13px",
    }

def _td(bold=False, highlight=False, color=None):
    return {
        "padding": "8px 14px",
        "borderBottom": "1px solid #e5e7eb",
        "background": "#f0fdf4" if highlight else "transparent",
        "fontWeight": "700" if (bold or highlight) else "400",
        "color": color if color else (SUCCESS if highlight else TEXT),
    }


# ── Layout ────────────────────────────────────────────────────────────────────

def payment_layout():
    if _META is None:
        return html.Div(
            "⚠️  Model not trained yet. Run  python payment_model.py  first.",
            style={"padding": "40px", "color": DANGER, "fontWeight": "600"},
        )

    report     = _META.get("model_report", {})
    best_model = _META.get("best_model", "")
    amt_min, amt_max = _META.get("amount_range", [0, 10000])

    # ── Model comparison section ──
    comparison_section = html.Div(
        style={
            "background": CARD_BG,
            "borderRadius": "12px",
            "padding": "24px 28px",
            "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
            "marginBottom": "24px",
        },
        children=[
            html.H3("Model Comparison",
                    style={"margin": "0 0 4px", "fontSize": "16px", "fontWeight": "700", "color": TEXT}),
            html.P(
                f"Three classifiers were trained and compared. "
                f"Best model: {best_model} (highest weighted F1 on held-out test set). "
                f"Green cells highlight the best score per metric.",
                style={"color": MUTED, "fontSize": "13px", "marginTop": "2px", "marginBottom": "16px"},
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                children=[
                    dcc.Graph(figure=_build_comparison_chart(report),
                              config={"displayModeBar": False}),
                    dcc.Graph(figure=_build_per_class_chart(report, best_model),
                              config={"displayModeBar": False}),
                ],
            ),
            _metrics_table(report),
        ],
    )

    # ── Prediction form ──
    form_grid = html.Div(
        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
        children=[
            _field("Bill Amount (₹)",
                html.Div([
                    dcc.Slider(
                        id="pay-amount",
                        min=int(amt_min), max=int(amt_max), step=50,
                        value=int((amt_min + amt_max) / 2),
                        marks={
                            int(amt_min): f"₹{int(amt_min):,}",
                            int(amt_max): f"₹{int(amt_max):,}",
                        },
                        tooltip={"placement": "bottom", "always_visible": True},
                    )
                ], style={"paddingTop": "10px"}),
            ),
            _field("Payment Method",
                dcc.Dropdown(id="pay-method",
                             options=_opts(_META["payment_method"]),
                             placeholder="Select…", style=_DD, clearable=False),
            ),
            _field("Treatment Type",
                dcc.Dropdown(id="pay-treatment",
                             options=_opts(_META["treatment_type"]),
                             placeholder="Select…", style=_DD, clearable=False),
            ),
            _field("Insurance Provider",
                dcc.Dropdown(id="pay-insurance",
                             options=_opts(_META["insurance_provider"]),
                             placeholder="Select…", style=_DD, clearable=False),
            ),
        ],
    )

    form_section = html.Div(
        style={
            "background": CARD_BG,
            "borderRadius": "12px",
            "padding": "24px 28px",
            "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
        },
        children=[
            html.H3("Predict Payment Status",
                    style={"margin": "0 0 4px", "fontSize": "16px", "fontWeight": "700", "color": TEXT}),
            html.P(
                "Enter the billing details below. The best-performing model will predict "
                "whether the bill is likely to be Paid, Pending, or Failed.",
                style={"color": MUTED, "fontSize": "13px", "marginTop": "2px", "marginBottom": "20px"},
            ),
            form_grid,
            html.Div(
                style={"marginTop": "24px", "textAlign": "right"},
                children=[
                    html.Button(
                        "Predict Payment Status",
                        id="pay-btn",
                        n_clicks=0,
                        style={
                            "background": TEAL,
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
            html.Div(id="pay-result", style={"marginTop": "24px"}),
        ],
    )

    return html.Div(children=[
        html.P(
            "Predicts whether a hospital bill will be Paid, Pending, or Failed using a "
            f"3-class classifier trained on {len(_META.get('payment_method', []))*0+200:.0f} billing records. "
            f"Best model: {best_model}.",
            style={"color": MUTED, "fontSize": "14px", "marginTop": "0", "marginBottom": "20px"},
        ),
        comparison_section,
        form_section,
    ])


# ── Callback ──────────────────────────────────────────────────────────────────

def register_payment_callbacks(app):
    if _MODEL is None:
        return

    @app.callback(
        Output("pay-result", "children"),
        Input("pay-btn", "n_clicks"),
        State("pay-amount",   "value"),
        State("pay-method",   "value"),
        State("pay-treatment","value"),
        State("pay-insurance","value"),
        prevent_initial_call=True,
    )
    def predict(n_clicks, amount, method, treatment, insurance):
        missing = [
            lbl for lbl, v in [
                ("Payment Method",    method),
                ("Treatment Type",    treatment),
                ("Insurance Provider",insurance),
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
            "amount":             float(amount) if amount is not None else 2500.0,
            "payment_method":     method,
            "treatment_type":     treatment,
            "insurance_provider": insurance,
        }])

        pred        = _MODEL.predict(row)[0]
        proba       = _MODEL.predict_proba(row)[0]
        classes     = _MODEL.classes_
        proba_dict  = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}

        main_color  = STATUS_COLOR.get(pred, ACCENT)
        main_icon   = STATUS_ICON.get(pred, "")

        # Probability bars
        prob_bars = html.Div(
            style={"marginTop": "16px"},
            children=[
                html.P("Prediction Probabilities",
                       style={"fontSize": "12px", "fontWeight": "600",
                              "color": MUTED, "margin": "0 0 10px"}),
                *[
                    html.Div(
                        style={"marginBottom": "10px"},
                        children=[
                            html.Div(
                                style={"display": "flex", "justifyContent": "space-between",
                                       "marginBottom": "4px"},
                                children=[
                                    html.Span(f"{STATUS_ICON.get(cls,'')} {cls}",
                                              style={"fontSize": "13px", "fontWeight": "600",
                                                     "color": STATUS_COLOR.get(cls, TEXT)}),
                                    html.Span(f"{proba_dict.get(cls, 0):.1f}%",
                                              style={"fontSize": "13px", "fontWeight": "700",
                                                     "color": STATUS_COLOR.get(cls, TEXT)}),
                                ],
                            ),
                            html.Div(
                                style={"background": "#e5e7eb", "borderRadius": "4px", "height": "8px"},
                                children=[
                                    html.Div(style={
                                        "width": f"{proba_dict.get(cls, 0)}%",
                                        "background": STATUS_COLOR.get(cls, ACCENT),
                                        "borderRadius": "4px",
                                        "height": "8px",
                                        "transition": "width 0.3s",
                                    })
                                ],
                            ),
                        ],
                    )
                    for cls in ["Paid", "Pending", "Failed"]
                ],
            ],
        )

        stat_strip = html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px",
                   "marginBottom": "16px"},
            children=[
                _stat_card("Predicted Status",    f"{main_icon} {pred}",         main_color),
                _stat_card("Model Used",           _META.get("best_model", ""),   ACCENT),
                _stat_card("Confidence",           f"{proba_dict.get(pred, 0):.1f}%", main_color),
            ],
        )

        input_summary = html.P(
            f"₹{amount:,.0f}  ·  {method}  ·  {treatment}  ·  {insurance}",
            style={"fontSize": "12px", "color": MUTED, "marginTop": "12px", "marginBottom": "0"},
        )

        return html.Div(
            style={
                "background": GREY_BG,
                "borderRadius": "12px",
                "padding": "20px 24px",
                "border": f"1px solid {main_color}",
            },
            children=[stat_strip, prob_bars, input_summary],
        )
