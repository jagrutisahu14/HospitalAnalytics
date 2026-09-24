"""
Hospital Revenue Forecasting — Dash Page
==========================================
Exports:
  revenue_layout()                 – Dash layout for the Revenue Forecast tab
  register_revenue_callbacks(app)  – registers interactive callbacks
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State

BASE = Path(__file__).parent

# ── Load artefacts ────────────────────────────────────────────────────────────
def _load():
    mp = BASE / "revenue_meta.pkl"
    if not mp.exists():
        return None
    with open(mp, "rb") as f:
        return pickle.load(f)

_META = _load()

# ── Palette ───────────────────────────────────────────────────────────────────
ACCENT  = "#3b82f6"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER  = "#ef4444"
PURPLE  = "#8b5cf6"
TEAL    = "#14b8a6"
ORANGE  = "#f97316"
TEXT    = "#1e293b"
MUTED   = "#64748b"
CARD_BG = "#ffffff"
GREY_BG = "#f1f5f9"

MODEL_COLORS = {
    "Linear Regression": ACCENT,
    "Holt-Winters":      PURPLE,
    "Random Forest":     TEAL,
}
METRIC_BETTER = "lower"   # MAE / RMSE / MAPE: lower is better


# ── Helpers ───────────────────────────────────────────────────────────────────
def _kpi(label, value, color=ACCENT, sub=""):
    return html.Div(
        style={
            "background": CARD_BG,
            "borderRadius": "12px",
            "padding": "16px 20px",
            "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
            "borderTop": f"4px solid {color}",
            "flex": "1",
            "minWidth": "140px",
        },
        children=[
            html.P(label, style={"margin": "0", "fontSize": "12px",
                                 "color": MUTED, "fontWeight": "600", "letterSpacing": ".3px"}),
            html.H3(value, style={"margin": "4px 0 0", "fontSize": "22px",
                                  "fontWeight": "800", "color": color}),
            html.P(sub, style={"margin": "2px 0 0", "fontSize": "11px", "color": MUTED})
            if sub else html.Span(),
        ],
    )


def _section(title, *children):
    return html.Div(
        style={
            "background": CARD_BG,
            "borderRadius": "12px",
            "padding": "22px 26px",
            "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
            "marginBottom": "20px",
        },
        children=[
            html.H3(title, style={"margin": "0 0 14px", "fontSize": "15px",
                                  "fontWeight": "700", "color": TEXT}),
            *children,
        ],
    )


def _th():
    return {"padding": "9px 14px", "background": "#1e293b", "color": "#fff",
            "fontWeight": "700", "fontSize": "13px", "textAlign": "left"}

def _td(highlight=False, bold=False):
    return {"padding": "8px 14px", "borderBottom": "1px solid #e5e7eb",
            "fontSize": "13px",
            "background": "#f0fdf4" if highlight else "transparent",
            "fontWeight": "700" if (bold or highlight) else "400",
            "color": SUCCESS if highlight else TEXT}


# ── Chart builders ────────────────────────────────────────────────────────────

def _forecast_chart(meta: dict, show_models: list, show_ci: bool) -> go.Figure:
    fig = go.Figure()

    hist_x = meta["hist_months"]
    paid   = meta["hist_paid_revenue"]
    billed = meta["hist_total_billed"]
    fut_x  = meta["future_months"]

    # Actual collected revenue (filled area)
    fig.add_trace(go.Scatter(
        x=hist_x, y=paid,
        name="Collected (Paid)", fill="tozeroy",
        fillcolor="rgba(34,197,94,0.12)",
        line=dict(color=SUCCESS, width=2),
        mode="lines+markers", marker=dict(size=5),
    ))
    # Total billed
    fig.add_trace(go.Scatter(
        x=hist_x, y=billed,
        name="Total Billed", line=dict(color=ACCENT, width=2, dash="dot"),
        mode="lines+markers", marker=dict(size=5),
    ))

    # Model forecasts (forward only)
    model_fcast = {
        "Linear Regression": meta["fcast_lr"],
        "Holt-Winters":      meta["fcast_hw"],
        "Random Forest":     meta["fcast_rf"],
    }
    best = meta["best_model"]

    for name in show_models:
        vals = model_fcast.get(name, [])
        color = MODEL_COLORS.get(name, ORANGE)
        width = 3 if name == best else 1.5
        dash  = "solid" if name == best else "dash"
        marker_sym = "star" if name == best else "circle-open"
        fig.add_trace(go.Scatter(
            x=fut_x, y=vals,
            name=f"{'★ ' if name==best else ''}{name} Forecast",
            line=dict(color=color, width=width, dash=dash),
            mode="lines+markers",
            marker=dict(size=8 if name == best else 5, symbol=marker_sym),
        ))

        # Confidence band (±15%) for best model
        if show_ci and name == best:
            upper = [v * 1.15 for v in vals]
            lower = [max(v * 0.85, 0) for v in vals]
            fig.add_trace(go.Scatter(
                x=fut_x + fut_x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=f"{color}22",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=True,
                name=f"{name} ±15% CI",
                hoverinfo="skip",
            ))

    # Vertical divider at forecast start (add_vline doesn't support string x-axes)
    fig.add_shape(
        type="line", xref="x", yref="paper",
        x0=hist_x[-1], x1=hist_x[-1], y0=0, y1=1,
        line=dict(dash="dash", color=MUTED, width=1),
    )
    fig.add_annotation(
        x=hist_x[-1], y=1, xref="x", yref="paper",
        text=" Forecast →", showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(color=MUTED, size=12),
    )

    fig.update_layout(
        title="Monthly Revenue — Historical & Forecast",
        xaxis_title="Month",
        yaxis_title="Revenue (₹)",
        yaxis_tickprefix="₹",
        yaxis_tickformat=",.0f",
        legend=dict(orientation="h", y=-0.22),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color=TEXT,
        margin=dict(l=20, r=20, t=50, b=80),
        height=420,
        hovermode="x unified",
    )
    return fig


def _model_comparison_chart(meta: dict) -> go.Figure:
    models  = list(meta["model_metrics"].keys())
    metrics = ["mae", "rmse", "mape"]
    labels  = ["MAE (₹)", "RMSE (₹)", "MAPE (%)"]
    colors  = [ACCENT, PURPLE, TEAL]
    best    = meta["best_model"]

    fig = go.Figure()
    for metric, label, color in zip(metrics, labels, colors):
        vals = [meta["model_metrics"][m][metric] for m in models]
        fig.add_trace(go.Bar(
            name=label, x=models, y=vals,
            marker_color=[color if m != best else SUCCESS for m in models],
            text=[f"{v:,.1f}" for v in vals], textposition="outside",
        ))
    fig.update_layout(
        title="Model Evaluation — Hold-out (last 3 months)",
        barmode="group",
        yaxis_title="Score",
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="white", paper_bgcolor="white", font_color=TEXT,
        margin=dict(l=20, r=20, t=50, b=70),
        height=300,
    )
    return fig


def _monthly_breakdown_chart(meta: dict) -> go.Figure:
    x      = meta["hist_months"]
    billed = meta["hist_total_billed"]
    paid   = meta["hist_paid_revenue"]
    failed = [b - p for b, p in zip(billed, paid)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Collected (Paid)",  x=x, y=paid,   marker_color=SUCCESS))
    fig.add_trace(go.Bar(name="Unpaid / Failed",   x=x, y=failed, marker_color=DANGER))
    fig.update_layout(
        title="Monthly Revenue Breakdown — Collected vs Unpaid",
        barmode="stack",
        yaxis_tickprefix="₹", yaxis_tickformat=",.0f",
        yaxis_title="Revenue (₹)", xaxis_title="",
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="white", paper_bgcolor="white", font_color=TEXT,
        margin=dict(l=20, r=20, t=50, b=70),
        height=300,
    )
    return fig


def _metrics_table(meta: dict) -> html.Div:
    models = list(meta["model_metrics"].keys())
    best   = meta["best_model"]
    rows   = [html.Tr([
        html.Th("Model",   style=_th()),
        html.Th("MAE (₹)", style=_th()),
        html.Th("RMSE (₹)",style=_th()),
        html.Th("MAPE (%)",style=_th()),
        html.Th("R²",      style=_th()),
        html.Th("",        style=_th()),
    ])]
    for m in models:
        mm  = meta["model_metrics"][m]
        is_best = m == best
        rows.append(html.Tr([
            html.Td(f"{'★ ' if is_best else ''}{m}", style=_td(bold=is_best)),
            html.Td(f"₹{mm['mae']:,.2f}",  style=_td(highlight=is_best)),
            html.Td(f"₹{mm['rmse']:,.2f}", style=_td(highlight=is_best)),
            html.Td(f"{mm['mape']:.1f}%",  style=_td(highlight=is_best)),
            html.Td(f"{mm['r2']:.4f}",     style=_td(highlight=is_best)),
            html.Td("Best ★" if is_best else "", style=_td(highlight=is_best)),
        ]))
    return html.Div(
        style={"overflowX": "auto", "marginTop": "12px"},
        children=[html.Table(
            style={"width": "100%", "borderCollapse": "collapse",
                   "fontSize": "13px", "color": TEXT},
            children=rows,
        )],
    )


# ── Layout ────────────────────────────────────────────────────────────────────

def revenue_layout():
    if _META is None:
        return html.Div(
            "⚠️  Model not trained yet. Run  python revenue_model.py  first.",
            style={"padding": "40px", "color": DANGER, "fontWeight": "600"},
        )

    meta     = _META
    best     = meta["best_model"]
    fut_x    = meta["future_months"]
    best_avg = meta["avg_next_6m"]
    best_fcast = meta["fcast_best"]
    models   = list(meta["model_metrics"].keys())

    # ── KPI row ──
    kpi_row = html.Div(
        style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "20px"},
        children=[
            _kpi("Total Annual Billed",    f"₹{meta['total_annual_billed']:,.0f}",  ACCENT,   "2023"),
            _kpi("Total Collected (Paid)", f"₹{meta['total_annual_paid']:,.0f}",    SUCCESS,  "2023"),
            _kpi("Avg Monthly Revenue",    f"₹{meta['avg_monthly_revenue']:,.0f}",  TEAL,     "historical"),
            _kpi("Peak Month",              meta["peak_month"],                      PURPLE,   "highest billed"),
            _kpi("Trough Month",            meta["trough_month"],                   WARNING,  "lowest billed"),
            _kpi(f"Avg Forecast (6m)",     f"₹{best_avg:,.0f}",                    ORANGE,   f"via {best}"),
        ],
    )

    # ── Forecast chart controls ──
    controls = html.Div(
        style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
               "alignItems": "center", "marginBottom": "14px"},
        children=[
            html.Div([
                html.Label("Show models:",
                           style={"fontSize": "13px", "fontWeight": "600",
                                  "color": TEXT, "marginRight": "8px"}),
                dcc.Checklist(
                    id="rev-model-toggle",
                    options=[{"label": f" {m}", "value": m} for m in models],
                    value=models,
                    inline=True,
                    inputStyle={"marginRight": "4px", "marginLeft": "10px"},
                    labelStyle={"fontSize": "13px", "color": TEXT},
                ),
            ]),
            html.Div([
                dcc.Checklist(
                    id="rev-ci-toggle",
                    options=[{"label": " Show ±15% confidence band", "value": "ci"}],
                    value=["ci"],
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"fontSize": "13px", "color": TEXT},
                ),
            ]),
        ],
    )

    forecast_section = _section(
        "📈 Revenue Forecast — Jan–Jun 2024",
        controls,
        dcc.Graph(id="rev-forecast-chart",
                  figure=_forecast_chart(meta, models, True),
                  config={"displayModeBar": False}),
    )

    breakdown_section = _section(
        "📊 Monthly Breakdown — Collected vs Unpaid",
        dcc.Graph(figure=_monthly_breakdown_chart(meta),
                  config={"displayModeBar": False}),
    )

    comparison_section = _section(
        "🔬 Model Comparison — Hold-out Evaluation (Oct–Dec 2023)",
        html.P(
            f"All three models were evaluated on the last 3 months of 2023 (held-out test set). "
            f"★ Best model: {best} (lowest MAE). "
            f"Green bars = best performer.",
            style={"color": MUTED, "fontSize": "13px", "marginTop": "-6px", "marginBottom": "12px"},
        ),
        dcc.Graph(figure=_model_comparison_chart(meta),
                  config={"displayModeBar": False}),
        _metrics_table(meta),
    )

    # ── What-if scenario form ──
    whatif_section = _section(
        "🔧 What-If Scenario — Adjust Forecast",
        html.P(
            "Simulate the impact of a volume or rate change on the 6-month forward revenue forecast.",
            style={"color": MUTED, "fontSize": "13px", "marginTop": "-6px", "marginBottom": "16px"},
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
            children=[
                html.Div([
                    html.Label("Appointment volume change (%)",
                               style={"fontSize": "13px", "fontWeight": "600",
                                      "color": TEXT, "display": "block", "marginBottom": "8px"}),
                    dcc.Slider(
                        id="rev-vol-slider",
                        min=-30, max=50, step=5, value=0,
                        marks={-30: "-30%", -10: "-10%", 0: "0%",
                                10: "+10%", 30: "+30%", 50: "+50%"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ]),
                html.Div([
                    html.Label("Average bill amount change (%)",
                               style={"fontSize": "13px", "fontWeight": "600",
                                      "color": TEXT, "display": "block", "marginBottom": "8px"}),
                    dcc.Slider(
                        id="rev-rate-slider",
                        min=-20, max=30, step=5, value=0,
                        marks={-20: "-20%", -10: "-10%", 0: "0%",
                                10: "+10%", 20: "+20%", 30: "+30%"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ]),
            ],
        ),
        html.Div(
            style={"marginTop": "18px", "textAlign": "right"},
            children=[
                html.Button(
                    "Apply Scenario",
                    id="rev-whatif-btn",
                    n_clicks=0,
                    style={
                        "background": ORANGE,
                        "color": "#fff",
                        "border": "none",
                        "borderRadius": "8px",
                        "padding": "10px 26px",
                        "fontSize": "14px",
                        "fontWeight": "700",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),
        html.Div(id="rev-whatif-result", style={"marginTop": "16px"}),
    )

    # ── Forecast table ──
    forecast_table_rows = [
        html.Tr([
            html.Th("Month",       style=_th()),
            html.Th("Best Forecast (₹)", style=_th()),
            html.Th("LR Forecast (₹)",   style=_th()),
            html.Th("HW Forecast (₹)",   style=_th()),
            html.Th("RF Forecast (₹)",   style=_th()),
        ])
    ]
    for i, month in enumerate(fut_x):
        forecast_table_rows.append(html.Tr([
            html.Td(month, style=_td(bold=True)),
            html.Td(f"₹{meta['fcast_best'][i]:,.0f}", style=_td(highlight=True)),
            html.Td(f"₹{meta['fcast_lr'][i]:,.0f}",   style=_td()),
            html.Td(f"₹{meta['fcast_hw'][i]:,.0f}",   style=_td()),
            html.Td(f"₹{meta['fcast_rf'][i]:,.0f}",   style=_td()),
        ]))

    forecast_table_section = _section(
        "📋 6-Month Forecast Table",
        html.Div(
            style={"overflowX": "auto"},
            children=[html.Table(
                style={"width": "100%", "borderCollapse": "collapse",
                       "fontSize": "13px", "color": TEXT},
                children=forecast_table_rows,
            )],
        ),
    )

    return html.Div(children=[
        html.P(
            f"Revenue forecasting using historical billing data (2023). "
            f"Three models compared: Linear Regression, Holt-Winters Exponential Smoothing, "
            f"and Random Forest. Best model: {best}.",
            style={"color": MUTED, "fontSize": "14px",
                   "marginTop": "0", "marginBottom": "20px"},
        ),
        kpi_row,
        forecast_section,
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
            children=[breakdown_section, comparison_section],
        ),
        forecast_table_section,
        whatif_section,
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_revenue_callbacks(app):
    if _META is None:
        return

    @app.callback(
        Output("rev-forecast-chart", "figure"),
        Input("rev-model-toggle", "value"),
        Input("rev-ci-toggle",    "value"),
    )
    def update_forecast_chart(selected_models, ci_toggle):
        show_ci = "ci" in (ci_toggle or [])
        return _forecast_chart(_META, selected_models or [], show_ci)

    @app.callback(
        Output("rev-whatif-result", "children"),
        Input("rev-whatif-btn", "n_clicks"),
        State("rev-vol-slider",  "value"),
        State("rev-rate-slider", "value"),
        prevent_initial_call=True,
    )
    def apply_whatif(n_clicks, vol_pct, rate_pct):
        vol_factor  = 1 + (vol_pct  or 0) / 100
        rate_factor = 1 + (rate_pct or 0) / 100
        combined    = vol_factor * rate_factor

        base_fcast = _META["fcast_best"]
        adj_fcast  = [round(v * combined, 2) for v in base_fcast]
        base_total = sum(base_fcast)
        adj_total  = sum(adj_fcast)
        delta      = adj_total - base_total
        delta_pct  = (delta / base_total * 100) if base_total else 0

        color  = SUCCESS if delta >= 0 else DANGER
        arrow  = "▲" if delta >= 0 else "▼"
        sign   = "+" if delta >= 0 else ""

        rows = [html.Tr([
            html.Th("Month",             style=_th()),
            html.Th("Base Forecast (₹)", style=_th()),
            html.Th("Adjusted (₹)",      style=_th()),
            html.Th("Δ Change (₹)",      style=_th()),
        ])]
        for month, base, adj in zip(_META["future_months"], base_fcast, adj_fcast):
            d = adj - base
            rows.append(html.Tr([
                html.Td(month,              style=_td(bold=True)),
                html.Td(f"₹{base:,.0f}",   style=_td()),
                html.Td(f"₹{adj:,.0f}",    style=_td(highlight=(d >= 0))),
                html.Td(f"{'+'if d>=0 else '-'}₹{abs(d):,.0f}",
                        style={**_td(), "color": SUCCESS if d >= 0 else DANGER,
                               "fontWeight": "700"}),
            ]))

        summary = html.Div(
            style={
                "display": "flex", "gap": "14px", "flexWrap": "wrap",
                "marginBottom": "14px",
            },
            children=[
                html.Div(style={
                    "background": CARD_BG, "borderRadius": "10px",
                    "padding": "14px 20px", "borderTop": f"4px solid {ORANGE}",
                    "flex": "1",
                }, children=[
                    html.P("Volume Adjustment",
                           style={"margin": "0", "fontSize": "11px",
                                  "color": MUTED, "fontWeight": "600"}),
                    html.H4(f"{'+' if vol_pct >= 0 else ''}{vol_pct}%",
                            style={"margin": "4px 0 0", "fontSize": "20px",
                                   "fontWeight": "800", "color": ORANGE}),
                ]),
                html.Div(style={
                    "background": CARD_BG, "borderRadius": "10px",
                    "padding": "14px 20px", "borderTop": f"4px solid {PURPLE}",
                    "flex": "1",
                }, children=[
                    html.P("Rate Adjustment",
                           style={"margin": "0", "fontSize": "11px",
                                  "color": MUTED, "fontWeight": "600"}),
                    html.H4(f"{'+' if rate_pct >= 0 else ''}{rate_pct}%",
                            style={"margin": "4px 0 0", "fontSize": "20px",
                                   "fontWeight": "800", "color": PURPLE}),
                ]),
                html.Div(style={
                    "background": CARD_BG, "borderRadius": "10px",
                    "padding": "14px 20px", "borderTop": f"4px solid {color}",
                    "flex": "1",
                }, children=[
                    html.P("Adjusted 6-Month Total",
                           style={"margin": "0", "fontSize": "11px",
                                  "color": MUTED, "fontWeight": "600"}),
                    html.H4(f"₹{adj_total:,.0f}",
                            style={"margin": "4px 0 0", "fontSize": "20px",
                                   "fontWeight": "800", "color": color}),
                ]),
                html.Div(style={
                    "background": CARD_BG, "borderRadius": "10px",
                    "padding": "14px 20px", "borderTop": f"4px solid {color}",
                    "flex": "1",
                }, children=[
                    html.P("vs Base Forecast",
                           style={"margin": "0", "fontSize": "11px",
                                  "color": MUTED, "fontWeight": "600"}),
                    html.H4(f"{arrow} {sign}₹{abs(delta):,.0f} ({sign}{delta_pct:.1f}%)",
                            style={"margin": "4px 0 0", "fontSize": "18px",
                                   "fontWeight": "800", "color": color}),
                ]),
            ],
        )

        table = html.Div(
            style={"overflowX": "auto"},
            children=[html.Table(
                style={"width": "100%", "borderCollapse": "collapse",
                       "fontSize": "13px"},
                children=rows,
            )],
        )
        return html.Div([summary, table])
