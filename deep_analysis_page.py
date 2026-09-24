"""
Hospital Deep Data Analysis Dashboard — Dash Page
===================================================
10 charts across 5 analytical dimensions, all driven by pre-computed
DataFrames from analysis.get_all_data(). No callbacks needed — fully
pre-rendered at startup.

Dimensions covered (none overlap with the existing Analytics tab):
  A. Scheduling Quality  — no-show rate by weekday, hour-of-day heatmap,
                           month × status calendar heatmap
  B. Financial Depth     — payment failure rate by method,
                           revenue by specialization × payment method,
                           billing heatmap (insurance × treatment)
  C. Patient Behaviour   — single vs repeat visitors (retention donut),
                           cost vs billed scatter (coloured by payment status)
  D. Doctor Efficiency   — bubble chart: appointments vs avg cost vs paid revenue
  E. Pipeline Health     — treatment → billing → paid funnel
"""

import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

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

STATUS_COLORS = {"Paid": SUCCESS, "Pending": WARNING, "Failed": DANGER,
                 "Completed": SUCCESS, "Scheduled": ACCENT,
                 "Cancelled": WARNING, "No-show": DANGER}

CM = dict(l=20, r=20, t=44, b=20)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _section(title, *children):
    return html.Div(
        style={
            "background": CARD_BG,
            "borderRadius": "12px",
            "padding": "20px 24px",
            "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
            "marginBottom": "20px",
        },
        children=[
            html.H3(title, style={"margin": "0 0 14px", "fontSize": "15px",
                                  "fontWeight": "700", "color": TEXT}),
            *children,
        ],
    )

def _grid(*children, cols="1fr 1fr"):
    return html.Div(
        style={"display": "grid", "gridTemplateColumns": cols, "gap": "16px"},
        children=list(children),
    )

def _graph(fig):
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ── A. Scheduling Quality ─────────────────────────────────────────────────────

def fig_noshow_weekday(data):
    df = data["noshow_weekday"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["weekday"], y=df["noshow_rate"],
        marker_color=[DANGER if r >= 30 else WARNING if r >= 20 else TEAL
                      for r in df["noshow_rate"]],
        text=[f"{r}%" for r in df["noshow_rate"]], textposition="outside",
        customdata=df[["total","noshow"]].values,
        hovertemplate="<b>%{x}</b><br>No-show rate: %{y}%<br>"
                      "Total appts: %{customdata[0]}<br>No-shows: %{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(title="No-Show Rate by Weekday", margin=CM,
                      yaxis_title="No-Show Rate (%)", yaxis_ticksuffix="%",
                      xaxis_title="", plot_bgcolor="white", paper_bgcolor="white",
                      font_color=TEXT)
    return fig


def fig_appt_hour(data):
    df = data["appt_hour"].sort_values("hour")
    fig = go.Figure(go.Bar(
        x=df["hour"], y=df["count"],
        marker_color=PURPLE,
        text=df["count"], textposition="outside",
    ))
    fig.update_layout(
        title="Appointment Volume by Hour of Day",
        xaxis=dict(title="Hour (24h)", tickmode="linear", dtick=1),
        yaxis_title="Appointments", margin=CM,
        plot_bgcolor="white", paper_bgcolor="white", font_color=TEXT,
    )
    return fig


def fig_status_month_heatmap(data):
    df = data["status_month_hm"]
    pivot = df.pivot(index="status", columns="month", values="count").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="Blues",
        text=pivot.values.astype(int),
        texttemplate="%{text}",
        hovertemplate="Month: %{x}<br>Status: %{y}<br>Count: %{z}<extra></extra>",
        showscale=True,
    ))
    fig.update_layout(
        title="Appointment Status × Month Heatmap",
        xaxis_title="Month", yaxis_title="Status",
        margin=dict(l=20, r=20, t=44, b=60),
        paper_bgcolor="white", font_color=TEXT,
        xaxis_tickangle=-45,
        height=280,
    )
    return fig


# ── B. Financial Depth ────────────────────────────────────────────────────────

def fig_payment_failure(data):
    df = data["payment_failure"]
    fig = go.Figure(go.Bar(
        x=df["payment_method"], y=df["failure_rate"],
        marker_color=[DANGER if r >= 35 else WARNING if r >= 25 else SUCCESS
                      for r in df["failure_rate"]],
        text=[f"{r}%" for r in df["failure_rate"]], textposition="outside",
        customdata=df[["total","failed"]].values,
        hovertemplate="<b>%{x}</b><br>Failure rate: %{y}%<br>"
                      "Total bills: %{customdata[0]}<br>Failed: %{customdata[1]}<extra></extra>",
    ))
    fig.update_layout(
        title="Payment Failure Rate by Method",
        yaxis_title="Failure Rate (%)", yaxis_ticksuffix="%",
        xaxis_title="", margin=CM,
        plot_bgcolor="white", paper_bgcolor="white", font_color=TEXT,
    )
    return fig


def fig_rev_spec_method(data):
    df = data["rev_spec_method"]
    methods = sorted(df["payment_method"].unique())
    colors  = [ACCENT, TEAL, PURPLE, ORANGE]
    fig = go.Figure()
    for i, method in enumerate(methods):
        sub = df[df["payment_method"] == method]
        fig.add_trace(go.Bar(
            name=method,
            x=sub["specialization"], y=sub["revenue"].round(0),
            marker_color=colors[i % len(colors)],
            text=sub["revenue"].apply(lambda v: f"₹{v:,.0f}"),
            textposition="outside",
        ))
    fig.update_layout(
        title="Paid Revenue by Specialization × Payment Method",
        barmode="group", yaxis_title="Revenue (₹)",
        yaxis_tickprefix="₹", yaxis_tickformat=",.0f",
        xaxis_title="", legend=dict(orientation="h", y=-0.25),
        margin=dict(l=20, r=20, t=44, b=60),
        plot_bgcolor="white", paper_bgcolor="white", font_color=TEXT,
        height=320,
    )
    return fig


def fig_bill_heatmap(data):
    df = data["bill_heatmap"]
    pivot = df.pivot(index="insurance_provider",
                     columns="treatment_type", values="avg_amount").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="YlOrRd",
        text=[[f"₹{v:,.0f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        hovertemplate="Insurance: %{y}<br>Treatment: %{x}<br>Avg Bill: %{text}<extra></extra>",
        showscale=True,
    ))
    fig.update_layout(
        title="Avg Bill Amount — Insurance × Treatment Type",
        xaxis_title="Treatment Type", yaxis_title="Insurance Provider",
        margin=dict(l=20, r=20, t=44, b=60),
        paper_bgcolor="white", font_color=TEXT,
        xaxis_tickangle=-30,
        height=300,
    )
    return fig


# ── C. Patient Behaviour ──────────────────────────────────────────────────────

def fig_patient_retention(data):
    df = data["patient_retention"]
    fig = go.Figure(go.Pie(
        labels=df["category"], values=df["patients"],
        marker=dict(colors=[ACCENT, SUCCESS]),
        hole=0.5,
        textinfo="label+percent+value",
    ))
    fig.update_layout(
        title="Patient Retention — Single vs Repeat Visitors",
        margin=CM, paper_bgcolor="white", font_color=TEXT,
    )
    return fig


def fig_cost_vs_billed(data):
    df = data["cost_vs_billed"]
    color_map = {"Paid": SUCCESS, "Pending": WARNING, "Failed": DANGER}
    fig = go.Figure()
    for status, color in color_map.items():
        sub = df[df["payment_status"] == status]
        fig.add_trace(go.Scatter(
            x=sub["cost"], y=sub["amount"],
            mode="markers",
            name=status,
            marker=dict(color=color, size=7, opacity=0.75),
            text=sub["treatment_type"],
            hovertemplate="<b>%{text}</b><br>Treatment cost: ₹%{x:,.0f}<br>"
                          "Bill amount: ₹%{y:,.0f}<br>Status: " + status + "<extra></extra>",
        ))
    # parity line
    all_vals = list(df["cost"]) + list(df["amount"])
    vmin, vmax = min(all_vals), max(all_vals)
    fig.add_trace(go.Scatter(
        x=[vmin, vmax], y=[vmin, vmax],
        mode="lines", name="Parity (cost = bill)",
        line=dict(color=MUTED, dash="dot", width=1),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title="Treatment Cost vs Bill Amount (by Payment Status)",
        xaxis_title="Treatment Cost (₹)", yaxis_title="Bill Amount (₹)",
        xaxis_tickprefix="₹", yaxis_tickprefix="₹",
        xaxis_tickformat=",.0f", yaxis_tickformat=",.0f",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=20, r=20, t=44, b=60),
        plot_bgcolor="white", paper_bgcolor="white", font_color=TEXT,
        height=340,
    )
    return fig


# ── D. Doctor Efficiency ──────────────────────────────────────────────────────

def fig_doctor_efficiency(data):
    df = data["doctor_efficiency"].head(10)
    spec_colors = {"Dermatology": ACCENT, "Pediatrics": TEAL, "Oncology": PURPLE}
    fig = go.Figure()
    for spec in df["specialization"].unique():
        sub = df[df["specialization"] == spec]
        fig.add_trace(go.Scatter(
            x=sub["appointments"],
            y=sub["avg_cost"],
            mode="markers",
            name=spec,
            marker=dict(
                size=sub["paid_revenue"].fillna(0) / sub["paid_revenue"].max() * 40 + 10,
                color=spec_colors.get(spec, ORANGE),
                opacity=0.8,
                line=dict(width=1, color="white"),
            ),
            text=sub["doctor_name"],
            customdata=sub[["paid_revenue"]].values,
            hovertemplate="<b>%{text}</b><br>Appointments: %{x}<br>"
                          "Avg Treatment Cost: ₹%{y:,.0f}<br>"
                          "Paid Revenue: ₹%{customdata[0]:,.0f}<extra></extra>",
        ))
    fig.update_layout(
        title="Doctor Efficiency — Appointments vs Avg Cost (bubble = paid revenue)",
        xaxis_title="Appointment Count",
        yaxis_title="Avg Treatment Cost (₹)",
        yaxis_tickprefix="₹", yaxis_tickformat=",.0f",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=20, r=20, t=44, b=60),
        plot_bgcolor="white", paper_bgcolor="white", font_color=TEXT,
        height=360,
    )
    return fig


# ── E. Pipeline Health ────────────────────────────────────────────────────────

def fig_pipeline_funnel(data):
    df = data["pipeline_funnel"]
    fig = go.Figure(go.Funnel(
        y=df["stage"], x=df["count"],
        textinfo="value+percent initial",
        marker=dict(color=[ACCENT, TEAL, PURPLE, SUCCESS]),
        connector=dict(line=dict(color=MUTED, width=1)),
    ))
    fig.update_layout(
        title="Patient Care Pipeline — Appointments → Paid",
        margin=CM, paper_bgcolor="white", font_color=TEXT,
        height=320,
    )
    return fig


# ── Summary insight bullets ───────────────────────────────────────────────────

def build_deep_insights(data):
    nw   = data["noshow_weekday"]
    pf   = data["payment_failure"]
    ret  = data["patient_retention"]
    doc  = data["doctor_efficiency"]
    fun  = data["pipeline_funnel"]

    worst_day  = nw.loc[nw["noshow_rate"].idxmax(), "weekday"]
    worst_day_rate = nw["noshow_rate"].max()
    worst_method   = pf.iloc[0]["payment_method"]
    worst_fail_rate = pf.iloc[0]["failure_rate"]
    repeat_pct = round(ret.loc[ret["category"]=="Repeat Visitor","patients"].values[0]
                       / ret["patients"].sum() * 100, 1)
    top_doc    = doc.iloc[0]["doctor_name"]
    top_doc_rev = doc.iloc[0]["paid_revenue"]
    funnel_conv = round(fun.loc[fun["stage"]=="Paid","count"].values[0]
                        / fun.loc[fun["stage"]=="Appointments","count"].values[0] * 100, 1)

    return [
        f"📅 {worst_day} has the highest no-show rate at {worst_day_rate}% — "
        f"consider targeted reminders for appointments booked on this day.",
        f"💳 {worst_method} payments fail at {worst_fail_rate}% — the highest among all "
        f"payment channels. Review gateway reliability or require pre-authorisation.",
        f"🔁 {repeat_pct}% of patients are repeat visitors — loyalty programs or "
        f"follow-up scheduling tools could improve this further.",
        f"🩺 {top_doc} generates ₹{top_doc_rev:,.0f} in paid revenue — the top earner among "
        f"all doctors. Analysing their scheduling pattern could inform best practices.",
        f"🏥 Only {funnel_conv}% of appointments convert to a paid bill — every stage of the "
        f"pipeline (no-shows, cancellations, payment failures) is eroding throughput.",
    ]


# ── Main layout ───────────────────────────────────────────────────────────────

def deep_analysis_layout(data):
    return html.Div(children=[
        html.P(
            "Deep cross-dimensional analysis across scheduling quality, financial patterns, "
            "patient behaviour, doctor efficiency, and the appointment-to-payment pipeline. "
            "All charts use live data from the five hospital datasets.",
            style={"color": MUTED, "fontSize": "14px",
                   "marginTop": "0", "marginBottom": "20px"},
        ),

        # A — Scheduling Quality
        _section("📅 A. Scheduling Quality",
            _grid(
                _graph(fig_noshow_weekday(data)),
                _graph(fig_appt_hour(data)),
            ),
            html.Div(style={"marginTop": "16px"},
                     children=[_graph(fig_status_month_heatmap(data))]),
        ),

        # B — Financial Depth
        _section("💰 B. Financial Depth",
            _grid(
                _graph(fig_payment_failure(data)),
                _graph(fig_patient_retention(data)),
            ),
            html.Div(style={"marginTop": "16px"},
                     children=[_graph(fig_rev_spec_method(data))]),
            html.Div(style={"marginTop": "16px"},
                     children=[_graph(fig_bill_heatmap(data))]),
        ),

        # C — Patient Behaviour
        _section("👥 C. Patient Behaviour",
            _graph(fig_cost_vs_billed(data)),
        ),

        # D — Doctor Efficiency
        _section("🩺 D. Doctor Efficiency",
            _graph(fig_doctor_efficiency(data)),
        ),

        # E — Pipeline Health
        _section("🏥 E. Appointment-to-Payment Pipeline",
            _graph(fig_pipeline_funnel(data)),
        ),

        # Key insights
        _section("🔍 Deep Analysis Insights",
            html.Ul(
                style={"margin": "0", "paddingLeft": "20px"},
                children=[
                    html.Li(ins, style={"marginBottom": "10px",
                                        "fontSize": "14px", "lineHeight": "1.6"})
                    for ins in build_deep_insights(data)
                ],
            ),
        ),
    ])
