"""
Hospital Analytics Dashboard — Frontend (Dash + Plotly)
Run:  python app.py
Then open http://127.0.0.1:8050 in your browser.

Tabs
  1. Analytics   – existing KPI + chart dashboard
  2. Bill Predictor – ML form to estimate bill amount
"""

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from analysis import get_all_data
from predict_page import prediction_layout, register_predict_callbacks
from noshow_page import noshow_layout, register_noshow_callbacks
from payment_page import payment_layout, register_payment_callbacks
from revenue_page import revenue_layout, register_revenue_callbacks
from deep_analysis_page import deep_analysis_layout

# ── Data ─────────────────────────────────────────────────────────────────────
data = get_all_data()
kpis = data["kpis"]

# ── Colour palette ────────────────────────────────────────────────────────────
ACCENT   = "#3b82f6"
SUCCESS  = "#22c55e"
WARNING  = "#f59e0b"
DANGER   = "#ef4444"
PURPLE   = "#8b5cf6"
TEAL     = "#14b8a6"
GREY_BG  = "#f1f5f9"
CARD_BG  = "#ffffff"
TEXT     = "#1e293b"
MUTED    = "#64748b"

STATUS_COLORS = {
    "Completed": SUCCESS,
    "Scheduled": ACCENT,
    "Cancelled": WARNING,
    "No-show":   DANGER,
}
PAYMENT_COLORS = {
    "Paid":    SUCCESS,
    "Pending": WARNING,
    "Failed":  DANGER,
}

CHART_MARGIN = dict(l=20, r=20, t=40, b=20)

# ── Helper: KPI card ──────────────────────────────────────────────────────────
def kpi_card(label, value, color=ACCENT, icon=""):
    return html.Div(
        style={
            "background": CARD_BG,
            "borderRadius": "12px",
            "padding": "20px 24px",
            "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
            "borderTop": f"4px solid {color}",
            "minWidth": "160px",
            "flex": "1",
        },
        children=[
            html.P(label, style={"margin": "0", "fontSize": "13px", "color": MUTED, "fontWeight": "600", "letterSpacing": ".4px"}),
            html.H2(str(value), style={"margin": "6px 0 0", "fontSize": "28px", "fontWeight": "700", "color": TEXT}),
        ],
    )


# ── Charts ────────────────────────────────────────────────────────────────────

def fig_appt_status():
    df = data["appt_status"]
    colors = [STATUS_COLORS.get(s, ACCENT) for s in df["status"]]
    fig = go.Figure(go.Bar(
        x=df["status"], y=df["count"],
        marker_color=colors, text=df["count"], textposition="outside",
    ))
    fig.update_layout(title="Appointments by Status", margin=CHART_MARGIN,
                      plot_bgcolor="white", paper_bgcolor="white",
                      font_color=TEXT, yaxis_title="Count", xaxis_title="")
    return fig


def fig_appt_month():
    df = data["appt_month"].sort_values("month")
    fig = px.line(df, x="month", y="count", markers=True,
                  color_discrete_sequence=[ACCENT],
                  title="Monthly Appointment Volume")
    fig.update_layout(margin=CHART_MARGIN, plot_bgcolor="white",
                      paper_bgcolor="white", font_color=TEXT,
                      xaxis_title="", yaxis_title="Appointments")
    fig.update_xaxes(tickangle=-45)
    return fig


def fig_appt_reason():
    df = data["appt_reason"].sort_values("count", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["count"], y=df["reason_for_visit"],
        orientation="h", marker_color=PURPLE,
        text=df["count"], textposition="outside",
    ))
    fig.update_layout(title="Appointments by Reason", margin=CHART_MARGIN,
                      plot_bgcolor="white", paper_bgcolor="white",
                      font_color=TEXT, xaxis_title="Count", yaxis_title="")
    return fig


def fig_billing_status():
    df = data["billing_status"]
    colors = [PAYMENT_COLORS.get(s, ACCENT) for s in df["payment_status"]]
    fig = go.Figure(go.Pie(
        labels=df["payment_status"],
        values=df["amount"].round(2),
        marker=dict(colors=colors),
        hole=0.45,
        textinfo="label+percent",
    ))
    fig.update_layout(title="Billing Status by Amount (₹)", margin=CHART_MARGIN,
                      paper_bgcolor="white", font_color=TEXT,
                      showlegend=True)
    return fig


def fig_revenue_method():
    df = data["revenue_method"].sort_values("revenue", ascending=False)
    fig = go.Figure(go.Bar(
        x=df["payment_method"], y=df["revenue"].round(2),
        marker_color=TEAL, text=df["revenue"].round(0).astype(int),
        textposition="outside",
    ))
    fig.update_layout(title="Paid Revenue by Payment Method", margin=CHART_MARGIN,
                      plot_bgcolor="white", paper_bgcolor="white",
                      font_color=TEXT, yaxis_title="Revenue (₹)", xaxis_title="")
    return fig


def fig_treatment_cost():
    df = data["treatment_cost"].sort_values("avg_cost", ascending=False)
    fig = go.Figure(go.Bar(
        x=df["treatment_type"], y=df["avg_cost"].round(2),
        marker_color=WARNING, text=df["avg_cost"].round(0).astype(int),
        textposition="outside",
    ))
    fig.update_layout(title="Avg Treatment Cost by Type", margin=CHART_MARGIN,
                      plot_bgcolor="white", paper_bgcolor="white",
                      font_color=TEXT, yaxis_title="Avg Cost (₹)", xaxis_title="")
    return fig


def fig_treatment_dist():
    df = data["treatment_type_dist"]
    fig = px.pie(df, names="treatment_type", values="count",
                 color_discrete_sequence=px.colors.qualitative.Pastel,
                 title="Treatment Type Distribution", hole=0.4)
    fig.update_layout(margin=CHART_MARGIN, paper_bgcolor="white", font_color=TEXT)
    return fig


def fig_doctor_workload():
    df = data["doctor_appts"].head(10)
    fig = go.Figure(go.Bar(
        x=df["appointments"], y=df["doctor_name"],
        orientation="h", marker_color=ACCENT,
        text=df["appointments"], textposition="outside",
        customdata=df["specialization"],
        hovertemplate="%{y}<br>Specialization: %{customdata}<br>Appointments: %{x}<extra></extra>",
    ))
    fig.update_layout(title="Doctor Workload (Top 10)", margin=CHART_MARGIN,
                      plot_bgcolor="white", paper_bgcolor="white",
                      font_color=TEXT, xaxis_title="Appointments", yaxis_title="")
    return fig


def fig_insurance():
    df = data["insurance_dist"].sort_values("patients", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["patients"], y=df["insurance_provider"],
        orientation="h", marker_color=PURPLE,
        text=df["patients"], textposition="outside",
    ))
    fig.update_layout(title="Patients by Insurance Provider", margin=CHART_MARGIN,
                      plot_bgcolor="white", paper_bgcolor="white",
                      font_color=TEXT, xaxis_title="Patients", yaxis_title="")
    return fig


def fig_age_group():
    df = data["age_group_dist"]
    fig = px.bar(df, x="age_group", y="patients",
                 color="age_group",
                 color_discrete_sequence=px.colors.sequential.Blues_r,
                 title="Patient Age Group Distribution",
                 text="patients")
    fig.update_layout(margin=CHART_MARGIN, plot_bgcolor="white",
                      paper_bgcolor="white", font_color=TEXT,
                      showlegend=False, xaxis_title="Age Group", yaxis_title="Patients")
    fig.update_traces(textposition="outside")
    return fig


def fig_monthly_revenue():
    df = data["monthly_revenue"].sort_values("month")
    fig = px.area(df, x="month", y="revenue",
                  color_discrete_sequence=[SUCCESS],
                  title="Monthly Collected Revenue (Paid Only)")
    fig.update_layout(margin=CHART_MARGIN, plot_bgcolor="white",
                      paper_bgcolor="white", font_color=TEXT,
                      xaxis_title="", yaxis_title="Revenue (₹)")
    fig.update_xaxes(tickangle=-45)
    return fig


def fig_specialization_appts():
    df = (data["master"]
          .groupby("specialization")["appointment_id"]
          .nunique()
          .reset_index(name="appointments")
          .sort_values("appointments", ascending=False))
    fig = px.pie(df, names="specialization", values="appointments",
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title="Appointments by Specialization", hole=0.4)
    fig.update_layout(margin=CHART_MARGIN, paper_bgcolor="white", font_color=TEXT)
    return fig


# ── Insight bullets ───────────────────────────────────────────────────────────

def build_insights():
    master  = data["master"]
    billing = master[master["payment_status"].notna()]

    top_doc   = data["doctor_appts"].iloc[0]["doctor_name"]
    top_spec  = data["doctor_appts"].iloc[0]["specialization"]
    top_treat = data["treatment_type_dist"].iloc[0]["treatment_type"]
    most_ins  = data["insurance_dist"].sort_values("patients", ascending=False).iloc[0]["insurance_provider"]

    failed_amt  = billing[billing["payment_status"] == "Failed"]["amount"].sum()
    paid_amt    = billing[billing["payment_status"] == "Paid"]["amount"].sum()
    pending_amt = billing[billing["payment_status"] == "Pending"]["amount"].sum()
    total_billed = paid_amt + failed_amt + pending_amt
    recovery = round(paid_amt / total_billed * 100, 1) if total_billed > 0 else 0.0

    insights = [
        f"📌 Appointment completion rate is {kpis['completion_rate']}% — "
        f"no-show rate of {kpis['no_show_rate']}% indicates scheduling friction worth addressing.",
        f"💰 ₹{kpis['total_revenue']:,.2f} collected so far; "
        f"₹{kpis['pending_revenue']:,.2f} still pending — revenue recovery stands at {recovery}%.",
        f"🩺 Dr. {top_doc} ({top_spec}) handles the most appointments — consider workload balancing.",
        f"💊 '{top_treat}' is the most frequently used treatment type.",
        f"🏥 '{most_ins}' is the largest insurance group among registered patients.",
        f"⚠️  ₹{failed_amt:,.2f} in billing failures — reviewing failed payment channels could recover revenue.",
    ]
    return insights


# ── Section wrapper ───────────────────────────────────────────────────────────

section_style = {
    "background": CARD_BG,
    "borderRadius": "12px",
    "padding": "20px 24px",
    "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
    "marginBottom": "24px",
}

def section(title_text, *children):
    return html.Div(style=section_style, children=[
        html.H3(title_text, style={"margin": "0 0 16px", "color": TEXT, "fontSize": "16px", "fontWeight": "700"}),
        *children,
    ])


# ── Analytics tab content (pre-rendered, no callbacks needed) ─────────────────

analytics_content = html.Div([
    # KPI Row
    html.Div(
        style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"},
        children=[
            kpi_card("Total Patients",        kpis["total_patients"],                   ACCENT),
            kpi_card("Total Doctors",          kpis["total_doctors"],                    PURPLE),
            kpi_card("Total Appointments",     kpis["total_appointments"],               TEAL),
            kpi_card("Completion Rate",        f"{kpis['completion_rate']}%",            SUCCESS),
            kpi_card("No-show Rate",           f"{kpis['no_show_rate']}%",               DANGER),
            kpi_card("Total Revenue (Paid)",   f"₹{kpis['total_revenue']:,.0f}",         SUCCESS),
            kpi_card("Pending Revenue",        f"₹{kpis['pending_revenue']:,.0f}",       WARNING),
            kpi_card("Avg Treatment Cost",     f"₹{kpis['avg_treatment_cost']:,.0f}",    TEAL),
        ],
    ),

    section("📅 Appointment Analysis",
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "16px"}, children=[
            dcc.Graph(figure=fig_appt_status(),  config={"displayModeBar": False}),
            dcc.Graph(figure=fig_appt_month(),   config={"displayModeBar": False}),
            dcc.Graph(figure=fig_appt_reason(),  config={"displayModeBar": False}),
        ])
    ),

    section("💳 Billing & Revenue",
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "16px"}, children=[
            dcc.Graph(figure=fig_billing_status(),  config={"displayModeBar": False}),
            dcc.Graph(figure=fig_revenue_method(),  config={"displayModeBar": False}),
            dcc.Graph(figure=fig_monthly_revenue(), config={"displayModeBar": False}),
        ])
    ),

    section("💊 Treatment Insights",
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
            dcc.Graph(figure=fig_treatment_cost(), config={"displayModeBar": False}),
            dcc.Graph(figure=fig_treatment_dist(), config={"displayModeBar": False}),
        ])
    ),

    section("🩺 Doctor & Specialization Analysis",
        html.Div(style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px"}, children=[
            dcc.Graph(figure=fig_doctor_workload(),      config={"displayModeBar": False}),
            dcc.Graph(figure=fig_specialization_appts(), config={"displayModeBar": False}),
        ])
    ),

    section("👥 Patient Demographics",
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
            dcc.Graph(figure=fig_age_group(),  config={"displayModeBar": False}),
            dcc.Graph(figure=fig_insurance(),  config={"displayModeBar": False}),
        ])
    ),

    section("🔍 Key Insights",
        html.Ul(
            style={"margin": "0", "paddingLeft": "20px"},
            children=[
                html.Li(insight, style={"marginBottom": "10px", "fontSize": "14px", "lineHeight": "1.6"})
                for insight in build_insights()
            ],
        )
    ),
])


# ── Tab styles ────────────────────────────────────────────────────────────────

TAB_STYLE = {
    "padding": "10px 24px",
    "fontWeight": "600",
    "fontSize": "14px",
    "borderBottom": "3px solid transparent",
    "color": MUTED,
    "background": "transparent",
    "border": "none",
    "cursor": "pointer",
}
TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "borderBottom": f"3px solid {ACCENT}",
    "color": ACCENT,
}


# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="Hospital Analytics Dashboard",
                suppress_callback_exceptions=True)
register_predict_callbacks(app)
register_noshow_callbacks(app)
register_payment_callbacks(app)
register_revenue_callbacks(app)

app.layout = html.Div(
    style={"fontFamily": "'Segoe UI', system-ui, sans-serif", "background": GREY_BG,
           "minHeight": "100vh", "color": TEXT},
    children=[

        # ── Header ──
        html.Div(
            style={"background": CARD_BG, "boxShadow": "0 1px 4px rgba(0,0,0,.08)",
                   "padding": "20px 40px 0", "marginBottom": "0"},
            children=[
                html.H1("🏥 Hospital Analytics Dashboard",
                        style={"margin": "0 0 4px", "fontSize": "22px", "fontWeight": "800", "color": TEXT}),
                html.P("Data-driven insights across patients, doctors, appointments, treatments & billing",
                       style={"margin": "0 0 16px", "color": MUTED, "fontSize": "13px"}),

                # Tabs
                dcc.Tabs(
                    id="main-tabs",
                    value="tab-analytics",
                    style={"borderBottom": "none"},
                    children=[
                        dcc.Tab(label="📊  Analytics",         value="tab-analytics",
                                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="🤖  Bill Predictor",    value="tab-predictor",
                                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="🚫  No-Show Predictor", value="tab-noshow",
                                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="💳  Payment Status",    value="tab-payment",
                                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="📈  Revenue Forecast",  value="tab-revenue",
                                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        dcc.Tab(label="🔬  Deep Analysis",     value="tab-deep",
                                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    ],
                ),
            ],
        ),

        # ── Tab content ──
        html.Div(id="tab-content", style={"padding": "32px 40px"}),

        # ── Footer ──
        html.Div(
            "Made with IBM Bob",
            style={"textAlign": "center", "color": MUTED, "fontSize": "12px",
                   "borderTop": "1px solid #e2e8f0", "paddingTop": "16px",
                   "marginTop": "8px", "paddingBottom": "24px"},
        ),
    ],
)


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab(tab):
    if tab == "tab-analytics":
        return analytics_content
    if tab == "tab-predictor":
        return html.Div([
            html.H2("Bill Amount Predictor",
                    style={"margin": "0 0 4px", "fontSize": "20px", "fontWeight": "800", "color": TEXT}),
            prediction_layout(),
        ])
    if tab == "tab-noshow":
        return html.Div([
            html.H2("No-Show Risk Predictor",
                    style={"margin": "0 0 4px", "fontSize": "20px", "fontWeight": "800", "color": TEXT}),
            noshow_layout(),
        ])
    if tab == "tab-payment":
        return html.Div([
            html.H2("Payment Status Predictor",
                    style={"margin": "0 0 4px", "fontSize": "20px", "fontWeight": "800", "color": TEXT}),
            payment_layout(),
        ])
    if tab == "tab-revenue":
        return html.Div([
            html.H2("Revenue Forecast",
                    style={"margin": "0 0 4px", "fontSize": "20px", "fontWeight": "800", "color": TEXT}),
            revenue_layout(),
        ])
    return html.Div([
        html.H2("Deep Data Analysis",
                style={"margin": "0 0 4px", "fontSize": "20px", "fontWeight": "800", "color": TEXT}),
        deep_analysis_layout(data),
    ])


if __name__ == "__main__":
    app.run(debug=True)
