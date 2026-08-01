
from __future__ import annotations

from datetime import date, timedelta
import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kurtosis, norm, skew, t as student_t
import streamlit as st
import yfinance as yf
from arch import arch_model


st.set_page_config(
    page_title="Real-Time VaR and Expected Shortfall Estimation | Mountain Path Academy",
    page_icon="〽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY, BLUE, GOLD, DARK_GOLD = "#0B2545", "#0B5CAD", "#F3C84B", "#D4A017"
TEAL, GREEN, RED, PURPLE, ORANGE = "#13A89E", "#2E8B57", "#E45756", "#7C3AED", "#F28E2B"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}.stApp{background:linear-gradient(180deg,#F7F9FC 0%,#EEF3F8 100%)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0B2545,#153F69)}[data-testid="stSidebar"] *{color:#F4F8FC!important}
.hero{background:linear-gradient(115deg,#081F3A 0%,#124A78 70%,#A97908 150%);padding:30px 34px;border-radius:20px;color:white;box-shadow:0 12px 32px rgba(6,27,52,.18);margin-bottom:18px}
.hero h1{font-size:2.15rem;margin:0 0 8px;color:white;font-weight:900}.hero p{margin:0;color:#D6E9F5;font-size:1.02rem;line-height:1.55}.eyebrow{color:#F3C84B;text-transform:uppercase;letter-spacing:.12em;font-weight:800;font-size:.75rem;margin-bottom:.55rem}
.section-title{font-size:1.42rem;font-weight:900;color:#0B2545;margin:18px 0 8px}.concept-card{background:white;border:1px solid #D9E5EF;border-top:5px solid #0B5CAD;padding:17px 18px;border-radius:15px;box-shadow:0 5px 16px rgba(18,54,84,.07);min-height:165px}.concept-card h3{color:#0B2545;font-size:1.05rem;margin:0 0 7px}.concept-card p{color:#3C5368;font-size:.91rem;line-height:1.5;margin:0}
.teaching-note{background:#EAF7F5;border-left:5px solid #13A89E;padding:13px 16px;border-radius:10px;color:#153C3A;margin:10px 0}.warning-note{background:#FFF3E8;border-left:5px solid #F28E2B;padding:13px 16px;border-radius:10px;color:#57300A;margin:10px 0}.formula{background:linear-gradient(135deg,#FFF9E6,#FFF1B8);border:1px solid #E8C45B;border-left:6px solid #D4A017;padding:14px 18px;border-radius:12px;color:#3D3006;font-weight:800;margin:8px 0 14px}
.selected-company-confirmation{margin:-4px 0 10px;padding:7px 10px;background:#F3C84B;border:1px solid #D4A017;border-radius:8px;color:#071A2F!important;-webkit-text-fill-color:#071A2F!important;font-size:.84rem;font-weight:800}.selected-company-confirmation *{color:#071A2F!important;-webkit-text-fill-color:#071A2F!important}
.profile-card{background:linear-gradient(135deg,#071A2F,#123B65);border:1px solid rgba(243,200,75,.42);border-radius:14px;padding:17px;margin:15px 0 8px;box-shadow:0 7px 20px rgba(0,0,0,.18)}.profile-card .name{color:#F3C84B!important;font-weight:850;font-size:1rem;margin:0 0 5px}.profile-card .title{color:#D7E9FA!important;font-size:.81rem;line-height:1.4;margin:0 0 8px}.profile-card .stats{color:#AFC7DE!important;font-size:.76rem;line-height:1.45;margin:4px 0}.profile-card .links{margin-top:11px;display:flex;gap:11px;flex-wrap:wrap}.profile-card .links a{color:#F3C84B!important;text-decoration:none;font-size:.78rem;font-weight:750}.profile-card .links a:hover{color:#FFF!important;text-decoration:underline}
.about-section{background:linear-gradient(125deg,#0B2545,#123F69);color:#EAF3FC;border:1px solid rgba(212,160,23,.45);border-radius:17px;padding:26px 30px;margin:24px 0 12px;box-shadow:0 10px 27px rgba(11,37,69,.16)}.about-section h3{color:#F3C84B!important;margin:0 0 11px}.about-section p{color:#EAF3FC;line-height:1.62;margin:7px 0}.about-section .highlight{color:#F3C84B;font-weight:800}.academy-link{display:inline-block;margin-top:13px;padding:8px 16px;background:#D4A017;color:#071A2F!important;border-radius:8px;text-decoration:none;font-weight:850}
.mp-footer{text-align:center;padding:23px 0 8px;margin-top:25px;border-top:1px solid rgba(212,160,23,.4);color:#64778B;font-size:.84rem}.mp-footer .footer-brand{color:#0B2545;font-size:1.12rem;font-weight:850}.mp-footer .footer-profile{color:#38556F;margin:5px 0 8px}.mp-footer a{color:#0B4F86;text-decoration:none;font-weight:750;margin:0 7px}.mp-footer a:hover{color:#A97908;text-decoration:underline}
[data-testid="stMetric"]{background:#FFF;border:1px solid #DDE8F1;padding:13px;border-radius:14px}.stTabs [data-baseweb="tab-list"]{gap:9px!important;flex-wrap:wrap!important;background:#D8E3ED!important;padding:10px!important;border-radius:14px!important}.stTabs button[data-baseweb="tab"]{flex:1 1 145px!important;min-height:52px!important;background:#0B2545!important;border:2px solid #F3C84B!important;border-radius:10px!important;color:#F3C84B!important}.stTabs button[data-baseweb="tab"] p{color:#F3C84B!important;font-weight:850!important}.stTabs button[data-baseweb="tab"][aria-selected="true"]{background:linear-gradient(135deg,#F3C84B,#D4A017)!important}.stTabs button[data-baseweb="tab"][aria-selected="true"] p{color:#071A2F!important}.stButton button,.stDownloadButton button{background:#0B3B67!important;color:white!important;border-radius:10px!important;font-weight:800!important}
.stTabs button[data-baseweb="tab"]{box-shadow:0 3px 8px rgba(11,37,69,.22)!important;transition:all .15s ease!important}.stTabs button[data-baseweb="tab"]:hover{background:#164E7A!important;border-color:#FFF1AC!important;transform:translateY(-1px)!important}.stTabs button[data-baseweb="tab"][aria-selected="true"]{border-color:#A97908!important;box-shadow:0 4px 12px rgba(169,121,8,.35)!important}.stTabs [data-baseweb="tab-highlight"]{display:none!important}.stTabs button[data-baseweb="tab"]:focus-visible{outline:4px solid #13A89E!important;outline-offset:3px!important}
.stButton button:hover,.stDownloadButton button:hover{background:#D4A017!important;color:#071A2F!important;border-color:#D4A017!important}
section[data-testid="stSidebar"] div[data-testid="stButton"] button{background:linear-gradient(135deg,#F3C84B,#D4A017)!important;color:#071A2F!important;-webkit-text-fill-color:#071A2F!important;border:2px solid #F9DC79!important;min-height:46px!important;border-radius:11px!important;font-weight:850!important;box-shadow:0 5px 14px rgba(0,0,0,.22)!important}section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover{background:#FFF1AC!important;border-color:#FFF1AC!important;transform:translateY(-1px)}section[data-testid="stSidebar"] div[data-testid="stButton"] button p{color:#071A2F!important;-webkit-text-fill-color:#071A2F!important;font-weight:850!important}
section[data-testid="stSidebar"] label p{color:#F3C84B!important;font-weight:850!important}section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"]>div{background:#FFF!important;border:2px solid #F3C84B!important}section[data-testid="stSidebar"] div[data-testid="stSelectbox"] *{color:#0B2545!important;-webkit-text-fill-color:#0B2545!important;font-weight:800!important}
/* Keep sidebar number inputs readable across Streamlit versions. */
section[data-testid="stSidebar"] div[data-testid="stNumberInput"]>div,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] [data-baseweb="input"]{background:#FFF!important;border:2px solid #F3C84B!important;border-radius:10px!important;overflow:hidden!important}
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input,
section[data-testid="stSidebar"] input[type="number"]{background:#FFF!important;color:#071A2F!important;-webkit-text-fill-color:#071A2F!important;caret-color:#071A2F!important;font-weight:850!important;opacity:1!important}
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button{background:#E8EEF5!important;color:#071A2F!important;border-color:#CBD8E5!important;opacity:1!important}
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button svg,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button svg path{color:#071A2F!important;fill:#071A2F!important;stroke:#071A2F!important;opacity:1!important}
@media(max-width:700px){.hero{padding:22px}.hero h1{font-size:1.7rem}}
</style>
""",
    unsafe_allow_html=True,
)

INSTRUMENTS = {
    "NIFTY 50 Index": "^NSEI", "Reliance Industries": "RELIANCE.NS", "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS", "Infosys": "INFY.NS", "TCS": "TCS.NS", "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "Larsen & Toubro": "LT.NS", "ITC": "ITC.NS", "Tata Motors": "TATAMOTORS.NS",
    "Adani Enterprises": "ADANIENT.NS", "Asian Paints": "ASIANPAINT.NS", "Bajaj Finance": "BAJFINANCE.NS",
    "Maruti Suzuki": "MARUTI.NS", "Sun Pharma": "SUNPHARMA.NS", "Tata Steel": "TATASTEEL.NS", "Wipro": "WIPRO.NS",
}


def section(title: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def note(text: str, warning: bool = False) -> None:
    css = "warning-note" if warning else "teaching-note"
    st.markdown(f"<div class='{css}'>{text}</div>", unsafe_allow_html=True)


def card(title: str, body: str, color: str = BLUE) -> str:
    return f"<div class='concept-card' style='border-top-color:{color}'><h3>{title}</h3><p>{body}</p></div>"


def style_fig(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", font=dict(family="Inter", color=NAVY), margin=dict(l=35, r=25, t=60, b=35), hoverlabel=dict(bgcolor="white", font_color=NAVY), legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(gridcolor="#E7EEF4"); fig.update_yaxes(gridcolor="#E7EEF4")
    return fig


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker: str, years: int) -> pd.DataFrame:
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=365 * years + 40)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        raise ValueError("No observations were returned by the market-data provider.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    data = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")}).dropna()
    data["Return"] = np.log(data["Close"] / data["Close"].shift(1))
    return data.dropna()


def demo_prices(years: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=max(252 * years, 300))
    ret = np.zeros(len(idx)); variance = np.full(len(idx), 0.00012)
    for i in range(1, len(idx)):
        shock_scale = 1.35 if rng.random() < 0.045 else 1.0
        ret[i] = 0.0003 + math.sqrt(variance[i - 1]) * rng.standard_t(6) / math.sqrt(1.5) * shock_scale
        variance[i] = 0.000003 + 0.09 * ret[i] ** 2 + 0.88 * variance[i - 1] + 0.05 * ret[i] ** 2 * (ret[i] < 0)
    return pd.DataFrame({"Close": 1000 * np.exp(np.cumsum(ret)), "Return": ret}, index=idx)


def volatility_acf(series: pd.Series, lags: int) -> pd.DataFrame:
    squared = series.pow(2)
    return pd.DataFrame({"Lag": range(1, lags + 1), "Squared-return autocorrelation": [squared.autocorr(i) for i in range(1, lags + 1)]})


@st.cache_data(show_spinner=False)
def fit_forecasting_models(returns: pd.Series, horizon: int, decay: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit four teaching models and return annualised percentage-volatility forecasts."""
    r = (returns.dropna() * 100).clip(-25, 25)
    if len(r) < 150:
        raise ValueError("At least 150 daily returns are required for stable model fitting.")
    models = {
        "ARCH(5)": arch_model(r, mean="Constant", vol="ARCH", p=5, dist="t", rescale=False),
        "GARCH(1,1)": arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False),
        "EGARCH(1,1)": arch_model(r, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False),
    }
    dates = pd.bdate_range(returns.index[-1] + pd.Timedelta(days=1), periods=horizon)
    latest_ewma = returns.ewm(alpha=1 - decay, adjust=False).var(bias=False).iloc[-1]
    forecasts = pd.DataFrame({"EWMA": np.repeat(np.sqrt(latest_ewma * 252) * 100, horizon)}, index=dates)
    diagnostics = [{"Model": "EWMA", "Persistence": decay, "AIC": np.nan, "Log likelihood": np.nan, "Fit status": "Fixed decay; not estimated"}]
    for name, specification in models.items():
        fit = specification.fit(disp="off")
        fit_note = "Converged" if fit.convergence_flag == 0 else f"Optimizer code {fit.convergence_flag}"
        if name == "EGARCH(1,1)":
            # arch provides an analytic EGARCH forecast only at h=1. For later steps,
            # the expected centred innovation terms are zero, leaving log variance
            # to mean-revert through omega + beta * log(previous variance).
            first_variance = float(fit.forecast(horizon=1, reindex=False).variance.iloc[-1, 0])
            variance_ceiling = max(float(r.var()) * 100, 1.0)
            if not np.isfinite(first_variance) or first_variance <= 0 or first_variance > variance_ceiling:
                first_variance = float(fit.conditional_volatility.iloc[-1] ** 2)
                fit_note += " · first step uses latest fitted variance"
            if not np.isfinite(first_variance) or first_variance <= 0 or first_variance > variance_ceiling:
                first_variance = float(r.var())
                fit_note += " · unstable fit fallback uses sample variance"
            omega, beta = float(fit.params.get("omega", 0)), float(fit.params.get("beta[1]", 0))
            variance = [first_variance]
            log_variance = math.log(max(first_variance, 1e-10))
            for _ in range(1, horizon):
                log_variance = float(np.clip(omega + beta * log_variance, -12, 8))
                variance.append(math.exp(log_variance))
            variance = np.asarray(variance)
        else:
            variance = fit.forecast(horizon=horizon, reindex=False).variance.iloc[-1].to_numpy()
        forecasts[name] = np.sqrt(np.maximum(variance, 0) * 252)
        params = fit.params
        if name == "ARCH(5)": persistence = sum(float(params.get(f"alpha[{i}]", 0)) for i in range(1, 6))
        else: persistence = float(params.get("beta[1]", np.nan)) + (float(params.get("alpha[1]", 0)) if name == "GARCH(1,1)" else 0)
        diagnostics.append({"Model": name, "Persistence": persistence, "AIC": fit.aic, "Log likelihood": fit.loglikelihood, "Fit status": fit_note})
    return forecasts, pd.DataFrame(diagnostics)


def build_excel_download(data: pd.DataFrame, asset: str, ticker: str, source: str, window: int, decay: float) -> bytes:
    """Build a polished, analysis-ready Excel workbook in memory."""
    output = io.BytesIO()
    daily = data.reset_index().copy()
    daily.columns = ["Date", "Adjusted Close (₹)", "Daily Log Return", "Rolling Volatility", "EWMA Volatility", "Absolute Return"]
    daily["Squared Return"] = daily["Daily Log Return"].pow(2)
    daily["Date"] = pd.to_datetime(daily["Date"]).dt.tz_localize(None)
    last_row = len(daily) + 1

    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="dd-mmm-yyyy") as writer:
        book = writer.book
        navy, blue, gold, pale_blue, pale_gold = "#0B2545", "#0B5CAD", "#F3C84B", "#EAF1F7", "#FFF1B8"
        title = book.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": navy, "font_size": 18, "valign": "vcenter"})
        subtitle = book.add_format({"font_color": "#DDEAF4", "bg_color": navy, "font_size": 10, "valign": "vcenter"})
        section_fmt = book.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": blue, "font_size": 11, "valign": "vcenter"})
        header = book.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": navy, "align": "center", "valign": "vcenter", "text_wrap": True})
        label = book.add_format({"bold": True, "font_color": navy, "bg_color": pale_blue})
        value = book.add_format({"font_color": navy})
        currency = book.add_format({"font_color": navy, "num_format": '₹#,##0.00;[Red](₹#,##0.00);-'})
        percent = book.add_format({"font_color": navy, "num_format": '0.00%;[Red](0.00%);-'})
        note_fmt = book.add_format({"font_color": "#3C5368", "bg_color": pale_gold, "text_wrap": True, "valign": "top"})

        summary = book.add_worksheet("Summary")
        writer.sheets["Summary"] = summary
        summary.hide_gridlines(2); summary.set_column("A:A", 31); summary.set_column("B:B", 25); summary.set_column("C:F", 15)
        summary.set_row(0, 30); summary.merge_range("A1:F1", "Real-Time VaR and Expected Shortfall Estimation", title)
        summary.merge_range("A2:F2", "Mountain Path Academy · Live-data learning workbook", subtitle)
        summary.merge_range("A4:F4", "Analysis profile", section_fmt)
        profile = [("Instrument", asset), ("Ticker", ticker), ("Source", source), ("Observations", len(data)), ("Rolling window", f"{window} trading days"), ("EWMA decay factor", decay)]
        for row, (item, item_value) in enumerate(profile, 4):
            summary.write(row, 0, item, label); summary.write(row, 1, item_value, value)
        summary.merge_range("A12:F12", "Formula-driven key outputs", section_fmt)
        summary.write("A13", "Latest adjusted close", label); summary.write_formula("B13", f"='Daily Data'!B{last_row + 1}", currency)
        summary.write("A14", "Annualised historic volatility", label); summary.write_formula("B14", f"=STDEV.S('Daily Data'!C3:C{last_row + 1})*SQRT(252)", percent)
        summary.write("A15", f"Latest {window}-day rolling volatility", label); summary.write_formula("B15", f"='Daily Data'!D{last_row + 1}", percent)
        summary.write("A16", "Lag-1 squared-return correlation", label); summary.write_formula("B16", f"=CORREL('Daily Data'!G4:G{last_row + 1},'Daily Data'!G3:G{last_row})", book.add_format({"num_format": "0.000"}))
        summary.merge_range("A18:F18", "Interpretation", section_fmt)
        summary.merge_range("A19:F22", "Volatility is time-varying, commonly clusters after large moves, can remain persistent, may react asymmetrically to negative returns, and often exhibits fat tails. These are sample diagnostics—not forecasts or trading recommendations.", note_fmt)
        summary.merge_range("A24:F24", "Source and limitations", section_fmt)
        summary.merge_range("A25:F28", f"Source: {source}. Prices may be delayed. Adjusted closes and statistical estimates depend on the selected sample, frequency, rolling window, and annualisation convention. Educational use only.", note_fmt)
        summary.freeze_panes(3, 0)

        daily.to_excel(writer, sheet_name="Daily Data", index=False, startrow=1)
        sheet = writer.sheets["Daily Data"]
        sheet.hide_gridlines(2); sheet.freeze_panes(2, 1); sheet.autofilter(1, 0, last_row, 6); sheet.set_row(0, 30)
        sheet.merge_range("A1:G1", f"{asset} ({ticker}) · Price, returns and volatility", title)
        for col, name in enumerate(daily.columns): sheet.write(1, col, name, header)
        sheet.set_column("A:A", 14, book.add_format({"num_format": "dd-mmm-yyyy"})); sheet.set_column("B:B", 20, currency)
        sheet.set_column("C:C", 18, percent); sheet.set_column("D:E", 20, percent); sheet.set_column("F:F", 17, percent); sheet.set_column("G:G", 17, book.add_format({"num_format": "0.000000"}))
        for excel_row, (_, record) in enumerate(daily.iterrows(), 2):
            sheet.write_number(excel_row, 2, float(record["Daily Log Return"]), percent)
            if pd.notna(record["Rolling Volatility"]): sheet.write_number(excel_row, 3, float(record["Rolling Volatility"]) / 100, percent)
            if pd.notna(record["EWMA Volatility"]): sheet.write_number(excel_row, 4, float(record["EWMA Volatility"]) / 100, percent)
            sheet.write_number(excel_row, 5, float(record["Absolute Return"]), percent)
            sheet.write_number(excel_row, 6, float(record["Squared Return"]), book.add_format({"num_format": "0.000000"}))
        chart = book.add_chart({"type": "line"})
        chart.add_series({"name": f"Rolling {window}D", "categories": ["Daily Data", 2, 0, last_row, 0], "values": ["Daily Data", 2, 3, last_row, 3], "line": {"color": "#7C3AED", "width": 2}})
        chart.add_series({"name": "EWMA", "categories": ["Daily Data", 2, 0, last_row, 0], "values": ["Daily Data", 2, 4, last_row, 4], "line": {"color": "#13A89E", "width": 2}})
        chart.set_title({"name": "Annualised volatility through time"}); chart.set_x_axis({"date_axis": True, "num_format": "dd-mmm"}); chart.set_y_axis({"name": "Volatility", "num_format": "0%"}); chart.set_legend({"position": "bottom"}); chart.set_style(10)
        sheet.insert_chart("H3", chart, {"x_scale": 1.35, "y_scale": 1.2})

        guide = book.add_worksheet("Learning Guide")
        guide.hide_gridlines(2); guide.set_column("A:A", 25); guide.set_column("B:B", 92); guide.set_row(0, 30)
        guide.merge_range("A1:B1", "VaR, Expected Shortfall and Forecast Volatility · Learning Guide", title)
        guide.write_row("A3", ["Characteristic", "Meaning and diagnostic"], header)
        lessons = [
            ("Time variation", "Volatility changes across calm and turbulent regimes. Compare rolling and EWMA estimates."),
            ("Clustering", "Large absolute returns tend to occur near other large absolute returns. Inspect the daily-return chart."),
            ("Persistence", "The effect of a shock decays gradually. Positive squared-return autocorrelation is supporting evidence."),
            ("Asymmetry", "Negative and positive returns of similar size may affect subsequent volatility differently."),
            ("Fat tails", "Extreme returns are more frequent than a fitted normal distribution predicts; excess kurtosis helps diagnose this."),
            ("Horizon sensitivity", "Results vary with frequency, sample period, rolling window, EWMA decay and the √252 convention."),
        ]
        body = book.add_format({"font_color": navy, "text_wrap": True, "valign": "top"})
        for row, (characteristic, explanation) in enumerate(lessons, 3):
            guide.write(row, 0, characteristic, label); guide.write(row, 1, explanation, body); guide.set_row(row, 42)
        guide.merge_range("A11:B11", "Important: volatility measures dispersion—not direction, expected return, Value at Risk, or maximum possible loss.", note_fmt)
        guide.freeze_panes(2, 0)
        book.set_properties({"title": "Real-Time VaR and Expected Shortfall Estimation (Basel Framework) Using Forecasted Volatilities", "author": "Mountain Path Academy", "comments": "Generated by the Streamlit learning studio"})
    return output.getvalue()


with st.sidebar:
    st.markdown("## 〽️ Analysis Controls")
    asset = st.selectbox("Choose a stock or index", list(INSTRUMENTS))
    ticker = INSTRUMENTS[asset]
    st.markdown(f'<div class="selected-company-confirmation">Selected: {asset}</div>', unsafe_allow_html=True)
    years = st.selectbox("Daily-price period", [1, 3, 5], index=1, format_func=lambda x: f"{x} year" if x == 1 else f"{x} years")
    window = st.slider("Rolling window (trading days)", 10, 90, 21)
    ewma_lambda = st.slider("EWMA decay factor (λ)", 0.80, 0.99, 0.94, 0.01)
    use_demo = st.toggle("Use classroom simulation", value=False)
    st.markdown("### Tail-risk controls")
    risk_confidence = st.select_slider("VaR / ES confidence level", options=[0.90, 0.95, 0.975, 0.99], value=0.975, format_func=lambda x: f"{x:.1%}")
    holding_days = st.select_slider("Holding period", options=[1, 5, 10, 20], value=10, format_func=lambda x: f"{x} trading day" if x == 1 else f"{x} trading days")
    portfolio_value = st.number_input("Illustrative portfolio value (₹)", 100_000, 100_000_000, 1_000_000, 100_000)
    st.markdown("---")
    st.caption("Prices: Yahoo Finance adjusted daily close. Values may be delayed or differ by provider convention.")
    cache_was_refreshed = st.session_state.pop("cache_was_refreshed", False)
    refresh_label = "✅ Data Refreshed" if cache_was_refreshed else "↻ Refresh Cached Data"
    if st.button(refresh_label, use_container_width=True):
        st.cache_data.clear()
        st.session_state["cache_was_refreshed"] = True
        st.rerun()
    st.markdown(
        """<div class='profile-card'>
        <p class='name'>Prof. V. Ravichandran</p>
        <p class='title'>Visiting Professor &amp; Professor of Practice at Leading Business Schools<br>
        Founder — The Mountain Path Academy</p>
        <p class='stats'>28+ years of industry experience<br>12+ years teaching Finance &amp; Financial Analytics</p>
        <div class='links'>
          <a href='https://themountainpathacademy.com' target='_blank'>🏔️ Academy</a>
          <a href='https://www.linkedin.com/in/trichyravis' target='_blank'>💼 LinkedIn</a>
          <a href='https://github.com/trichyravis' target='_blank'>💻 GitHub</a>
        </div></div>""", unsafe_allow_html=True)

source = "Reproducible classroom simulation"
if use_demo:
    prices = demo_prices(years)
else:
    try:
        prices = load_prices(ticker, years)
        source = "Yahoo Finance · adjusted daily close"
    except Exception as exc:
        prices = demo_prices(years)
        st.warning(f"Live data is temporarily unavailable ({exc}). Showing the classroom simulation instead.")

prices["Rolling volatility"] = prices["Return"].rolling(window).std() * np.sqrt(252) * 100
prices["EWMA volatility"] = prices["Return"].ewm(alpha=1 - ewma_lambda, adjust=False).std(bias=False) * np.sqrt(252) * 100
prices["Absolute return"] = prices["Return"].abs()
annual_vol = prices["Return"].std() * np.sqrt(252) * 100
negative_vol = prices.loc[prices["Return"] < 0, "Return"].std() * np.sqrt(252) * 100
positive_vol = prices.loc[prices["Return"] >= 0, "Return"].std() * np.sqrt(252) * 100
acf1 = prices["Return"].pow(2).autocorr(1)
excess_kurtosis = kurtosis(prices["Return"], fisher=True, bias=False)

st.markdown("<div class='hero'><div class='eyebrow'>The Mountain Path Academy · Applied Finance</div><h1>Real-Time VaR and Expected Shortfall Estimation (Basel Framework) Using Forecasted Volatilities (EWMA, ARCH, GARCH, EGARCH)</h1><p>Transform live market observations and conditional-volatility forecasts into transparent tail-risk estimates, model comparisons and Basel-oriented learning outputs.</p></div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest adjusted close", f"₹{prices['Close'].iloc[-1]:,.2f}")
m2.metric("Annualised volatility", f"{annual_vol:.2f}%")
m3.metric(f"Latest {window}D volatility", f"{prices['Rolling volatility'].dropna().iloc[-1]:.2f}%")
m4.metric("Last market date", prices.index[-1].strftime("%d %b %Y"))
st.caption(f"{asset} ({ticker}) · {len(prices):,} daily observations · {source}")

tabs = st.tabs(["🎓 Educational", "🧭 Characteristics map", "🌊 Time variation", "🔗 Clustering & persistence", "⚖️ Asymmetry", "📊 Fat tails", "🔮 Forecasting models", "🛡️ VaR & Basel ES", "🧮 10 solved illustrations", "🧪 Learning check"])

with tabs[0]:
    section("What volatility is—and what it is not")
    left, right = st.columns(2)
    with left:
        st.markdown(card("Core definition", "Volatility measures the dispersion of returns. Higher volatility means returns are spread more widely around their average; it says nothing by itself about whether price will rise or fall.", BLUE), unsafe_allow_html=True)
        st.markdown("<div class='formula'>Daily log return: rₜ = ln(Pₜ / Pₜ₋₁)<br>Annualised historic volatility: σ = SD(rₜ) × √252</div>", unsafe_allow_html=True)
    with right:
        st.markdown(card("Why characteristics matter", "Constant-volatility models are convenient, but markets commonly show changing regimes, clusters of large moves, persistence, asymmetry and unusually frequent extremes. Recognising these patterns improves model interpretation.", GOLD), unsafe_allow_html=True)
        note("A volatility estimate is conditional on the data frequency, lookback period, model and annualisation convention.")
    section("A practical observation sequence")
    steps = st.columns(4)
    for col, item in zip(steps, [
        ("1 · Observe", "Plot returns and rolling volatility to identify calm and turbulent periods.", BLUE),
        ("2 · Diagnose", "Check squared-return correlation, asymmetry, skewness and excess kurtosis.", PURPLE),
        ("3 · Compare", "Change the window and EWMA decay factor to test sensitivity.", TEAL),
        ("4 · Interpret", "Describe evidence carefully without treating a sample pattern as certainty.", ORANGE),
    ]): col.markdown(card(*item), unsafe_allow_html=True)
    section("Historic, conditional and implied volatility")
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("Historic / realised", "Computed from past returns. Useful for describing what happened, but sensitive to the chosen sample.", BLUE), unsafe_allow_html=True)
    c2.markdown(card("Conditional", "A rolling, EWMA or GARCH-style estimate that changes with information available through time.", TEAL), unsafe_allow_html=True)
    c3.markdown(card("Implied", "Backed out from option prices using a pricing model. It reflects market pricing and model assumptions—not a guaranteed forecast.", GOLD), unsafe_allow_html=True)
    note("Use the remaining tabs as a guided evidence lab. Every chart responds to the instrument and assumptions selected in the sidebar.")

with tabs[1]:
    section("Six ideas to carry into every volatility analysis")
    rows = [
        [("Time-varying", "Risk is not constant. Calm and turbulent regimes appear in the same return series.", BLUE), ("Clustering", "Large moves tend to follow large moves, and small moves tend to follow small moves.", PURPLE), ("Persistence", "A shock can influence conditional risk for many sessions before gradually fading.", TEAL)],
        [("Asymmetry", "Negative and positive news of similar size need not have the same effect on future volatility.", RED), ("Fat tails", "Extreme observations occur more often than a normal distribution predicts.", ORANGE), ("Horizon-sensitive", "The measured risk changes with the sampling frequency, window, and annualisation rule.", GREEN)],
    ]
    for row in rows:
        cols = st.columns(3)
        for col, item in zip(cols, row): col.markdown(card(*item), unsafe_allow_html=True)
    note("Volatility describes the magnitude and variability of returns—not their direction, expected return, or maximum possible loss.")

with tabs[2]:
    section("Volatility moves through regimes")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.10, row_heights=[.52, .48], subplot_titles=("Adjusted price", "Annualised rolling and EWMA volatility"))
    fig.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name="Adjusted close", line=dict(color=BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=prices["Rolling volatility"], name=f"Rolling {window}D", line=dict(color=PURPLE, width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=prices["EWMA volatility"], name=f"EWMA λ={ewma_lambda:.2f}", line=dict(color=TEAL, width=2)), row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1); fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    st.plotly_chart(style_fig(fig, 590), use_container_width=True)
    st.markdown("<div class='formula'>Annualised volatility = daily standard deviation × √252</div>", unsafe_allow_html=True)
    note("Change the rolling window in the sidebar. Short windows react quickly but are noisy; long windows are smoother but slow to recognise regime changes.")

with tabs[3]:
    section("Large moves leave an observable footprint")
    left, right = st.columns([1.55, 1])
    with left:
        colors = np.where(prices["Return"] >= 0, TEAL, RED)
        fig = go.Figure(go.Bar(x=prices.index, y=prices["Return"] * 100, marker_color=colors, name="Daily return"))
        fig.update_layout(title="Daily log returns"); fig.update_yaxes(title="Return (%)")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        acf = volatility_acf(prices["Return"], 20)
        fig = go.Figure(go.Bar(x=acf["Lag"], y=acf.iloc[:, 1], marker_color=PURPLE))
        fig.add_hline(y=0, line_color=NAVY); fig.update_layout(title="Memory in squared returns"); fig.update_xaxes(title="Trading-day lag"); fig.update_yaxes(title="Autocorrelation")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Lag-1 squared-return correlation", f"{acf1:.3f}")
    c2.metric("EWMA half-life", f"{math.log(.5) / math.log(ewma_lambda):.1f} trading days")
    note("Positive correlations in squared returns support volatility clustering. Persistence means today's volatility contains information about near-future volatility; it does not imply that return direction is predictable.")

with tabs[4]:
    section("Does the sign of the move matter?")
    bucket = pd.qcut(prices["Return"], q=10, duplicates="drop")
    response = prices.assign(Bucket=bucket).groupby("Bucket", observed=True).agg(mean_return=("Return", "mean"), next_abs_return=("Absolute return", lambda x: prices["Absolute return"].shift(-1).reindex(x.index).mean()))
    fig = go.Figure(go.Scatter(x=response["mean_return"] * 100, y=response["next_abs_return"] * 100, mode="markers+lines", marker=dict(size=11, color=np.where(response["mean_return"] < 0, RED, TEAL))))
    fig.update_layout(title="Today's return bucket versus next-day absolute return"); fig.update_xaxes(title="Average current return (%)"); fig.update_yaxes(title="Average next-day |return| (%)")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Volatility on negative-return days", f"{negative_vol:.2f}%")
    b.metric("Volatility on positive-return days", f"{positive_vol:.2f}%")
    c.metric("Negative / positive ratio", f"{negative_vol / positive_vol:.2f}×")
    note("A ratio above 1 is sample evidence of stronger dispersion on negative days. The leverage effect is a dynamic claim, so this visual is a diagnostic—not proof of causality.", warning=True)

with tabs[5]:
    section("Compare observed returns with the normal curve")
    values = prices["Return"].dropna() * 100
    x = np.linspace(values.quantile(.005), values.quantile(.995), 250)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, histnorm="probability density", nbinsx=55, name="Observed returns", marker_color=BLUE, opacity=.68))
    fig.add_trace(go.Scatter(x=x, y=norm.pdf(x, values.mean(), values.std()), name="Fitted normal", line=dict(color=GOLD, width=3)))
    fig.update_layout(title="Return distribution and fitted normal density", barmode="overlay"); fig.update_xaxes(title="Daily log return (%)"); fig.update_yaxes(title="Density")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Skewness", f"{skew(values, bias=False):.2f}")
    b.metric("Excess kurtosis", f"{excess_kurtosis:.2f}")
    threshold = 3 * values.std(); c.metric("Moves beyond ±3σ", f"{(values.abs() > threshold).sum()}")
    note("A normal distribution has excess kurtosis of 0. A positive estimate indicates heavier tails in this sample, making extreme returns more common than the fitted bell curve suggests.")

with tabs[6]:
    section("Forecast volatility with four conditional models")
    horizon = st.slider("Forward forecast horizon (trading days)", 1, 30, 10, key="forecast_horizon")
    st.markdown("<div class='formula'>All plotted forecasts are annualised: forecast daily standard deviation × √252</div>", unsafe_allow_html=True)
    try:
        with st.spinner("Fitting ARCH-family models…"):
            model_forecasts, model_diagnostics = fit_forecasting_models(prices["Return"], horizon, ewma_lambda)
        fig = go.Figure()
        forecast_colors = {"EWMA": GOLD, "ARCH(5)": PURPLE, "GARCH(1,1)": BLUE, "EGARCH(1,1)": RED}
        for model in model_forecasts:
            fig.add_trace(go.Scatter(x=model_forecasts.index, y=model_forecasts[model], mode="lines+markers", name=model, line=dict(color=forecast_colors[model], width=3)))
        fig.update_layout(title=f"{horizon}-day annualised volatility forecast · {asset}", hovermode="x unified")
        fig.update_xaxes(title="Forecast business date"); fig.update_yaxes(title="Annualised volatility (%)")
        st.plotly_chart(style_fig(fig, 480), use_container_width=True)
        metric_cols = st.columns(4)
        for col, model in zip(metric_cols, model_forecasts.columns):
            col.metric(f"{model} · Day 1", f"{model_forecasts[model].iloc[0]:.2f}%", delta=f"Day {horizon}: {model_forecasts[model].iloc[-1]:.2f}%")
        with st.expander("Model diagnostics and comparison", expanded=False):
            st.dataframe(model_diagnostics.style.format({"Persistence": "{:.3f}", "AIC": "{:,.1f}", "Log likelihood": "{:,.1f}"}, na_rep="—"), use_container_width=True, hide_index=True)
            st.caption("A lower AIC is preferable only when models use the same return sample. Persistence close to 1 implies slow decay, but definitions differ across model families.")
    except Exception as exc:
        st.error(f"The forecasting models could not be fitted for this sample: {exc}")

    section("How the forecasting models work")
    model_rows = [
        [("EWMA", "Updates variance with a fixed decay: σ²ₜ = λσ²ₜ₋₁ + (1−λ)r²ₜ₋₁. It reacts quickly and needs no numerical fitting.", GOLD), ("ARCH(5)", "Forecasts variance from a constant plus five recent squared shocks. Each lag receives its own estimated weight.", PURPLE)],
        [("GARCH(1,1)", "Combines the latest squared shock with the previous conditional variance, producing a compact and persistent forecast.", BLUE), ("EGARCH(1,1)", "Models log variance and includes a sign term, allowing negative and positive shocks to affect future volatility differently.", RED)],
    ]
    for row in model_rows:
        cols = st.columns(2)
        for col, item in zip(cols, row): col.markdown(card(*item), unsafe_allow_html=True)

    section("Limitations to state with every forecast")
    limitations = st.columns(4)
    for col, item in zip(limitations, [
        ("EWMA", "The decay factor is imposed, forecasts are flat without mean reversion, and shock direction is ignored.", GOLD),
        ("ARCH", "Many lags consume parameters, estimates can be unstable, and long-memory behaviour is represented inefficiently.", PURPLE),
        ("GARCH", "The standard form treats equal-sized positive and negative shocks symmetrically and can miss jumps or structural breaks.", BLUE),
        ("EGARCH", "Results depend on specification and distribution; multi-step expected log-variance recursion can smooth away future shock uncertainty.", RED),
    ]): col.markdown(card(*item), unsafe_allow_html=True)
    note("A forecast is conditional on the selected history and model. Compare models, examine residual diagnostics, and re-estimate after major regime changes; never treat a single estimate as certainty.", warning=True)

with tabs[7]:
    section("Compare Value at Risk and Expected Shortfall")
    st.markdown("<div class='formula'>VaR is a loss threshold. Expected Shortfall is the average loss beyond that threshold.</div>", unsafe_allow_html=True)
    try:
        with st.spinner("Generating volatility-conditioned tail-risk estimates…"):
            risk_forecasts, _ = fit_forecasting_models(prices["Return"], holding_days, ewma_lambda)

        confidence, tail_probability = risk_confidence, 1 - risk_confidence
        z = norm.ppf(confidence)
        clean_returns = prices["Return"].dropna()
        fitted_df, _, _ = student_t.fit(clean_returns, floc=0)
        fitted_df = max(float(fitted_df), 2.1)
        t_lower_q = student_t.ppf(tail_probability, fitted_df)
        t_standardiser = math.sqrt((fitted_df - 2) / fitted_df)
        t_var_multiplier = -t_lower_q * t_standardiser
        t_es_multiplier = t_standardiser * ((fitted_df + t_lower_q ** 2) / (fitted_df - 1)) * student_t.pdf(t_lower_q, fitted_df) / tail_probability

        rows = []
        for forecast_model in risk_forecasts.columns:
            annual_path = risk_forecasts[forecast_model].to_numpy() / 100
            horizon_sigma = math.sqrt(float(np.sum((annual_path ** 2) / 252)))
            rows.extend([
                {"Volatility forecast": forecast_model, "Tail model": "Normal parametric", "VaR (₹)": portfolio_value * z * horizon_sigma, "Expected Shortfall (₹)": portfolio_value * norm.pdf(z) / tail_probability * horizon_sigma, "Horizon volatility": horizon_sigma},
                {"Volatility forecast": forecast_model, "Tail model": f"Student-t (ν={fitted_df:.1f})", "VaR (₹)": portfolio_value * t_var_multiplier * horizon_sigma, "Expected Shortfall (₹)": portfolio_value * t_es_multiplier * horizon_sigma, "Horizon volatility": horizon_sigma},
            ])

        horizon_returns = clean_returns.rolling(holding_days).sum().dropna() if holding_days > 1 else clean_returns
        loss_returns = -horizon_returns
        hist_var_rate = float(loss_returns.quantile(confidence))
        hist_tail = loss_returns[loss_returns >= hist_var_rate]
        hist_es_rate = float(hist_tail.mean()) if not hist_tail.empty else hist_var_rate
        rows.append({"Volatility forecast": "Observed sample", "Tail model": "Historical simulation", "VaR (₹)": portfolio_value * hist_var_rate, "Expected Shortfall (₹)": portfolio_value * hist_es_rate, "Horizon volatility": float(horizon_returns.std())})
        risk_table = pd.DataFrame(rows)

        a, b, c, d = st.columns(4)
        selected_row = risk_table[(risk_table["Volatility forecast"] == "GARCH(1,1)") & (risk_table["Tail model"] == "Normal parametric")].iloc[0]
        a.metric("Confidence", f"{confidence:.1%}")
        b.metric("Holding period", f"{holding_days}D")
        c.metric("GARCH normal VaR", f"₹{selected_row['VaR (₹)']:,.0f}")
        d.metric("GARCH normal ES", f"₹{selected_row['Expected Shortfall (₹)']:,.0f}")

        plot_table = risk_table.copy()
        plot_table["Method"] = plot_table["Volatility forecast"] + " · " + plot_table["Tail model"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=plot_table["Method"], y=plot_table["VaR (₹)"], name="VaR", marker_color=BLUE))
        fig.add_trace(go.Bar(x=plot_table["Method"], y=plot_table["Expected Shortfall (₹)"], name="Expected Shortfall", marker_color=RED))
        fig.update_layout(title=f"Tail-risk comparison · {confidence:.1%} confidence · {holding_days}-day horizon", barmode="group")
        fig.update_xaxes(title=""); fig.update_yaxes(title="Illustrative loss (₹)", tickformat=",")
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)

        section("Loss distribution and tail area")
        distribution_losses = loss_returns * portfolio_value
        fig = go.Figure(go.Histogram(x=distribution_losses, nbinsx=55, histnorm="probability density", marker_color=BLUE, opacity=.72, name="Observed horizon losses"))
        fig.add_vline(x=portfolio_value * hist_var_rate, line_color=GOLD, line_width=3, annotation_text="Historical VaR", annotation_position="top")
        fig.add_vline(x=portfolio_value * hist_es_rate, line_color=RED, line_width=3, annotation_text="Historical ES", annotation_position="top right")
        fig.update_layout(title=f"Observed {holding_days}-day loss distribution with the worst {tail_probability:.1%} tail")
        fig.update_xaxes(title="Loss (₹; negative values are gains)"); fig.update_yaxes(title="Density")
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)

        st.dataframe(risk_table.style.format({"VaR (₹)": "₹{:,.0f}", "Expected Shortfall (₹)": "₹{:,.0f}", "Horizon volatility": "{:.2%}"}), use_container_width=True, hide_index=True)

        section("Basel Expected Shortfall view")
        basel_returns = clean_returns.rolling(10).sum().dropna()
        basel_losses = -basel_returns
        basel_var_rate = float(basel_losses.quantile(.975))
        basel_es_rate = float(basel_losses[basel_losses >= basel_var_rate].mean())
        ba, bb, bc = st.columns(3)
        ba.metric("Basel confidence convention", "97.5% one-tailed")
        bb.metric("Base liquidity horizon", "10 trading days")
        bc.metric("Simplified historical ES", f"₹{portfolio_value * basel_es_rate:,.0f}")
        note("Basel's market-risk internal-model approach uses daily ES at a 97.5% one-tailed confidence level and starts liquidity-horizon scaling from a 10-day base horizon. The figure here is a simplified single-equity teaching benchmark; it is not a regulatory capital calculation because it does not implement stressed calibration, risk-factor liquidity buckets, reduced-factor scaling, modellability tests, desk approval or capital multipliers.", warning=True)
        st.markdown("Official reference: [Basel Framework MAR33 — Internal models approach](https://www.bis.org/basel_framework/chapter/MAR/33.htm)")
    except Exception as exc:
        st.error(f"Tail-risk estimates could not be calculated for this sample: {exc}")

    section("Model meaning and limitations")
    r1, r2, r3 = st.columns(3)
    r1.markdown(card("Historical VaR / ES", "Uses realised overlapping horizon losses. It captures the sample's non-normal shape but cannot represent events absent from history and is sensitive to the chosen window.", PURPLE), unsafe_allow_html=True)
    r2.markdown(card("Normal parametric", "Uses the selected volatility forecast and a normal tail. It is transparent and fast, but often understates fat-tail and asymmetry risk.", BLUE), unsafe_allow_html=True)
    r3.markdown(card("Student-t parametric", "Combines forecast volatility with heavier fitted tails. It can improve tail sensitivity, but results depend strongly on degrees of freedom and distribution fit.", ORANGE), unsafe_allow_html=True)
    note("VaR does not describe how severe losses become after the threshold is breached. ES addresses that gap by averaging tail losses, but it remains model- and sample-dependent.")

    section("VaR and Expected Shortfall learning deck · solved problems")
    tail_lessons = [
        ("1 · One-day normal VaR", "A ₹10,00,000 portfolio has forecast daily volatility of 1.25%. Find 99% one-day VaR, assuming zero mean.", "VaR = ₹10,00,000 × 2.326 × 1.25% = ₹29,075. Interpretation: under the model, only 1% of days are expected to lose more than this threshold."),
        ("2 · Scale VaR to 10 days", "Using the ₹29,075 one-day VaR above, estimate 10-day VaR under independent, identically distributed returns.", "10-day VaR = ₹29,075 × √10 = ₹91,946. The square-root rule is an approximation and can fail when volatility changes, returns are autocorrelated or positions are nonlinear."),
        ("3 · Normal Expected Shortfall", "A ₹10,00,000 portfolio has 10-day volatility of 4%. Calculate 97.5% normal ES.", "For 97.5%, z=1.960 and φ(z)/(1−.975)=2.338. ES = ₹10,00,000 × 4% × 2.338 = ₹93,520. This exceeds VaR of ₹78,400 because ES averages losses beyond the cutoff."),
        ("4 · Historical VaR", "Ten ordered scenario losses are ₹5k, ₹8k, ₹10k, ₹12k, ₹15k, ₹18k, ₹22k, ₹25k, ₹31k and ₹40k. Illustrate 90% historical VaR.", "With this deliberately small teaching sample, the 90th-percentile cutoff lies near the worst observation; a nearest-rank estimate is ₹40,000. Production estimates need much larger samples and a documented quantile convention."),
        ("5 · Historical Expected Shortfall", "Using the same losses and a teaching cutoff of ₹31,000, compute tail-average ES.", "Losses at or beyond the cutoff are ₹31,000 and ₹40,000. ES = (₹31,000+₹40,000)/2 = ₹35,500. ES reports tail severity, while VaR reports only the threshold."),
        ("6 · Forecast-model comparison", "EWMA and GARCH forecast 10-day volatility of 3.6% and 4.2%. Compare 99% normal VaR on ₹20,00,000.", "EWMA VaR = ₹20,00,000×2.326×3.6%=₹1,67,472. GARCH VaR=₹20,00,000×2.326×4.2%=₹1,95,384. The ₹27,912 difference comes entirely from the conditional-volatility forecast."),
        ("7 · Fat-tail adjustment", "Normal 99% VaR uses a multiplier of 2.326, while a fitted standardised Student-t model uses 2.75. Compare VaR at 5% horizon volatility on ₹50,00,000.", "Normal VaR=₹50,00,000×5%×2.326=₹5,81,500. Student-t VaR=₹50,00,000×5%×2.75=₹6,87,500. The heavy-tail assumption adds ₹1,06,000."),
        ("8 · Basel-oriented interpretation", "Why is a 97.5% 10-day ES shown separately from a user-selected 95% one-day VaR?", "They answer different questions. The user-selected measure supports scenario exploration; Basel's internal-model convention uses 97.5% one-tailed ES with a 10-day base liquidity horizon plus further stressed calibration and liquidity-horizon adjustments. This app's benchmark is educational, not regulatory capital."),
    ]
    for title_text, problem, answer in tail_lessons:
        with st.expander(title_text, expanded=False):
            st.markdown(f"**Problem.** {problem}")
            st.markdown(f"**Solved answer.** {answer}")

with tabs[8]:
    section("Ten solved illustrations")
    note("Each illustration uses daily returns and the √252 annualisation convention. Percentage-return calculations are shown in percentage points unless stated otherwise.")
    illustrations = [
        ("1 · EWMA one-step variance update", "Given λ = 0.94, yesterday's volatility = 1.20%, and yesterday's return = −2.00%.", "σ²ₜ = 0.94(1.20²) + 0.06(2.00²) = 1.5936. Therefore σₜ = √1.5936 = 1.262% daily, or 1.262% × √252 = 20.03% annualised."),
        ("2 · EWMA half-life", "Find how long a shock retains half its original weight when λ = 0.94.", "Half-life = ln(0.5) / ln(0.94) = 11.20 trading days. Thus the shock's weight falls to about 50% after 11 sessions."),
        ("3 · Compare two EWMA decay factors", "A 3% shock follows a 1% variance estimate. Compare λ = 0.94 with λ = 0.97.", "λ=.94: new variance = .94(1²)+.06(3²)=1.48, so volatility=1.217%. λ=.97: variance=.97(1²)+.03(3²)=1.24, so volatility=1.114%. The lower λ reacts more strongly."),
        ("4 · ARCH(2) one-step forecast", "Use ω = 0.05, α₁ = 0.30, α₂ = 0.20, with the last two returns 2% and −1%.", "σ²ₜ₊₁ = 0.05 + 0.30(2²) + 0.20(−1²) = 1.45. Forecast daily volatility = √1.45 = 1.204%, or 19.11% annualised."),
        ("5 · ARCH persistence", "An ARCH(3) has α₁=.25, α₂=.20 and α₃=.15. Interpret persistence.", "Sum of ARCH weights = .25+.20+.15=.60. Sixty percent of recent squared-shock influence carries through the lag structure; because it is below 1, shocks decay rather than explode."),
        ("6 · GARCH(1,1) one-step forecast", "Use ω=.04, α=.10, β=.85, yesterday's return=−2%, and variance=1.44.", "σ²ₜ₊₁=.04+.10(−2²)+.85(1.44)=1.664. Daily volatility=√1.664=1.290%, or 20.48% annualised."),
        ("7 · GARCH long-run variance", "For ω=.04, α=.10 and β=.85, calculate the unconditional variance.", "Long-run variance = ω/(1−α−β) = .04/(1−.95)=.80. Long-run daily volatility=√.80=.894%, or 14.20% annualised. This exists because α+β<1."),
        ("8 · GARCH shock half-life", "A fitted GARCH model has α+β=.95. Estimate the variance-shock half-life.", "Half-life = ln(.5)/ln(.95)=13.51 trading days. The model expects half the shock effect to remain after roughly 14 sessions."),
        ("9 · EGARCH leverage interpretation", "An EGARCH sign coefficient γ = −0.12 is estimated. What does it imply?", "Because γ is negative, a negative standardised shock increases log variance more than an equal-sized positive shock. This is evidence consistent with a leverage/asymmetry effect; significance and residual diagnostics must still be checked."),
        ("10 · Select and communicate a forecast", "EWMA, ARCH, GARCH and EGARCH forecast 18%, 22%, 20% and 24% annualised volatility. Which is correct?", "None is automatically 'correct'. Report the range (18%–24%), compare out-of-sample forecast errors, review AIC only as in-sample evidence, and prefer a model whose assumptions match the observed clustering/asymmetry. Communicate 21% as a simple model-average benchmark only with that caveat."),
    ]
    for title, question, solution in illustrations:
        with st.expander(title, expanded=False):
            st.markdown(f"**Problem.** {question}")
            st.markdown(f"**Solved answer.** {solution}")
    section("Connect the illustrations to the live forecast")
    note("Illustrations 1–3 explain the sidebar EWMA decay control; 4–5 explain ARCH; 6–8 explain GARCH persistence and mean reversion; 9 motivates EGARCH; and 10 shows how to compare the live model outputs responsibly.")

with tabs[9]:
    section("Test the interpretation, not the arithmetic")
    questions = [
        ("Volatility clustering means…", ["large moves tend to follow large moves", "returns must reverse tomorrow", "prices always fall after a shock"], 0),
        ("A shorter rolling window is usually…", ["more responsive and noisier", "always more accurate", "unaffected by recent shocks"], 0),
        ("Positive excess kurtosis suggests…", ["more tail mass than a normal model", "a guaranteed positive return", "constant variance"], 0),
        ("Volatility measures…", ["return dispersion, not direction", "the maximum possible loss", "the expected return"], 0),
    ]
    answers = [st.radio(f"{i + 1}. {q}", options, index=None, key=f"q{i}") for i, (q, options, _) in enumerate(questions)]
    if st.button("Score my answers"):
        if any(answer is None for answer in answers):
            st.warning("Please answer every question first.")
        else:
            score = sum(answer == options[correct] for answer, (_, options, correct) in zip(answers, questions))
            st.success(f"Score: {score}/{len(questions)}")
    section("Download the formatted learning workbook")
    workbook_bytes = build_excel_download(prices, asset, ticker, source, window, ewma_lambda)
    st.download_button("Download formatted Excel workbook", workbook_bytes, file_name=f"{ticker.replace('^','')}_real_time_var_expected_shortfall.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("""<div class='about-section'>
<h3>About This Project</h3>
<p>Developed by <span class='highlight'>Prof. V. Ravichandran</span>, Visiting Professor &amp;
Professor of Practice at Leading Business Schools and founder of
<span class='highlight'>The Mountain Path Academy</span>.</p>
<p>Drawing on <span class='highlight'>28+ years of industry experience</span> and
<span class='highlight'>12+ years of teaching</span>, this dashboard turns volatility theory into a
practical, live-data learning experience covering regimes, clustering, persistence, asymmetry and fat tails.</p>
<a class='academy-link' href='https://themountainpathacademy.com' target='_blank'>🏔️ Visit The Mountain Path Academy</a>
</div>
<div class='mp-footer'>
  <div class='footer-brand'>🏔️ The Mountain Path Academy</div>
  <div class='footer-profile'>Prof. V. Ravichandran · Visiting Professor &amp; Professor of Practice at Leading Business Schools</div>
  <div><a href='https://themountainpathacademy.com' target='_blank'>themountainpathacademy.com</a></div>
  <div style='margin-top:8px'><a href='https://www.linkedin.com/in/trichyravis' target='_blank'>LinkedIn</a><a href='https://github.com/trichyravis' target='_blank'>GitHub</a></div>
  <div style='margin-top:10px;font-size:.77rem'>Educational analytics project · Not investment advice · © 2026 The Mountain Path Academy</div>
</div>""", unsafe_allow_html=True)
