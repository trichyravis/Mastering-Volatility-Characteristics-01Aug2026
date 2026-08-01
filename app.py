from __future__ import annotations

from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import kurtosis, norm, skew
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Characteristics of Volatility | Mountain Path Academy",
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
html,body,[class*="css"]{font-family:'Inter',sans-serif}.stApp{background:linear-gradient(180deg,#F8FAFD,#EAF1F7)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#081F3A,#124A78);color:#F7FAFC}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] p,[data-testid="stSidebar"] li{color:#F7FAFC}
.hero{background:linear-gradient(120deg,#071A2F 0%,#0B3B67 58%,#A97908 145%);padding:30px 34px;border-radius:22px;color:white;box-shadow:0 14px 34px rgba(7,26,47,.22);margin-bottom:16px;border:1px solid rgba(243,200,75,.35)}
.hero h1{font-size:2.25rem;margin:0 0 8px;color:white;font-weight:900}.hero p{margin:0;color:#DDEAF4;line-height:1.55}.eyebrow{color:#F3C84B;text-transform:uppercase;letter-spacing:.14em;font-weight:900;font-size:.76rem;margin-bottom:.55rem}
.section-title{font-size:1.42rem;font-weight:900;color:#0B2545;margin:18px 0 8px}.concept-card{background:white;border:1px solid #D9E5EF;border-top:5px solid #0B5CAD;padding:17px 18px;border-radius:15px;box-shadow:0 5px 16px rgba(18,54,84,.07);min-height:165px}.concept-card h3{color:#0B2545;font-size:1.05rem;margin:0 0 7px}.concept-card p{color:#3C5368;font-size:.91rem;line-height:1.5;margin:0}
.teaching-note{background:#EAF7F5;border-left:5px solid #13A89E;padding:13px 16px;border-radius:10px;color:#153C3A;margin:10px 0}.warning-note{background:#FFF3E8;border-left:5px solid #F28E2B;padding:13px 16px;border-radius:10px;color:#57300A;margin:10px 0}.formula{background:linear-gradient(135deg,#FFF9E6,#FFF1B8);border:1px solid #E8C45B;border-left:6px solid #D4A017;padding:14px 18px;border-radius:12px;color:#3D3006;font-weight:800;margin:8px 0 14px}
.footer{background:linear-gradient(115deg,#081F3A,#124A78);color:#E6F1F8;padding:22px;border-radius:16px;margin-top:28px;text-align:center;border-top:4px solid #F3C84B}.footer a{color:#F3C84B!important;font-weight:800}
[data-testid="stMetric"]{background:#FFF;border:1px solid #DDE8F1;padding:13px;border-radius:14px}.stTabs [data-baseweb="tab-list"]{gap:9px!important;flex-wrap:wrap!important;background:#D8E3ED!important;padding:10px!important;border-radius:14px!important}.stTabs button[data-baseweb="tab"]{flex:1 1 145px!important;min-height:52px!important;background:#0B2545!important;border:2px solid #F3C84B!important;border-radius:10px!important;color:#F3C84B!important}.stTabs button[data-baseweb="tab"] p{color:#F3C84B!important;font-weight:850!important}.stTabs button[data-baseweb="tab"][aria-selected="true"]{background:linear-gradient(135deg,#F3C84B,#D4A017)!important}.stTabs button[data-baseweb="tab"][aria-selected="true"] p{color:#071A2F!important}.stButton button,.stDownloadButton button{background:#0B3B67!important;color:white!important;border-radius:10px!important;font-weight:800!important}
section[data-testid="stSidebar"] label p{color:#F3C84B!important;font-weight:850!important}section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"]>div{background:#FFF!important;border:2px solid #F3C84B!important}section[data-testid="stSidebar"] div[data-testid="stSelectbox"] *{color:#0B2545!important;-webkit-text-fill-color:#0B2545!important;font-weight:800!important}
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


with st.sidebar:
    st.markdown("## 〽️ Volatility Characteristics")
    st.caption("MP · Interactive learning studio")
    asset = st.selectbox("Choose a stock or index", list(INSTRUMENTS))
    ticker = INSTRUMENTS[asset]
    years = st.selectbox("Daily-price period", [1, 3, 5], index=1, format_func=lambda x: f"{x} year" if x == 1 else f"{x} years")
    window = st.slider("Rolling window (trading days)", 10, 90, 21)
    ewma_lambda = st.slider("EWMA decay factor (λ)", 0.80, 0.99, 0.94, 0.01)
    use_demo = st.toggle("Use classroom simulation", value=False)
    if st.button("Refresh market data", use_container_width=True):
        st.cache_data.clear()
    st.markdown("---")
    st.caption("Live prices are supplied by Yahoo Finance and may be delayed. Educational use only.")

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

st.markdown("<div class='hero'><div class='eyebrow'>Mountain Path Academy · Applied Finance</div><h1>Understanding Characteristics of Volatility</h1><p>Use current market observations to see why volatility changes through time, arrives in clusters, persists after shocks, reacts differently to bad news, and produces more extreme returns than a normal model expects.</p></div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest adjusted close", f"₹{prices['Close'].iloc[-1]:,.2f}")
m2.metric("Annualised volatility", f"{annual_vol:.2f}%")
m3.metric(f"Latest {window}D volatility", f"{prices['Rolling volatility'].dropna().iloc[-1]:.2f}%")
m4.metric("Last market date", prices.index[-1].strftime("%d %b %Y"))
st.caption(f"{asset} ({ticker}) · {len(prices):,} daily observations · {source}")

tabs = st.tabs(["🧭 Characteristics map", "🌊 Time variation", "🔗 Clustering & persistence", "⚖️ Asymmetry", "📊 Fat tails", "🧪 Learning check"])

with tabs[0]:
    section("Six ideas to carry into every volatility analysis")
    rows = [
        [("Time-varying", "Risk is not constant. Calm and turbulent regimes appear in the same return series.", BLUE), ("Clustering", "Large moves tend to follow large moves, and small moves tend to follow small moves.", PURPLE), ("Persistence", "A shock can influence conditional risk for many sessions before gradually fading.", TEAL)],
        [("Asymmetry", "Negative and positive news of similar size need not have the same effect on future volatility.", RED), ("Fat tails", "Extreme observations occur more often than a normal distribution predicts.", ORANGE), ("Horizon-sensitive", "The measured risk changes with the sampling frequency, window, and annualisation rule.", GREEN)],
    ]
    for row in rows:
        cols = st.columns(3)
        for col, item in zip(cols, row): col.markdown(card(*item), unsafe_allow_html=True)
    note("Volatility describes the magnitude and variability of returns—not their direction, expected return, or maximum possible loss.")

with tabs[1]:
    section("Volatility moves through regimes")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.10, row_heights=[.52, .48], subplot_titles=("Adjusted price", "Annualised rolling and EWMA volatility"))
    fig.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name="Adjusted close", line=dict(color=BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=prices["Rolling volatility"], name=f"Rolling {window}D", line=dict(color=PURPLE, width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=prices["EWMA volatility"], name=f"EWMA λ={ewma_lambda:.2f}", line=dict(color=TEAL, width=2)), row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1); fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    st.plotly_chart(style_fig(fig, 590), use_container_width=True)
    st.markdown("<div class='formula'>Annualised volatility = daily standard deviation × √252</div>", unsafe_allow_html=True)
    note("Change the rolling window in the sidebar. Short windows react quickly but are noisy; long windows are smoother but slow to recognise regime changes.")

with tabs[2]:
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

with tabs[3]:
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

with tabs[4]:
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

with tabs[5]:
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
    section("Download the analysed observations")
    export = prices.reset_index().rename(columns={prices.index.name or "index": "Date"})
    st.download_button("Download analysis as CSV", export.to_csv(index=False).encode(), file_name=f"{ticker.replace('^','')}_volatility_characteristics.csv", mime="text/csv")

st.markdown("<div class='footer'><strong>Mountain Path Academy</strong><br>Learn the pattern, test the evidence, respect the limitations.<br><small>Educational material only—not investment or trading advice.</small></div>", unsafe_allow_html=True)
