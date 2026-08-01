# Understanding Characteristics of Volatility

An interactive Streamlit learning studio in the Mountain Path Academy navy-and-gold design language. It uses adjusted daily market prices to make six properties of volatility visible:

- time variation and volatility regimes;
- clustering of large and small moves;
- persistence and gradual decay after shocks;
- asymmetric behaviour around positive and negative returns;
- fat-tailed return distributions; and
- sensitivity to the measurement horizon and method.

Learners can select a major Nifty instrument, change the sample and rolling-window assumptions, compare rolling and EWMA estimates, inspect squared-return autocorrelations, explore asymmetry and tail behaviour, complete a short knowledge check, and download the analysed data.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app requests adjusted daily prices from Yahoo Finance. If the provider is unavailable, it automatically switches to a reproducible classroom simulation so every lesson remains usable. Prices may be delayed. Educational material only—not investment or trading advice.
