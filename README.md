# Real-Time VaR and Expected Shortfall Estimation (Basel Framework) Using Forecasted Volatilities (EWMA, ARCH, GARCH, EGARCH)

An interactive Streamlit learning studio in the Mountain Path Academy navy-and-gold design language. It uses adjusted daily market prices to make six properties of volatility visible:

- time variation and volatility regimes;
- clustering of large and small moves;
- persistence and gradual decay after shocks;
- asymmetric behaviour around positive and negative returns;
- fat-tailed return distributions; and
- sensitivity to the measurement horizon and method.

Learners can select a major Nifty instrument, study a dedicated educational primer, change the sample and rolling-window assumptions, compare rolling and EWMA estimates, inspect squared-return autocorrelations, explore asymmetry and tail behaviour, forecast with EWMA, ARCH(5), GARCH(1,1) and EGARCH(1,1), compare historical/normal/Student-t VaR and Expected Shortfall using configurable confidence and holding-period controls, review a simplified Basel-oriented 97.5%/10-day ES view, work through ten solved illustrations, complete a short knowledge check, and download a professionally formatted Excel workbook with a summary, daily observations, formulas, chart and learning guide.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app requests adjusted daily prices from Yahoo Finance. If the provider is unavailable, it automatically switches to a reproducible classroom simulation so every lesson remains usable. Prices may be delayed. Educational material only—not investment or trading advice.
