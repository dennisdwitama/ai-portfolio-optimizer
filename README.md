# RiskLens

RiskLens is an AI-assisted portfolio analytics app that combines **financial data engineering**, **machine learning**, **portfolio optimization**, **backtesting**, and **interactive Streamlit deployment**.

It uses `yfinance` to download historical stock data, applies **K-Means clustering** to group stocks by risk profile, trains **Random Forest** models to estimate next-day return signals, and feeds those insights into a **mean-variance optimizer**. The result is an interactive dashboard for comparing AI-assisted and classical portfolio construction approaches.

Live app: https://risklens-ai-portfolio-optimizer.streamlit.app/

## Highlights

- Download up to 5 years of historical market data with `yfinance`
- Engineer technical indicators for each stock
- Cluster assets into **Low / Medium / High** risk groups using K-Means
- Predict next-day returns and upside probability using Random Forest
- Construct portfolios with **mean-variance optimization**
- Compare:
  - AI-Optimized strategy
  - Classical Mean-Variance strategy
  - Equal-Weight benchmark
- Backtest strategies with monthly rebalancing
- Visualize results with Plotly inside a Streamlit app

## Default Universe

The default stock universe includes 10 large-cap US equities:

- AAPL
- MSFT
- GOOG
- NVDA
- TSLA
- AMZN
- META
- JPM
- UNH
- XOM

## Project Structure

```text
ai_portfolio_optimizer/
|-- app.py
|-- README.md
|-- requirements.txt
|-- notebooks/
|   `-- portfolio_optimizer_walkthrough.ipynb
|-- data/
|   `-- cache/
`-- src/
    |-- __init__.py
    |-- backtest.py
    |-- config.py
    |-- data_loader.py
    |-- features.py
    |-- ml_models.py
    |-- optimizer.py
    |-- pipeline.py
    `-- visuals.py
```

## Workflow

### 1. Data collection

Historical stock prices are downloaded with `yfinance`. A local cache in `data/cache` helps reduce repeated calls and makes reruns more stable.

### 2. Feature engineering

Technical indicators are computed for each stock, including:

- 1-day, 5-day, and 10-day returns
- 10-day and 21-day rolling volatility
- Price-to-SMA ratios for 10, 21, and 50 days
- RSI(14)
- MACD and MACD signal
- Bollinger Band width
- 5-day volume change

### 3. Risk clustering

Each stock is summarized by annualized return, volatility, Sharpe ratio, and downside volatility. K-Means then assigns stocks into 3 clusters that are labeled **Low**, **Medium**, and **High** risk.

### 4. AI prediction layer

RiskLens trains:

- a Random Forest regressor to estimate next-day return
- a Random Forest classifier to estimate the probability of a positive next-day move

These predictions are converted into an AI return signal used by the optimizer.

### 5. Portfolio construction

The classical strategy uses historical expected returns and covariance only.

The AI-optimized strategy blends historical expected returns with the ML-predicted alpha signal before optimization.

### 6. Backtesting

The app backtests monthly rebalancing and reports:

- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown

## Installation

```bash
git clone https://github.com/dennisdwitama/ai-portfolio-optimizer.git
cd ai_portfolio_optimizer
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Streamlit App

The deployed version is available at:

https://risklens-ai-portfolio-optimizer.streamlit.app/

## Notebook Walkthrough

To explore the project step by step:

```bash
jupyter notebook notebooks/portfolio_optimizer_walkthrough.ipynb
```

## App Features

RiskLens lets users:

- choose stock tickers
- select a risk profile: Conservative, Balanced, or Aggressive
- run the end-to-end analysis interactively
- inspect K-Means risk clusters
- view Random Forest predictions
- compare AI and classical optimized weights
- review backtest performance across strategies

## Notes and Limitations

- `yfinance` depends on Yahoo Finance public endpoints and may occasionally fail or rate-limit requests.
- The app trains models locally with `scikit-learn`; it does not require an external AI API key.
- Results are sensitive to date range, market regime, and the instability of short-horizon return forecasting.
- This project is educational and should not be treated as investment advice.

## Future Improvements

- Add more robust walk-forward validation
- Add transaction costs and turnover penalties
- Support ETFs and international equities
- Add alternative optimizers such as Black-Litterman or Hierarchical Risk Parity
- Expand the live dashboard with richer portfolio diagnostics
