# AI-Driven Portfolio Optimizer

A portfolio project that combines **financial data engineering**, **machine learning**, **portfolio optimization**, **backtesting**, and **interactive web deployment**.

This project uses `yfinance` to download historical stock data, applies **K-Means clustering** to group stocks by risk profile, trains a **Random Forest** model to predict next-day returns from technical indicators, then feeds those insights into a **mean-variance optimizer** built with `PyPortfolioOpt`. The final solution includes a **5-year backtest**, **interactive Plotly charts**, and a **Streamlit web app**.

## Project Highlights

- Download 5 years of historical market data with `yfinance`
- Build technical indicators for each stock
- Cluster assets into **Low / Medium / High** risk groups using K-Means
- Predict next-day returns and upward-move probability using Random Forest
- Construct portfolios using **mean-variance optimization**
- Compare:
  - AI-Optimized strategy
  - Classical Mean-Variance strategy
  - Equal-Weight benchmark
- Backtest strategies over time with monthly rebalancing
- Visualize outputs with Plotly
- Explore results through an interactive Streamlit dashboard

## Default Universe

The project includes 10 large-cap US stocks by default:

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

```bash
ai_portfolio_optimizer/
│
├── app.py
├── requirements.txt
├── README.md
├── notebooks/
│   └── portfolio_optimizer_walkthrough.ipynb
├── data/
│   └── cache/
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── features.py
    ├── ml_models.py
    ├── optimizer.py
    ├── backtest.py
    ├── visuals.py
    └── pipeline.py
```

## End-to-End Workflow

### 1. Data collection
Historical stock prices are downloaded using `yfinance`. A local cache is used to reduce repeated API calls and make the app more stable.

### 2. Feature engineering
Technical indicators are computed for each stock, including:

- 1-day, 5-day, and 10-day returns
- 10-day and 21-day rolling volatility
- Price-to-SMA ratios (10, 21, 50)
- RSI(14)
- MACD and MACD signal
- Bollinger Band width
- 5-day volume change

### 3. K-Means risk clustering
Each stock is described using annualized return, volatility, Sharpe ratio, and downside volatility. K-Means groups them into 3 clusters, then clusters are labeled as Low, Medium, or High risk based on average volatility.

### 4. Random Forest prediction
A Random Forest regressor predicts **next-day returns**, while a Random Forest classifier estimates the probability that the next-day return is positive.

### 5. Portfolio optimization
The optimizer uses `PyPortfolioOpt` to solve a **mean-variance optimization** problem. The AI strategy blends historical expected returns with ML-predicted alpha before optimizing.

### 6. Backtesting
Strategies are rebalanced monthly over a rolling window. The project evaluates:

- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown

## Installation

```bash
git clone <your-repo-url>
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

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## Notebook Walkthrough

Open the notebook to explore the project step by step:

```bash
jupyter notebook notebooks/portfolio_optimizer_walkthrough.ipynb
```

## Example Streamlit Features

The app lets users:

- choose stock tickers
- set risk profile: Conservative / Balanced / Aggressive
- run the full pipeline interactively
- inspect K-Means clusters
- view Random Forest predictions
- compare optimized weights
- analyze backtest performance

## Suggested Resume / Portfolio Description

**AI-Driven Portfolio Optimizer** — Built an end-to-end portfolio analytics app using `yfinance`, `scikit-learn`, `PyPortfolioOpt`, `Plotly`, and `Streamlit`. Engineered technical indicators, clustered stocks by risk using K-Means, predicted next-day returns with Random Forest, optimized allocations with mean-variance optimization, and backtested AI-assisted strategies over a 5-year horizon.

## Notes and Limitations

- `yfinance` depends on Yahoo Finance public endpoints and may occasionally fail or rate-limit requests. The project includes a local cache to reduce repeated downloads. citeturn598449search0turn598449search3
- `PyPortfolioOpt` uses expected returns and risk models such as sample covariance as inputs to mean-variance optimization. citeturn598449search1turn598449search4turn598449search16
- Streamlit supports Plotly natively with `st.plotly_chart`, which is used throughout the dashboard. citeturn598449search2turn598449search11
- This is an educational project, not investment advice.

## Future Improvements

- Add LSTM or XGBoost return forecasting
- Add transaction costs and turnover penalties
- Support ETFs and international equities
- Add Black-Litterman or Hierarchical Risk Parity
- Deploy to Streamlit Community Cloud
