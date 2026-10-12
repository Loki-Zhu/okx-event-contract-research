# OKX Event Contract 5-Minute Strategy

Research and reproducible code for predicting OKX 5-minute Up/Down event-contract outcomes using OKX spot and index data.

## Contents

- `okx_strategy.py` — feature engineering, model training, out-of-sample backtesting, and latest-signal generation.
- `okx_model_pool.pkl` — pretrained ensemble of five LightGBM classifiers.

Raw market CSV files are intentionally not included. The script expects them under `../data/`:

- `okx_spot_ETH-USDT_5m.csv`
- `okx_spot_BTC-USDT_5m.csv`
- `okx_index_ETH-USDT_5m.csv`
- `okx_index_BTC-USDT_5m.csv`

## Setup

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python okx_strategy.py train
python okx_strategy.py backtest ETH
python okx_strategy.py backtest BTC
python okx_strategy.py signal ETH
```

## Current assumptions

- Forecast target: direction of the second future 5-minute index candle (`t+2`).
- Default entry price: `0.51` USDT per share.
- Default stake: `32` USDT per trade.
- Default confidence filter: absolute model probability distance from `0.5` greater than `0.04`.
- Taker fee formula: `0.045 × contracts × price × (1 − price)`.
- Positions are assumed to be held to settlement.

The backtest does not model order-book depth, slippage, partial fills, or failed fills. This repository is research material, not financial advice.

