"""
Configuration settings for the Solana Trend Predictor
"""

# Data settings
DATA_DIR = "data"
MODEL_DIR = "models"

# Model settings
MODEL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "num_leaves": 31,
    "random_state": 42
}

# Feature settings
TECHNICAL_INDICATORS = {
    "rsi_period": 14,
    "ma_short": 5,
    "ma_long": 15,
    "volatility_period": 5,
    "atr_period": 14,
    "large_trade_threshold": 5.0
}

# Trading settings
TRADING_PARAMS = {
    "initial_capital": 10000,
    "fee_rate": 0.001,
    "buy_threshold": 0.6,
    "sell_threshold": 0.4,
    "stop_loss": 0.05,  # 5% 止损
    "take_profit": 0.10,  # 10% 止盈
    "position_size": 0.2,  # 每次交易使用20%的资金
    "execution_delay": 1,  # 执行延迟（分钟）
    "slippage": 0.001,  # 滑点率
    "market_impact": 0.0005  # 市场影响率
} 