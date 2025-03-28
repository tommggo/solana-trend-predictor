"""
Technical indicators for Solana Trend Predictor
实现各种技术指标的计算
"""

import numpy as np
import pandas as pd

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算ATR (Average True Range)
    
    Args:
        df (pd.DataFrame): 包含high, low, close列的DataFrame
        period (int): ATR周期
        
    Returns:
        pd.Series: ATR值
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # 计算True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 计算ATR
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算RSI (Relative Strength Index)
    
    Args:
        df (pd.DataFrame): 包含close列的DataFrame
        period (int): RSI周期
        
    Returns:
        pd.Series: RSI值
    """
    close = df['close']
    
    # 计算价格变化
    delta = close.diff()
    
    # 分离上涨和下跌
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # 计算RS和RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_ema_ratio(df: pd.DataFrame, short_period: int = 5, long_period: int = 15) -> pd.Series:
    """
    计算短期和长期EMA的比值
    
    Args:
        df (pd.DataFrame): 包含close列的DataFrame
        short_period (int): 短期EMA周期
        long_period (int): 长期EMA周期
        
    Returns:
        pd.Series: EMA比值
    """
    close = df['close']
    
    # 计算短期和长期EMA
    ema_short = close.ewm(span=short_period, adjust=False).mean()
    ema_long = close.ewm(span=long_period, adjust=False).mean()
    
    # 计算比值
    ema_ratio = ema_short / ema_long
    
    return ema_ratio

def calculate_large_trade_marker(df: pd.DataFrame, threshold: float = 5.0) -> pd.Series:
    """
    标记大单交易
    
    Args:
        df (pd.DataFrame): 包含volume列的DataFrame
        threshold (float): 大单阈值（相对于平均值的倍数）
        
    Returns:
        pd.Series: 大单标记（1表示大单，0表示普通交易）
    """
    volume = df['volume']
    
    # 计算移动平均
    volume_ma = volume.rolling(window=20).mean()
    
    # 标记大单
    large_trade = (volume > volume_ma * threshold).astype(int)
    
    return large_trade

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标
    
    Args:
        df (pd.DataFrame): 原始数据DataFrame
        
    Returns:
        pd.DataFrame: 包含所有技术指标的DataFrame
    """
    # 复制数据以避免修改原始数据
    data = df.copy()
    
    # 计算ATR
    data['atr'] = calculate_atr(data)
    
    # 计算RSI
    data['rsi'] = calculate_rsi(data)
    
    # 计算EMA比值
    data['ema_ratio'] = calculate_ema_ratio(data)
    
    # 计算大单标记
    data['large_trade'] = calculate_large_trade_marker(data)
    
    return data 