"""
Data generator for Solana Trend Predictor
Generate synthetic market data for testing and development
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .config import TECHNICAL_INDICATORS

class DataGenerator:
    def __init__(self, start_price=100, volatility=0.02):
        """
        Initialize data generator
        
        Args:
            start_price (float): Initial price of SOL
            volatility (float): Price volatility parameter
        """
        self.start_price = start_price
        self.volatility = volatility
        
    def generate_ohlcv(self, n_points=1000, freq='1min'):
        """
        Generate OHLCV (Open, High, Low, Close, Volume) data
        
        Args:
            n_points (int): Number of data points to generate
            freq (str): Frequency of data points
            
        Returns:
            pd.DataFrame: DataFrame containing OHLCV data
        """
        # 生成时间戳
        end_time = datetime.now()
        timestamps = pd.date_range(end=end_time, periods=n_points, freq=freq)
        
        # 生成价格数据（使用随机游走）
        prices = np.random.normal(0, self.volatility, n_points).cumsum() + self.start_price
        prices = np.maximum(prices, 0)  # 确保价格为正
        
        # 生成OHLCV数据
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
            'high': prices * (1 + np.random.uniform(0, 0.002, n_points)),
            'low': prices * (1 - np.random.uniform(0, 0.002, n_points)),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_points)
        })
        
        # 确保high >= open,close >= low
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 0.001, n_points)
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 0.001, n_points)
        
        return df
    
    def generate_orderbook(self, timestamps, prices):
        """
        Generate orderbook data
        
        Args:
            timestamps: 时间戳数组
            prices: 价格数组
            
        Returns:
            pd.DataFrame: DataFrame containing orderbook data
        """
        n_points = len(timestamps)
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'bid_volume': np.random.lognormal(10, 1, n_points),
            'ask_volume': np.random.lognormal(10, 1, n_points),
            'bid_count': np.random.randint(10, 100, n_points),
            'ask_count': np.random.randint(10, 100, n_points)
        })
        
        return df
    
    def generate_chain_data(self, timestamps):
        """
        Generate on-chain data
        
        Args:
            timestamps: 时间戳数组
            
        Returns:
            pd.DataFrame: DataFrame containing on-chain data
        """
        n_points = len(timestamps)
        df = pd.DataFrame({
            'timestamp': timestamps,
            'stake_ratio': np.random.normal(0.7, 0.01, n_points),  # 质押率
            'network_congestion': np.random.uniform(0, 1, n_points),  # 网络拥堵度
            'large_transfers': np.random.poisson(2, n_points)  # 大额转账数量
        })
        
        return df
    
    def generate_all_data(self, n_points=1000):
        """
        Generate all types of data
        
        Args:
            n_points (int): Number of data points to generate
            
        Returns:
            pd.DataFrame: DataFrame containing all data
        """
        # 首先生成OHLCV数据
        ohlcv = self.generate_ohlcv(n_points)
        
        # 使用相同的时间戳生成其他数据
        orderbook = self.generate_orderbook(ohlcv['timestamp'], ohlcv['close'])
        chain = self.generate_chain_data(ohlcv['timestamp'])
        
        # 合并数据
        df = pd.merge(ohlcv, orderbook, on='timestamp', how='left')
        df = pd.merge(df, chain, on='timestamp', how='left')
        
        return df
    
    def load_sample_data(self):
        """
        Load sample data for model training
        
        Returns:
            tuple: (X, y) where X is feature matrix and y is target vector
        """
        df = self.generate_all_data(1000)
        
        # 准备特征
        X = df[['close', 'volume', 'bid_volume', 'ask_volume', 
                'bid_count', 'ask_count', 'stake_ratio', 
                'network_congestion', 'large_transfers']]
        
        # 准备标签（未来5分钟是否上涨）
        y = (df['close'].shift(-5) > df['close']).astype(int)
        
        # 删除最后5行（因为没有未来数据）
        X = X.iloc[:-5]
        y = y.iloc[:-5]
        
        return X, y 