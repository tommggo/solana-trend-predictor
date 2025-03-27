"""
Test script for data generator
验证数据生成器的功能
"""

import pandas as pd
import numpy as np
from src.data_generator import DataGenerator

def test_data_generator():
    """测试数据生成器的各项功能"""
    print("开始测试数据生成器...")
    
    # 创建生成器实例
    generator = DataGenerator(start_price=100, volatility=0.02)
    
    # 1. 测试OHLCV数据生成
    print("\n1. 测试OHLCV数据生成:")
    ohlcv = generator.generate_ohlcv(n_points=5)
    print(ohlcv)
    print("\nOHLCV数据统计:")
    print(ohlcv.describe())
    
    # 2. 测试订单簿数据生成
    print("\n2. 测试订单簿数据生成:")
    orderbook = generator.generate_orderbook(ohlcv['timestamp'], ohlcv['close'])
    print(orderbook)
    print("\n订单簿数据统计:")
    print(orderbook.describe())
    
    # 3. 测试链上数据生成
    print("\n3. 测试链上数据生成:")
    chain = generator.generate_chain_data(ohlcv['timestamp'])
    print(chain)
    print("\n链上数据统计:")
    print(chain.describe())
    
    # 4. 测试完整数据生成
    print("\n4. 测试完整数据生成:")
    all_data = generator.generate_all_data(n_points=5)
    print(all_data)
    print("\n完整数据统计:")
    print(all_data.describe())
    
    # 5. 测试样本数据加载
    print("\n5. 测试样本数据加载:")
    X, y = generator.load_sample_data()
    print("\n特征矩阵形状:", X.shape)
    print("标签向量形状:", y.shape)
    print("\n特征矩阵前5行:")
    print(X.head())
    print("\n标签向量前5个值:")
    print(y.head())
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_data_generator() 