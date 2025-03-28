"""
LightGBM model for Solana Trend Predictor
实现模型训练、预测、评估等功能
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from .config import MODEL_DIR, MODEL_PARAMS
from .indicators import calculate_all_indicators

class SolanaPredictor:
    def __init__(self, model_params=None):
        """
        初始化预测器
        
        Args:
            model_params (dict): LightGBM模型参数，如果为None则使用默认参数
        """
        self.model_params = model_params or MODEL_PARAMS
        self.model = None
        self.feature_columns = None
        
    def prepare_data(self, df):
        """
        准备特征和标签
        
        Args:
            df (pd.DataFrame): 原始数据
            
        Returns:
            tuple: (特征DataFrame, 标签Series)
        """
        # 复制数据以避免修改原始数据
        data = df.copy()
        
        # 计算技术指标
        data = calculate_all_indicators(data)
        
        # 计算基础特征
        # 1. 价格变化率
        data['price_change'] = data['close'].pct_change()
        
        # 2. 成交量变化率
        data['volume_change'] = data['volume'].pct_change()
        
        # 3. 买卖盘压力
        data['order_pressure'] = (data['bid_volume'] - data['ask_volume']) / (data['bid_volume'] + data['ask_volume'])
        
        # 4. 订单数量比
        data['order_count_ratio'] = data['bid_count'] / data['ask_count']
        
        # 5. 价格波动率（使用过去5个时间点的标准差）
        data['price_volatility'] = data['close'].rolling(window=5).std()
        
        # 6. 网络活跃度（结合网络拥堵度和大额转账）
        data['network_activity'] = data['network_congestion'] * (1 + data['large_transfers'] / 10)
        
        # 7. 质押率变化
        data['stake_ratio_change'] = data['stake_ratio'].pct_change()
        
        # 计算标签（未来5分钟的价格变化）
        data['future_price'] = data['close'].shift(-5)
        data['label'] = (data['future_price'] > data['close']).astype(int)
        
        # 删除包含NaN的行
        data = data.dropna()
        
        # 选择特征列
        self.feature_columns = [
            'price_change', 'volume_change', 'order_pressure',
            'order_count_ratio', 'price_volatility', 'network_activity',
            'stake_ratio_change', 'stake_ratio', 'network_congestion',
            'large_transfers', 'atr', 'rsi', 'ema_ratio', 'large_trade'
        ]
        
        return data[self.feature_columns], data['label']
    
    def train(self, X, y, validation_split=0.2):
        """
        训练模型
        
        Args:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 标签数据
            validation_split (float): 验证集比例
            
        Returns:
            dict: 训练历史
        """
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 训练模型
        self.model = lgb.train(
            self.model_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # 返回训练历史
        return {
            'train_accuracy': accuracy_score(y_train, self.predict(X_train)),
            'val_accuracy': accuracy_score(y_val, self.predict(X_val))
        }
    
    def predict(self, X):
        """
        预测
        
        Args:
            X (pd.DataFrame): 特征数据
            
        Returns:
            np.ndarray: 预测结果
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        return (self.model.predict(X) > 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        预测概率
        
        Args:
            X (pd.DataFrame): 特征数据
            
        Returns:
            np.ndarray: 预测概率
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Args:
            X (pd.DataFrame): 特征数据
            y (pd.Series): 真实标签
            
        Returns:
            dict: 评估指标
        """
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
    
    def save_model(self, model_name='solana_predictor.pkl'):
        """
        保存模型
        
        Args:
            model_name (str): 模型文件名
        """
        if self.model is None:
            raise ValueError("No model to save!")
            
        # 确保模型目录存在
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(MODEL_DIR, model_name)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, model_path)
    
    def load_model(self, model_name='solana_predictor.pkl'):
        """
        加载模型
        
        Args:
            model_name (str): 模型文件名
        """
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # 加载模型
        saved_model = joblib.load(model_path)
        self.model = saved_model['model']
        self.feature_columns = saved_model['feature_columns'] 