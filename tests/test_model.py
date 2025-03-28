# -*- coding: utf-8 -*-
"""
Test cases for the SolanaPredictor model
"""

import unittest
import numpy as np
import pandas as pd
from src.model import SolanaPredictor
from src.config import MODEL_PARAMS

class TestSolanaPredictor(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(100, 10, 100),
            'high': np.random.normal(105, 10, 100),
            'low': np.random.normal(95, 10, 100),
            'close': np.random.normal(100, 10, 100),
            'volume': np.random.normal(1000, 100, 100),
            'bid_volume': np.random.normal(500, 50, 100),
            'ask_volume': np.random.normal(500, 50, 100),
            'bid_count': np.random.normal(100, 10, 100),
            'ask_count': np.random.normal(100, 10, 100),
            'network_congestion': np.random.uniform(0, 1, 100),
            'large_transfers': np.random.normal(10, 2, 100),
            'stake_ratio': np.random.uniform(0.6, 0.8, 100)
        })
        
        # 初始化模型
        self.model = SolanaPredictor(model_params=MODEL_PARAMS)
    
    def test_prepare_data(self):
        """Test data preparation"""
        features, labels = self.model.prepare_data(self.test_data)
        
        # 检查特征列是否存在
        expected_features = [
            'price_change', 'volume_change', 'order_pressure',
            'order_count_ratio', 'price_volatility', 'network_activity',
            'stake_ratio_change', 'stake_ratio', 'network_congestion',
            'large_transfers'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        # 检查是否有NaN值
        self.assertFalse(features.isna().any().any())
        self.assertFalse(labels.isna().any())
        
        # 检查特征和标签的长度是否一致
        self.assertEqual(len(features), len(labels))
        
        # 检查标签是否为0或1
        self.assertTrue(all(labels.isin([0, 1])))
    
    def test_train_and_predict(self):
        """Test model training and prediction"""
        # 准备特征和标签
        features, labels = self.model.prepare_data(self.test_data)
        
        # 训练模型
        history = self.model.train(features, labels)
        
        # 检查训练历史
        self.assertIn('train_accuracy', history)
        self.assertIn('val_accuracy', history)
        
        # 测试预测
        predictions = self.model.predict(features)
        probas = self.model.predict_proba(features)
        
        # 检查预测结果
        self.assertEqual(len(predictions), len(features))
        self.assertTrue(all(np.isin(predictions, [0, 1])))
        self.assertTrue(all((probas >= 0) & (probas <= 1)))
    
    def test_evaluate(self):
        """Test model evaluation"""
        # 准备特征和标签
        features, labels = self.model.prepare_data(self.test_data)
        
        # 训练模型
        self.model.train(features, labels)
        
        # 评估模型
        metrics = self.model.evaluate(features, labels)
        
        # 检查评估指标
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # 检查指标值是否在合理范围内
        for metric in metrics.values():
            self.assertTrue(0 <= metric <= 1)
    
    def test_save_and_load(self):
        """Test model saving and loading"""
        # 准备特征和标签
        features, labels = self.model.prepare_data(self.test_data)
        
        # 训练模型
        self.model.train(features, labels)
        
        # 保存模型
        self.model.save_model('test_model.pkl')
        
        # 创建新模型实例
        new_model = SolanaPredictor(model_params=MODEL_PARAMS)
        
        # 加载模型
        new_model.load_model('test_model.pkl')
        
        # 检查预测结果是否一致
        original_predictions = self.model.predict(features)
        loaded_predictions = new_model.predict(features)
        
        np.testing.assert_array_equal(original_predictions, loaded_predictions)

if __name__ == '__main__':
    unittest.main()
