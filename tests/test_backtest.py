import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtest import Backtest
from src.model import SolanaPredictor
from src.config import MODEL_PARAMS, TRADING_PARAMS

class TestBacktest(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(90, 110, 100),  # 使用均匀分布，确保价格在合理范围内
            'high': np.random.uniform(95, 115, 100),
            'low': np.random.uniform(85, 105, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.normal(1000, 100, 100),
            'bid_volume': np.random.normal(500, 50, 100),
            'ask_volume': np.random.normal(500, 50, 100),
            'bid_count': np.random.normal(100, 10, 100),
            'ask_count': np.random.normal(100, 10, 100),
            'network_congestion': np.random.uniform(0, 1, 100),
            'large_transfers': np.random.normal(10, 2, 100),
            'stake_ratio': np.random.uniform(0.6, 0.8, 100)
        })
        
        # 确保价格为正数且合理
        self.test_data[['open', 'high', 'low', 'close']] = self.test_data[['open', 'high', 'low', 'close']].clip(lower=90, upper=110)
        
        # 创建标签（随机生成用于测试）
        self.test_data['target'] = np.random.randint(0, 2, size=100)
        
        # 初始化模型和回测系统
        self.model = SolanaPredictor(model_params=MODEL_PARAMS)
        
        # 训练模型
        features, labels = self.model.prepare_data(self.test_data)
        self.model.train(features, labels)
        
        self.backtest = Backtest(
            initial_capital=TRADING_PARAMS['initial_capital'],
            fee_rate=TRADING_PARAMS['fee_rate'],
            stop_loss=TRADING_PARAMS['stop_loss'],
            take_profit=TRADING_PARAMS['take_profit'],
            position_size=TRADING_PARAMS['position_size'],
            execution_delay=TRADING_PARAMS['execution_delay'],
            slippage=TRADING_PARAMS['slippage'],
            market_impact=TRADING_PARAMS['market_impact']
        )
    
    def test_initialization(self):
        """Test backtest initialization"""
        self.assertEqual(self.backtest.initial_capital, TRADING_PARAMS['initial_capital'])
        self.assertEqual(self.backtest.fee_rate, TRADING_PARAMS['fee_rate'])
        self.assertEqual(self.backtest.stop_loss, TRADING_PARAMS['stop_loss'])
        self.assertEqual(self.backtest.take_profit, TRADING_PARAMS['take_profit'])
        self.assertEqual(self.backtest.position_size, TRADING_PARAMS['position_size'])
        self.assertEqual(self.backtest.execution_delay, TRADING_PARAMS['execution_delay'])
        self.assertEqual(self.backtest.slippage, TRADING_PARAMS['slippage'])
        self.assertEqual(self.backtest.market_impact, TRADING_PARAMS['market_impact'])
        self.assertEqual(self.backtest.capital, TRADING_PARAMS['initial_capital'])
        self.assertEqual(self.backtest.position, 0)
        self.assertEqual(len(self.backtest.trades), 0)
        self.assertEqual(len(self.backtest.equity_curve), 1)
        self.assertEqual(len(self.backtest.pending_orders), 0)
    
    def test_slippage_and_market_impact(self):
        """Test slippage and market impact calculations"""
        # 测试滑点
        price = 100.0
        buy_price = self.backtest._apply_slippage(price, 'buy')
        sell_price = self.backtest._apply_slippage(price, 'sell')
        
        self.assertGreater(buy_price, price)  # 买入价格应该高于原始价格
        self.assertLess(sell_price, price)  # 卖出价格应该低于原始价格
        
        # 测试市场影响
        quantity = 1.0
        impacted_price = self.backtest._apply_market_impact(price, quantity)
        self.assertGreater(impacted_price, price)  # 市场影响应该导致价格上升
    
    def test_order_execution_delay(self):
        """Test order execution delay"""
        # 创建测试订单
        current_time = datetime.now()
        order = {
            'type': 'open',
            'price': 100.0,
            'quantity': 1.0,
            'timestamp': current_time,
            'direction': 'long'
        }
        
        # 添加订单
        self.backtest.pending_orders.append(order)
        
        # 测试立即执行（应该不执行）
        self.backtest._execute_pending_orders(100.0, current_time)
        self.assertEqual(len(self.backtest.pending_orders), 1)
        
        # 测试延迟后执行
        delayed_time = current_time + timedelta(minutes=self.backtest.execution_delay)
        self.backtest._execute_pending_orders(100.0, delayed_time)
        self.assertEqual(len(self.backtest.pending_orders), 0)
    
    def test_position_management(self):
        """Test position management"""
        # 运行回测
        results = self.backtest.run(self.test_data, self.model)
        
        # 检查交易记录
        trades = results['trades']
        
        for trade in trades:
            # 检查交易记录的必要字段
            self.assertIn('timestamp', trade)
            self.assertIn('type', trade)
            self.assertIn('price', trade)
            self.assertIn('quantity', trade)
            self.assertIn('fee', trade)
            self.assertIn('capital', trade)
            
            # 检查数值的有效性
            self.assertGreater(trade['price'], 0)
            self.assertGreater(trade['quantity'], 0)
            self.assertGreater(trade['fee'], 0)
            self.assertGreater(trade['capital'], 0)
            
            # 检查交易类型
            self.assertIn(trade['type'], ['open', 'close'])
    
    def test_risk_management(self):
        """Test risk management"""
        # 运行回测
        results = self.backtest.run(self.test_data, self.model)
        
        # 检查止损止盈
        close_trades = [t for t in results['trades'] if t['type'] == 'close']
        
        for trade in close_trades:
            if 'reason' in trade:
                self.assertIn(trade['reason'], ['stop_loss', 'take_profit'])
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # 运行回测
        results = self.backtest.run(self.test_data, self.model)
        
        # 检查性能指标
        self.assertIn('win_rate', results)
        self.assertIn('max_drawdown', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('annual_return', results)
        self.assertIn('avg_slippage', results)
        self.assertIn('avg_execution_delay', results)
        
        # 检查指标值的合理性
        self.assertGreaterEqual(results['win_rate'], 0)
        self.assertLessEqual(results['win_rate'], 1)
        self.assertGreaterEqual(results['max_drawdown'], 0)
        self.assertLessEqual(results['max_drawdown'], 1)
        self.assertGreaterEqual(results['final_capital'], 0)

if __name__ == '__main__':
    unittest.main()