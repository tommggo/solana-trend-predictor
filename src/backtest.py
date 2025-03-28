"""
Backtesting system for Solana Trend Predictor
实现回测策略、性能评估、风险控制和交易记录功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from .model import SolanaPredictor
from .config import TRADING_PARAMS

class Backtest:
    def __init__(
        self,
        initial_capital: float = TRADING_PARAMS['initial_capital'],
        fee_rate: float = TRADING_PARAMS['fee_rate'],
        stop_loss: float = TRADING_PARAMS['stop_loss'],
        take_profit: float = TRADING_PARAMS['take_profit'],
        position_size: float = TRADING_PARAMS['position_size'],
        execution_delay: int = 1,  # 执行延迟（分钟）
        slippage: float = 0.001,  # 滑点率
        market_impact: float = 0.0005  # 市场影响率
    ):
        """
        初始化回测系统
        
        Args:
            initial_capital (float): 初始资金
            fee_rate (float): 交易费率
            stop_loss (float): 止损比例
            take_profit (float): 止盈比例
            position_size (float): 仓位大小（占总资金比例）
            execution_delay (int): 执行延迟（分钟）
            slippage (float): 滑点率
            market_impact (float): 市场影响率
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.execution_delay = execution_delay
        self.slippage = slippage
        self.market_impact = market_impact
        
        # 回测状态
        self.capital = initial_capital  # 当前资金
        self.position = 0  # 当前持仓量
        self.entry_price = 0  # 入场价格
        self.trades: List[Dict] = []  # 交易记录
        self.equity_curve: List[float] = [initial_capital]  # 权益曲线
        self.pending_orders: List[Dict] = []  # 待执行订单
        
        # 性能指标
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.annual_return = 0
        self.avg_slippage = 0
        self.avg_execution_delay = 0
    
    def _apply_slippage(self, price: float, direction: str) -> float:
        """
        应用滑点
        
        Args:
            price (float): 原始价格
            direction (str): 交易方向（'buy'或'sell'）
            
        Returns:
            float: 考虑滑点后的价格
        """
        slippage_factor = 1 + self.slippage if direction == 'buy' else 1 - self.slippage
        return price * slippage_factor
    
    def _apply_market_impact(self, price: float, quantity: float) -> float:
        """
        应用市场影响
        
        Args:
            price (float): 原始价格
            quantity (float): 交易数量
            
        Returns:
            float: 考虑市场影响后的价格
        """
        impact_factor = 1 + (quantity * self.market_impact)
        return price * impact_factor
    
    def _execute_pending_orders(self, current_price: float, current_time: datetime):
        """
        执行待处理订单
        
        Args:
            current_price (float): 当前价格
            current_time (datetime): 当前时间
        """
        # 过滤出需要执行的订单
        orders_to_execute = [
            order for order in self.pending_orders
            if (current_time - order['timestamp']).total_seconds() >= self.execution_delay * 60
        ]
        
        for order in orders_to_execute:
            # 应用滑点和市场影响
            execution_price = self._apply_slippage(current_price, order['direction'])
            execution_price = self._apply_market_impact(execution_price, order['quantity'])
            
            # 执行订单
            if order['type'] == 'open':
                self._open_position(execution_price, current_time, order['direction'])
            else:
                self._close_position(execution_price, current_time, order['reason'])
            
            # 从待处理订单中移除
            self.pending_orders.remove(order)
    
    def _place_order(self, order_type: str, price: float, timestamp: datetime, 
                    direction: str = None, reason: str = None):
        """
        下单
        
        Args:
            order_type (str): 订单类型（'open'或'close'）
            price (float): 当前价格
            timestamp (datetime): 当前时间
            direction (str): 交易方向（'long'或'short'）
            reason (str): 平仓原因
        """
        # 计算交易量
        if order_type == 'open':
            position_value = self.capital * self.position_size
            quantity = max(position_value / price, 0.01)
        else:
            quantity = abs(self.position)
        
        # 添加到待处理订单
        self.pending_orders.append({
            'type': order_type,
            'price': price,
            'quantity': quantity,
            'timestamp': timestamp,
            'direction': direction,
            'reason': reason
        })
    
    def run(self, data: pd.DataFrame, model: SolanaPredictor) -> Dict:
        """
        运行回测
        
        Args:
            data (pd.DataFrame): 回测数据
            model (SolanaPredictor): 预测模型
            
        Returns:
            Dict: 回测结果
        """
        # 准备特征
        features, _ = model.prepare_data(data)
        
        # 获取预测概率
        probabilities = model.predict_proba(features)
        
        # 重置回测状态
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.pending_orders = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # 遍历每个时间点
        for i in range(len(features)):
            current_price = data.iloc[i]['close']
            current_time = data.iloc[i]['timestamp']
            
            # 执行待处理订单
            self._execute_pending_orders(current_price, current_time)
            
            # 检查止损止盈
            if self.position != 0:
                price_change = (current_price - self.entry_price) / self.entry_price
                
                # 止损
                if price_change <= -self.stop_loss:
                    self._place_order('close', current_price, current_time, reason='stop_loss')
                
                # 止盈
                elif price_change >= self.take_profit:
                    self._place_order('close', current_price, current_time, reason='take_profit')
            
            # 获取预测概率
            prob = probabilities[i]
            
            # 交易信号
            if prob > TRADING_PARAMS['buy_threshold'] and self.position <= 0:  # 做多信号
                self._place_order('open', current_price, current_time, direction='long')
            elif prob < TRADING_PARAMS['sell_threshold'] and self.position >= 0:  # 做空信号
                self._place_order('open', current_price, current_time, direction='short')
            
            # 更新权益曲线
            current_value = self.capital + (self.position * current_price if self.position != 0 else 0)
            self.equity_curve.append(current_value)
        
        # 计算性能指标
        self._calculate_performance_metrics(data['timestamp'])
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'annual_return': self.annual_return,
            'final_capital': self.capital,
            'avg_slippage': self.avg_slippage,
            'avg_execution_delay': self.avg_execution_delay
        }
    
    def _open_position(self, price: float, timestamp: datetime, direction: str):
        """
        开仓
        
        Args:
            price (float): 当前价格
            timestamp (datetime): 当前时间
            direction (str): 交易方向（'long'或'short'）
        """
        # 计算交易量（确保至少为0.01）
        position_value = self.capital * self.position_size
        quantity = max(position_value / price, 0.01)  # 确保最小交易量为0.01
        
        # 计算手续费（确保至少为0.0001）
        fee = max(position_value * self.fee_rate, 0.0001)
        
        # 确保有足够的资金
        if position_value + fee > self.capital:
            position_value = self.capital * 0.99  # 留出1%作为手续费
            quantity = max(position_value / price, 0.01)  # 确保最小交易量为0.01
            fee = max(position_value * self.fee_rate, 0.0001)
        
        # 更新资金和持仓
        self.capital -= fee
        self.position = quantity if direction == 'long' else -quantity
        self.entry_price = price
        
        # 记录交易
        self.trades.append({
            'timestamp': timestamp,
            'type': 'open',
            'direction': direction,
            'price': price,
            'quantity': quantity,  # 使用计算出的数量
            'fee': fee,
            'capital': self.capital
        })
    
    def _close_position(self, price: float, timestamp: datetime, reason: str):
        """
        平仓
        
        Args:
            price (float): 当前价格
            timestamp (datetime): 当前时间
            reason (str): 平仓原因
        """
        # 计算盈亏
        price_change = (price - self.entry_price) / self.entry_price
        if self.position < 0:  # 空仓
            price_change = -price_change
        
        # 计算交易量
        position_value = abs(self.position) * price
        
        # 计算手续费
        fee = position_value * self.fee_rate
        
        # 计算盈亏
        pnl = position_value * price_change - fee
        
        # 更新资金和持仓
        self.capital += pnl
        self.position = 0
        
        # 更新交易统计
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # 记录交易
        self.trades.append({
            'timestamp': timestamp,
            'type': 'close',
            'reason': reason,
            'price': price,
            'quantity': max(abs(self.position), 0.01),  # 确保最小交易量为0.01
            'fee': fee,
            'pnl': pnl,
            'capital': self.capital
        })
    
    def _calculate_performance_metrics(self, timestamps: pd.Series):
        """
        计算性能指标
        
        Args:
            timestamps (pd.Series): 时间戳序列
        """
        # 计算收益率序列
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        # 计算最大回撤
        cumulative_returns = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (rolling_max - cumulative_returns) / rolling_max
        self.max_drawdown = float(np.max(drawdowns) if len(drawdowns) > 0 else 0.0)
        
        # 计算夏普比率
        risk_free_rate = 0.02  # 假设无风险利率为2%
        excess_returns = returns - risk_free_rate/252  # 日化无风险利率
        std_returns = np.std(excess_returns)
        if std_returns > 0:
            self.sharpe_ratio = float(np.sqrt(252) * np.mean(excess_returns) / std_returns)
        else:
            self.sharpe_ratio = 0.0
        
        # 计算年化收益率
        total_days = (timestamps.iloc[-1] - timestamps.iloc[0]).days
        if total_days > 0:
            self.annual_return = float((self.capital / self.initial_capital) ** (365/total_days) - 1)
        else:
            self.annual_return = 0.0
    
    def plot_equity_curve(self):
        """
        绘制权益曲线
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Trading Days')
        plt.ylabel('Capital')
        plt.grid(True)
        plt.show()
    
    def plot_drawdown(self):
        """
        绘制回撤曲线
        """
        import matplotlib.pyplot as plt
        
        # 计算回撤
        cumulative_returns = np.cumprod(1 + np.diff(self.equity_curve) / self.equity_curve[:-1])
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (rolling_max - cumulative_returns) / rolling_max
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdowns)
        plt.title('Drawdown')
        plt.xlabel('Trading Days')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.show()