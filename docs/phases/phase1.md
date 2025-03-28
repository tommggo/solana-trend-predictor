# 第一阶段：基础架构搭建

## 1. 数据收集模块

### 1.1 数据生成器 (DataGenerator)
- 负责数据收集和预处理
- 生成 OHLCV、订单簿和链上数据
- 数据清洗和标准化

### 1.2 数据源
- OHLCV 数据
- 订单簿数据
- 链上数据

### 1.3 数据预处理
- 数据清洗
- 缺失值处理
- 异常值检测

## 2. 特征工程

### 2.1 技术指标模块 (indicators.py)
- 计算技术分析指标
  - ATR(14)
  - RSI(14)
  - EMA(5)/EMA(15) 比率
  - 大单标记功能
- 生成市场微观结构特征
- 处理链上数据特征

### 2.2 特征列表
```python
feature_columns = [
    # 价格相关特征
    'price_change',          # 价格变化率
    'price_volatility',      # 价格波动率
    
    # 成交量相关特征
    'volume_change',         # 成交量变化率
    'order_count_ratio',     # 订单数量比率
    'large_trade',           # 大单交易标记
    
    # 订单簿特征
    'order_pressure',        # 订单压力（买卖盘口压力差）
    
    # 链上数据特征
    'network_activity',      # 网络活跃度
    'stake_ratio',           # 质押比率
    'stake_ratio_change',    # 质押比率变化
    'network_congestion',    # 网络拥堵度
    'large_transfers',       # 大额转账数量
    
    # 技术指标
    'atr',                   # 平均真实波幅 (Average True Range)
    'rsi',                   # 相对强弱指标 (Relative Strength Index)
    'ema_ratio'             # EMA(5)/EMA(15) 比率
]
```

## 3. 基础模型

### 3.1 预测模型 (SolanaPredictor)
- LightGBM 模型实现
- 特征工程和模型训练
- 预测和评估功能

### 3.2 模型参数
```python
MODEL_PARAMS = {
    # 模型目标
    'objective': 'binary',           # 二分类问题（预测上涨/下跌）
    'metric': 'binary_logloss',      # 评估指标：二元对数损失
    
    # 模型结构
    'boosting_type': 'gbdt',         # 梯度提升决策树
    'num_leaves': 31,                # 叶子节点数量
    'learning_rate': 0.05,           # 学习率
    'feature_fraction': 0.9          # 特征采样比例
}
```

### 3.3 模型功能
- 数据准备
- 模型训练
- 预测
- 评估
- 模型保存/加载

## 4. 回测系统

### 4.1 回测引擎 (Backtest)
- 模拟交易执行
- 风险控制
- 性能评估

### 4.2 交易参数
```python
TRADING_PARAMS = {
    # 资金管理
    'initial_capital': 10000,        # 初始资金
    'position_size': 0.1,            # 单次交易仓位比例
    
    # 交易成本
    'fee_rate': 0.001,               # 交易手续费率
    
    # 风险控制
    'stop_loss': 0.02,               # 止损比例
    'take_profit': 0.03,             # 止盈比例
    
    # 交易信号
    'buy_threshold': 0.6,            # 买入信号阈值
    'sell_threshold': 0.4            # 卖出信号阈值
}
```

### 4.3 回测功能
- 订单执行
- 仓位管理
- 风险控制
- 性能评估
- 可视化分析

## 5. 使用示例

### 5.1 系统初始化
```python
# 初始化系统组件
data_generator = DataGenerator()
model = SolanaPredictor()
backtest = Backtest()
```

### 5.2 数据准备
```python
# 生成数据
data = data_generator.generate_data()

# 准备特征
features, labels = model.prepare_data(data)
```

### 5.3 模型训练
```python
# 训练模型
model.train(features, labels)

# 保存模型
model.save_model()
```

### 5.4 回测分析
```python
# 运行回测
results = backtest.run(data, model)

# 分析结果
backtest.plot_equity_curve()
backtest.plot_drawdown()
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## 6. 完成状态

### 6.1 已完成功能
- [x] 数据生成器框架
- [x] 技术指标计算
- [x] 基础模型实现
- [x] 回测系统
- [x] 性能评估指标

### 6.2 待完成功能
- [ ] 实际数据收集实现
- [ ] 实时数据更新机制
- [ ] Parquet 格式数据存储
- [ ] 增量训练机制

## 7. 注意事项

1. 数据质量
   - 确保数据完整性
   - 处理异常值
   - 保持数据一致性

2. 模型使用
   - 定期重训练
   - 监控模型性能
   - 避免过拟合

3. 风险控制
   - 严格执行止损
   - 控制仓位大小
   - 监控系统风险

4. 系统维护
   - 定期备份数据
   - 更新依赖包
   - 检查系统日志 