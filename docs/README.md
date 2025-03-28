# Solana 趋势预测系统文档

## 项目概述

Solana 趋势预测系统是一个基于机器学习的加密货币交易系统，专门用于预测 Solana (SOL) 的短期价格趋势。系统采用 5 分钟时间周期的数据进行预测，结合了技术分析、市场微观结构和链上数据等多个维度的信息。

## 系统要求

- Python 3.8+
- 依赖包：
  - pandas
  - numpy
  - lightgbm
  - scikit-learn
  - matplotlib

## 目录结构

```
solana-trend-predictor/
├── src/                    # 源代码
├── tests/                  # 测试用例
├── docs/                   # 文档
│   ├── phases/            # 阶段文档
│   │   ├── phase1.md      # 第一阶段文档
│   │   ├── phase2.md      # 第二阶段文档
│   │   └── phase3.md      # 第三阶段文档
│   └── README.md          # 主文档
└── data/                   # 数据存储
```

## 开发阶段

系统开发分为三个主要阶段，每个阶段都有详细的实现文档：

### 第一阶段：基础架构搭建
- 数据收集模块
- 特征工程
- 基础模型
- 回测系统
[详细文档](phases/phase1.md)

### 第二阶段：模型优化
- 模型集成
- 参数优化
- 特征选择
- 性能提升
[详细文档](phases/phase2.md)

### 第三阶段：系统完善
- 实时交易
- 风险控制
- 监控系统
- 部署运维
[详细文档](phases/phase3.md)

## 快速开始

1. 克隆项目
```bash
git clone https://github.com/yourusername/solana-trend-predictor.git
cd solana-trend-predictor
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行示例
```python
from src.data_generator import DataGenerator
from src.model import SolanaPredictor
from src.backtest import Backtest

# 初始化系统组件
data_generator = DataGenerator()
model = SolanaPredictor()
backtest = Backtest()

# 运行系统
data = data_generator.generate_data()
features, labels = model.prepare_data(data)
model.train(features, labels)
results = backtest.run(data, model)
```

## 文档导航

- [第一阶段文档](phases/phase1.md)：基础架构搭建
- [第二阶段文档](phases/phase2.md)：模型优化
- [第三阶段文档](phases/phase3.md)：系统完善

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License 