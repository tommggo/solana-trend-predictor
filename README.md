# Solana Trend Predictor

一个基于机器学习的 Solana (SOL) 5分钟趋势预测系统。

## 项目结构

```
.
├── src/                    # 源代码目录
│   ├── __init__.py        # 包初始化文件
│   ├── model.py           # 模型实现
│   ├── backtest.py        # 回测系统
│   ├── data_generator.py  # 数据生成器
│   └── config.py          # 配置文件
├── examples/              # 示例代码
│   └── backtest_demo.py   # 回测示例
├── tests/                 # 测试代码
├── notebooks/             # Jupyter notebooks
├── data/                  # 数据目录
├── models/                # 模型保存目录
├── setup.py              # 包安装配置
└── pyproject.toml        # 项目配置
```

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/solana-trend-predictor.git
cd solana-trend-predictor
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -e .
```

## 使用

运行回测示例：
```bash
python examples/backtest_demo.py
```

## 开发计划

详细的开发计划请参见 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)。

## 许可证

MIT License