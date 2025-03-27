from setuptools import setup, find_packages

setup(
    name="solana-trend-predictor",
    version="0.1.0",
    description="A machine learning system for predicting SOL price trends",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "lightgbm>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.17.0",
        "scikit-learn>=1.0.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
) 