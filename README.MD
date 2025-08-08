# Cryptocurrency Portfolio Management System

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.36:8501

A 4-stage pipeline system for cryptocurrency portfolio optimization.

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the System
```bash
python run.py
```

Then select an option:
- **1**: Run all stages (complete pipeline)
- **2**: Run Stage 1 only (Data Collection)
- **3**: Run Stage 2 only (Feature Engineering)
- **4**: Run Stage 3 only (Model Design)
- **5**: Run Stage 4 only (Dashboard)
- **6**: Run stages 1-3 (without Dashboard)

## Project Structure

```

├── .env
├── .gitignore
├── .idea
│   ├── .gitignore
│   ├── fintech_crypto_project.iml
│   ├── inspectionProfiles
│   │   ├── Project_Default.xml
│   │   └── profiles_settings.xml
│   ├── misc.xml
│   ├── modules.xml
│   └── workspace.xml
├── 1.py
├── README
├── __pycache__
│   └── config.cpython-312.pyc
├── config.py
├── crypto_portfolio.db
├── data
│   ├── cache
│   │   └── crypto_cache.db
│   ├── features
│   │   ├── engineered_features.csv
│   │   ├── engineered_features.parquet
│   │   ├── feature_report.json
│   │   └── news_features.csv
│   ├── processed
│   │   ├── collection_report.json
│   │   ├── news_data.csv
│   │   ├── news_data.parquet
│   │   ├── price_data.csv
│   │   └── price_data.parquet
│   └── results
│       ├── best_strategy_holdings.csv
│       ├── cumulative_returns.html
│       ├── metrics_comparison.html
│       ├── optimal_weights.csv
│       ├── optimization_results.json
│       ├── plots
│       │   ├── cumret_W-WED.png
│       │   ├── cumulative_returns.png
│       │   ├── metrics_comparison.png
│       │   ├── portfolio_allocation_analysis.png
│       │   ├── portfolio_weights.png
│       │   ├── risk_return_scatter.png
│       │   ├── sharpe_W-WED.png
│       │   ├── sharpe_comparison.png
│       │   ├── strategy_comparison.png
│       │   └── strategy_performance_grid.png
│       ├── risk_return_scatter.html
│       ├── sentiment_analysis
│       │   ├── sentiment_wordcloud.png
│       │   └── sentiment_wordcloud_combined.png
│       ├── sharpe_comparison.html
│       ├── strategy_comparison.csv
│       ├── strategy_comparison.html
│       └── weights_heatmap.html
├── logs
│   └── data_collection.log
├── reports
│   └── stage2
│       ├── feature_engineering_report.json
│       ├── plots
│       │   ├── correlation_heatmap.png
│       │   └── feature_distributions.png
│       └── tables
│           ├── feature_summary.csv
│           ├── feature_summary.tex
│           ├── feature_summary.txt
│           ├── missing_values.csv
│           ├── missing_values.tex
│           ├── missing_values.txt
│           ├── sentiment_statistics.csv
│           ├── sentiment_statistics.tex
│           └── sentiment_statistics.txt
├── requirements.txt
├── run.py
├── run_app.bat
└── src
    ├── __init__.py
    ├── __pycache__
    │   └── __init__.cpython-312.pyc
    └── pipeline
        ├── __init__.py
        ├── __pycache__
        │   ├── __init__.cpython-312.pyc
        │   ├── enhanced_visualizer.cpython-312.pyc
        │   └── stage3_model_design.cpython-312.pyc
        ├── enhanced_visualizer.py
        ├── stage1_data_collection.py
        ├── stage2_feature_engineering.py
        ├── stage3_model_design.py
        └── stage4_app_implementation.py
```

## Pipeline Stages

### Stage 1: Data Collection
- Fetches top 200 cryptocurrencies data from CoinDesk API
- Collects historical OHLCV (Open, High, Low, Close, Volume) data
- Saves to `data/processed/stage_1_crypto_data.csv`

### Stage 2: Feature Engineering
- Calculates technical indicators (momentum, volatility, volume shocks)
- Generates weekly returns and risk metrics
- Outputs to `data/processed/stage_2_crypto_data.csv`

### Stage 3: Model Design
- Performs portfolio optimization
- Implements multiple strategies (mean-variance, risk parity, etc.)
- Saves portfolio weights and performance metrics

### Stage 4: Application
- Launches interactive Streamlit dashboard
- Displays portfolio performance and allocations
- Provides real-time monitoring and analysis

## Configuration

Edit `config.py` to modify:
- API keys
- Date ranges
- Number of cryptocurrencies
- Risk parameters
- Portfolio constraints

## Common Issues

1. **Import Error**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Not Found**: Run stages in order (1→2→3→4)
   ```bash
   python run.py
   # Select option 1 to run all stages
   ```

3. **Dashboard Not Opening**: The dashboard runs on http://localhost:8501
   - Check if port 8501 is available
   - Try running directly: `streamlit run src/pipeline/stage4_app_implementation.py`

## Data Output

- **Stage 1**: `data/processed/stage_1_crypto_data.csv`
- **Stage 2**: `data/processed/stage_2_crypto_data.csv`
- **Stage 3**: `data/results/` (portfolio weights and metrics)
- **Stage 4**: Interactive dashboard on localhost:8501

## Notes

- The system uses CoinDesk API for data (API key included in config.py)
- First run may take longer due to data download
- Dashboard updates every 60 seconds by default
- Sentiment analysis uses VADER by default

## Support

For issues or questions, check:
1. Log files in `logs/` directory
2. Ensure all stages completed successfully
3. Verify data files exist in `data/processed/`
