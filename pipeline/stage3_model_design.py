"""
Stage 3: Model Design and Implementation
Implements portfolio optimization models enhanced with sentiment analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import cvxpy as cp
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization"""
    min_weight: float = 0.0
    max_weight: float = 0.4
    target_volatility: float = 0.15
    risk_free_rate: float = 0.02
    rebalance_frequency: str = 'W'
    transaction_cost: float = 0.001
    sentiment_weight: float = 0.3
    lookback_period: int = 60


class SentimentAnalyzer:
    """Advanced sentiment analysis for portfolio construction"""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.sentiment_factors = [
            'news_vader_compound_mean',
            'news_emotion_intensity_mean',
            'news_sentiment_volatility_mean',
            'sentiment_momentum_alignment'
        ]

    def calculate_sentiment_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite sentiment scores for each asset"""
        # Ensure we have sentiment data
        available_factors = [f for f in self.sentiment_factors if f in df.columns]

        if not available_factors:
            logger.warning("No sentiment factors found, using neutral sentiment")
            df['sentiment_score'] = 0.5
            return df

        # Normalize sentiment factors
        for factor in available_factors:
            df[f'{factor}_norm'] = self._normalize_factor(df[factor])

        # Calculate weighted sentiment score
        weights = self._get_sentiment_weights(len(available_factors))
        df['sentiment_score'] = sum(
            df[f'{factor}_norm'] * weight
            for factor, weight in zip(available_factors, weights)
        )

        # Calculate sentiment momentum
        df['sentiment_momentum'] = df.groupby('symbol')['sentiment_score'].diff(5)

        # Calculate sentiment dispersion (disagreement)
        df['sentiment_dispersion'] = df.groupby('symbol')['sentiment_score'].rolling(20).std()

        return df

    def _normalize_factor(self, series: pd.Series) -> pd.Series:
        """Normalize factor to [0, 1] range"""
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val > 0:
            return (series - min_val) / (max_val - min_val)
        else:
            return pd.Series(0.5, index=series.index)

    def _get_sentiment_weights(self, n_factors: int) -> List[float]:
        """Get weights for sentiment factors"""
        # Give more weight to compound sentiment
        if n_factors == 4:
            return [0.4, 0.3, 0.2, 0.1]
        elif n_factors == 3:
            return [0.5, 0.3, 0.2]
        elif n_factors == 2:
            return [0.6, 0.4]
        else:
            return [1.0]

    def get_sentiment_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals based on sentiment"""
        latest_data = df.groupby('symbol').last()

        signals = {}
        for symbol in latest_data.index:
            score = latest_data.loc[symbol, 'sentiment_score']
            momentum = latest_data.loc[symbol, 'sentiment_momentum']
            dispersion = latest_data.loc[symbol, 'sentiment_dispersion']

            # Combine factors for final signal
            if pd.notna(momentum) and pd.notna(dispersion):
                # Strong positive signal: high score, positive momentum, low dispersion
                signal = score + 0.3 * momentum - 0.2 * dispersion
            else:
                signal = score

            signals[symbol] = np.clip(signal, 0, 1)

        return signals


class RiskModel:
    """Advanced risk modeling with sentiment integration"""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.estimator = LedoitWolf()

    def estimate_covariance(self, returns: pd.DataFrame,
                            sentiment_scores: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Estimate covariance matrix with sentiment adjustment"""
        # Base covariance from returns
        base_cov = self.estimator.fit(returns).covariance_

        if sentiment_scores is not None:
            # Adjust correlation based on sentiment similarity
            sentiment_corr = sentiment_scores.corr()

            # Blend return correlation with sentiment correlation
            return_corr = np.corrcoef(returns.T)
            blended_corr = (1 - self.config.sentiment_weight) * return_corr + \
                           self.config.sentiment_weight * sentiment_corr.values

            # Ensure positive definite
            eigenvalues, eigenvectors = np.linalg.eigh(blended_corr)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            blended_corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Convert correlation back to covariance
            std_devs = np.sqrt(np.diag(base_cov))
            cov_matrix = np.outer(std_devs, std_devs) * blended_corr
        else:
            cov_matrix = base_cov

        return cov_matrix

    def calculate_risk_metrics(self, weights: np.ndarray,
                               returns: pd.DataFrame,
                               cov_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        portfolio_return = np.dot(weights, returns.mean()) * 252
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        # Downside risk
        portfolio_returns = returns @ weights
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Maximum drawdown
        cum_returns = (1 + portfolio_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'downside_volatility': downside_vol,
            'max_drawdown': max_drawdown
        }


class PortfolioOptimizer:
    """Main portfolio optimization engine"""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.risk_model = RiskModel(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)

    def optimize_portfolio(self, df: pd.DataFrame,
                           optimization_type: str = 'sentiment_enhanced') -> Dict:
        """Main optimization function"""
        # Prepare data
        returns, features, sentiment_signals = self._prepare_data(df)

        if returns.empty:
            logger.error("No valid returns data for optimization")
            return {}

        # Estimate covariance
        sentiment_df = df.pivot(index='date', columns='symbol', values='sentiment_score').iloc[-60:]
        cov_matrix = self.risk_model.estimate_covariance(returns, sentiment_df)

        # Select optimization method
        if optimization_type == 'sentiment_enhanced':
            weights = self._sentiment_enhanced_optimization(
                returns, cov_matrix, sentiment_signals
            )
        elif optimization_type == 'risk_parity':
            weights = self._risk_parity_optimization(returns, cov_matrix)
        elif optimization_type == 'min_variance':
            weights = self._min_variance_optimization(cov_matrix)
        elif optimization_type == 'max_sharpe':
            weights = self._max_sharpe_optimization(returns, cov_matrix)
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")

        # Calculate metrics
        metrics = self.risk_model.calculate_risk_metrics(weights, returns, cov_matrix)

        return {
            'weights': dict(zip(returns.columns, weights)),
            'metrics': metrics,
            'sentiment_signals': sentiment_signals
        }

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Prepare data for optimization"""
        # Calculate sentiment scores
        df = self.sentiment_analyzer.calculate_sentiment_scores(df)

        # Get returns matrix
        returns = df.pivot(index='date', columns='symbol', values='returns_1d')
        returns = returns.iloc[-self.config.lookback_period:].dropna(axis=1)

        # Get features matrix
        feature_cols = [col for col in df.columns if col.endswith('_norm') or col.endswith('_zscore')]
        if feature_cols:
            features = df.pivot(index='date', columns='symbol', values=feature_cols[0])
            features = features.iloc[-self.config.lookback_period:]
        else:
            features = pd.DataFrame()

        # Get sentiment signals
        sentiment_signals = self.sentiment_analyzer.get_sentiment_signals(df)

        return returns, features, sentiment_signals

    def _sentiment_enhanced_optimization(self, returns: pd.DataFrame,
                                         cov_matrix: np.ndarray,
                                         sentiment_signals: Dict) -> np.ndarray:
        """Optimization with sentiment integration"""
        n_assets = len(returns.columns)

        # Adjust expected returns based on sentiment
        base_returns = returns.mean().values
        sentiment_adjustment = np.array([
            sentiment_signals.get(symbol, 0.5) - 0.5
            for symbol in returns.columns
        ])

        # Scale adjustment to be meaningful but not dominant
        adjusted_returns = base_returns + 0.1 * sentiment_adjustment * returns.std().values

        # Optimization problem
        w = cp.Variable(n_assets)

        # Objective: maximize sentiment-adjusted Sharpe ratio
        ret = adjusted_returns @ w
        risk = cp.quad_form(w, cov_matrix)
        objective = cp.Maximize(ret - 0.5 * self.config.target_volatility * risk)

        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= self.config.min_weight,
            w <= self.config.max_weight
        ]

        # Add risk constraint
        constraints.append(risk <= self.config.target_volatility ** 2)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        if problem.status == 'optimal':
            return w.value
        else:
            logger.warning(f"Optimization problem status: {problem.status}")
            # Fall back to equal weights
            return np.ones(n_assets) / n_assets

    def _risk_parity_optimization(self, returns: pd.DataFrame,
                                  cov_matrix: np.ndarray) -> np.ndarray:
        """Risk parity optimization"""
        n_assets = len(returns.columns)

        def risk_contribution(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib

        def objective(weights):
            contrib = risk_contribution(weights)
            # Minimize variance of risk contributions
            return np.var(contrib)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]

        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets

        # Initial guess
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        else:
            logger.warning("Risk parity optimization failed")
            return w0

    def _min_variance_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Minimum variance optimization"""
        n_assets = cov_matrix.shape[0]

        w = cp.Variable(n_assets)
        risk = cp.quad_form(w, cov_matrix)

        objective = cp.Minimize(risk)
        constraints = [
            cp.sum(w) == 1,
            w >= self.config.min_weight,
            w <= self.config.max_weight
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        if problem.status == 'optimal':
            return w.value
        else:
            return np.ones(n_assets) / n_assets

    def _max_sharpe_optimization(self, returns: pd.DataFrame,
                                 cov_matrix: np.ndarray) -> np.ndarray:
        """Maximum Sharpe ratio optimization"""
        n_assets = len(returns.columns)
        expected_returns = returns.mean().values

        def negative_sharpe(weights):
            portfolio_return = expected_returns @ weights
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol
            return -sharpe

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]

        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets

        # Initial guess: equal weights
        w0 = np.ones(n_assets) / n_assets

        result = minimize(negative_sharpe, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if result.success:
            return result.x
        else:
            return w0


class DynamicRiskBudgeting:
    """Dynamic risk budgeting based on market conditions and sentiment"""

    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.base_risk_budget = config.target_volatility

    def calculate_risk_budget(self, market_data: pd.DataFrame,
                              sentiment_data: Dict[str, float]) -> float:
        """Calculate dynamic risk budget based on market conditions"""
        # Market volatility regime
        market_vol = market_data['returns_1d'].std() * np.sqrt(252)
        vol_percentile = market_data['hvol_20'].rank(pct=True).iloc[-1]

        # Sentiment regime
        avg_sentiment = np.mean(list(sentiment_data.values()))
        sentiment_dispersion = np.std(list(sentiment_data.values()))

        # Risk budget adjustments
        vol_adjustment = 1.0
        if vol_percentile > 0.8:  # High volatility regime
            vol_adjustment = 0.7
        elif vol_percentile < 0.2:  # Low volatility regime
            vol_adjustment = 1.3

        sentiment_adjustment = 1.0
        if avg_sentiment < 0.3 and sentiment_dispersion > 0.2:  # Fear with disagreement
            sentiment_adjustment = 0.6
        elif avg_sentiment > 0.7 and sentiment_dispersion < 0.1:  # Greed with consensus
            sentiment_adjustment = 0.8

        # Final risk budget
        risk_budget = self.base_risk_budget * vol_adjustment * sentiment_adjustment

        return np.clip(risk_budget, 0.05, 0.25)  # Keep within reasonable bounds


class BacktestEngine:
    """Backtesting engine for portfolio strategies"""

    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.results = {}

    def backtest_strategy(self, df: pd.DataFrame,
                          strategy_func,
                          rebalance_freq: str = 'W',
                          transaction_cost: float = 0.001) -> Dict[str, Any]:
        """Backtest a portfolio strategy"""
        # Group by rebalance frequency
        df['period'] = df['date'].dt.to_period(rebalance_freq)
        periods = df['period'].unique()

        portfolio_values = [self.initial_capital]
        weights_history = []
        trades_history = []

        prev_weights = {}

        for period in periods[:-1]:  # Skip last period (no forward returns)
            # Get data for this period
            period_data = df[df['period'] == period].copy()

            # Run strategy
            optimization_result = strategy_func(period_data)

            if not optimization_result:
                continue

            new_weights = optimization_result['weights']

            # Calculate trades
            trades = self._calculate_trades(prev_weights, new_weights, portfolio_values[-1])
            trades_history.append(trades)

            # Get next period returns
            next_period_data = df[df['period'] == periods[periods.get_loc(period) + 1]]

            # Calculate portfolio return
            portfolio_return = self._calculate_portfolio_return(
                new_weights, next_period_data, transaction_cost, trades
            )

            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)

            weights_history.append({
                'date': period.to_timestamp(),
                'weights': new_weights.copy()
            })

            prev_weights = new_weights.copy()

        # Calculate performance metrics
        returns = pd.Series(portfolio_values[1:]) / pd.Series(portfolio_values[:-1]) - 1

        results = {
            'portfolio_values': portfolio_values,
            'returns': returns,
            'weights_history': weights_history,
            'trades_history': trades_history,
            'metrics': self._calculate_metrics(returns, portfolio_values),
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        }

        return results

    def _calculate_trades(self, old_weights: Dict[str, float],
                          new_weights: Dict[str, float],
                          portfolio_value: float) -> Dict[str, float]:
        """Calculate trades needed to rebalance"""
        trades = {}

        all_symbols = set(old_weights.keys()) | set(new_weights.keys())

        for symbol in all_symbols:
            old_value = old_weights.get(symbol, 0) * portfolio_value
            new_value = new_weights.get(symbol, 0) * portfolio_value
            trade_value = new_value - old_value

            if abs(trade_value) > 0.01 * portfolio_value:  # Only trade if > 1%
                trades[symbol] = trade_value

        return trades

    def _calculate_portfolio_return(self, weights: Dict[str, float],
                                    next_period_data: pd.DataFrame,
                                    transaction_cost: float,
                                    trades: Dict[str, float]) -> float:
        """Calculate portfolio return including transaction costs"""
        # Asset returns
        returns_by_symbol = {}
        for symbol in next_period_data['symbol'].unique():
            symbol_data = next_period_data[next_period_data['symbol'] == symbol]
            if not symbol_data.empty:
                returns_by_symbol[symbol] = symbol_data['returns_1d'].mean()

        # Portfolio return before costs
        gross_return = sum(
            weights.get(symbol, 0) * returns_by_symbol.get(symbol, 0)
            for symbol in weights.keys()
        )

        # Transaction costs
        total_trades = sum(abs(trade) for trade in trades.values())
        cost = transaction_cost * total_trades / sum(abs(w) for w in weights.values())

        return gross_return - cost

    def _calculate_metrics(self, returns: pd.Series,
                           portfolio_values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        # Annualized metrics
        periods_per_year = 252 if len(returns) > 252 else len(returns)

        metrics = {
            'annual_return': returns.mean() * periods_per_year,
            'annual_volatility': returns.std() * np.sqrt(periods_per_year),
            'sharpe_ratio': (returns.mean() * periods_per_year) / (returns.std() * np.sqrt(periods_per_year)),
            'sortino_ratio': self._calculate_sortino_ratio(returns, periods_per_year),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'calmar_ratio': (returns.mean() * periods_per_year) / abs(self._calculate_max_drawdown(portfolio_values)),
            'win_rate': (returns > 0).mean(),
            'best_period': returns.max(),
            'worst_period': returns.min(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': returns.quantile(0.05),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean()
        }

        return metrics

    def _calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(periods_per_year)
            return (returns.mean() * periods_per_year) / downside_vol
        return np.inf

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        values = pd.Series(portfolio_values)
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max
        return drawdown.min()


async def run_portfolio_optimization(config: Dict) -> Dict[str, Any]:
    """Main function for Stage 3"""
    logger.info("Starting Stage 3: Portfolio Optimization")

    # Load features
    features_path = Path(config['features_path'])
    df = pd.read_csv(features_path, parse_dates=['date'])

    # Initialize components
    portfolio_config = PortfolioConfig(
        min_weight=config.get('min_weight', 0.0),
        max_weight=config.get('max_weight', 0.4),
        target_volatility=config.get('target_volatility', 0.15),
        sentiment_weight=config.get('sentiment_weight', 0.3),
        rebalance_frequency=config.get('rebalance_frequency', 'D'),
        lookback_period=config.get('lookback_period', 60)
    )

    optimizer = PortfolioOptimizer(portfolio_config)
    risk_budgeting = DynamicRiskBudgeting(portfolio_config)
    backtest_engine = BacktestEngine()

    # Define strategy
    def sentiment_enhanced_strategy(period_data: pd.DataFrame) -> Dict:
        # Update risk budget
        sentiment_signals = optimizer.sentiment_analyzer.get_sentiment_signals(period_data)
        portfolio_config.target_volatility = risk_budgeting.calculate_risk_budget(
            period_data, sentiment_signals
        )

        # Optimize portfolio
        return optimizer.optimize_portfolio(
            period_data,
            optimization_type='sentiment_enhanced'
        )

    # Run backtest
    backtest_results = backtest_engine.backtest_strategy(
        df,
        sentiment_enhanced_strategy,
        rebalance_freq=portfolio_config.rebalance_frequency,
        transaction_cost=portfolio_config.transaction_cost
    )

    # Generate current portfolio
    latest_data = df.sort_values('date').groupby('symbol').tail(portfolio_config.lookback_period)
    current_portfolio = optimizer.optimize_portfolio(
        latest_data,
        optimization_type='sentiment_enhanced'
    )

    # Prepare results
    results = {
        'current_weights': current_portfolio['weights'],
        'current_metrics': current_portfolio['metrics'],
        'sentiment_signals': current_portfolio['sentiment_signals'],
        'backtest_results': backtest_results,
        'optimization_config': portfolio_config.__dict__,
        'timestamp': datetime.now().isoformat()
    }

    # Save results
    output_dir = Path(config.get('output_dir', 'data/results'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save portfolio weights
    pd.DataFrame([current_portfolio['weights']]).T.to_csv(
        output_dir / 'current_weights.csv',
        header=['weight']
    )

    # Save backtest results
    pd.DataFrame(backtest_results['weights_history']).to_csv(
        output_dir / 'weights_history.csv',
        index=False
    )

    # Save complete results
    with open(output_dir / 'portfolio_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Stage 3 completed successfully!")
    logger.info(f"Current Sharpe Ratio: {current_portfolio['metrics']['sharpe_ratio']:.3f}")
    logger.info(f"Backtest Total Return: {backtest_results['total_return']:.2%}")

    return results


if __name__ == "__main__":
    config = {
        'features_path': 'data/processed/stage2_features.csv',
        'output_dir': 'data/results',
        'min_weight': 0.0,
        'max_weight': 0.4,
        'target_volatility': 0.15,
        'sentiment_weight': 0.3,
        'rebalance_frequency': 'D',
        'lookback_period': 60
    }

    results = asyncio.run(run_portfolio_optimization(config))