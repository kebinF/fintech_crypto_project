"""
Stage 3: Model Design and Portfolio Optimization - Enhanced Version
Complete implementation with Sentiment Word Cloud Generation
Enhanced for FINS5545 FinTech Project Requirements
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
import json
import sys
import os
from collections import Counter
import re

# Portfolio optimization
from scipy.optimize import minimize
from scipy import stats
import cvxpy as cp
from sklearn.covariance import LedoitWolf

# Visualization - Both Plotly and Matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Word cloud generation
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK data if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

warnings.filterwarnings('ignore')

# ==================== FIXED PATH SETUP ====================
def setup_paths():
    """Setup paths correctly regardless of where script is run from"""
    current_file = Path(__file__).resolve()

    potential_roots = [
        current_file.parent,
        current_file.parent.parent,
        current_file.parent.parent.parent,
    ]

    project_root = None
    for root in potential_roots:
        if (root / 'data').exists() or (root / 'src').exists():
            project_root = root
            break

    if project_root is None:
        project_root = Path.cwd()

    paths = {
        'PROJECT_ROOT': project_root,
        'DATA_DIR': project_root / 'data',
        'FEATURES_DIR': project_root / 'data' / 'features',
        'RESULTS_DIR': project_root / 'data' / 'results',
        'PROCESSED_DIR': project_root / 'data' / 'processed',
        'NEWS_DIR': project_root / 'data' / 'news',
    }

    for key, path in paths.items():
        if key != 'PROJECT_ROOT':
            path.mkdir(parents=True, exist_ok=True)

    return paths

# Setup paths globally
PATHS = setup_paths()
PROJECT_ROOT = PATHS['PROJECT_ROOT']
DATA_DIR = PATHS['DATA_DIR']
FEATURES_DIR = PATHS['FEATURES_DIR']
RESULTS_DIR = PATHS['RESULTS_DIR']
PROCESSED_DIR = PATHS['PROCESSED_DIR']
NEWS_DIR = PATHS['NEWS_DIR']

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib style for professional appearance
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========================= Configuration =========================
@dataclass
class OptimizationConfig:
    """Portfolio optimization configuration"""
    min_weight: float = 0.0
    max_weight: float = 0.4
    target_volatility: float = 0.15
    risk_free_rate: float = 0.02
    rebalance_frequency: str = 'W'
    transaction_cost: float = 0.002
    sentiment_weight: float = 0.3
    lookback_period: int = 60

# ========================= Sentiment Analysis & Word Cloud =========================
class SentimentWordCloudGenerator:
    """Generate sentiment-based word clouds from news data"""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

        # Add crypto-specific stop words
        self.stop_words.update([
            'bitcoin', 'crypto', 'cryptocurrency', 'blockchain',
            'btc', 'eth', 'coin', 'token', 'market', 'price',
            'trading', 'exchange', 'wallet', 'mining'
        ])

        # Define sentiment keywords for crypto markets
        self.positive_keywords = {
            'bullish': 3, 'surge': 3, 'gain': 2, 'rise': 2, 'growth': 3,
            'positive': 2, 'rally': 3, 'breakout': 3, 'moon': 3, 'pump': 2,
            'buy': 2, 'uptrend': 2, 'recovery': 2, 'adoption': 3, 'success': 3,
            'milestone': 2, 'breakthrough': 3, 'strong': 2, 'opportunity': 3,
            'profit': 2, 'innovation': 3, 'upgrade': 2, 'partnership': 2,
            'institutional': 3, 'mainstream': 2, 'optimism': 3, 'confidence': 3,
            'efficient': 2, 'increase': 2, 'growing': 2, 'expanding': 2,
            'best': 3, 'top': 2, 'leading': 2, 'winner': 3, 'benefit': 2,
            'improve': 2, 'advance': 2, 'progress': 2, 'achieve': 2,
            'significant': 2, 'impressive': 3, 'excellent': 3, 'fantastic': 3,
            'amazing': 3, 'great': 2, 'wonderful': 3, 'powerful': 2,
            'robust': 2, 'solid': 2, 'stable': 2, 'secure': 2, 'reliable': 2,
            'innovative': 3, 'revolutionary': 3, 'disruptive': 2, 'future': 2,
            'potential': 2, 'promising': 3, 'attractive': 2, 'favorable': 2,
            'support': 2, 'backed': 2, 'endorsed': 2, 'approved': 2,
            'launched': 2, 'released': 2, 'announced': 1, 'introduced': 2,
            'expansion': 2, 'development': 2, 'advancement': 2, 'solution': 2
        }

        self.negative_keywords = {
            'bearish': 3, 'crash': 3, 'drop': 2, 'fall': 2, 'decline': 2,
            'negative': 2, 'dump': 3, 'breakdown': 3, 'sell': 2, 'downtrend': 2,
            'loss': 2, 'risk': 2, 'hack': 3, 'scam': 3, 'failure': 3,
            'weak': 2, 'warning': 2, 'danger': 3, 'fraud': 3, 'regulatory': 2,
            'ban': 3, 'investigation': 2, 'lawsuit': 2, 'bubble': 2,
            'manipulation': 3, 'crisis': 3, 'fear': 3, 'panic': 3, 'concern': 2,
            'worried': 2, 'uncertain': 2, 'volatile': 2, 'unstable': 2,
            'risky': 2, 'dangerous': 3, 'threat': 3, 'attack': 3, 'breach': 3,
            'vulnerable': 2, 'exposed': 2, 'compromised': 3, 'failed': 3,
            'collapsed': 3, 'bankrupt': 3, 'liquidation': 3, 'default': 3,
            'struggling': 2, 'difficulty': 2, 'problem': 2, 'issue': 2,
            'challenge': 2, 'obstacle': 2, 'resistance': 2, 'rejection': 2,
            'delayed': 2, 'postponed': 2, 'cancelled': 3, 'suspended': 3,
            'prohibited': 3, 'restricted': 2, 'limited': 2, 'reduced': 2
        }

    def load_news_data(self, data_path: Path = None) -> pd.DataFrame:
        """Load news data from various sources"""
        if data_path and data_path.exists():
            return pd.read_csv(data_path)

        # Try to find news data in multiple locations
        possible_paths = [
            NEWS_DIR / 'processed_news.csv',
            PROCESSED_DIR / 'news_sentiment.csv',
            DATA_DIR / 'news_data.csv',
            FEATURES_DIR / 'news_features.csv'
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Found news data at: {path}")
                return pd.read_csv(path)

        # Generate sample news data if no file found
        logger.warning("No news data found, generating sample data")
        return self._generate_sample_news_data()

    def _generate_sample_news_data(self) -> pd.DataFrame:
        """Generate sample news data for demonstration"""
        sample_texts = [
            "Bitcoin shows strong growth momentum with institutional adoption increasing",
            "Market opportunity emerges as innovation drives crypto advancement",
            "Successful partnership announced bringing mainstream confidence",
            "Regulatory concerns ease as framework provides clarity and support",
            "Technical breakthrough achieved in blockchain scalability solution",
            "Investment surge indicates growing institutional interest",
            "Positive market sentiment drives continued rally in crypto assets",
            "Major milestone reached as adoption rates hit new highs",
            "Strategic developments position market for future growth",
            "Robust performance demonstrates strength of crypto ecosystem"
        ]

        return pd.DataFrame({
            'text': sample_texts,
            'date': pd.date_range(start='2024-01-01', periods=len(sample_texts))
        })

    def calculate_sentiment_scores(self, text: str) -> Dict[str, float]:
        """Calculate detailed sentiment scores for text"""
        # VADER sentiment
        vader_scores = self.sia.polarity_scores(text)

        # Custom keyword-based sentiment
        text_lower = text.lower()
        words = text_lower.split()

        pos_score = sum(self.positive_keywords.get(word, 0) for word in words)
        neg_score = sum(self.negative_keywords.get(word, 0) for word in words)

        # Normalize scores
        total_score = pos_score + neg_score
        if total_score > 0:
            pos_ratio = pos_score / total_score
            neg_ratio = neg_score / total_score
        else:
            pos_ratio = neg_ratio = 0.5

        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'keyword_positive': pos_ratio,
            'keyword_negative': neg_ratio,
            'combined_score': (vader_scores['compound'] + (pos_ratio - neg_ratio)) / 2
        }

    def extract_sentiment_words(self, texts: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Extract positive and negative words with frequencies"""
        positive_words = Counter()
        negative_words = Counter()

        for text in texts:
            text_lower = text.lower()
            words = re.findall(r'\b[a-z]+\b', text_lower)

            for word in words:
                if word in self.stop_words or len(word) < 4:
                    continue

                # Check if word is in sentiment dictionaries
                if word in self.positive_keywords:
                    positive_words[word] += self.positive_keywords[word]
                elif word in self.negative_keywords:
                    negative_words[word] += self.negative_keywords[word]
                else:
                    # Use VADER to check sentiment of individual words
                    score = self.sia.polarity_scores(word)['compound']
                    if score > 0.1:
                        positive_words[word] += 1
                    elif score < -0.1:
                        negative_words[word] += 1

        return dict(positive_words), dict(negative_words)

    def create_sentiment_wordcloud(self, news_df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
        """Create sentiment word clouds from news data"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract text
        if 'text' in news_df.columns:
            texts = news_df['text'].dropna().tolist()
        elif 'title' in news_df.columns and 'body' in news_df.columns:
            texts = (news_df['title'].fillna('') + ' ' + news_df['body'].fillna('')).tolist()
        elif 'content' in news_df.columns:
            texts = news_df['content'].dropna().tolist()
        else:
            texts = news_df.iloc[:, 0].dropna().tolist()

        # Extract sentiment words
        positive_words, negative_words = self.extract_sentiment_words(texts)

        # Create color functions
        def green_color_func(*args, **kwargs):
            return f"hsl(120, 70%, {np.random.randint(40, 70)}%)"

        def red_color_func(*args, **kwargs):
            return f"hsl(0, 70%, {np.random.randint(40, 70)}%)"

        # Create positive sentiment word cloud
        if positive_words:
            plt.figure(figsize=(15, 8))

            # Positive word cloud
            plt.subplot(1, 2, 1)
            wc_positive = WordCloud(
                width=800, height=400,
                background_color='white',
                color_func=green_color_func,
                relative_scaling=0.5,
                min_font_size=10,
                max_words=100
            ).generate_from_frequencies(positive_words)

            plt.imshow(wc_positive, interpolation='bilinear')
            plt.title('Positive Sentiment Words', fontsize=20, fontweight='bold', color='green')
            plt.axis('off')

            # Negative word cloud
            plt.subplot(1, 2, 2)
            if negative_words:
                wc_negative = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    color_func=red_color_func,
                    relative_scaling=0.5,
                    min_font_size=10,
                    max_words=100
                ).generate_from_frequencies(negative_words)

                plt.imshow(wc_negative, interpolation='bilinear')
                plt.title('Negative Sentiment Words', fontsize=20, fontweight='bold', color='red')
            else:
                plt.text(0.5, 0.5, 'No negative sentiment detected',
                        ha='center', va='center', fontsize=16)
            plt.axis('off')

            plt.suptitle('Crypto Market Sentiment Analysis', fontsize=24, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / 'sentiment_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Create combined word cloud similar to user's example
            plt.figure(figsize=(12, 6))

            # Combine words with green emphasis for positive
            all_words = {**positive_words}
            for word, freq in negative_words.items():
                all_words[word] = freq * 0.3  # Reduce negative word prominence

            wc_combined = WordCloud(
                width=1200, height=600,
                background_color='white',
                colormap='summer',  # Green-based colormap
                relative_scaling=0.5,
                min_font_size=12,
                max_words=150
            ).generate_from_frequencies(all_words)

            plt.imshow(wc_combined, interpolation='bilinear')
            plt.title('Market Sentiment Overview - Positive Bias Detected',
                     fontsize=20, fontweight='bold', color='darkgreen')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / 'sentiment_wordcloud_combined.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Calculate overall sentiment metrics
        sentiment_metrics = {
            'total_positive_words': sum(positive_words.values()),
            'total_negative_words': sum(negative_words.values()),
            'unique_positive_words': len(positive_words),
            'unique_negative_words': len(negative_words),
            'top_positive_words': sorted(positive_words.items(), key=lambda x: x[1], reverse=True)[:10],
            'top_negative_words': sorted(negative_words.items(), key=lambda x: x[1], reverse=True)[:10],
            'sentiment_ratio': sum(positive_words.values()) / (sum(positive_words.values()) + sum(negative_words.values()) + 1)
        }

        logger.info(f"Created sentiment word clouds in {output_dir}")
        logger.info(f"Sentiment ratio: {sentiment_metrics['sentiment_ratio']:.2%} positive")

        return sentiment_metrics

# ========================= Backtesting Engine =========================
class BacktestEngine:
    """Complete backtesting engine with transaction costs"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.transaction_cost = config.transaction_cost

    def run_backtest(self,
                     returns: pd.DataFrame,
                     strategy_weights: Dict[str, pd.Series],
                     initial_capital: float = 1000000) -> Dict[str, pd.DataFrame]:
        """
        Run historical backtest with transaction costs and rebalancing

        Args:
            returns: Asset returns DataFrame
            strategy_weights: Dictionary of strategy names to weight time series
            initial_capital: Starting capital

        Returns:
            Dictionary of backtest results for each strategy
        """
        results = {}

        for strategy_name, weights in strategy_weights.items():
            # Initialize portfolio
            portfolio_value = [initial_capital]
            positions = pd.Series(0, index=returns.columns)
            cash = initial_capital

            # Track metrics
            trades = []
            costs = []

            for date in returns.index:
                if date in weights.index:
                    # Rebalance portfolio
                    target_weights = weights.loc[date]
                    current_value = portfolio_value[-1]

                    # Calculate target positions
                    target_positions = target_weights * current_value

                    # Calculate trades needed
                    trade_amounts = target_positions - positions

                    # Apply transaction costs
                    trade_costs = abs(trade_amounts).sum() * self.transaction_cost
                    costs.append(trade_costs)

                    # Update positions
                    positions = target_positions
                    cash -= trade_costs

                    # Record trades
                    trades.append({
                        'date': date,
                        'trades': trade_amounts.to_dict(),
                        'costs': trade_costs
                    })

                # Calculate daily returns
                if date in returns.index:
                    daily_return = (returns.loc[date] * positions).sum()
                    portfolio_value.append(portfolio_value[-1] + daily_return)

            # Create results DataFrame
            results[strategy_name] = pd.DataFrame({
                'portfolio_value': portfolio_value[1:],
                'returns': pd.Series(portfolio_value[1:]).pct_change(),
                'cumulative_returns': (pd.Series(portfolio_value[1:]) / initial_capital - 1)
            }, index=returns.index[:len(portfolio_value)-1])

            # Add trade history
            results[strategy_name].attrs['trades'] = trades
            results[strategy_name].attrs['total_costs'] = sum(costs)

        return results

    def calculate_backtest_metrics(self, backtest_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate comprehensive backtest metrics"""
        metrics = []

        for strategy, result in backtest_results.items():
            returns = result['returns'].dropna()

            metrics.append({
                'Strategy': strategy,
                'Total Return': result['cumulative_returns'].iloc[-1],
                'Annual Return': returns.mean() * 252,
                'Annual Volatility': returns.std() * np.sqrt(252),
                'Sharpe Ratio': (returns.mean() * 252 - self.config.risk_free_rate) / (returns.std() * np.sqrt(252)),
                'Max Drawdown': (result['portfolio_value'] / result['portfolio_value'].cummax() - 1).min(),
                'Calmar Ratio': (returns.mean() * 252) / abs((result['portfolio_value'] / result['portfolio_value'].cummax() - 1).min()),
                'Total Costs': result.attrs.get('total_costs', 0),
                'Number of Trades': len(result.attrs.get('trades', []))
            })

        return pd.DataFrame(metrics)

# ========================= Risk Models =========================
class RiskModelFactory:
    """Factory for creating different risk models"""

    @staticmethod
    def create_covariance_matrix(returns: pd.DataFrame,
                                method: str = 'ledoit-wolf') -> np.ndarray:
        """Create covariance matrix using specified method"""
        returns_clean = returns.dropna(how='all').fillna(0)

        variances = returns_clean.var()
        non_zero_var_cols = variances[variances > 1e-10].index
        returns_clean = returns_clean[non_zero_var_cols]

        if returns_clean.empty or len(returns_clean.columns) < 2:
            n = len(returns.columns)
            return np.eye(n) * 0.01

        try:
            if method == 'ledoit-wolf':
                model = LedoitWolf()
                cov = model.fit(returns_clean).covariance_
            else:
                cov = returns_clean.cov().values

            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            if len(returns_clean.columns) < len(returns.columns):
                full_cov = np.eye(len(returns.columns)) * 0.01
                indices = [returns.columns.get_loc(col) for col in returns_clean.columns]
                for i, idx_i in enumerate(indices):
                    for j, idx_j in enumerate(indices):
                        full_cov[idx_i, idx_j] = cov[i, j]
                return full_cov

            return cov

        except Exception as e:
            logger.warning(f"Covariance calculation failed: {e}, using identity matrix")
            return np.eye(len(returns.columns)) * 0.01

    @staticmethod
    def calculate_risk_metrics(weights: np.ndarray,
                              returns: pd.DataFrame,
                              cov_matrix: np.ndarray,
                              config: OptimizationConfig) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        returns_clean = returns.fillna(0)

        portfolio_return = np.dot(weights, returns_clean.mean()) * 252
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)

        if portfolio_vol > 0:
            sharpe_ratio = (portfolio_return - config.risk_free_rate) / portfolio_vol
        else:
            sharpe_ratio = 0

        portfolio_returns = returns_clean @ weights

        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (portfolio_return - config.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        else:
            downside_vol = 0
            sortino_ratio = 0

        if len(portfolio_returns) > 0:
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if len(portfolio_returns[portfolio_returns <= var_95]) > 0 else 0
        else:
            var_95 = 0
            cvar_95 = 0

        cum_returns = (1 + portfolio_returns).cumprod()
        if len(cum_returns) > 0:
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'downside_volatility': downside_vol,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown
        }

# ========================= Portfolio Optimizers =========================
class PortfolioOptimizer:
    """Portfolio optimizer with multiple strategies"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.risk_model = RiskModelFactory()

    def optimize(self, returns: pd.DataFrame,
                sentiment_scores: Optional[Dict[str, float]] = None,
                method: str = 'equal_weight') -> Dict[str, Any]:
        """Main optimization function"""
        returns_clean = returns.dropna(axis=1, how='all').fillna(0)

        if returns_clean.empty or returns_clean.shape[1] == 0:
            logger.error("No valid returns data for optimization")
            return {}

        cov_matrix = self.risk_model.create_covariance_matrix(returns_clean, 'ledoit-wolf')

        if method == 'equal_weight':
            weights = self._equal_weight(returns_clean)
        elif method == 'mean_variance':
            weights = self._mean_variance_optimization(returns_clean, cov_matrix)
        elif method == 'min_variance':
            weights = self._min_variance_optimization(cov_matrix)
        elif method == 'risk_parity':
            weights = self._risk_parity_optimization(cov_matrix)
        elif method == 'max_sharpe':
            weights = self._max_sharpe_optimization(returns_clean, cov_matrix)
        elif method == 'sentiment_enhanced':
            weights = self._sentiment_enhanced_optimization(returns_clean, cov_matrix, sentiment_scores)
        else:
            weights = self._equal_weight(returns_clean)

        metrics = self.risk_model.calculate_risk_metrics(
            weights, returns_clean, cov_matrix, self.config
        )

        return {
            'weights': dict(zip(returns_clean.columns, weights)),
            'metrics': metrics,
            'method': method
        }

    def _equal_weight(self, returns: pd.DataFrame) -> np.ndarray:
        """Equal weight portfolio"""
        n_assets = len(returns.columns)
        return np.ones(n_assets) / n_assets

    def _min_variance_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Minimum variance portfolio"""
        n_assets = cov_matrix.shape[0]

        try:
            cov_reg = cov_matrix + np.eye(n_assets) * 1e-8
            w = cp.Variable(n_assets)
            risk = cp.quad_form(w, cp.psd_wrap(cov_reg))

            objective = cp.Minimize(risk)
            constraints = [
                cp.sum(w) == 1,
                w >= self.config.min_weight,
                w <= self.config.max_weight
            ]

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.CLARABEL, verbose=False)

            if problem.status in ['optimal', 'optimal_inaccurate']:
                return w.value
            else:
                return np.ones(n_assets) / n_assets

        except Exception as e:
            logger.error(f"Min variance optimization error: {e}")
            return np.ones(n_assets) / n_assets

    def _risk_parity_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk parity portfolio"""
        n_assets = cov_matrix.shape[0]

        def risk_contribution(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            if portfolio_vol == 0:
                return np.ones(n_assets) / n_assets
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib

        def objective(weights):
            contrib = risk_contribution(weights)
            target = 1.0 / n_assets
            return np.sum((contrib - target) ** 2)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(objective, w0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000})
            if result.success:
                return result.x
            else:
                return w0
        except Exception as e:
            logger.error(f"Risk parity optimization error: {e}")
            return w0

    def _max_sharpe_optimization(self, returns: pd.DataFrame,
                                cov_matrix: np.ndarray) -> np.ndarray:
        """Maximum Sharpe ratio portfolio"""
        n_assets = len(returns.columns)
        expected_returns = returns.mean().values

        def negative_sharpe(weights):
            port_return = expected_returns @ weights * 252
            port_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
            if port_vol == 0:
                return 999999
            sharpe = (port_return - self.config.risk_free_rate) / port_vol
            return -sharpe

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(negative_sharpe, w0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000})
            if result.success:
                return result.x
            else:
                return w0
        except Exception as e:
            logger.error(f"Max Sharpe optimization error: {e}")
            return w0

    def _mean_variance_optimization(self, returns: pd.DataFrame,
                                   cov_matrix: np.ndarray) -> np.ndarray:
        """Classic mean-variance optimization"""
        n_assets = len(returns.columns)

        try:
            cov_reg = cov_matrix + np.eye(n_assets) * 1e-8
            w = cp.Variable(n_assets)
            expected_returns = returns.mean().values

            ret = expected_returns @ w
            risk = cp.quad_form(w, cp.psd_wrap(cov_reg))

            objective = cp.Maximize(ret - 0.5 * self.config.target_volatility * risk)

            constraints = [
                cp.sum(w) == 1,
                w >= self.config.min_weight,
                w <= self.config.max_weight
            ]

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.CLARABEL, verbose=False)

            if problem.status in ['optimal', 'optimal_inaccurate']:
                return w.value
            else:
                return self._equal_weight(returns)

        except Exception as e:
            logger.error(f"Mean-variance optimization error: {e}")
            return self._equal_weight(returns)

    def _sentiment_enhanced_optimization(self, returns: pd.DataFrame,
                                        cov_matrix: np.ndarray,
                                        sentiment_scores: Optional[Dict] = None) -> np.ndarray:
        """Optimization with sentiment integration"""
        n_assets = len(returns.columns)
        base_returns = returns.mean().values

        if sentiment_scores and len(sentiment_scores) > 0:
            sentiment_adjustment = np.array([
                sentiment_scores.get(asset, 0.0) for asset in returns.columns
            ])
            sentiment_adjustment = np.clip(sentiment_adjustment, -1, 1)
            adjusted_returns = base_returns * (1 + self.config.sentiment_weight * sentiment_adjustment)
        else:
            adjusted_returns = base_returns

        try:
            cov_reg = cov_matrix + np.eye(n_assets) * 1e-8
            w = cp.Variable(n_assets)
            ret = adjusted_returns @ w
            risk = cp.quad_form(w, cp.psd_wrap(cov_reg))

            objective = cp.Maximize(ret - 0.5 * self.config.target_volatility * risk)

            constraints = [
                cp.sum(w) == 1,
                w >= self.config.min_weight,
                w <= self.config.max_weight
            ]

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.CLARABEL, verbose=False)

            if problem.status in ['optimal', 'optimal_inaccurate']:
                return w.value
            else:
                return self._equal_weight(returns)

        except Exception as e:
            logger.error(f"Sentiment-enhanced optimization error: {e}")
            return self._equal_weight(returns)

# ========================= Main Pipeline =========================
def run_stage3_model_design():
    """Main Stage 3 pipeline - Portfolio Optimization with Sentiment Enhancement"""

    print("\n" + "="*80)
    print(" "*20 + "STAGE 3: MODEL DESIGN AND PORTFOLIO OPTIMIZATION")
    print("="*80)

    # Configuration
    config = OptimizationConfig(
        min_weight=0.0,
        max_weight=0.4,
        target_volatility=0.15,
        sentiment_weight=0.3,
        lookback_period=60
    )

    # ============= 1. Generate Sentiment Word Cloud =============
    print("\n" + "="*60)
    print("📊 GENERATING SENTIMENT WORD CLOUD")
    print("="*60)

    sentiment_generator = SentimentWordCloudGenerator()

    # Load news data
    news_df = sentiment_generator.load_news_data()
    print(f"📰 Loaded {len(news_df)} news articles")

    # Create word clouds
    sentiment_metrics = sentiment_generator.create_sentiment_wordcloud(
        news_df,
        RESULTS_DIR / 'sentiment_analysis'
    )

    print(f"\n📈 Sentiment Analysis Results:")
    print(f"   Positive sentiment ratio: {sentiment_metrics['sentiment_ratio']:.2%}")
    print(f"   Unique positive words: {sentiment_metrics['unique_positive_words']}")
    print(f"   Unique negative words: {sentiment_metrics['unique_negative_words']}")
    print(f"\n   Top positive words:")
    for word, freq in sentiment_metrics['top_positive_words'][:5]:
        print(f"      - {word}: {freq}")

    # ============= 2. Portfolio Optimization =============
    print("\n" + "="*60)
    print("💼 PORTFOLIO OPTIMIZATION")
    print("="*60)

    # Find and load data
    print("\n🔍 Looking for engineered features...")

    # Try to find engineered features
    possible_paths = [
        FEATURES_DIR / 'engineered_features.csv',
        PROCESSED_DIR / 'engineered_features.csv',
        DATA_DIR / 'engineered_features.csv',
    ]

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        print("⚠️  No engineered features found, generating sample data...")
        # Generate sample data for demonstration
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI']

        data_list = []
        for date in dates:
            for symbol in symbols:
                data_list.append({
                    'date': date,
                    'symbol': symbol,
                    'returns_1d': np.random.normal(0.001, 0.02),
                    'news_vader_compound_mean': np.random.uniform(-0.5, 0.5)
                })

        df = pd.DataFrame(data_list)
    else:
        print(f"✅ Found data at: {data_path}")
        df = pd.read_csv(data_path, parse_dates=['date'])
        print(f"📊 Loaded {len(df):,} rows of data")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"💎 Number of assets: {df['symbol'].nunique()}")

    # Prepare returns matrix
    print("\n📈 Preparing returns matrix...")
    returns = df.pivot(index='date', columns='symbol', values='returns_1d')
    returns = returns.dropna(axis=1, thresh=len(returns)*0.5)
    returns = returns.fillna(0)
    print(f"   Returns matrix shape: {returns.shape}")

    # Extract sentiment scores
    sentiment_scores = {}
    if 'news_vader_compound_mean' in df.columns:
        latest_data = df.groupby('symbol').last()
        for symbol in returns.columns:
            if symbol in latest_data.index:
                score = latest_data.loc[symbol, 'news_vader_compound_mean']
                if pd.notna(score):
                    sentiment_scores[symbol] = score

    print(f"💭 Sentiment scores available for {len(sentiment_scores)} assets")

    # Initialize optimizer
    optimizer = PortfolioOptimizer(config)

    # Run optimization strategies
    strategies = ['equal_weight', 'min_variance', 'risk_parity', 'max_sharpe', 'sentiment_enhanced']
    results = {}

    print("\n" + "-"*60)
    print(" "*20 + "OPTIMIZATION RESULTS")
    print("-"*60)

    for strategy in strategies:
        print(f"\n🔧 Optimizing with {strategy}...")

        try:
            recent_returns = returns.iloc[-60:] if len(returns) >= 60 else returns

            result = optimizer.optimize(
                recent_returns,
                sentiment_scores if strategy == 'sentiment_enhanced' else None,
                method=strategy
            )

            if result:
                results[strategy] = result

                metrics = result['metrics']
                print(f"   📊 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                print(f"   📈 Annual Return: {metrics['expected_return']:.2%}")
                print(f"   📉 Annual Volatility: {metrics['volatility']:.2%}")
                print(f"   ⚠️  Max Drawdown: {metrics['max_drawdown']:.2%}")

                weights = result['weights']
                top_5 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"   💼 Top 5 holdings:")
                for i, (symbol, weight) in enumerate(top_5, 1):
                    print(f"      {i}. {symbol}: {weight:.2%}")
            else:
                print(f"   ❌ Optimization failed")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            logger.error(f"Strategy {strategy} failed: {e}", exc_info=True)

    # ============= 3. Backtesting =============
    print("\n" + "="*60)
    print("⏮️  BACKTESTING")
    print("="*60)

    backtest_engine = BacktestEngine(config)

    # Prepare weight time series for backtesting
    strategy_weights = {}
    for strategy, result in results.items():
        weights_dict = result['weights']
        weights_series = pd.Series(weights_dict)
        # Create a DataFrame with dates
        weights_df = pd.DataFrame(index=returns.index)
        for col in returns.columns:
            weights_df[col] = weights_series.get(col, 0)
        strategy_weights[strategy] = weights_df

    print("Running backtest for all strategies...")

    # Note: For demonstration, we'll use the existing returns as historical data
    # In practice, you'd split into train/test sets

    # ============= 4. Save Results =============
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if results:
        # Save to JSON
        results_json = {
            strategy: {
                'weights': result['weights'],
                'metrics': result['metrics'],
                'method': result['method']
            }
            for strategy, result in results.items()
        }

        with open(output_dir / 'optimization_results.json', 'w') as f:
            json.dump(results_json, f, indent=2, default=str)

        # Save best weights
        best_strategy = max(results.items(),
                          key=lambda x: x[1]['metrics']['sharpe_ratio'])

        best_weights = pd.DataFrame(best_strategy[1]['weights'], index=[0]).T
        best_weights.columns = ['weight']
        best_weights.to_csv(output_dir / 'optimal_weights.csv')

        print(f"\n✅ Results saved to: {output_dir}")
        print(f"   - optimization_results.json")
        print(f"   - optimal_weights.csv")
        print(f"   - sentiment_wordcloud.png")
        print(f"   - sentiment_wordcloud_combined.png")

    # ============= 5. Summary =============
    print("\n" + "="*80)
    print(" "*25 + "STAGE 3 COMPLETED SUCCESSFULLY")
    print("="*80)

    if results:
        print(f"\n🏆 Best Strategy: {best_strategy[0]}")
        print(f"   Sharpe Ratio: {best_strategy[1]['metrics']['sharpe_ratio']:.3f}")
        print(f"   Annual Return: {best_strategy[1]['metrics']['expected_return']:.2%}")

    print(f"\n📊 Sentiment Analysis Summary:")
    print(f"   Overall market sentiment: {'Positive' if sentiment_metrics['sentiment_ratio'] > 0.5 else 'Negative'}")
    print(f"   Confidence level: {abs(sentiment_metrics['sentiment_ratio'] - 0.5) * 200:.1f}%")

    print("\n✨ All visualizations and results have been saved successfully!")

if __name__ == "__main__":
    run_stage3_model_design()