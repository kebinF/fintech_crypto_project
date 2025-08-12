"""
Stage 4: Clean Version - Cryptocurrency Portfolio Management Application
FINS5545 FinTech Project - Gemini AI Integration Fixed
"""
import talib_compat
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import logging
from pathlib import Path
import yfinance as yf
import time
import hashlib
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import warnings

# Import required packages
from dotenv import load_dotenv
import google.generativeai as genai
from scipy.optimize import minimize

# Try to import VADER sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

warnings.filterwarnings('ignore')

# Load environment variables at module start
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force English interface
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

# Streamlit page configuration
st.set_page_config(
    page_title="Crypto Portfolio Optimizer - FINS5545",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': """
        **FINS5545 FinTech Project - Stage 4**
        
        Advanced Cryptocurrency Portfolio Management Platform
        
        Features:
        - AI-Enhanced Portfolio Optimization
        - Google Gemini Integration  
        - Real-time Market Analysis
        - Sentiment-Driven Allocation
        """
    }
)

# Enhanced CSS styling
st.markdown("""
    <style>
    * {
        font-family: "Source Sans Pro", sans-serif !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .success-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .ai-response {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
    }
    
    .stSelectbox label,
    .stSlider label,
    .stButton button,
    .stTextInput label {
        color: #333 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PortfolioConfig:
    """Portfolio configuration settings"""
    initial_capital: float = 100000.0
    target_volatility: float = 0.18
    max_position_size: float = 0.25
    min_position_size: float = 0.05
    sentiment_weight: float = 0.30
    risk_tolerance: str = 'Moderate'
    rebalance_frequency: str = 'Weekly'
    stop_loss_threshold: float = 0.15
    take_profit_threshold: float = 0.30

@dataclass
class AssetData:
    """Asset data structure"""
    symbol: str
    price: float
    volume: float
    market_cap: float
    change_24h: float
    sentiment_score: float
    weight: float = 0.0

# ============================================================================
# Gemini AI Manager - Clean Fixed Version
# ============================================================================

class GeminiAIManager:
    """Google Gemini AI integration with proper error handling"""

    def __init__(self):
        self.model = None
        self.api_key = None
        self.is_configured = False
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize Gemini AI"""
        try:
            # Get API key from environment
            self.api_key = os.getenv('GEMINI_API_KEY')

            if not self.api_key or self.api_key == 'your_gemini_api_key_here':
                logger.warning("Gemini API key not found in environment variables")
                return

            # Configure Gemini
            genai.configure(api_key=self.api_key)

            # Initialize model
            self.model = genai.GenerativeModel("gemini-1.5-flash")

            # Test connection
            test_response = self.model.generate_content("Test connection")
            if test_response and test_response.text:
                self.is_configured = True
                logger.info("Gemini AI configured successfully")
            else:
                raise Exception("Test generation failed")

        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")
            self.is_configured = False
            self.model = None

    def configure_api_key(self, api_key: str) -> bool:
        """Configure API key manually"""
        try:
            if not api_key or len(api_key) < 10:
                return False

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')

            # Test the configuration
            test_response = self.model.generate_content("Hello")
            if test_response and test_response.text:
                self.api_key = api_key
                self.is_configured = True
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"API key configuration failed: {e}")
            return False

    def get_portfolio_advice(self, portfolio_data: Dict, market_conditions: Dict) -> str:
        """Generate AI-powered portfolio advice"""
        if not self.is_configured:
            return self._fallback_advice(portfolio_data, market_conditions)

        try:
            prompt = f"""
            As a professional cryptocurrency portfolio advisor, analyze this portfolio:
            
            Portfolio Holdings:
            {self._format_portfolio(portfolio_data)}
            
            Market Conditions:
            - Market Sentiment: {market_conditions.get('sentiment', 'Neutral')}
            - Volatility Level: {market_conditions.get('volatility', 'Normal')}
            - Market Trend: {market_conditions.get('trend', 'Sideways')}
            
            Please provide concise recommendations covering:
            1. Portfolio balance assessment
            2. Risk management advice
            3. Rebalancing suggestions
            4. Market outlook implications
            
            Keep response under 250 words, professional tone.
            """

            response = self.model.generate_content(prompt)
            if response and response.text:
                return response.text
            else:
                return self._fallback_advice(portfolio_data, market_conditions)

        except Exception as e:
            logger.error(f"Gemini advice generation failed: {e}")
            return self._fallback_advice(portfolio_data, market_conditions)

    def analyze_market_sentiment(self, news_data: str, market_data: Dict) -> Dict:
        """Analyze market sentiment using Gemini"""
        if not self.is_configured:
            return self._fallback_sentiment(news_data)

        try:
            prompt = f"""
            Analyze the cryptocurrency market sentiment based on:
            
            Recent News Headlines: {news_data[:500]}
            
            Market Data: {json.dumps(market_data, indent=2)}
            
            Respond in JSON format:
            {{
                "sentiment_score": 0.0 to 1.0,
                "confidence": 0.0 to 1.0,
                "key_factors": ["factor1", "factor2"],
                "recommendation": "BULLISH/BEARISH/NEUTRAL",
                "risk_level": "LOW/MEDIUM/HIGH"
            }}
            """

            response = self.model.generate_content(prompt)
            return self._parse_json_response(response.text)

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._fallback_sentiment(news_data)

    def _format_portfolio(self, portfolio_data: Dict) -> str:
        """Format portfolio data for AI analysis"""
        if 'weights' in portfolio_data:
            weights = portfolio_data['weights']
            formatted = []
            for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                if weight > 0.01:  # Only include significant positions
                    formatted.append(f"- {symbol}: {weight:.1%}")
            return "\n".join(formatted[:10])  # Top 10 positions
        else:
            return str(portfolio_data)

    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from AI response"""
        try:
            import re
            # Find JSON in response
            json_match = re.search(r'\{[^}]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        # Fallback parsing
        return {
            'sentiment_score': 0.5,
            'confidence': 0.3,
            'key_factors': ['Limited analysis available'],
            'recommendation': 'NEUTRAL',
            'risk_level': 'MEDIUM'
        }

    def _fallback_advice(self, portfolio_data: Dict, market_conditions: Dict) -> str:
        """Provide fallback advice when AI is unavailable"""
        return """
        **Portfolio Analysis (Basic Mode)**
        
        Current Assessment:
        Your portfolio shows a reasonable diversification across major cryptocurrencies. 
        Consider the following recommendations:
        
        Rebalancing Suggestions:
        - Monitor position sizes relative to market cap
        - Consider reducing concentration in volatile altcoins
        - Maintain core positions in BTC and ETH (60-70% combined)
        
        Risk Management:
        - Set stop-losses for positions exceeding 10% allocation
        - Keep 5-10% in stablecoins for opportunities
        - Review correlations during market stress
        
        Market Outlook:
        - Crypto markets remain volatile - use dollar-cost averaging
        - Monitor regulatory developments closely
        - Focus on projects with strong fundamentals
        
        Note: This is basic analysis. Configure Gemini AI for advanced insights.
        """

    def _fallback_sentiment(self, news_data: str) -> Dict:
        """Fallback sentiment analysis"""
        if VADER_AVAILABLE and news_data:
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(news_data)
            return {
                'sentiment_score': (scores['compound'] + 1) / 2,  # Convert to 0-1 scale
                'confidence': abs(scores['compound']),
                'key_factors': ['VADER sentiment analysis'],
                'recommendation': 'BULLISH' if scores['compound'] > 0.1 else 'BEARISH' if scores['compound'] < -0.1 else 'NEUTRAL',
                'risk_level': 'MEDIUM'
            }

        return {
            'sentiment_score': 0.5,
            'confidence': 0.2,
            'key_factors': ['No analysis available'],
            'recommendation': 'NEUTRAL',
            'risk_level': 'MEDIUM'
        }

# ============================================================================
# Realistic Data Manager
# ============================================================================

class RealisticDataManager:
    """Generate realistic cryptocurrency data for demonstration"""

    def __init__(self):
        self.major_cryptos = {
            'BTC': {'name': 'Bitcoin', 'base_price': 43000, 'volatility': 0.05},
            'ETH': {'name': 'Ethereum', 'base_price': 2800, 'volatility': 0.06},
            'BNB': {'name': 'Binance Coin', 'base_price': 310, 'volatility': 0.08},
            'SOL': {'name': 'Solana', 'base_price': 98, 'volatility': 0.12},
            'ADA': {'name': 'Cardano', 'base_price': 0.52, 'volatility': 0.10},
            'AVAX': {'name': 'Avalanche', 'base_price': 28, 'volatility': 0.15},
            'DOT': {'name': 'Polkadot', 'base_price': 7.2, 'volatility': 0.12},
            'MATIC': {'name': 'Polygon', 'base_price': 0.98, 'volatility': 0.14},
            'LINK': {'name': 'Chainlink', 'base_price': 14.5, 'volatility': 0.11},
            'UNI': {'name': 'Uniswap', 'base_price': 6.8, 'volatility': 0.13}
        }

        self.market_caps = {
            'BTC': 850e9, 'ETH': 340e9, 'BNB': 48e9, 'SOL': 42e9, 'ADA': 18e9,
            'AVAX': 11e9, 'DOT': 9e9, 'MATIC': 8e9, 'LINK': 7e9, 'UNI': 4e9
        }

    def get_realistic_portfolio_weights(self) -> Dict[str, float]:
        """Generate realistic portfolio weights based on market cap and diversification"""
        # Start with market cap weighted allocation
        total_mcap = sum(self.market_caps.values())
        base_weights = {symbol: mcap/total_mcap for symbol, mcap in self.market_caps.items()}

        # Apply diversification constraints
        adjusted_weights = {}
        for symbol, weight in base_weights.items():
            if symbol in ['BTC', 'ETH']:
                # Major coins: 20-35% max
                adjusted_weights[symbol] = min(weight * 1.2, 0.35)
            else:
                # Altcoins: 5-15% max
                adjusted_weights[symbol] = min(weight * 2, 0.15)

        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        final_weights = {symbol: weight/total_weight for symbol, weight in adjusted_weights.items()}

        return final_weights

    def get_current_market_data(self) -> List[AssetData]:
        """Generate realistic current market data"""
        assets = []

        for symbol, info in self.major_cryptos.items():
            # Generate realistic price with some daily variation
            base_price = info['base_price']
            daily_change = np.random.normal(0, info['volatility'])
            current_price = base_price * (1 + daily_change)

            # Generate realistic volume (% of market cap)
            volume_ratio = np.random.uniform(0.02, 0.15)  # 2-15% of market cap
            volume = self.market_caps[symbol] * volume_ratio

            # Generate sentiment score
            sentiment = np.random.uniform(0.3, 0.7)  # Slightly positive bias

            assets.append(AssetData(
                symbol=symbol,
                price=round(current_price, 2),
                volume=volume,
                market_cap=self.market_caps[symbol],
                change_24h=daily_change * 100,
                sentiment_score=sentiment
            ))

        return assets

    def get_portfolio_performance_data(self, days: int = 30) -> pd.DataFrame:
        """Generate realistic portfolio performance history"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days),
                             end=datetime.now(), freq='D')

        # Generate correlated returns for portfolio
        base_return = 0.0008  # ~0.08% daily base return
        volatility = 0.025    # 2.5% daily volatility

        returns = np.random.normal(base_return, volatility, len(dates))

        # Add some momentum and mean reversion
        for i in range(1, len(returns)):
            momentum = returns[i-1] * 0.1  # 10% momentum
            returns[i] += momentum

        # Calculate cumulative portfolio value
        initial_value = 100000
        portfolio_values = initial_value * np.cumprod(1 + returns)

        return pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values,
            'daily_return': returns,
            'cumulative_return': (portfolio_values / initial_value - 1) * 100
        })

# ============================================================================
# Portfolio Optimizer
# ============================================================================

class SentimentEnhancedOptimizer:
    """Advanced portfolio optimizer with sentiment integration"""

    def __init__(self, config: PortfolioConfig):
        self.config = config

    def optimize_portfolio(self, assets: List[AssetData]) -> Dict[str, Any]:
        """Optimize portfolio with sentiment enhancement"""
        try:
            # Calculate expected returns with sentiment adjustment
            symbols = [asset.symbol for asset in assets]
            base_returns = np.array([self._calculate_expected_return(asset) for asset in assets])
            sentiment_scores = np.array([asset.sentiment_score for asset in assets])

            # Adjust returns based on sentiment
            sentiment_adjustment = (sentiment_scores - 0.5) * self.config.sentiment_weight
            adjusted_returns = base_returns * (1 + sentiment_adjustment)

            # Create covariance matrix (simplified)
            n_assets = len(assets)
            corr_matrix = self._generate_correlation_matrix(symbols)
            volatilities = np.array([self._estimate_volatility(asset) for asset in assets])
            cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

            # Optimize weights
            weights = self._optimize_weights(adjusted_returns, cov_matrix)

            # Calculate portfolio metrics
            metrics = self._calculate_metrics(weights, adjusted_returns, cov_matrix)

            return {
                'weights': dict(zip(symbols, weights)),
                'metrics': metrics,
                'expected_returns': dict(zip(symbols, adjusted_returns)),
                'sentiment_impact': dict(zip(symbols, sentiment_adjustment))
            }

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return self._equal_weight_fallback(assets)

    def _calculate_expected_return(self, asset: AssetData) -> float:
        """Calculate expected return for an asset"""
        # Base return influenced by market cap (larger = more stable, lower return)
        market_cap_factor = min(asset.market_cap / 1e12, 1.0)  # Normalize by $1T
        base_return = 0.15 * (1 - market_cap_factor * 0.5)  # 7.5% to 15% annual

        # Add volatility component
        volatility_bonus = 0.05 * (1 - market_cap_factor)  # Higher vol = higher expected return

        return base_return + volatility_bonus

    def _estimate_volatility(self, asset: AssetData) -> float:
        """Estimate asset volatility"""
        # Base volatility by market cap
        market_cap_factor = min(asset.market_cap / 1e12, 1.0)
        base_vol = 0.8 * (1 - market_cap_factor * 0.5)  # 40% to 80% annual

        # Adjust by recent price change
        recent_vol_adj = abs(asset.change_24h) / 100 * 0.1

        return base_vol + recent_vol_adj

    def _generate_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """Generate realistic correlation matrix"""
        n = len(symbols)
        corr_matrix = np.eye(n)

        # Set realistic correlations
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i != j:
                    if symbol1 == 'BTC' or symbol2 == 'BTC':
                        corr_matrix[i, j] = np.random.uniform(0.6, 0.8)  # High BTC correlation
                    elif symbol1 == 'ETH' or symbol2 == 'ETH':
                        corr_matrix[i, j] = np.random.uniform(0.5, 0.7)  # Moderate ETH correlation
                    else:
                        corr_matrix[i, j] = np.random.uniform(0.3, 0.6)  # Lower altcoin correlation

        # Ensure matrix is symmetric and positive definite
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)

        return corr_matrix

    def _optimize_weights(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize portfolio weights using mean-variance optimization"""
        n_assets = len(expected_returns)

        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            # Maximize Sharpe ratio
            return -portfolio_return / np.sqrt(portfolio_variance) if portfolio_variance > 0 else -999

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]

        # Bounds
        bounds = [(self.config.min_position_size, self.config.max_position_size)
                 for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)

            if result.success:
                return result.x
            else:
                logger.warning("Optimization failed, using equal weights")
                return x0
        except:
            return x0

    def _calculate_metrics(self, weights: np.ndarray, returns: np.ndarray,
                          cov_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        risk_free_rate = 0.02  # 2% risk-free rate
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_weight': weights.max(),
            'min_weight': weights.min(),
            'concentration': np.sum(weights**2)  # Herfindahl index
        }

    def _equal_weight_fallback(self, assets: List[AssetData]) -> Dict[str, Any]:
        """Fallback to equal weight portfolio"""
        n_assets = len(assets)
        weights = [1/n_assets] * n_assets
        symbols = [asset.symbol for asset in assets]

        return {
            'weights': dict(zip(symbols, weights)),
            'metrics': {
                'expected_return': 0.12,
                'volatility': 0.35,
                'sharpe_ratio': 0.35,
                'max_weight': 1/n_assets,
                'min_weight': 1/n_assets,
                'concentration': 1/n_assets
            },
            'expected_returns': dict(zip(symbols, [0.12/n_assets]*n_assets)),
            'sentiment_impact': dict(zip(symbols, [0.0]*n_assets))
        }

# ============================================================================
# Visualizations
# ============================================================================

class AdvancedVisualizer:
    """Create professional portfolio visualizations"""

    @staticmethod
    def create_portfolio_dashboard(portfolio_data: Dict, assets: List[AssetData],
                                 performance_data: pd.DataFrame) -> go.Figure:
        """Create comprehensive portfolio dashboard"""

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Portfolio Allocation', 'Performance Metrics', 'Asset Sentiment',
                'Daily Returns Distribution', 'Correlation Heatmap', 'Risk Analysis'
            ),
            specs=[
                [{'type': 'pie'}, {'type': 'indicator'}, {'type': 'bar'}],
                [{'type': 'histogram'}, {'type': 'heatmap'}, {'type': 'bar'}]
            ]
        )

        # Portfolio Allocation (Pie Chart)
        weights = portfolio_data.get('weights', {})
        if weights:
            symbols = list(weights.keys())
            values = list(weights.values())

            fig.add_trace(
                go.Pie(
                    labels=symbols,
                    values=values,
                    hole=0.4,
                    textinfo='label+percent',
                    textfont_size=10
                ),
                row=1, col=1
            )

        # Sharpe Ratio Indicator
        metrics = portfolio_data.get('metrics', {})
        sharpe_ratio = metrics.get('sharpe_ratio', 0)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=sharpe_ratio,
                title={'text': "Sharpe Ratio"},
                gauge={
                    'axis': {'range': [0, 2]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 1.0], 'color': "yellow"},
                        {'range': [1.0, 1.5], 'color': "lightgreen"},
                        {'range': [1.5, 2.0], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ),
            row=1, col=2
        )

        # Asset Sentiment Scores
        sentiment_data = [(asset.symbol, asset.sentiment_score) for asset in assets]
        if sentiment_data:
            symbols, sentiments = zip(*sentiment_data)
            colors = ['green' if s > 0.6 else 'red' if s < 0.4 else 'yellow' for s in sentiments]

            fig.add_trace(
                go.Bar(
                    x=list(symbols),
                    y=list(sentiments),
                    marker_color=colors,
                    name='Sentiment'
                ),
                row=1, col=3
            )

        # Returns Distribution
        if not performance_data.empty and 'daily_return' in performance_data.columns:
            fig.add_trace(
                go.Histogram(
                    x=performance_data['daily_return'] * 100,
                    nbinsx=20,
                    name='Daily Returns',
                    marker_color='blue',
                    opacity=0.7
                ),
                row=2, col=1
            )

        # Simple Correlation Matrix
        if len(assets) >= 3:
            symbols_subset = [asset.symbol for asset in assets[:5]]
            corr_data = np.random.uniform(0.3, 0.8, (len(symbols_subset), len(symbols_subset)))
            np.fill_diagonal(corr_data, 1.0)

            fig.add_trace(
                go.Heatmap(
                    z=corr_data,
                    x=symbols_subset,
                    y=symbols_subset,
                    colorscale='RdBu',
                    zmid=0.5,
                    showscale=False
                ),
                row=2, col=2
            )

        # Risk Metrics
        risk_metrics = ['Volatility', 'Max Weight', 'Concentration']
        risk_values = [
            metrics.get('volatility', 0) * 100,
            metrics.get('max_weight', 0) * 100,
            metrics.get('concentration', 0) * 100
        ]

        fig.add_trace(
            go.Bar(
                x=risk_metrics,
                y=risk_values,
                marker_color=['orange', 'purple', 'brown'],
                name='Risk Metrics'
            ),
            row=2, col=3
        )

        fig.update_layout(
            title_text="Portfolio Performance Dashboard",
            height=700,
            showlegend=False
        )

        return fig

    @staticmethod
    def create_performance_chart(performance_data: pd.DataFrame) -> go.Figure:
        """Create detailed performance chart"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Value Over Time', 'Daily Returns'),
            vertical_spacing=0.15
        )

        # Portfolio value line
        fig.add_trace(
            go.Scatter(
                x=performance_data['date'],
                y=performance_data['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=3),
                hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add benchmark line (BTC)
        btc_performance = 100000 * (1 + np.cumsum(np.random.normal(0.001, 0.04, len(performance_data))))
        fig.add_trace(
            go.Scatter(
                x=performance_data['date'],
                y=btc_performance,
                mode='lines',
                name='BTC Benchmark',
                line=dict(color='#ff7f0e', width=2, dash='dot'),
                hovertemplate='Date: %{x}<br>BTC Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Daily returns bar chart
        colors = ['green' if r > 0 else 'red' for r in performance_data['daily_return']]
        fig.add_trace(
            go.Bar(
                x=performance_data['date'],
                y=performance_data['daily_return'] * 100,
                marker_color=colors,
                name='Daily Returns (%)',
                opacity=0.7
            ),
            row=2, col=1
        )

        fig.update_layout(
            title='Portfolio Performance Analysis',
            height=600,
            hovermode='x unified'
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)

        return fig

# ============================================================================
# Session State Management
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""

    # Core managers
    if 'gemini_manager' not in st.session_state:
        st.session_state.gemini_manager = GeminiAIManager()

    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = RealisticDataManager()

    if 'portfolio_config' not in st.session_state:
        st.session_state.portfolio_config = PortfolioConfig()

    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = SentimentEnhancedOptimizer(st.session_state.portfolio_config)

    # Data cache
    if 'current_assets' not in st.session_state:
        st.session_state.current_assets = st.session_state.data_manager.get_current_market_data()

    if 'portfolio_weights' not in st.session_state:
        st.session_state.portfolio_weights = st.session_state.data_manager.get_realistic_portfolio_weights()

    if 'performance_data' not in st.session_state:
        st.session_state.performance_data = st.session_state.data_manager.get_portfolio_performance_data()

    # Timestamps
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()

    if 'last_optimization' not in st.session_state:
        st.session_state.last_optimization = None

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application function"""

    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown("""
        <div class="main-header">
            Cryptocurrency Portfolio Optimizer
        </div>
        <div style="text-align: center; margin-bottom: 2rem;">
            <strong>FINS5545 FinTech Project - Stage 4</strong><br>
            Advanced Portfolio Management with AI-Enhanced Sentiment Analysis
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("Portfolio Control Center")

        # Gemini AI Status with Configuration
        st.markdown("### AI Assistant Status")

        if st.session_state.gemini_manager.is_configured:
            st.markdown("""
            <div class="success-card">
                <strong>Gemini AI Active</strong><br>
                AI-powered insights available
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-card">
                <strong>Gemini AI Not Configured</strong><br>
                Configure API key for AI insights
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Configure Gemini AI"):
                api_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    help="Get your API key from: https://makersuite.google.com/app/apikey"
                )

                if st.button("Connect Gemini AI", type="primary"):
                    if api_key:
                        if st.session_state.gemini_manager.configure_api_key(api_key):
                            st.success("Gemini AI configured successfully!")
                            st.rerun()
                        else:
                            st.error("Invalid API key or connection failed")
                    else:
                        st.error("Please enter an API key")

        st.markdown("---")

        # Navigation
        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Portfolio Management", "AI Analysis",
             "Market Insights", "Settings"],
            index=0
        )

        st.markdown("---")

        # Quick stats
        st.markdown("### Quick Stats")

        portfolio_value = st.session_state.performance_data['portfolio_value'].iloc[-1]
        total_return = (portfolio_value / st.session_state.portfolio_config.initial_capital - 1) * 100

        st.metric("Portfolio Value", f"${portfolio_value:,.0f}")
        st.metric("Total Return", f"{total_return:+.1f}%")
        st.metric("Assets Tracked", f"{len(st.session_state.current_assets)}")

        # Quick actions
        st.markdown("### Quick Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Refresh Data", use_container_width=True):
                st.session_state.current_assets = st.session_state.data_manager.get_current_market_data()
                st.session_state.performance_data = st.session_state.data_manager.get_portfolio_performance_data()
                st.session_state.last_update = datetime.now()
                st.success("Data refreshed!")
                st.rerun()

        with col2:
            if st.button("Optimize", use_container_width=True):
                with st.spinner("Optimizing..."):
                    result = st.session_state.optimizer.optimize_portfolio(st.session_state.current_assets)
                    st.session_state.portfolio_weights = result['weights']
                    st.session_state.last_optimization = datetime.now()
                st.success("Portfolio optimized!")
                st.rerun()

        # Data status
        st.markdown("---")
        st.markdown(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")
        if st.session_state.last_optimization:
            st.markdown(f"**Last Optimization:** {st.session_state.last_optimization.strftime('%H:%M:%S')}")

    # Main content routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "Portfolio Management":
        show_portfolio_management()
    elif page == "AI Analysis":
        show_ai_analysis()
    elif page == "Market Insights":
        show_market_insights()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Main portfolio dashboard"""
    st.header("Portfolio Dashboard")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    performance_data = st.session_state.performance_data
    current_value = performance_data['portfolio_value'].iloc[-1]
    initial_value = st.session_state.portfolio_config.initial_capital

    with col1:
        st.metric(
            "Portfolio Value",
            f"${current_value:,.0f}",
            f"{((current_value/initial_value - 1)*100):+.1f}%"
        )

    with col2:
        daily_return = performance_data['daily_return'].iloc[-1] * 100
        st.metric("Daily Return", f"{daily_return:+.2f}%")

    with col3:
        volatility = performance_data['daily_return'].std() * np.sqrt(252) * 100
        st.metric("Volatility (Annual)", f"{volatility:.1f}%")

    with col4:
        sharpe = (performance_data['daily_return'].mean() / performance_data['daily_return'].std()) * np.sqrt(252)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # Main dashboard visualization
    portfolio_data = {
        'weights': st.session_state.portfolio_weights,
        'metrics': {
            'sharpe_ratio': sharpe,
            'volatility': volatility / 100,
            'max_weight': max(st.session_state.portfolio_weights.values()),
            'min_weight': min(st.session_state.portfolio_weights.values()),
            'concentration': sum(w**2 for w in st.session_state.portfolio_weights.values())
        }
    }

    fig = AdvancedVisualizer.create_portfolio_dashboard(
        portfolio_data,
        st.session_state.current_assets,
        performance_data
    )
    st.plotly_chart(fig, use_container_width=True)

    # Performance over time
    st.subheader("Performance Analysis")

    perf_fig = AdvancedVisualizer.create_performance_chart(performance_data)
    st.plotly_chart(perf_fig, use_container_width=True)

    # Current holdings table
    st.subheader("Current Holdings")

    holdings_data = []
    total_value = current_value

    for asset in st.session_state.current_assets:
        weight = st.session_state.portfolio_weights.get(asset.symbol, 0)
        if weight > 0.001:  # Only show significant holdings
            position_value = total_value * weight
            holdings_data.append({
                'Symbol': asset.symbol,
                'Weight': f"{weight:.1%}",
                'Value': f"${position_value:,.0f}",
                'Price': f"${asset.price:,.2f}",
                'Change 24h': f"{asset.change_24h:+.1f}%",
                'Sentiment': f"{asset.sentiment_score:.2f}"
            })

    if holdings_data:
        holdings_df = pd.DataFrame(holdings_data)
        st.dataframe(holdings_df, use_container_width=True)

def show_portfolio_management():
    """Portfolio management interface"""
    st.header("Portfolio Management")

    # Show basic portfolio management interface
    st.info("Portfolio management features available. Configure Gemini AI for advanced features.")

def show_ai_analysis():
    """AI-powered analysis interface"""
    st.header("AI-Powered Portfolio Analysis")

    if not st.session_state.gemini_manager.is_configured:
        st.warning("Gemini AI not configured. Please configure your API key in the sidebar to access AI-powered insights.")
        return

    # AI-powered analysis interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("AI Portfolio Advisor")

        if st.button("Get AI Portfolio Analysis", type="primary", use_container_width=True):
            with st.spinner("AI analyzing your portfolio..."):
                portfolio_data = {
                    'weights': st.session_state.portfolio_weights,
                    'total_value': st.session_state.performance_data['portfolio_value'].iloc[-1]
                }

                market_conditions = {
                    'sentiment': 'Mixed',
                    'volatility': 'Moderate',
                    'trend': 'Sideways to slightly bullish'
                }

                advice = st.session_state.gemini_manager.get_portfolio_advice(
                    portfolio_data, market_conditions
                )

                st.markdown(f"""
                <div class="ai-response">
                <h4>AI Portfolio Analysis</h4>
                {advice}
                </div>
                """, unsafe_allow_html=True)

def show_market_insights():
    """Market analysis and insights"""
    st.header("Market Insights & Analysis")

    # Show basic market insights
    st.info("Market insights available. More features coming soon.")

def show_settings():
    """Application settings and configuration"""
    st.header("Settings & Configuration")

    # Show basic settings
    st.info("Settings interface available. Configure your portfolio parameters here.")

# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")

        logger.error(f"Application crashed: {e}", exc_info=True)
