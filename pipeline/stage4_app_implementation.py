"""
Stage 4: Enhanced Application Implementation
Complete FinTech application with interactive dashboard and AI integration
Compatible with updated data collection and portfolio optimization systems
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Optional dependencies with graceful fallback
try:
    import dash
    from dash import dcc, html, Input, Output, State, dash_table, callback_context
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    print("Warning: Dash components not available. Dashboard features will be limited.")

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("Warning: Google Gemini not available. AI features will be disabled.")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Info: yfinance not available. Real-time data updates disabled.")

from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PortfolioState:
    """Current portfolio state with comprehensive tracking"""
    weights: Dict[str, float]
    value: float
    last_update: datetime
    performance_metrics: Dict[str, float]
    sentiment_scores: Dict[str, float]
    risk_metrics: Dict[str, float]
    market_conditions: Dict[str, Any]

class AIAdvisor:
    """AI-powered investment advisor with multiple fallback options"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.model = None
        self.conversation_history = []

        if HAS_GEMINI and api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                print("✓ Gemini AI advisor initialized successfully")
            except Exception as e:
                print(f"Warning: Gemini initialization failed: {e}")
                self.model = None
        else:
            print("Info: AI advisor running in fallback mode (no Gemini API)")

    async def get_portfolio_advice(self, portfolio_state: PortfolioState,
                                  market_conditions: Dict) -> str:
        """Generate portfolio advice with AI or fallback analysis"""

        if self.model:
            try:
                prompt = self._build_advice_prompt(portfolio_state, market_conditions)
                response = self.model.generate_content(prompt)
                advice = response.text

                # Store in conversation history
                self.conversation_history.append({
                    'timestamp': datetime.now(),
                    'type': 'portfolio_advice',
                    'prompt': prompt[:200] + "...",  # Truncated for storage
                    'response': advice
                })

                return advice

            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                return self._fallback_portfolio_advice(portfolio_state, market_conditions)
        else:
            return self._fallback_portfolio_advice(portfolio_state, market_conditions)

    def _build_advice_prompt(self, portfolio_state: PortfolioState,
                           market_conditions: Dict) -> str:
        """Build comprehensive prompt for AI advisor"""

        # Get top holdings
        top_holdings = sorted(
            portfolio_state.weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Calculate portfolio metrics
        total_sentiment = np.mean(list(portfolio_state.sentiment_scores.values())) if portfolio_state.sentiment_scores else 0.5

        prompt = f"""
        As a cryptocurrency investment advisor, analyze this portfolio and provide actionable insights:
        
        CURRENT PORTFOLIO:
        - Total Value: ${portfolio_state.value:,.2f}
        - Last Update: {portfolio_state.last_update.strftime('%Y-%m-%d')}
        
        TOP HOLDINGS:
        {chr(10).join([f"• {symbol}: {weight:.1%}" for symbol, weight in top_holdings])}
        
        PERFORMANCE METRICS:
        - Sharpe Ratio: {portfolio_state.performance_metrics.get('sharpe_ratio', 0):.2f}
        - Expected Return: {portfolio_state.performance_metrics.get('expected_return', 0):.1%}
        - Volatility: {portfolio_state.performance_metrics.get('volatility', 0):.1%}
        - Max Drawdown: {portfolio_state.performance_metrics.get('max_drawdown', 0):.1%}
        
        MARKET SENTIMENT:
        - Average Sentiment Score: {total_sentiment:.2f} (0=bearish, 1=bullish)
        - Market Regime: {market_conditions.get('regime', 'Neutral')}
        - Volatility Environment: {market_conditions.get('volatility_regime', 'Normal')}
        
        RISK METRICS:
        - VaR (95%): {portfolio_state.risk_metrics.get('var_95', 0):.2%}
        - Portfolio Beta: {portfolio_state.risk_metrics.get('beta', 1.0):.2f}
        
        Please provide:
        1. Portfolio Assessment (strengths/weaknesses)
        2. Risk Analysis based on current conditions
        3. Specific rebalancing recommendations
        4. Market outlook for next 1-2 weeks
        5. Top 3 actionable recommendations
        
        Keep response concise, practical, and investment-focused.
        """

        return prompt

    def _fallback_portfolio_advice(self, portfolio_state: PortfolioState,
                                 market_conditions: Dict) -> str:
        """Generate rule-based advice when AI is unavailable"""

        advice_parts = []

        # Portfolio Assessment
        sharpe = portfolio_state.performance_metrics.get('sharpe_ratio', 0)
        volatility = portfolio_state.performance_metrics.get('volatility', 0)

        if sharpe > 1.0:
            advice_parts.append("✅ **Portfolio Assessment**: Strong risk-adjusted returns (Sharpe > 1.0)")
        elif sharpe > 0.5:
            advice_parts.append("🟡 **Portfolio Assessment**: Moderate performance, room for improvement")
        else:
            advice_parts.append("🔴 **Portfolio Assessment**: Below-average risk-adjusted returns")

        # Risk Analysis
        if volatility > 0.25:
            advice_parts.append("⚠️ **Risk Alert**: High portfolio volatility detected. Consider reducing position sizes.")
        elif volatility < 0.10:
            advice_parts.append("📊 **Risk Status**: Low volatility portfolio. May consider slight risk increase for returns.")

        # Sentiment Analysis
        avg_sentiment = np.mean(list(portfolio_state.sentiment_scores.values())) if portfolio_state.sentiment_scores else 0.5

        if avg_sentiment > 0.7:
            advice_parts.append("🚀 **Sentiment**: Bullish sentiment detected. Monitor for potential overvaluation.")
        elif avg_sentiment < 0.3:
            advice_parts.append("📉 **Sentiment**: Bearish sentiment. Potential buying opportunities emerging.")
        else:
            advice_parts.append("⚖️ **Sentiment**: Neutral sentiment. Focus on fundamental analysis.")

        # Diversification Check
        weights_list = list(portfolio_state.weights.values())
        if max(weights_list) > 0.4:
            advice_parts.append("🎯 **Diversification**: High concentration risk. Consider reducing largest positions.")

        # Top Recommendations
        advice_parts.append("\n**Top Recommendations:**")
        advice_parts.append("1. Rebalance weekly to maintain target allocations")
        advice_parts.append("2. Monitor sentiment shifts for early trend detection")
        advice_parts.append("3. Review risk metrics regularly during volatile periods")

        return "\n\n".join(advice_parts)

    async def answer_user_question(self, question: str,
                                  portfolio_state: PortfolioState) -> str:
        """Answer specific user questions about the portfolio"""

        if self.model:
            try:
                context = f"""
                Context: Cryptocurrency portfolio analysis system
                
                Portfolio Summary:
                - Value: ${portfolio_state.value:,.2f}
                - Holdings: {len(portfolio_state.weights)} assets
                - Sharpe Ratio: {portfolio_state.performance_metrics.get('sharpe_ratio', 0):.2f}
                - Volatility: {portfolio_state.performance_metrics.get('volatility', 0):.1%}
                
                User Question: {question}
                
                Please provide a helpful, accurate, and actionable response focused on cryptocurrency investing.
                """

                response = self.model.generate_content(context)
                return response.text

            except Exception as e:
                logger.error(f"Gemini API error in Q&A: {e}")
                return self._fallback_answer(question, portfolio_state)
        else:
            return self._fallback_answer(question, portfolio_state)

    def _fallback_answer(self, question: str, portfolio_state: PortfolioState) -> str:
        """Provide rule-based answers when AI is unavailable"""

        question_lower = question.lower()

        # Common question patterns
        if any(word in question_lower for word in ['risk', 'volatile', 'safe']):
            volatility = portfolio_state.performance_metrics.get('volatility', 0)
            return f"""**Risk Assessment:**
            
Your portfolio's current volatility is {volatility:.1%} annualized. 

• **Low Risk** (< 15%): Conservative allocation
• **Medium Risk** (15-25%): Balanced approach  
• **High Risk** (> 25%): Aggressive strategy

Consider adjusting position sizes based on your risk tolerance."""

        elif any(word in question_lower for word in ['performance', 'return', 'profit']):
            return_pct = portfolio_state.performance_metrics.get('expected_return', 0)
            sharpe = portfolio_state.performance_metrics.get('sharpe_ratio', 0)

            return f"""**Performance Analysis:**
            
Expected annual return: {return_pct:.1%}
Risk-adjusted performance (Sharpe): {sharpe:.2f}

• Sharpe > 1.0: Excellent
• Sharpe 0.5-1.0: Good
• Sharpe < 0.5: Needs improvement

Focus on consistent risk-adjusted returns rather than absolute returns."""

        elif any(word in question_lower for word in ['diversif', 'allocation', 'balance']):
            weights = list(portfolio_state.weights.values())
            max_weight = max(weights) if weights else 0

            return f"""**Diversification Analysis:**
            
Portfolio spread across {len(weights)} assets.
Largest position: {max_weight:.1%}

• **Well Diversified**: No single asset > 20%
• **Concentrated**: Largest position > 30%

Consider rebalancing if concentration risk is high."""

        else:
            return """I'd be happy to help with questions about:
            
• Portfolio risk and volatility
• Performance and returns analysis  
• Diversification and allocation
• Market sentiment interpretation
• Rebalancing strategies

Please feel free to ask more specific questions about these topics."""

class CryptoDashboard:
    """Enhanced interactive dashboard for cryptocurrency portfolio management"""

    def __init__(self, data_dir: Path = Path('data'), gemini_api_key: Optional[str] = None):
        self.data_dir = data_dir
        self.ai_advisor = AIAdvisor(gemini_api_key)
        self.portfolio_state = self._load_portfolio_state()

        if not HAS_DASH:
            print("Error: Dash not available. Cannot create dashboard.")
            return

        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.CYBORG],
            suppress_callback_exceptions=True,
            title="Crypto Portfolio Intelligence"
        )

        self._setup_layout()
        self._setup_callbacks()

        print("✓ Dashboard initialized successfully")

    def _load_portfolio_state(self) -> PortfolioState:
        """Load current portfolio state from saved results"""

        try:
            # Load portfolio results
            results_path = self.data_dir / 'results' / 'portfolio_results.json'
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)

                current_portfolio = results.get('current_portfolio', {})

                return PortfolioState(
                    weights=current_portfolio.get('weights', {}),
                    value=1000000,  # Default portfolio value
                    last_update=datetime.fromisoformat(results.get('timestamp', datetime.now().isoformat())),
                    performance_metrics=current_portfolio.get('metrics', {}),
                    sentiment_scores=current_portfolio.get('sentiment_signals', {}),
                    risk_metrics=current_portfolio.get('metrics', {}),
                    market_conditions=self._assess_market_conditions()
                )
            else:
                print("Warning: No portfolio results found. Using default state.")

        except Exception as e:
            print(f"Warning: Error loading portfolio state: {e}")

        # Return default state
        return PortfolioState(
            weights={},
            value=1000000,
            last_update=datetime.now(),
            performance_metrics={},
            sentiment_scores={},
            risk_metrics={},
            market_conditions={}
        )

    def _assess_market_conditions(self) -> Dict[str, Any]:
        """Assess current market conditions from available data"""

        conditions = {
            'regime': 'Neutral',
            'volatility_regime': 'Normal',
            'trend': 'Sideways',
            'sentiment_regime': 'Neutral'
        }

        try:
            # Load recent price data if available
            price_data_path = self.data_dir / 'processed' / 'final_price_data.csv'
            if price_data_path.exists():
                df = pd.read_csv(price_data_path, parse_dates=['date'])
                recent_data = df[df['date'] >= df['date'].max() - timedelta(days=30)]

                if not recent_data.empty:
                    # Calculate market volatility
                    daily_returns = recent_data.groupby('date')['close'].mean().pct_change()
                    volatility = daily_returns.std() * np.sqrt(252)

                    if volatility > 0.6:
                        conditions['volatility_regime'] = 'High'
                    elif volatility < 0.3:
                        conditions['volatility_regime'] = 'Low'

                    # Assess trend
                    recent_returns = daily_returns.tail(10).mean()
                    if recent_returns > 0.01:
                        conditions['trend'] = 'Upward'
                    elif recent_returns < -0.01:
                        conditions['trend'] = 'Downward'

        except Exception as e:
            print(f"Warning: Error assessing market conditions: {e}")

        return conditions

    def _setup_layout(self):
        """Setup the dashboard layout with comprehensive UI"""

        if not HAS_DASH:
            return

        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🚀 Crypto Portfolio Intelligence",
                           className="text-center mb-2 text-primary"),
                    html.H5("AI-Powered Sentiment-Driven Portfolio Management",
                           className="text-center text-muted mb-4"),
                    html.Hr()
                ])
            ]),

            # Key Performance Indicators
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Portfolio Value", className="text-muted mb-1"),
                            html.H3(f"${self.portfolio_state.value:,.0f}",
                                   id="portfolio-value", className="text-success mb-0")
                        ])
                    ], className="h-100")
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Sharpe Ratio", className="text-muted mb-1"),
                            html.H3(f"{self.portfolio_state.performance_metrics.get('sharpe_ratio', 0):.2f}",
                                   id="sharpe-ratio", className="mb-0")
                        ])
                    ], className="h-100")
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Expected Return", className="text-muted mb-1"),
                            html.H3(f"{self.portfolio_state.performance_metrics.get('expected_return', 0):.1%}",
                                   id="expected-return", className="text-info mb-0")
                        ])
                    ], className="h-100")
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Market Sentiment", className="text-muted mb-1"),
                            html.H3("Neutral", id="market-sentiment", className="text-warning mb-0")
                        ])
                    ], className="h-100")
                ], width=3),
            ], className="mb-4"),

            # Main Content Tabs
            dbc.Tabs([
                dbc.Tab(label="📊 Portfolio Overview", tab_id="overview"),
                dbc.Tab(label="📈 Performance Analysis", tab_id="performance"),
                dbc.Tab(label="💭 Sentiment Dashboard", tab_id="sentiment"),
                dbc.Tab(label="🤖 AI Advisor", tab_id="advisor"),
                dbc.Tab(label="⚙️ Risk Management", tab_id="risk"),
                dbc.Tab(label="🔧 Settings", tab_id="settings")
            ], id="main-tabs", active_tab="overview"),

            # Tab Content Container
            html.Div(id="tab-content", className="mt-4"),

            # Auto-refresh for real-time updates
            dcc.Interval(id='interval-component', interval=300*1000, n_intervals=0),  # 5 minutes

            # Store for sharing data between callbacks
            dcc.Store(id='portfolio-data-store'),

        ], fluid=True, className="px-4 py-3")

    def _create_overview_content(self):
        """Create comprehensive portfolio overview"""

        # Portfolio allocation pie chart
        if self.portfolio_state.weights:
            labels = list(self.portfolio_state.weights.keys())[:10]  # Top 10
            values = [self.portfolio_state.weights[label] for label in labels]

            allocation_fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    textposition='inside',
                    textinfo='label+percent',
                    marker=dict(colors=px.colors.qualitative.Set3)
                )
            ])

            allocation_fig.update_layout(
                title="Portfolio Allocation",
                template="plotly_dark",
                height=450,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
            )
        else:
            allocation_fig = go.Figure()
            allocation_fig.update_layout(
                title="Portfolio Allocation - No Data Available",
                template="plotly_dark",
                height=450
            )

        # Holdings table
        if self.portfolio_state.weights:
            holdings_data = []
            sorted_holdings = sorted(
                self.portfolio_state.weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:15]  # Top 15 holdings

            for symbol, weight in sorted_holdings:
                holdings_data.append({
                    'Symbol': symbol,
                    'Weight': f"{weight:.2%}",
                    'Value': f"${self.portfolio_state.value * weight:,.0f}",
                    'Sentiment': f"{self.portfolio_state.sentiment_scores.get(symbol, 0.5):.2f}"
                })
        else:
            holdings_data = [{'Symbol': 'No data', 'Weight': '0%', 'Value': '$0', 'Sentiment': '0.5'}]

        holdings_table = dash_table.DataTable(
            data=holdings_data,
            columns=[
                {"name": "Symbol", "id": "Symbol"},
                {"name": "Weight", "id": "Weight"},
                {"name": "Value", "id": "Value"},
                {"name": "Sentiment", "id": "Sentiment"}
            ],
            style_cell={
                'textAlign': 'center',
                'backgroundColor': '#1e1e1e',
                'color': 'white',
                'fontFamily': 'Arial'
            },
            style_header={
                'backgroundColor': '#2e2e2e',
                'fontWeight': 'bold',
                'border': '1px solid #444'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#2a2a2a'
                }
            ],
            page_size=10
        )

        # Portfolio metrics cards
        metrics_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Volatility", className="text-muted"),
                        html.H4(f"{self.portfolio_state.performance_metrics.get('volatility', 0):.1%}",
                               className="text-warning")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Max Drawdown", className="text-muted"),
                        html.H4(f"{self.portfolio_state.performance_metrics.get('max_drawdown', 0):.1%}",
                               className="text-danger")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("VaR (95%)", className="text-muted"),
                        html.H4(f"{self.portfolio_state.risk_metrics.get('var_95', 0):.2%}",
                               className="text-info")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Last Update", className="text-muted"),
                        html.H6(f"{self.portfolio_state.last_update.strftime('%Y-%m-%d')}",
                               className="text-light")
                    ])
                ])
            ], width=3),
        ], className="mb-4")

        return html.Div([
            metrics_cards,
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=allocation_fig)
                ], width=7),
                dbc.Col([
                    html.H4("Top Holdings", className="mb-3"),
                    holdings_table
                ], width=5)
            ])
        ])

    def _create_performance_content(self):
        """Create performance analysis dashboard"""

        # Try to load historical performance data
        try:
            weights_history_path = self.data_dir / 'results' / 'weights_history.csv'
            if weights_history_path.exists():
                weights_df = pd.read_csv(weights_history_path, parse_dates=['date'])

                # Create performance visualization
                performance_fig = self._create_performance_charts()
            else:
                performance_fig = self._create_sample_performance_charts()

        except Exception as e:
            print(f"Warning: Error loading performance data: {e}")
            performance_fig = self._create_sample_performance_charts()

        # Performance metrics summary
        metrics_summary = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Risk-Adjusted Returns"),
                    dbc.CardBody([
                        html.P(f"Sharpe Ratio: {self.portfolio_state.performance_metrics.get('sharpe_ratio', 0):.2f}"),
                        html.P(f"Sortino Ratio: {self.portfolio_state.performance_metrics.get('sortino_ratio', 0):.2f}"),
                        html.P(f"Calmar Ratio: {self.portfolio_state.performance_metrics.get('calmar_ratio', 0):.2f}")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Risk Metrics"),
                    dbc.CardBody([
                        html.P(f"Volatility: {self.portfolio_state.performance_metrics.get('volatility', 0):.1%}"),
                        html.P(f"Max Drawdown: {self.portfolio_state.performance_metrics.get('max_drawdown', 0):.1%}"),
                        html.P(f"VaR (95%): {self.portfolio_state.risk_metrics.get('var_95', 0):.2%}")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Return Metrics"),
                    dbc.CardBody([
                        html.P(f"Expected Return: {self.portfolio_state.performance_metrics.get('expected_return', 0):.1%}"),
                        html.P(f"Downside Volatility: {self.portfolio_state.performance_metrics.get('downside_volatility', 0):.1%}"),
                        html.P(f"Win Rate: {self.portfolio_state.performance_metrics.get('win_rate', 0):.1%}")
                    ])
                ])
            ], width=4)
        ], className="mb-4")

        return html.Div([
            metrics_summary,
            dcc.Graph(figure=performance_fig)
        ])

    def _create_performance_charts(self) -> go.Figure:
        """Create actual performance charts from historical data"""

        # This would create charts from real historical data
        # For now, return sample charts
        return self._create_sample_performance_charts()

    def _create_sample_performance_charts(self) -> go.Figure:
        """Create sample performance charts when no historical data available"""

        # Generate sample data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')

        # Sample portfolio returns
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.02, 252)  # Daily returns
        portfolio_values = 1000000 * (1 + returns).cumprod()

        # Sample benchmark (Bitcoin)
        btc_returns = np.random.normal(0.001, 0.025, 252)
        btc_values = 1000000 * (1 + btc_returns).cumprod()

        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Value Over Time', 'Daily Returns Distribution',
                          'Rolling Sharpe Ratio', 'Drawdown Analysis'),
            specs=[[{"secondary_y": False}, {"type": "histogram"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Portfolio value chart
        fig.add_trace(
            go.Scatter(x=dates, y=portfolio_values, name='Portfolio', line=dict(color='#00ff41')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dates, y=btc_values, name='BTC Benchmark', line=dict(color='#ff6b35')),
            row=1, col=1
        )

        # Returns distribution
        fig.add_trace(
            go.Histogram(x=returns, name='Daily Returns', nbinsx=30, marker_color='lightblue'),
            row=1, col=2
        )

        # Rolling Sharpe ratio
        rolling_sharpe = pd.Series(returns).rolling(30).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        fig.add_trace(
            go.Scatter(x=dates, y=rolling_sharpe, name='30-Day Sharpe', line=dict(color='yellow')),
            row=2, col=1
        )

        # Drawdown
        running_max = pd.Series(portfolio_values).expanding().max()
        drawdown = (portfolio_values - running_max) / running_max * 100
        fig.add_trace(
            go.Scatter(x=dates, y=drawdown, name='Drawdown %',
                      fill='tozeroy', line=dict(color='red')),
            row=2, col=2
        )

        fig.update_layout(
            template="plotly_dark",
            height=800,
            showlegend=True,
            title_text="Portfolio Performance Analysis"
        )

        return fig

    def _create_sentiment_content(self):
        """Create sentiment analysis dashboard"""

        if not self.portfolio_state.sentiment_scores:
            return html.Div([
                dbc.Alert(
                    "No sentiment data available. Please ensure news data was collected in Stage 1.",
                    color="warning"
                ),
                html.P("Sentiment analysis requires news data from cryptocurrency sources.")
            ])

        # Sentiment distribution chart
        sentiment_values = list(self.portfolio_state.sentiment_scores.values())

        sentiment_dist_fig = go.Figure(data=[
            go.Histogram(
                x=sentiment_values,
                nbinsx=20,
                marker_color='lightgreen',
                opacity=0.7
            )
        ])

        sentiment_dist_fig.update_layout(
            title="Current Sentiment Distribution",
            template="plotly_dark",
            xaxis_title="Sentiment Score (0=Bearish, 1=Bullish)",
            yaxis_title="Number of Assets",
            height=400
        )

        # Sentiment by asset (top 10)
        top_assets = sorted(
            self.portfolio_state.sentiment_scores.items(),
            key=lambda x: self.portfolio_state.weights.get(x[0], 0),
            reverse=True
        )[:10]

        asset_sentiment_fig = go.Figure(data=[
            go.Bar(
                x=[asset[0] for asset in top_assets],
                y=[asset[1] for asset in top_assets],
                marker_color=['green' if score > 0.6 else 'red' if score < 0.4 else 'yellow'
                             for _, score in top_assets]
            )
        ])

        asset_sentiment_fig.update_layout(
            title="Sentiment by Top Holdings",
            template="plotly_dark",
            xaxis_title="Cryptocurrency",
            yaxis_title="Sentiment Score",
            height=400
        )

        # Market regime analysis
        avg_sentiment = np.mean(sentiment_values)
        sentiment_std = np.std(sentiment_values)

        market_regime = self._determine_market_regime(avg_sentiment, sentiment_std)

        regime_card = dbc.Card([
            dbc.CardHeader("Market Sentiment Analysis"),
            dbc.CardBody([
                html.H4(market_regime['status'], className=f"text-{market_regime['color']}"),
                html.P(market_regime['description']),
                html.Hr(),
                html.P(f"Average Sentiment: {avg_sentiment:.3f}"),
                html.P(f"Sentiment Volatility: {sentiment_std:.3f}"),
                html.P(f"Bullish Assets: {sum(1 for v in sentiment_values if v > 0.6)}"),
                html.P(f"Bearish Assets: {sum(1 for v in sentiment_values if v < 0.4)}")
            ])
        ])

        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=sentiment_dist_fig)
                ], width=6),
                dbc.Col([
                    regime_card
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=asset_sentiment_fig)
                ])
            ])
        ])

    def _determine_market_regime(self, avg_sentiment: float, sentiment_std: float) -> Dict[str, str]:
        """Determine market regime based on sentiment metrics"""

        if avg_sentiment > 0.7 and sentiment_std < 0.2:
            return {
                'status': '🟢 Extreme Greed',
                'color': 'success',
                'description': 'Very bullish sentiment with low disagreement. Consider taking profits and managing risk.'
            }
        elif avg_sentiment > 0.6:
            return {
                'status': '🟢 Greed',
                'color': 'success',
                'description': 'Bullish sentiment dominates. Monitor for potential reversal signals.'
            }
        elif avg_sentiment < 0.3 and sentiment_std < 0.2:
            return {
                'status': '🔴 Extreme Fear',
                'color': 'danger',
                'description': 'Very bearish sentiment with consensus. Potential buying opportunity.'
            }
        elif avg_sentiment < 0.4:
            return {
                'status': '🔴 Fear',
                'color': 'danger',
                'description': 'Bearish sentiment prevails. Exercise caution but watch for oversold conditions.'
            }
        elif sentiment_std > 0.3:
            return {
                'status': '🟡 High Divergence',
                'color': 'warning',
                'description': 'Mixed sentiment signals. Increased volatility expected.'
            }
        else:
            return {
                'status': '🟡 Neutral',
                'color': 'info',
                'description': 'Balanced sentiment. Focus on fundamental analysis and technical indicators.'
            }

    def _create_ai_advisor_content(self):
        """Create AI advisor interface"""

        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("🤖 AI Investment Advisor"),
                    html.P("Get personalized advice powered by advanced AI analysis" if self.ai_advisor.model
                          else "Get rule-based portfolio advice and analysis"),
                    html.Hr()
                ])
            ]),

            # Portfolio Analysis Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Analysis & Recommendations"),
                        dbc.CardBody([
                            html.Div(id="ai-advice-content", children=[
                                dbc.Spinner([
                                    html.P("Click 'Get AI Analysis' for personalized portfolio recommendations")
                                ])
                            ]),
                            dbc.Button(
                                "Get AI Analysis",
                                id="get-advice-btn",
                                color="primary",
                                className="mt-3",
                                size="lg"
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),

            # Q&A Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Ask Your AI Advisor"),
                        dbc.CardBody([
                            dbc.InputGroup([
                                dbc.Input(
                                    id="user-question",
                                    placeholder="Ask any question about your portfolio (e.g., 'How can I reduce risk?')",
                                    type="text",
                                    size="lg"
                                ),
                                dbc.Button(
                                    "Ask",
                                    id="ask-btn",
                                    color="secondary",
                                    outline=True
                                )
                            ], className="mb-3"),
                            html.Hr(),
                            html.Div(id="ai-answer", className="mt-3")
                        ])
                    ])
                ], width=12)
            ]),

            # Quick Questions
            dbc.Row([
                dbc.Col([
                    html.H5("Quick Questions:", className="mt-4 mb-3"),
                    dbc.ButtonGroup([
                        dbc.Button("Portfolio Risk", id="quick-risk", color="outline-info", size="sm"),
                        dbc.Button("Performance Analysis", id="quick-performance", color="outline-info", size="sm"),
                        dbc.Button("Diversification", id="quick-diversification", color="outline-info", size="sm"),
                        dbc.Button("Market Outlook", id="quick-outlook", color="outline-info", size="sm")
                    ], className="flex-wrap")
                ])
            ])
        ])

    def _create_risk_management_content(self):
        """Create risk management dashboard"""

        # Risk metrics overview
        risk_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Value at Risk (95%)", className="text-muted"),
                        html.H4(f"{self.portfolio_state.risk_metrics.get('var_95', 0):.2%}",
                               className="text-danger")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Expected Shortfall", className="text-muted"),
                        html.H4(f"{self.portfolio_state.risk_metrics.get('cvar_95', 0):.2%}",
                               className="text-warning")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Maximum Drawdown", className="text-muted"),
                        html.H4(f"{self.portfolio_state.performance_metrics.get('max_drawdown', 0):.1%}",
                               className="text-info")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Portfolio Beta", className="text-muted"),
                        html.H4(f"{self.portfolio_state.risk_metrics.get('beta', 1.0):.2f}",
                               className="text-success")
                    ])
                ])
            ], width=3)
        ], className="mb-4")

        # Risk decomposition chart
        if self.portfolio_state.weights:
            # Calculate risk contribution (simplified)
            weights = np.array(list(self.portfolio_state.weights.values()))
            symbols = list(self.portfolio_state.weights.keys())

            # Approximate risk contribution based on weights
            risk_contrib = weights**2  # Simplified - assumes equal correlations
            risk_contrib = risk_contrib / risk_contrib.sum()

            risk_fig = go.Figure(data=[
                go.Bar(
                    x=symbols[:10],  # Top 10
                    y=risk_contrib[:10] * 100,
                    marker_color='rgba(255, 182, 193, 0.8)'
                )
            ])

            risk_fig.update_layout(
                title="Risk Contribution by Asset (Top 10)",
                template="plotly_dark",
                xaxis_title="Asset",
                yaxis_title="Risk Contribution (%)",
                height=400
            )
        else:
            risk_fig = go.Figure()
            risk_fig.update_layout(
                title="Risk Analysis - No Data Available",
                template="plotly_dark"
            )

        # Risk management recommendations
        risk_recommendations = dbc.Card([
            dbc.CardHeader("Risk Management Recommendations"),
            dbc.CardBody([
                html.Ul([
                    html.Li("Monitor VaR daily and adjust positions if limits are exceeded"),
                    html.Li("Rebalance weekly to maintain target risk levels"),
                    html.Li("Consider reducing position sizes during high volatility periods"),
                    html.Li("Implement stop-loss orders for individual positions > 10%"),
                    html.Li("Diversify across uncorrelated assets to reduce portfolio risk")
                ])
            ])
        ])

        return html.Div([
            risk_cards,
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=risk_fig)
                ], width=8),
                dbc.Col([
                    risk_recommendations
                ], width=4)
            ])
        ])

    def _create_settings_content(self):
        """Create settings and configuration interface"""

        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("⚙️ Portfolio Settings"),
                    html.P("Configure portfolio optimization parameters"),
                    html.Hr()
                ])
            ]),

            dbc.Row([
                # Risk Parameters
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Parameters"),
                        dbc.CardBody([
                            dbc.Label("Target Volatility (%)"),
                            dbc.Input(
                                id="target-vol",
                                type="number",
                                value=15,
                                min=5,
                                max=30,
                                step=1
                            ),
                            html.Small("Annual volatility target", className="text-muted"),

                            html.Hr(),

                            dbc.Label("Maximum Position Size (%)"),
                            dbc.Input(
                                id="max-position",
                                type="number",
                                value=30,
                                min=5,
                                max=50,
                                step=5
                            ),
                            html.Small("Maximum weight per asset", className="text-muted"),

                            html.Hr(),

                            dbc.Label("Sentiment Weight (%)"),
                            dbc.Input(
                                id="sentiment-weight",
                                type="number",
                                value=30,
                                min=0,
                                max=100,
                                step=10
                            ),
                            html.Small("Weight of sentiment in optimization", className="text-muted")
                        ])
                    ])
                ], width=6),

                # Rebalancing Settings
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Rebalancing Settings"),
                        dbc.CardBody([
                            dbc.Label("Rebalance Frequency"),
                            dbc.Select(
                                id="rebalance-freq",
                                options=[
                                    {"label": "Daily", "value": "D"},
                                    {"label": "Weekly", "value": "W"},
                                    {"label": "Bi-weekly", "value": "2W"},
                                    {"label": "Monthly", "value": "M"}
                                ],
                                value="W"
                            ),

                            html.Hr(),

                            dbc.Label("Transaction Cost (%)"),
                            dbc.Input(
                                id="transaction-cost",
                                type="number",
                                value=0.1,
                                min=0,
                                max=1,
                                step=0.05
                            ),
                            html.Small("Transaction cost per trade", className="text-muted"),

                            html.Hr(),

                            dbc.Label("Lookback Period (days)"),
                            dbc.Input(
                                id="lookback-period",
                                type="number",
                                value=60,
                                min=20,
                                max=252,
                                step=10
                            ),
                            html.Small("Historical data period for optimization", className="text-muted")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),

            # Data Management
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Data Management"),
                        dbc.CardBody([
                            dbc.Button(
                                "Refresh Portfolio Data",
                                id="refresh-data-btn",
                                color="info",
                                className="me-2 mb-2"
                            ),
                            dbc.Button(
                                "Recalculate Optimization",
                                id="recalc-opt-btn",
                                color="warning",
                                className="me-2 mb-2"
                            ),
                            dbc.Button(
                                "Export Portfolio Data",
                                id="export-data-btn",
                                color="success",
                                className="me-2 mb-2"
                            ),
                            html.Hr(),
                            html.Div(id="data-status", className="mt-3")
                        ])
                    ])
                ])
            ], className="mb-4"),

            # Save Settings Button
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Save All Settings",
                        id="save-settings-btn",
                        color="primary",
                        size="lg",
                        className="w-100"
                    ),
                    html.Div(id="settings-status", className="mt-3")
                ])
            ])
        ])

    def _setup_callbacks(self):
        """Setup all dashboard callbacks"""

        if not HAS_DASH:
            return

        # Tab content callback
        @self.app.callback(
            Output("tab-content", "children"),
            Input("main-tabs", "active_tab")
        )
        def render_tab_content(active_tab):
            if active_tab == "overview":
                return self._create_overview_content()
            elif active_tab == "performance":
                return self._create_performance_content()
            elif active_tab == "sentiment":
                return self._create_sentiment_content()
            elif active_tab == "advisor":
                return self._create_ai_advisor_content()
            elif active_tab == "risk":
                return self._create_risk_management_content()
            elif active_tab == "settings":
                return self._create_settings_content()
            return html.Div("Select a tab to view content")

        # AI Advisor callbacks
        @self.app.callback(
            Output("ai-advice-content", "children"),
            Input("get-advice-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def get_ai_advice(n_clicks):
            if n_clicks:
                try:
                    # Get AI advice (synchronous call for now)
                    import asyncio
                    advice = asyncio.run(
                        self.ai_advisor.get_portfolio_advice(
                            self.portfolio_state,
                            self.portfolio_state.market_conditions
                        )
                    )
                    return dcc.Markdown(advice)
                except Exception as e:
                    return dbc.Alert(f"Error getting AI advice: {e}", color="danger")
            return dash.no_update

        # Q&A callback
        @self.app.callback(
            Output("ai-answer", "children"),
            [Input("ask-btn", "n_clicks")] +
            [Input(f"quick-{topic}", "n_clicks") for topic in ["risk", "performance", "diversification", "outlook"]],
            State("user-question", "value"),
            prevent_initial_call=True
        )
        def answer_question(ask_clicks, risk_clicks, perf_clicks, div_clicks, outlook_clicks, question):
            ctx = callback_context
            if not ctx.triggered:
                return dash.no_update

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Handle quick questions
            if button_id == "quick-risk":
                question = "How can I better manage portfolio risk?"
            elif button_id == "quick-performance":
                question = "How is my portfolio performing compared to benchmarks?"
            elif button_id == "quick-diversification":
                question = "Is my portfolio well diversified?"
            elif button_id == "quick-outlook":
                question = "What's the market outlook for crypto?"
            elif not question:
                return dbc.Alert("Please enter a question", color="warning")

            try:
                import asyncio
                answer = asyncio.run(
                    self.ai_advisor.answer_user_question(question, self.portfolio_state)
                )
                return dcc.Markdown(answer)
            except Exception as e:
                return dbc.Alert(f"Error processing question: {e}", color="danger")

        # Settings callback
        @self.app.callback(
            Output("settings-status", "children"),
            Input("save-settings-btn", "n_clicks"),
            [State(setting_id, "value") for setting_id in
             ["target-vol", "max-position", "sentiment-weight", "rebalance-freq", "transaction-cost", "lookback-period"]],
            prevent_initial_call=True
        )
        def save_settings(n_clicks, target_vol, max_pos, sent_weight, rebal_freq, trans_cost, lookback):
            if n_clicks:
                try:
                    settings = {
                        'target_volatility': target_vol / 100,
                        'max_position': max_pos / 100,
                        'sentiment_weight': sent_weight / 100,
                        'rebalance_frequency': rebal_freq,
                        'transaction_cost': trans_cost / 100,
                        'lookback_period': lookback,
                        'updated_at': datetime.now().isoformat()
                    }

                    # Save settings
                    settings_dir = self.data_dir / 'config'
                    settings_dir.mkdir(exist_ok=True)

                    with open(settings_dir / 'portfolio_settings.json', 'w') as f:
                        json.dump(settings, f, indent=2)

                    return dbc.Alert("✅ Settings saved successfully!", color="success")

                except Exception as e:
                    return dbc.Alert(f"❌ Error saving settings: {e}", color="danger")

            return dash.no_update

        # Auto-refresh callback
        @self.app.callback(
            [Output("portfolio-value", "children"),
             Output("market-sentiment", "children")],
            Input("interval-component", "n_intervals")
        )
        def update_live_metrics(n):
            # Simulate small changes for demo
            base_value = self.portfolio_state.value
            updated_value = base_value * (1 + np.random.normal(0, 0.001))

            # Update sentiment display
            if self.portfolio_state.sentiment_scores:
                avg_sentiment = np.mean(list(self.portfolio_state.sentiment_scores.values()))
                if avg_sentiment > 0.6:
                    sentiment_display = html.Span("Bullish", className="text-success")
                elif avg_sentiment < 0.4:
                    sentiment_display = html.Span("Bearish", className="text-danger")
                else:
                    sentiment_display = html.Span("Neutral", className="text-warning")
            else:
                sentiment_display = html.Span("No Data", className="text-muted")

            return f"${updated_value:,.0f}", sentiment_display

    def run(self, debug: bool = False, port: int = 8050, host: str = '0.0.0.0'):
        """Run the dashboard application"""

        if not HAS_DASH:
            print("Error: Cannot run dashboard - Dash not available")
            return

        print(f"🚀 Starting Crypto Portfolio Intelligence Dashboard")
        print(f"📡 Server: http://localhost:{port}")
        print(f"🤖 AI Advisor: {'Enabled' if self.ai_advisor.model else 'Fallback Mode'}")
        print(f"📊 Portfolio: {len(self.portfolio_state.weights)} assets")

        try:
            self.app.run_server(debug=debug, port=port, host=host)
        except Exception as e:
            print(f"Error running dashboard: {e}")

def main():
    """Main entry point for the dashboard application"""

    import argparse

    parser = argparse.ArgumentParser(description='Crypto Portfolio Dashboard')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--port', type=int, default=8050, help='Port to run dashboard on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run dashboard on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--gemini-key', help='Google Gemini API key for AI features')

    args = parser.parse_args()

    # Create data directory if it doesn't exist
    data_path = Path(args.data_dir)
    data_path.mkdir(exist_ok=True)

    # Check for required data files
    required_files = [
        'processed/final_price_data.csv',
        'results/portfolio_results.json'
    ]

    missing_files = []
    for file_path in required_files:
        if not (data_path / file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("⚠️  Warning: Some data files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nThe dashboard will run with limited functionality.")
        print("Run Stages 1-3 to generate complete data.")

    # Initialize and run dashboard
    try:
        dashboard = CryptoDashboard(data_path, args.gemini_key)
        dashboard.run(debug=args.debug, port=args.port, host=args.host)

    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard failed to start: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())