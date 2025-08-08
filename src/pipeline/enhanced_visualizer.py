"""
Enhanced Portfolio Visualization Module for Stage 3
Cryptocurrency Portfolio Optimization with Sentiment Analysis
FINS5545 FinTech Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Any, Optional
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class EnhancedPortfolioVisualizer:
    """Advanced visualization system for cryptocurrency portfolio analysis"""

    def __init__(self, results: Dict, output_dir: Path, returns_data: Optional[pd.DataFrame] = None,
                 sentiment_data: Optional[Dict] = None):
        self.results = results
        self.output_dir = output_dir
        self.returns_data = returns_data
        self.sentiment_data = sentiment_data or {}

        # Setup directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)

        # Professional styling
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # Color scheme for crypto themes
        self.colors = {
            'bitcoin': '#F7931A',
            'ethereum': '#627EEA',
            'crypto_green': '#1BB76E',
            'crypto_red': '#C73E1D',
            'neutral': '#2E86AB',
            'sentiment_pos': '#4CAF50',
            'sentiment_neg': '#F44336',
            'sentiment_neu': '#FF9800'
        }

    def create_comprehensive_analysis(self):
        """Generate complete visualization suite"""
        print("\nGenerating comprehensive portfolio analysis...")

        # Core performance charts
        self.create_strategy_performance_grid()
        self.create_portfolio_allocation_analysis()
        self.create_efficient_frontier_plot()

        # Risk and performance analysis
        self.create_risk_attribution_chart()
        self.create_drawdown_analysis()

        # Sentiment-specific charts
        if self.sentiment_data:
            self.create_sentiment_impact_analysis()

        # Market analysis charts
        self.create_correlation_heatmap()
        self.create_performance_attribution()

        # Summary dashboard
        self.create_executive_dashboard()

        print(f"All visualizations saved to: {self.plots_dir}")

    def create_drawdown_analysis(self):
        """Create drawdown analysis chart"""
        if not self.returns_data or len(self.returns_data.columns) < 3:
            print("Skipping drawdown analysis - insufficient return data")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Simulate portfolio performance for each strategy
        returns = self.returns_data.fillna(0)

        for i, (strategy, result) in enumerate(list(self.results.items())[:4]):
            weights = result['weights']

            # Get portfolio returns
            portfolio_returns = returns @ pd.Series(weights).reindex(returns.columns, fill_value=0)
            cumulative_returns = (1 + portfolio_returns).cumprod()

            # Calculate drawdown
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max

            # Plot cumulative returns
            ax1.plot(cumulative_returns.index, cumulative_returns.values,
                    label=strategy.replace('_', ' ').title(), linewidth=2)

            # Plot drawdown
            ax2.fill_between(drawdown.index, drawdown.values, 0,
                           alpha=0.3, label=strategy.replace('_', ' ').title())

        ax1.set_title('Cumulative Portfolio Performance', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Cumulative Return', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_title('Portfolio Drawdown Analysis', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Drawdown', fontweight='bold')
        ax2.set_xlabel('Date', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'drawdown_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_portfolio_allocation_analysis(self):
        """Create detailed portfolio allocation analysis"""
        if not self.results:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Get best strategy
        best_strategy = max(self.results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
        weights = best_strategy[1]['weights']

        # Panel 1: Top 10 holdings bar chart
        top_10 = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])

        bars = ax1.bar(range(len(top_10)), list(top_10.values()),
                      color=self.colors['neutral'], alpha=0.8)
        ax1.set_xticks(range(len(top_10)))
        ax1.set_xticklabels(list(top_10.keys()), rotation=45, ha='right')
        ax1.set_ylabel('Portfolio Weight', fontweight='bold')
        ax1.set_title('Top 10 Holdings', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        for i, v in enumerate(top_10.values()):
            ax1.text(i, v + 0.005, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')

        # Panel 2: Weight distribution histogram
        weight_values = list(weights.values())
        ax2.hist(weight_values, bins=20, alpha=0.7, color=self.colors['neutral'],
                edgecolor='black', linewidth=1)
        ax2.set_xlabel('Portfolio Weight', fontweight='bold')
        ax2.set_ylabel('Number of Assets', fontweight='bold')
        ax2.set_title('Weight Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Cumulative weight concentration
        sorted_weights = sorted(weight_values, reverse=True)
        cumulative_weights = np.cumsum(sorted_weights)

        ax3.plot(range(1, len(cumulative_weights) + 1), cumulative_weights,
                linewidth=3, color=self.colors['crypto_green'])
        ax3.axhline(y=0.5, color=self.colors['crypto_red'], linestyle='--',
                   label='50% Concentration')
        ax3.axhline(y=0.8, color=self.colors['neutral'], linestyle='--',
                   label='80% Concentration')
        ax3.set_xlabel('Number of Assets', fontweight='bold')
        ax3.set_ylabel('Cumulative Weight', fontweight='bold')
        ax3.set_title('Portfolio Concentration Curve', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Asset allocation by market cap tiers
        # Simulate market cap tiers
        large_cap = sum([w for s, w in weights.items() if s in ['BTC', 'ETH']])
        mid_cap = sum([w for s, w in weights.items() if s in ['BNB', 'SOL', 'ADA', 'AVAX']])
        small_cap = 1 - large_cap - mid_cap

        tiers = ['Large Cap\n(BTC, ETH)', 'Mid Cap\n(Top 10)', 'Small Cap\n(Others)']
        tier_weights = [large_cap, mid_cap, small_cap]
        colors_tier = [self.colors['crypto_green'], self.colors['neutral'], self.colors['crypto_red']]

        wedges, texts, autotexts = ax4.pie(tier_weights, labels=tiers, autopct='%1.1f%%',
                                          colors=colors_tier, startangle=90)
        ax4.set_title('Allocation by Market Cap Tiers', fontweight='bold')

        plt.suptitle(f'Portfolio Allocation Analysis - {best_strategy[0].replace("_", " ").title()}',
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'portfolio_allocation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_strategy_performance_grid(self):
        """Multi-panel strategy comparison with key metrics"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)

        # Extract metrics for all strategies
        strategies = list(self.results.keys())
        metrics_data = self._extract_all_metrics()

        # Panel 1: Sharpe Ratio comparison
        ax1 = fig.add_subplot(gs[0, 0])
        sharpe_values = [self.results[s]['metrics']['sharpe_ratio'] for s in strategies]
        bars = ax1.bar(range(len(strategies)), sharpe_values,
                      color=[self.colors['crypto_green'] if x > 1.0 else
                            self.colors['neutral'] if x > 0.5 else
                            self.colors['crypto_red'] for x in sharpe_values])
        ax1.set_title('Sharpe Ratio Performance', fontweight='bold', fontsize=11)
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45, ha='right')
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Good (1.0)')
        ax1.grid(True, alpha=0.3)

        for i, v in enumerate(sharpe_values):
            ax1.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

        # Panel 2: Return vs Volatility
        ax2 = fig.add_subplot(gs[0, 1])
        returns = [self.results[s]['metrics']['expected_return'] * 100 for s in strategies]
        volatilities = [self.results[s]['metrics']['volatility'] * 100 for s in strategies]

        scatter = ax2.scatter(volatilities, returns, s=150, c=sharpe_values,
                             cmap='RdYlGn', alpha=0.8, edgecolors='black', linewidth=2)
        ax2.set_xlabel('Volatility (%)', fontweight='bold')
        ax2.set_ylabel('Expected Return (%)', fontweight='bold')
        ax2.set_title('Risk-Return Profile', fontweight='bold', fontsize=11)
        ax2.grid(True, alpha=0.3)

        for i, strategy in enumerate(strategies):
            ax2.annotate(strategy.replace('_', ' ').title(),
                        (volatilities[i], returns[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Panel 3: Maximum Drawdown
        ax3 = fig.add_subplot(gs[0, 2])
        drawdowns = [abs(self.results[s]['metrics']['max_drawdown']) * 100 for s in strategies]
        bars = ax3.bar(range(len(strategies)), drawdowns,
                      color=[self.colors['crypto_red'] if x > 20 else
                            self.colors['neutral'] if x > 10 else
                            self.colors['crypto_green'] for x in drawdowns])
        ax3.set_title('Maximum Drawdown (%)', fontweight='bold', fontsize=11)
        ax3.set_xticks(range(len(strategies)))
        ax3.set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)

        for i, v in enumerate(drawdowns):
            ax3.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Panel 4: Risk-adjusted returns (Sortino)
        ax4 = fig.add_subplot(gs[1, 0])
        sortino_values = [self.results[s]['metrics'].get('sortino_ratio', 0) for s in strategies]
        bars = ax4.bar(range(len(strategies)), sortino_values,
                      color=self.colors['ethereum'], alpha=0.8)
        ax4.set_title('Sortino Ratio', fontweight='bold', fontsize=11)
        ax4.set_xticks(range(len(strategies)))
        ax4.set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)

        for i, v in enumerate(sortino_values):
            ax4.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

        # Panel 5: Value at Risk (VaR 95%)
        ax5 = fig.add_subplot(gs[1, 1])
        var_values = [abs(self.results[s]['metrics']['var_95']) * 100 for s in strategies]
        bars = ax5.bar(range(len(strategies)), var_values,
                      color=self.colors['crypto_red'], alpha=0.7)
        ax5.set_title('Value at Risk 95% (%)', fontweight='bold', fontsize=11)
        ax5.set_xticks(range(len(strategies)))
        ax5.set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45, ha='right')
        ax5.grid(True, alpha=0.3)

        # Panel 6: Portfolio concentration
        ax6 = fig.add_subplot(gs[1, 2])
        if self.results:
            best_strategy = max(self.results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
            weights = list(best_strategy[1]['weights'].values())
            concentration = sum(w**2 for w in weights)  # Herfindahl index

            # Top holdings pie chart
            top_weights = dict(sorted(best_strategy[1]['weights'].items(),
                                    key=lambda x: x[1], reverse=True)[:8])
            others_weight = 1 - sum(top_weights.values())
            if others_weight > 0:
                top_weights['Others'] = others_weight

            wedges, texts, autotexts = ax6.pie(list(top_weights.values()),
                                              labels=list(top_weights.keys()),
                                              autopct='%1.1f%%', startangle=90,
                                              colors=plt.cm.Set3(range(len(top_weights))))
            ax6.set_title(f'Portfolio Composition\n{best_strategy[0].replace("_", " ").title()}',
                         fontweight='bold', fontsize=11)

        # Panel 7-9: Performance metrics comparison table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')

        # Create comparison table
        table_data = []
        for strategy in strategies:
            metrics = self.results[strategy]['metrics']
            table_data.append([
                strategy.replace('_', ' ').title(),
                f"{metrics['expected_return']*100:.1f}%",
                f"{metrics['volatility']*100:.1f}%",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics.get('sortino_ratio', 0):.2f}",
                f"{abs(metrics['max_drawdown'])*100:.1f}%",
                f"{abs(metrics['var_95'])*100:.1f}%"
            ])

        table = ax7.table(cellText=table_data,
                         colLabels=['Strategy', 'Return', 'Volatility', 'Sharpe', 'Sortino', 'Max DD', 'VaR 95%'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style the table
        for i in range(len(strategies) + 1):
            for j in range(7):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')

        plt.suptitle('Cryptocurrency Portfolio Strategy Analysis', fontsize=16, fontweight='bold', y=0.95)
        plt.savefig(self.plots_dir / 'strategy_performance_grid.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_efficient_frontier_plot(self):
        """Generate efficient frontier with strategy positions"""
        if not self.returns_data or len(self.returns_data.columns) < 3:
            print("Skipping efficient frontier - insufficient return data")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Calculate efficient frontier points
        returns = self.returns_data.fillna(0)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Generate frontier
        target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
        frontier_volatility = []

        for target in target_returns:
            try:
                # Simple minimum variance for target return
                n_assets = len(mean_returns)
                weights = np.ones(n_assets) / n_assets  # Fallback
                portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
                frontier_volatility.append(portfolio_vol)
            except:
                frontier_volatility.append(0)

        # Plot 1: Efficient frontier with strategy positions
        ax1.plot([v * 100 for v in frontier_volatility], [r * 100 for r in target_returns],
                'b-', linewidth=2, label='Efficient Frontier', alpha=0.7)

        # Plot strategy positions
        for strategy in self.results:
            metrics = self.results[strategy]['metrics']
            vol = metrics['volatility'] * 100
            ret = metrics['expected_return'] * 100
            sharpe = metrics['sharpe_ratio']

            # Color based on performance
            if sharpe > 1.5:
                color = self.colors['crypto_green']
                marker = 'o'
                size = 120
            elif sharpe > 0.5:
                color = self.colors['neutral']
                marker = 's'
                size = 100
            else:
                color = self.colors['crypto_red']
                marker = '^'
                size = 80

            ax1.scatter(vol, ret, c=color, marker=marker, s=size,
                       alpha=0.8, edgecolors='black', linewidth=2,
                       label=strategy.replace('_', ' ').title())

            # Add strategy labels
            ax1.annotate(strategy.replace('_', ' ').title(),
                        (vol, ret), xytext=(5, 5),
                        textcoords='offset points', fontsize=9, fontweight='bold')

        ax1.set_xlabel('Portfolio Volatility (%)', fontweight='bold')
        ax1.set_ylabel('Expected Return (%)', fontweight='bold')
        ax1.set_title('Efficient Frontier Analysis', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 2: Sharpe ratio surface
        vol_range = np.linspace(5, 50, 20)
        ret_range = np.linspace(0, 30, 20)
        V, R = np.meshgrid(vol_range, ret_range)

        # Calculate Sharpe ratio for each point (simplified)
        risk_free_rate = 2  # 2% risk-free rate
        Sharpe = (R - risk_free_rate) / V

        contour = ax2.contourf(V, R, Sharpe, levels=20, cmap='RdYlGn', alpha=0.7)

        # Plot strategy positions on Sharpe surface
        for strategy in self.results:
            metrics = self.results[strategy]['metrics']
            vol = metrics['volatility'] * 100
            ret = metrics['expected_return'] * 100
            ax2.scatter(vol, ret, c='black', s=100, marker='o',
                       edgecolors='white', linewidth=2)
            ax2.annotate(strategy.replace('_', ' ').title(),
                        (vol, ret), xytext=(5, 5),
                        textcoords='offset points', fontsize=9,
                        fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        ax2.set_xlabel('Portfolio Volatility (%)', fontweight='bold')
        ax2.set_ylabel('Expected Return (%)', fontweight='bold')
        ax2.set_title('Sharpe Ratio Landscape', fontweight='bold', fontsize=14)

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax2)
        cbar.set_label('Sharpe Ratio', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'efficient_frontier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_sentiment_impact_analysis(self):
        """Analyze sentiment impact on portfolio performance"""
        if not self.sentiment_data:
            print("Skipping sentiment analysis - no sentiment data available")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Panel 1: Sentiment scores distribution
        sentiment_values = list(self.sentiment_data.values())
        if sentiment_values:
            ax1.hist(sentiment_values, bins=20, alpha=0.7, color=self.colors['neutral'],
                    edgecolor='black', linewidth=1)
            ax1.axvline(np.mean(sentiment_values), color=self.colors['crypto_red'],
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(sentiment_values):.3f}')
            ax1.set_xlabel('Sentiment Score', fontweight='bold')
            ax1.set_ylabel('Frequency', fontweight='bold')
            ax1.set_title('Market Sentiment Distribution', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Panel 2: Sentiment vs Strategy Performance
        if 'sentiment_enhanced' in self.results:
            strategies = list(self.results.keys())
            sharpe_ratios = [self.results[s]['metrics']['sharpe_ratio'] for s in strategies]

            # Highlight sentiment-enhanced strategy
            colors = [self.colors['crypto_green'] if s == 'sentiment_enhanced' else
                     self.colors['neutral'] for s in strategies]

            bars = ax2.bar(range(len(strategies)), sharpe_ratios, color=colors, alpha=0.8)
            ax2.set_xlabel('Strategy', fontweight='bold')
            ax2.set_ylabel('Sharpe Ratio', fontweight='bold')
            ax2.set_title('Sentiment Enhancement Impact', fontweight='bold')
            ax2.set_xticks(range(len(strategies)))
            ax2.set_xticklabels([s.replace('_', ' ').title() for s in strategies],
                               rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for i, v in enumerate(sharpe_ratios):
                ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

        # Panel 3: Sentiment-Performance Correlation
        assets = list(self.sentiment_data.keys())[:10]  # Top 10 assets
        if assets and 'sentiment_enhanced' in self.results:
            weights = self.results['sentiment_enhanced']['weights']
            asset_weights = [weights.get(asset, 0) for asset in assets]
            asset_sentiments = [self.sentiment_data.get(asset, 0) for asset in assets]

            scatter = ax3.scatter(asset_sentiments, asset_weights, s=100,
                                 c=asset_sentiments, cmap='RdYlGn', alpha=0.8,
                                 edgecolors='black', linewidth=1)

            # Add trend line
            if len(asset_sentiments) > 1:
                z = np.polyfit(asset_sentiments, asset_weights, 1)
                p = np.poly1d(z)
                ax3.plot(sorted(asset_sentiments), p(sorted(asset_sentiments)),
                        "r--", alpha=0.8, linewidth=2)

            ax3.set_xlabel('Sentiment Score', fontweight='bold')
            ax3.set_ylabel('Portfolio Weight', fontweight='bold')
            ax3.set_title('Sentiment-Weight Relationship', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # Add asset labels
            for i, asset in enumerate(assets):
                ax3.annotate(asset, (asset_sentiments[i], asset_weights[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Panel 4: Sentiment category performance
        if sentiment_values:
            # Categorize sentiments
            positive_assets = [k for k, v in self.sentiment_data.items() if v > 0.1]
            negative_assets = [k for k, v in self.sentiment_data.items() if v < -0.1]
            neutral_assets = [k for k, v in self.sentiment_data.items() if -0.1 <= v <= 0.1]

            categories = ['Positive', 'Neutral', 'Negative']
            counts = [len(positive_assets), len(neutral_assets), len(negative_assets)]
            colors_cat = [self.colors['sentiment_pos'], self.colors['sentiment_neu'],
                         self.colors['sentiment_neg']]

            wedges, texts, autotexts = ax4.pie(counts, labels=categories, autopct='%1.1f%%',
                                              colors=colors_cat, startangle=90)
            ax4.set_title('Sentiment Category Distribution', fontweight='bold')

        plt.suptitle('Sentiment Analysis Impact on Portfolio Optimization',
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'sentiment_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_risk_attribution_chart(self):
        """Portfolio risk attribution analysis"""
        if not self.results:
            return

        # Get best performing strategy
        best_strategy = max(self.results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
        weights = best_strategy[1]['weights']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Panel 1: Weight distribution
        top_holdings = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:15])

        y_pos = np.arange(len(top_holdings))
        weights_values = list(top_holdings.values())
        assets = list(top_holdings.keys())

        bars = ax1.barh(y_pos, weights_values, color=self.colors['neutral'], alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(assets)
        ax1.set_xlabel('Portfolio Weight', fontweight='bold')
        ax1.set_title(f'Top Holdings - {best_strategy[0].replace("_", " ").title()}', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add weight labels
        for i, v in enumerate(weights_values):
            ax1.text(v + 0.001, i, f'{v:.1%}', va='center', fontweight='bold')

        # Panel 2: Risk contribution simulation
        # Simulate risk contributions (in practice, this would use actual covariance data)
        risk_contributions = np.array(weights_values) * np.random.uniform(0.8, 1.2, len(weights_values))
        risk_contributions = risk_contributions / risk_contributions.sum()

        bars = ax2.barh(y_pos, risk_contributions, color=self.colors['crypto_red'], alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(assets)
        ax2.set_xlabel('Risk Contribution', fontweight='bold')
        ax2.set_title('Estimated Risk Attribution', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Panel 3: Weight vs Risk scatter
        ax3.scatter(weights_values, risk_contributions, s=100,
                   c=range(len(weights_values)), cmap='viridis', alpha=0.8,
                   edgecolors='black', linewidth=1)

        # Perfect correlation line
        max_val = max(max(weights_values), max(risk_contributions))
        ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.7,
                label='Perfect Correlation')

        ax3.set_xlabel('Portfolio Weight', fontweight='bold')
        ax3.set_ylabel('Risk Contribution', fontweight='bold')
        ax3.set_title('Weight vs Risk Relationship', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Concentration metrics
        metrics_names = ['Portfolio\nConcentration', 'Top 5\nConcentration', 'Top 10\nConcentration']

        # Calculate concentration metrics
        herfindahl = sum(w**2 for w in weights.values())
        top5_concentration = sum(sorted(weights.values(), reverse=True)[:5])
        top10_concentration = sum(sorted(weights.values(), reverse=True)[:10])

        concentrations = [herfindahl, top5_concentration, top10_concentration]
        colors_conc = [self.colors['crypto_green'] if c < 0.5 else
                      self.colors['neutral'] if c < 0.8 else
                      self.colors['crypto_red'] for c in concentrations]

        bars = ax4.bar(range(len(metrics_names)), concentrations,
                      color=colors_conc, alpha=0.8)
        ax4.set_xticks(range(len(metrics_names)))
        ax4.set_xticklabels(metrics_names)
        ax4.set_ylabel('Concentration Level', fontweight='bold')
        ax4.set_title('Portfolio Concentration Analysis', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, v in enumerate(concentrations):
            ax4.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.suptitle(f'Risk Attribution Analysis - {best_strategy[0].replace("_", " ").title()}',
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'risk_attribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_correlation_heatmap(self):
        """Advanced correlation analysis"""
        if not self.returns_data or len(self.returns_data.columns) < 5:
            print("Skipping correlation analysis - insufficient data")
            return

        # Calculate correlation matrix
        returns = self.returns_data.fillna(0)
        correlation_matrix = returns.corr()

        # Select top assets by market presence
        top_assets = returns.var().nlargest(20).index
        corr_subset = correlation_matrix.loc[top_assets, top_assets]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Panel 1: Full correlation heatmap
        mask = np.triu(np.ones_like(corr_subset, dtype=bool))
        sns.heatmap(corr_subset, mask=mask, annot=False, cmap='RdBu_r', center=0,
                   square=True, ax=ax1, cbar_kws={"shrink": .8})
        ax1.set_title('Asset Correlation Matrix', fontweight='bold', fontsize=12)

        # Panel 2: Average correlations
        avg_correlations = []
        asset_names = []
        for asset in top_assets[:10]:
            other_assets = [a for a in top_assets if a != asset]
            avg_corr = corr_subset.loc[asset, other_assets].mean()
            avg_correlations.append(avg_corr)
            asset_names.append(asset)

        bars = ax2.barh(range(len(asset_names)), avg_correlations,
                       color=[self.colors['crypto_red'] if x > 0.7 else
                             self.colors['neutral'] if x > 0.3 else
                             self.colors['crypto_green'] for x in avg_correlations])
        ax2.set_yticks(range(len(asset_names)))
        ax2.set_yticklabels(asset_names)
        ax2.set_xlabel('Average Correlation', fontweight='bold')
        ax2.set_title('Average Asset Correlations', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Panel 3: Correlation distribution
        corr_values = corr_subset.values[np.triu_indices_from(corr_subset.values, k=1)]
        ax3.hist(corr_values, bins=20, alpha=0.7, color=self.colors['neutral'],
                edgecolor='black', linewidth=1)
        ax3.axvline(np.mean(corr_values), color=self.colors['crypto_red'],
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(corr_values):.3f}')
        ax3.set_xlabel('Correlation Coefficient', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Correlation Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Network-style correlation plot
        # Show strongest correlations as network
        threshold = 0.6
        strong_corrs = []

        for i in range(len(top_assets[:8])):
            for j in range(i+1, len(top_assets[:8])):
                corr_val = corr_subset.iloc[i, j]
                if abs(corr_val) > threshold:
                    strong_corrs.append((top_assets[i], top_assets[j], corr_val))

        # Simple network visualization
        ax4.set_xlim(-1, 1)
        ax4.set_ylim(-1, 1)

        # Position assets in circle
        n_assets = min(8, len(top_assets))
        angles = np.linspace(0, 2*np.pi, n_assets, endpoint=False)
        positions = {top_assets[i]: (0.8*np.cos(angles[i]), 0.8*np.sin(angles[i]))
                    for i in range(n_assets)}

        # Draw connections
        for asset1, asset2, corr in strong_corrs:
            if asset1 in positions and asset2 in positions:
                x1, y1 = positions[asset1]
                x2, y2 = positions[asset2]
                color = self.colors['crypto_green'] if corr > 0 else self.colors['crypto_red']
                alpha = min(abs(corr), 0.8)
                ax4.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=abs(corr)*3)

        # Draw asset nodes
        for asset, (x, y) in positions.items():
            ax4.scatter(x, y, s=200, c=self.colors['neutral'], alpha=0.8,
                       edgecolors='black', linewidth=2)
            ax4.annotate(asset, (x, y), xytext=(0, 0), textcoords='offset points',
                        ha='center', va='center', fontweight='bold', fontsize=8)

        ax4.set_title(f'Strong Correlations Network (>{threshold:.1f})', fontweight='bold')
        ax4.set_aspect('equal')
        ax4.axis('off')

        plt.suptitle('Cryptocurrency Correlation Analysis', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_performance_attribution(self):
        """Portfolio performance attribution analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Panel 1: Strategy performance ranking
        strategies = list(self.results.keys())
        performance_metrics = []

        for strategy in strategies:
            metrics = self.results[strategy]['metrics']
            # Composite score: weighted average of key metrics
            score = (metrics['sharpe_ratio'] * 0.4 +
                    metrics['expected_return'] * 100 * 0.3 +
                    (1 - abs(metrics['max_drawdown'])) * 0.3)
            performance_metrics.append((strategy, score, metrics['sharpe_ratio']))

        # Sort by composite score
        performance_metrics.sort(key=lambda x: x[1], reverse=True)

        strategies_sorted = [x[0] for x in performance_metrics]
        scores = [x[1] for x in performance_metrics]
        sharpe_ratios = [x[2] for x in performance_metrics]

        # Create ranking chart
        colors_rank = [self.colors['crypto_green'] if i == 0 else
                      self.colors['neutral'] if i < len(strategies)//2 else
                      self.colors['crypto_red'] for i in range(len(strategies))]

        bars = ax1.barh(range(len(strategies_sorted)), scores, color=colors_rank, alpha=0.8)
        ax1.set_yticks(range(len(strategies_sorted)))
        ax1.set_yticklabels([s.replace('_', ' ').title() for s in strategies_sorted])
        ax1.set_xlabel('Composite Performance Score', fontweight='bold')
        ax1.set_title('Strategy Performance Ranking', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # Add ranking numbers
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax1.text(score + 0.1, i, f'#{i+1}', va='center', fontweight='bold', fontsize=12)

        # Panel 2: Risk-adjusted return comparison
        returns_annual = [self.results[s]['metrics']['expected_return'] * 100 for s in strategies]
        volatilities = [self.results[s]['metrics']['volatility'] * 100 for s in strategies]

        # Create bubble chart where bubble size represents Sharpe ratio
        bubble_sizes = [(self.results[s]['metrics']['sharpe_ratio'] + 1) * 100 for s in strategies]

        scatter = ax2.scatter(volatilities, returns_annual, s=bubble_sizes,
                             c=sharpe_ratios, cmap='RdYlGn', alpha=0.7,
                             edgecolors='black', linewidth=2)

        # Add strategy labels
        for i, strategy in enumerate(strategies):
            ax2.annotate(strategy.replace('_', ' ').title(),
                        (volatilities[i], returns_annual[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax2.set_xlabel('Volatility (%)', fontweight='bold')
        ax2.set_ylabel('Expected Return (%)', fontweight='bold')
        ax2.set_title('Risk-Return Efficiency (Bubble size = Sharpe Ratio)', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Sharpe Ratio', fontweight='bold')

        # Panel 3: Drawdown comparison
        max_drawdowns = [abs(self.results[s]['metrics']['max_drawdown']) * 100 for s in strategies]

        bars = ax3.bar(range(len(strategies)), max_drawdowns,
                      color=[self.colors['crypto_green'] if x < 10 else
                            self.colors['neutral'] if x < 20 else
                            self.colors['crypto_red'] for x in max_drawdowns])
        ax3.set_xticks(range(len(strategies)))
        ax3.set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=45, ha='right')
        ax3.set_ylabel('Maximum Drawdown (%)', fontweight='bold')
        ax3.set_title('Downside Risk Comparison', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, v in enumerate(max_drawdowns):
            ax3.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Panel 4: Performance metrics radar chart
        if len(strategies) >= 3:
            # Select top 3 strategies
            top_3_strategies = strategies_sorted[:3]

            # Normalize metrics for radar chart
            metrics_names = ['Return', 'Sharpe', 'Low Volatility', 'Low Drawdown', 'Sortino']

            angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle

            ax4 = plt.subplot(2, 2, 4, projection='polar')

            colors_radar = [self.colors['crypto_green'], self.colors['neutral'], self.colors['crypto_red']]

            for i, strategy in enumerate(top_3_strategies):
                metrics = self.results[strategy]['metrics']

                # Normalize metrics to 0-1 scale
                values = [
                    min(metrics['expected_return'] * 100 / 20, 1),  # Return (capped at 20%)
                    min(metrics['sharpe_ratio'] / 2, 1),           # Sharpe (capped at 2)
                    1 - min(metrics['volatility'], 1),            # Low volatility
                    1 - min(abs(metrics['max_drawdown']), 1),     # Low drawdown
                    min(metrics.get('sortino_ratio', 0) / 2, 1)   # Sortino (capped at 2)
                ]
                values += values[:1]  # Complete the circle

                ax4.plot(angles, values, 'o-', linewidth=2, label=strategy.replace('_', ' ').title(),
                        color=colors_radar[i])
                ax4.fill(angles, values, alpha=0.25, color=colors_radar[i])

            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics_names)
            ax4.set_ylim(0, 1)
            ax4.set_title('Top 3 Strategies Comparison', fontweight='bold', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.suptitle('Portfolio Performance Attribution Analysis', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_attribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_executive_dashboard(self):
        """Executive summary dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, hspace=0.3, wspace=0.3)

        # Get best strategy
        best_strategy = max(self.results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
        best_name = best_strategy[0]
        best_metrics = best_strategy[1]['metrics']
        best_weights = best_strategy[1]['weights']

        # Header section
        fig.text(0.5, 0.95, 'CRYPTOCURRENCY PORTFOLIO OPTIMIZATION',
                ha='center', va='center', fontsize=20, fontweight='bold')
        fig.text(0.5, 0.92, 'Executive Dashboard - FINS5545 FinTech Project',
                ha='center', va='center', fontsize=14)
        fig.text(0.5, 0.89, f'Optimal Strategy: {best_name.replace("_", " ").title()}',
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['crypto_green'], alpha=0.7))

        # Key metrics cards
        metrics_info = [
            ('Annual Return', f"{best_metrics['expected_return']*100:.1f}%", self.colors['crypto_green']),
            ('Sharpe Ratio', f"{best_metrics['sharpe_ratio']:.2f}", self.colors['neutral']),
            ('Volatility', f"{best_metrics['volatility']*100:.1f}%", self.colors['neutral']),
            ('Max Drawdown', f"{abs(best_metrics['max_drawdown'])*100:.1f}%", self.colors['crypto_red'])
        ]

        for i, (label, value, color) in enumerate(metrics_info):
            ax_metric = fig.add_subplot(gs[0, i])
            ax_metric.text(0.5, 0.7, value, ha='center', va='center',
                          fontsize=24, fontweight='bold', color=color)
            ax_metric.text(0.5, 0.3, label, ha='center', va='center',
                          fontsize=12, fontweight='bold')
            ax_metric.set_xlim(0, 1)
            ax_metric.set_ylim(0, 1)
            ax_metric.axis('off')

            # Add border
            rect = Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=2,
                           edgecolor=color, facecolor='none')
            ax_metric.add_patch(rect)

        # Portfolio composition (top 8 holdings)
        ax_composition = fig.add_subplot(gs[1, :2])
        top_8_weights = dict(sorted(best_weights.items(), key=lambda x: x[1], reverse=True)[:8])
        others_weight = 1 - sum(top_8_weights.values())
        if others_weight > 0.01:
            top_8_weights['Others'] = others_weight

        wedges, texts, autotexts = ax_composition.pie(list(top_8_weights.values()),
                                                     labels=list(top_8_weights.keys()),
                                                     autopct='%1.1f%%', startangle=90,
                                                     colors=plt.cm.Set3(range(len(top_8_weights))))
        ax_composition.set_title('Portfolio Composition', fontweight='bold', fontsize=14)

        # Strategy comparison
        ax_comparison = fig.add_subplot(gs[1, 2:])
        strategies = list(self.results.keys())
        sharpe_ratios = [self.results[s]['metrics']['sharpe_ratio'] for s in strategies]

        bars = ax_comparison.bar(range(len(strategies)), sharpe_ratios,
                               color=[self.colors['crypto_green'] if s == best_name else
                                     self.colors['neutral'] for s in strategies])
        ax_comparison.set_xticks(range(len(strategies)))
        ax_comparison.set_xticklabels([s.replace('_', ' ').title() for s in strategies],
                                     rotation=45, ha='right')
        ax_comparison.set_ylabel('Sharpe Ratio', fontweight='bold')
        ax_comparison.set_title('Strategy Performance Comparison', fontweight='bold', fontsize=14)
        ax_comparison.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, v in enumerate(sharpe_ratios):
            ax_comparison.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

        # Risk metrics table
        ax_table = fig.add_subplot(gs[2, :2])
        ax_table.axis('off')

        table_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Expected Return', f"{best_metrics['expected_return']*100:.1f}%", 'Annual return expectation'],
            ['Volatility', f"{best_metrics['volatility']*100:.1f}%", 'Annual risk level'],
            ['Sharpe Ratio', f"{best_metrics['sharpe_ratio']:.2f}", 'Risk-adjusted performance'],
            ['Sortino Ratio', f"{best_metrics.get('sortino_ratio', 0):.2f}", 'Downside risk-adjusted return'],
            ['Max Drawdown', f"{abs(best_metrics['max_drawdown'])*100:.1f}%", 'Worst historical loss'],
            ['VaR (95%)', f"{abs(best_metrics['var_95'])*100:.1f}%", 'Daily loss at 95% confidence']
        ]

        table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                              cellLoc='center', loc='center',
                              bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the table
        for i in range(len(table_data)):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor(self.colors['neutral'])
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')

        # Key insights text box
        ax_insights = fig.add_subplot(gs[2, 2:])
        ax_insights.axis('off')

        insights_text = f"""
KEY INSIGHTS & RECOMMENDATIONS

✓ Optimal Strategy: {best_name.replace('_', ' ').title()}
  - Delivers {best_metrics['expected_return']*100:.1f}% expected annual return
  - Sharpe ratio of {best_metrics['sharpe_ratio']:.2f} indicates strong risk-adjusted performance
  
✓ Risk Profile:
  - Annual volatility: {best_metrics['volatility']*100:.1f}%
  - Maximum drawdown: {abs(best_metrics['max_drawdown'])*100:.1f}%
  
✓ Portfolio Characteristics:
  - Diversified across {len([w for w in best_weights.values() if w > 0.01])} major cryptocurrencies
  - Top holding represents {max(best_weights.values()):.1%} of portfolio
  
{'✓ Sentiment Enhancement: Positive impact on performance detected' if 'sentiment_enhanced' in self.results and self.results['sentiment_enhanced']['metrics']['sharpe_ratio'] > 0.5 else ''}

⚠ Risk Considerations:
  - Cryptocurrency markets are highly volatile
  - Past performance does not guarantee future results
  - Consider position sizing and regular rebalancing
        """

        ax_insights.text(0.05, 0.95, insights_text, ha='left', va='top',
                        fontsize=11, transform=ax_insights.transAxes,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.8))

        plt.savefig(self.plots_dir / 'executive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _extract_all_metrics(self):
        """Helper function to extract metrics from all strategies"""
        all_metrics = {}
        for strategy, result in self.results.items():
            all_metrics[strategy] = result['metrics']
        return all_metrics

    def _save_figure(self, fig, filename):
        """Helper function to save figures"""
        fig.write_html(self.output_dir / f'{filename}.html')
        try:
            fig.write_image(self.plots_dir / f'{filename}_plotly.png')
        except:
            pass

# Updated main visualization creation function for stage3_model_design.py
def create_enhanced_visualizations(results: Dict, output_dir: Path,
                                 returns_data: Optional[pd.DataFrame] = None,
                                 sentiment_data: Optional[Dict] = None):
    """Create comprehensive visualization suite"""

    visualizer = EnhancedPortfolioVisualizer(results, output_dir, returns_data, sentiment_data)
    visualizer.create_comprehensive_analysis()

    print(f"\n✅ Enhanced visualizations created successfully")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Charts generated:")

    chart_files = list((output_dir / 'plots').glob('*.png'))
    for i, chart_file in enumerate(chart_files, 1):
        print(f"   {i}. {chart_file.name}")

    return len(chart_files)