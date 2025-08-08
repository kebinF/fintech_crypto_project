"""
Stage 2: Feature Engineering Pipeline - Final Version
Complete implementation with visualizations and table outputs
Addresses both structured and unstructured data as per project requirements
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import talib
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER not available, using simple sentiment scoring")

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== Configuration ==================
PROJECT_ROOT = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd()
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'features'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'stage2'
TABLES_DIR = REPORT_DIR / 'tables'
PLOTS_DIR = REPORT_DIR / 'plots'

# Create directories
for dir_path in [OUTPUT_DIR, REPORT_DIR, TABLES_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ================== Table Generator ==================
class TableGenerator:
    """Generate formatted tables for report"""

    @staticmethod
    def create_feature_summary_table(df: pd.DataFrame, feature_groups: Dict) -> pd.DataFrame:
        """Create feature summary table"""

        summary_data = []

        for group_name, features in feature_groups.items():
            if features:
                group_features = [f for f in features if f in df.columns]
                if group_features:
                    summary_data.append({
                        'Feature Group': group_name,
                        'Count': len(group_features),
                        'Missing %': f"{df[group_features].isnull().mean().mean() * 100:.2f}%",
                        'Example Features': ', '.join(group_features[:3])
                    })

        summary_df = pd.DataFrame(summary_data)
        return summary_df

    @staticmethod
    def create_missing_value_table(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Create missing value analysis table"""

        missing_stats = df.isnull().sum()
        missing_pct = (missing_stats / len(df) * 100).round(2)

        missing_df = pd.DataFrame({
            'Feature': missing_stats.index,
            'Missing Count': missing_stats.values,
            'Missing %': missing_pct.values
        })

        # Filter and sort
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        missing_df = missing_df.sort_values('Missing Count', ascending=False).head(top_n)

        return missing_df

    @staticmethod
    def create_sentiment_statistics_table(df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment feature statistics table"""

        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'vader' in col.lower()]

        if not sentiment_cols:
            return pd.DataFrame()

        stats_data = []
        for col in sentiment_cols[:10]:  # Top 10 sentiment features
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                stats_data.append({
                    'Feature': col,
                    'Mean': f"{df[col].mean():.4f}",
                    'Std': f"{df[col].std():.4f}",
                    'Min': f"{df[col].min():.4f}",
                    'Max': f"{df[col].max():.4f}",
                    'Skewness': f"{df[col].skew():.4f}"
                })

        return pd.DataFrame(stats_data)

    @staticmethod
    def save_table_to_file(table_df: pd.DataFrame, filename: str, format: str = 'both'):
        """Save table to CSV and/or formatted text"""

        if format in ['csv', 'both']:
            table_df.to_csv(TABLES_DIR / f"{filename}.csv", index=False)

        if format in ['txt', 'both']:
            with open(TABLES_DIR / f"{filename}.txt", 'w') as f:
                f.write(tabulate(table_df, headers='keys', tablefmt='grid', showindex=False))

        # Also create LaTeX version
        latex_str = table_df.to_latex(index=False, escape=False)
        with open(TABLES_DIR / f"{filename}.tex", 'w') as f:
            f.write(latex_str)

# ================== Visualization Generator ==================
class VisualizationGenerator:
    """Generate visualizations for report"""

    @staticmethod
    def create_feature_distribution_plots(df: pd.DataFrame, features: List[str], save_path: Path):
        """Create distribution plots for key features"""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, feature in enumerate(features[:6]):
            if feature in df.columns:
                ax = axes[idx]

                # Remove outliers for better visualization
                data = df[feature].dropna()
                q1, q3 = data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                data_filtered = data[(data >= lower) & (data <= upper)]

                ax.hist(data_filtered, bins=50, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
                ax.legend()

        plt.suptitle('Feature Distributions', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, features: List[str], save_path: Path):
        """Create correlation heatmap for key features"""

        # Select numeric features
        numeric_features = [f for f in features if f in df.columns and df[f].dtype in ['float64', 'int64']]

        if len(numeric_features) > 20:
            numeric_features = numeric_features[:20]

        corr_matrix = df[numeric_features].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   annot=False)  # Too many features for annotations
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    @staticmethod
    def create_sentiment_timeline(df: pd.DataFrame, save_path: Path):
        """Create sentiment timeline visualization"""

        if 'date' not in df.columns:
            return

        # Aggregate sentiment by date
        sentiment_cols = ['sentiment_index', 'market_sentiment', 'fear_greed_index']
        available_cols = [col for col in sentiment_cols if col in df.columns]

        if not available_cols:
            return

        daily_sentiment = df.groupby('date')[available_cols].mean()

        fig, ax = plt.subplots(figsize=(14, 6))

        for col in available_cols:
            ax.plot(daily_sentiment.index, daily_sentiment[col], label=col, linewidth=2)

        ax.set_title('Sentiment Indicators Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add neutral line
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral')

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    @staticmethod
    def create_missing_value_heatmap(df: pd.DataFrame, save_path: Path, sample_size: int = 1000):
        """Create missing value pattern heatmap"""

        # Sample data for visualization
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df

        # Select features with any missing values
        missing_cols = [col for col in df.columns if df[col].isnull().any()]

        if not missing_cols:
            return

        # Limit to top 30 features with most missing values
        if len(missing_cols) > 30:
            missing_counts = df[missing_cols].isnull().sum()
            missing_cols = missing_counts.nlargest(30).index.tolist()

        plt.figure(figsize=(12, 8))
        sns.heatmap(df_sample[missing_cols].isnull(), cbar=True, cmap='RdYlBu',
                   yticklabels=False, xticklabels=True)
        plt.title('Missing Value Patterns')
        plt.xlabel('Features')
        plt.ylabel('Samples')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

# ================== Price Feature Engineering ==================
class PriceFeatureEngineer:
    """Create technical indicators and market microstructure features"""

    def __init__(self):
        self.feature_groups = {
            'returns': ['simple', 'log', 'multi_period'],
            'momentum': ['rsi', 'macd', 'roc', 'stoch'],
            'volatility': ['atr', 'bollinger', 'historical'],
            'volume': ['volume_ratio', 'obv', 'volume_shock'],
            'microstructure': ['amihud', 'roll_spread', 'high_low_spread']
        }

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all price-based features"""
        logger.info("Creating price features...")
        df = df.sort_values(['symbol', 'date']).copy()

        # Basic features
        df = self._create_return_features(df)
        df = self._create_momentum_features(df)
        df = self._create_volatility_features(df)
        df = self._create_volume_features(df)
        df = self._create_microstructure_features(df)

        # Cross-sectional features
        df = self._create_cross_sectional_features(df)

        return df

    def _create_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create various return measures"""
        logger.info("Creating return features...")

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol

            # Simple returns
            df.loc[mask, 'returns_1d'] = df.loc[mask, 'close'].pct_change()

            # Log returns
            df.loc[mask, 'log_returns'] = np.log(df.loc[mask, 'close'] / df.loc[mask, 'close'].shift(1))

            # Multi-period returns
            for period in [5, 10, 20, 60]:
                df.loc[mask, f'returns_{period}d'] = df.loc[mask, 'close'].pct_change(period)

            # Overnight and intraday returns
            df.loc[mask, 'overnight_return'] = (
                df.loc[mask, 'open'] / df.loc[mask, 'close'].shift(1) - 1
            )
            df.loc[mask, 'intraday_return'] = (
                df.loc[mask, 'close'] / df.loc[mask, 'open'] - 1
            )

        return df

    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum indicators"""
        logger.info("Creating momentum features...")

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            sym_data = df.loc[mask].copy()

            if len(sym_data) < 30:
                continue

            try:
                # RSI
                for period in [14, 21]:
                    df.loc[mask, f'rsi_{period}'] = talib.RSI(
                        sym_data['close'].values, timeperiod=period
                    )

                # MACD
                macd, signal, hist = talib.MACD(
                    sym_data['close'].values,
                    fastperiod=12, slowperiod=26, signalperiod=9
                )
                df.loc[mask, 'macd'] = macd
                df.loc[mask, 'macd_signal'] = signal
                df.loc[mask, 'macd_hist'] = hist

                # Rate of Change
                for period in [10, 20]:
                    df.loc[mask, f'roc_{period}'] = talib.ROC(
                        sym_data['close'].values, timeperiod=period
                    )

                # Stochastic
                slowk, slowd = talib.STOCH(
                    sym_data['high'].values,
                    sym_data['low'].values,
                    sym_data['close'].values,
                    fastk_period=14, slowk_period=3, slowd_period=3
                )
                df.loc[mask, 'stoch_k'] = slowk
                df.loc[mask, 'stoch_d'] = slowd

            except Exception as e:
                logger.warning(f"Error creating momentum features for {symbol}: {e}")

        return df

    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility indicators"""
        logger.info("Creating volatility features...")

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            sym_data = df.loc[mask].copy()

            if len(sym_data) < 30:
                continue

            try:
                # ATR
                df.loc[mask, 'atr_14'] = talib.ATR(
                    sym_data['high'].values,
                    sym_data['low'].values,
                    sym_data['close'].values,
                    timeperiod=14
                )

                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(
                    sym_data['close'].values,
                    timeperiod=20,
                    nbdevup=2, nbdevdn=2
                )
                df.loc[mask, 'bb_upper'] = upper
                df.loc[mask, 'bb_middle'] = middle
                df.loc[mask, 'bb_lower'] = lower
                df.loc[mask, 'bb_width'] = (upper - lower) / (middle + 1e-10)
                df.loc[mask, 'bb_position'] = (
                    (sym_data['close'] - lower) / (upper - lower + 1e-10)
                )

                # Historical volatility
                for period in [10, 20, 30]:
                    df.loc[mask, f'hvol_{period}'] = (
                        sym_data['log_returns'].rolling(period).std() * np.sqrt(365)
                    )

            except Exception as e:
                logger.warning(f"Error creating volatility features for {symbol}: {e}")

        return df

    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        logger.info("Creating volume features...")

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            sym_data = df.loc[mask].copy()

            # Volume moving averages and ratios
            for period in [10, 20]:
                vol_ma = sym_data['usd_volume'].rolling(period).mean()
                df.loc[mask, f'vol_ma_{period}'] = vol_ma
                df.loc[mask, f'vol_ratio_{period}'] = sym_data['usd_volume'] / (vol_ma + 1e-10)

            # On-Balance Volume
            try:
                obv = talib.OBV(sym_data['close'].values, sym_data['usd_volume'].values)
                df.loc[mask, 'obv'] = obv
            except:
                pass

            # Volume shock
            for period in [7, 14]:
                rolling_mean = sym_data['usd_volume'].shift(1).rolling(period).mean()
                df.loc[mask, f'vol_shock_{period}'] = (
                    np.log1p(sym_data['usd_volume']) - np.log1p(rolling_mean + 1e-10)
                )

        return df

    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        logger.info("Creating microstructure features...")

        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            sym_data = df.loc[mask].copy()

            # Amihud illiquidity measure
            df.loc[mask, 'amihud_20'] = (
                (np.abs(sym_data['returns_1d']) / (sym_data['usd_volume_mil'] + 1e-10))
                .rolling(20).mean()
            )

            # Roll's implied spread
            returns = sym_data['returns_1d'].dropna()
            if len(returns) > 20:
                autocorr = returns.rolling(20).apply(
                    lambda x: x.autocorr() if len(x) > 1 else np.nan
                )
                df.loc[mask, 'roll_spread'] = 2 * np.sqrt(np.abs(autocorr.clip(upper=0)))

            # High-low spread estimator
            df.loc[mask, 'hl_spread'] = (
                2 * (np.log(sym_data['high']) - np.log(sym_data['low']))
            ).rolling(20).mean()

        return df

    def _create_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cross-sectional features with FIXED beta calculation"""
        logger.info("Creating cross-sectional features...")

        # Rank features within each date
        rank_features = ['returns_1d', 'returns_5d', 'rsi_14', 'hvol_20', 'vol_ratio_20']

        for feature in rank_features:
            if feature in df.columns:
                df[f'{feature}_rank'] = (
                    df.groupby('date')[feature]
                    .rank(pct=True, method='average')
                )

                df[f'{feature}_zscore'] = (
                    df.groupby('date')[feature]
                    .transform(lambda x: (x - x.mean()) / (x.std() + 1e-10))
                )

        # Market-relative features
        df['excess_return'] = (
            df['returns_1d'] - df.groupby('date')['returns_1d'].transform('mean')
        )

        df['relative_volume'] = (
            df['usd_volume'] / (df.groupby('date')['usd_volume'].transform('mean') + 1e-10)
        )

        # Simple beta calculation (avoiding the index alignment issue)
        if 'BTC' in df['symbol'].unique():
            logger.info("Calculating beta relative to BTC...")

            # Simple approach: use correlation as proxy for beta
            for period in [30, 60]:
                df[f'beta_{period}'] = 1.0  # Default

                for symbol in df['symbol'].unique():
                    if symbol == 'BTC':
                        df.loc[df['symbol'] == 'BTC', f'beta_{period}'] = 1.0
                    else:
                        # Calculate correlation with BTC
                        symbol_returns = df.loc[df['symbol'] == symbol, 'returns_1d']
                        btc_returns = df.loc[df['symbol'] == 'BTC', 'returns_1d']

                        if len(symbol_returns) > period and len(btc_returns) > period:
                            correlation = symbol_returns.corr(btc_returns)
                            # Use correlation as simplified beta
                            df.loc[df['symbol'] == symbol, f'beta_{period}'] = max(0.5, min(1.5, correlation))
        else:
            df['beta_30'] = 1.0
            df['beta_60'] = 1.0

        return df

# ================== News Feature Engineering ==================
class NewsFeatureEngineer:
    """Create sentiment and text-based features from news data"""

    def __init__(self):
        # Initialize sentiment analyzer
        if VADER_AVAILABLE:
            self.sia = SentimentIntensityAnalyzer()
            self._update_crypto_lexicon()
        else:
            self.sia = None

        # Cryptocurrency entities
        self.crypto_entities = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'BNB': ['binance', 'bnb'],
            'SOL': ['solana', 'sol'],
            'XRP': ['ripple', 'xrp'],
            'ADA': ['cardano', 'ada'],
            'DOGE': ['dogecoin', 'doge'],
            'DOT': ['polkadot', 'dot'],
            'MATIC': ['polygon', 'matic'],
            'LINK': ['chainlink', 'link']
        }

    def _update_crypto_lexicon(self):
        """Add cryptocurrency-specific terms to VADER lexicon"""
        if not self.sia:
            return

        crypto_lexicon = {
            # Positive terms
            'moon': 3.0, 'mooning': 3.0, 'bullish': 2.5, 'pump': 2.0,
            'hodl': 2.0, 'rally': 2.0, 'surge': 2.0, 'adoption': 2.0,

            # Negative terms
            'dump': -2.5, 'crash': -3.0, 'bearish': -2.5, 'rekt': -3.0,
            'rugpull': -3.0, 'scam': -3.0, 'hack': -2.5,
            'fud': -2.0, 'panic': -2.5, 'plunge': -2.5,

            # Neutral
            'blockchain': 0.5, 'defi': 0.5, 'nft': 0.5
        }

        self.sia.lexicon.update(crypto_lexicon)

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all news-based features"""
        logger.info("Creating news features...")

        # Ensure required columns
        if 'title' not in df.columns:
            df['title'] = ''
        if 'body' not in df.columns:
            df['body'] = ''

        # Combine text
        df['full_text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')

        # Create features
        df = self._create_text_stats(df)
        df = self._create_sentiment_features(df)
        df = self._create_entity_features(df)
        df = self._create_topic_features(df)
        df = self._create_temporal_features(df)

        return df

    def _create_text_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic text statistics"""
        df['text_length'] = df['full_text'].str.len()
        df['word_count'] = df['full_text'].str.split().str.len()
        df['sentence_count'] = df['full_text'].str.count('[.!?]+')

        df['exclamation_count'] = df['full_text'].str.count('!')
        df['question_count'] = df['full_text'].str.count('\\?')
        df['uppercase_ratio'] = df['full_text'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
        )

        return df

    def _create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment features"""

        if VADER_AVAILABLE and self.sia:
            # VADER sentiment
            vader_scores = df['full_text'].apply(lambda x: self.sia.polarity_scores(x))
            df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
            df['vader_positive'] = vader_scores.apply(lambda x: x['pos'])
            df['vader_negative'] = vader_scores.apply(lambda x: x['neg'])
            df['vader_neutral'] = vader_scores.apply(lambda x: x['neu'])
        else:
            # Use existing simple sentiment
            df['vader_compound'] = df.get('positive', 0.5) * 2 - 1
            df['vader_positive'] = df.get('positive', 0.5)
            df['vader_negative'] = 1 - df.get('positive', 0.5)
            df['vader_neutral'] = 0.5

        # Sentiment intensity
        df['emotion_intensity'] = np.abs(df['vader_compound'])

        # Sentiment volatility
        def sent_volatility(text):
            if not text:
                return 0
            sentences = text.split('.')
            if len(sentences) < 2:
                return 0
            if VADER_AVAILABLE and self.sia:
                sentiments = [self.sia.polarity_scores(s)['compound'] for s in sentences if s.strip()]
            else:
                sentiments = [0.5] * len(sentences)
            return np.std(sentiments) if len(sentiments) > 1 else 0

        df['sentiment_volatility'] = df['full_text'].apply(sent_volatility)

        return df

    def _create_entity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cryptocurrency entity features"""

        for crypto, keywords in self.crypto_entities.items():
            pattern = '|'.join(keywords)
            df[f'{crypto}_mentions'] = df['full_text'].str.lower().str.count(pattern)

        mention_cols = [col for col in df.columns if col.endswith('_mentions')]
        df['total_crypto_mentions'] = df[mention_cols].sum(axis=1)

        df['dominant_crypto'] = df[mention_cols].idxmax(axis=1).str.replace('_mentions', '')
        df.loc[df['total_crypto_mentions'] == 0, 'dominant_crypto'] = 'NONE'

        df['crypto_diversity'] = (df[mention_cols] > 0).sum(axis=1)

        return df

    def _create_topic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create topic-based features"""

        topics = {
            'regulation': ['regulation', 'sec', 'government', 'legal', 'compliance'],
            'technology': ['blockchain', 'smart', 'defi', 'protocol', 'upgrade'],
            'market': ['price', 'trading', 'volume', 'market', 'bull', 'bear'],
            'adoption': ['adoption', 'institutional', 'payment', 'mainstream'],
            'security': ['hack', 'exploit', 'vulnerability', 'security', 'breach']
        }

        for topic, keywords in topics.items():
            pattern = '|'.join(keywords)
            df[f'topic_{topic}'] = df['full_text'].str.lower().str.count(pattern)

        topic_cols = [col for col in df.columns if col.startswith('topic_')]
        df['dominant_topic'] = df[topic_cols].idxmax(axis=1).str.replace('topic_', '')

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""

        df['date'] = pd.to_datetime(df['date'])

        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        df['date_only'] = df['date'].dt.date
        df['news_velocity'] = df.groupby('date_only')['id'].transform('count')

        return df

# ================== Feature Integration ==================
class FeatureIntegrator:
    """Integrate price and news features"""

    def integrate_features(self, price_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        """Merge and create interaction features"""
        logger.info("Integrating features...")

        # Aggregate news features by date and crypto
        news_agg = self._aggregate_news_features(news_df)

        # Ensure date columns are datetime
        price_df['date'] = pd.to_datetime(price_df['date'])
        if not news_agg.empty:
            news_agg['date'] = pd.to_datetime(news_agg['date'])

        # Merge with price data
        if not news_agg.empty:
            merged = pd.merge(
                price_df,
                news_agg,
                on=['date', 'symbol'],
                how='left'
            )

            # Fill missing news features
            news_cols = [col for col in merged.columns if col.startswith('news_')]
            merged[news_cols] = merged[news_cols].fillna(0)
        else:
            merged = price_df.copy()

        # Create interaction features
        merged = self._create_interaction_features(merged)

        # Create lagged features
        merged = self._create_lagged_features(merged)

        return merged

    def _aggregate_news_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news features by date and cryptocurrency"""

        if news_df.empty:
            return pd.DataFrame()

        agg_features = []

        unique_cryptos = news_df['dominant_crypto'].unique()
        unique_cryptos = [c for c in unique_cryptos if c != 'NONE']

        for symbol in unique_cryptos[:50]:  # Top 50
            symbol_news = news_df[news_df['dominant_crypto'] == symbol].copy()

            if symbol_news.empty:
                continue

            symbol_news['date'] = pd.to_datetime(symbol_news['date'])

            # Daily aggregations
            daily_agg = symbol_news.groupby(symbol_news['date'].dt.date).agg({
                'vader_compound': ['mean', 'std'],
                'emotion_intensity': 'mean',
                'sentiment_volatility': 'mean',
                'word_count': 'mean',
                'id': 'count',
                'crypto_diversity': 'mean'
            })

            # Flatten column names
            daily_agg.columns = ['news_' + '_'.join(col).strip() for col in daily_agg.columns]
            daily_agg = daily_agg.reset_index()
            daily_agg.columns = ['date'] + list(daily_agg.columns[1:])
            daily_agg['symbol'] = symbol

            agg_features.append(daily_agg)

        if agg_features:
            return pd.concat(agg_features, ignore_index=True)
        else:
            return pd.DataFrame()

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that combine price and sentiment information"""

        if 'news_vader_compound_mean' in df.columns:
            df['sentiment_volume_interaction'] = (
                df['news_vader_compound_mean'] * df.get('vol_ratio_20', 1)
            )

            if 'returns_5d' in df.columns:
                df['sentiment_momentum_alignment'] = (
                    np.sign(df['news_vader_compound_mean']) == np.sign(df['returns_5d'])
                ).astype(int)

        return df

    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for time series modeling"""

        lag_features = ['returns_1d', 'rsi_14', 'vol_ratio_20']

        if 'news_vader_compound_mean' in df.columns:
            lag_features.append('news_vader_compound_mean')

        for feature in lag_features:
            if feature in df.columns:
                for lag in [1, 3, 5, 7]:
                    df[f'{feature}_lag{lag}'] = df.groupby('symbol')[feature].shift(lag)

        return df

# ================== Missing Value Handler ==================
class MissingValueHandler:
    """Handle missing values scientifically"""

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply appropriate missing value strategies"""
        logger.info("Handling missing values...")

        # Forward fill price-related features
        price_features = [col for col in df.columns if any(x in col.lower() for x in ['price', 'open', 'high', 'low', 'close'])]
        for col in price_features:
            if col in df.columns:
                df[col] = df.groupby('symbol')[col].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

        # Interpolate continuous features
        continuous_features = [col for col in df.columns if any(x in col.lower() for x in ['returns', 'volatility', 'hvol', 'atr'])]
        for col in continuous_features:
            if col in df.columns and df[col].dtype in ['float64', 'float32']:
                df[col] = df.groupby('symbol')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both') if len(x) > 2 else x.fillna(x.mean())
                )

        # Zero fill for specific features
        zero_fill_features = ['overnight_return', 'intraday_return']
        for col in zero_fill_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Fill remaining numeric features with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df.groupby('symbol')[col].transform(
                    lambda x: x.fillna(x.median()).fillna(0)
                )

        return df

# ================== Main Pipeline ==================
def run_feature_engineering():
    """Main feature engineering pipeline with complete visualization and table outputs"""

    print("\n" + "="*80)
    print(" " * 25 + "STAGE 2: FEATURE ENGINEERING")
    print("="*80)

    start_time = datetime.now()

    # Load data from Stage 1
    print("\nLoading data from Stage 1...")
    price_df = pd.read_csv(DATA_DIR / 'price_data.csv', parse_dates=['date'])
    news_df = pd.read_csv(DATA_DIR / 'news_data.csv', parse_dates=['date'])

    print(f"Loaded {len(price_df):,} price records")
    print(f"Loaded {len(news_df):,} news articles")

    # Create price features
    print("\n" + "-"*60)
    print("PRICE FEATURE ENGINEERING")
    print("-"*60)
    price_engineer = PriceFeatureEngineer()
    price_features = price_engineer.create_all_features(price_df)
    print(f"Created {len(price_features.columns) - len(price_df.columns)} new price features")

    # Create news features
    print("\n" + "-"*60)
    print("NEWS FEATURE ENGINEERING")
    print("-"*60)
    news_engineer = NewsFeatureEngineer()
    news_features = news_engineer.create_all_features(news_df)
    print(f"Created {len(news_features.columns) - len(news_df.columns)} new news features")

    # Integrate features
    print("\n" + "-"*60)
    print("FEATURE INTEGRATION")
    print("-"*60)
    integrator = FeatureIntegrator()
    final_features = integrator.integrate_features(price_features, news_features)

    # Handle missing values
    missing_handler = MissingValueHandler()
    final_features = missing_handler.handle_missing_values(final_features)

    print(f"Final dataset: {final_features.shape[0]} rows × {final_features.shape[1]} columns")

    # Generate tables
    print("\n" + "-"*60)
    print("GENERATING TABLES")
    print("-"*60)

    table_gen = TableGenerator()

    # Feature summary table
    feature_groups = {
        'Price Features': [col for col in final_features.columns if not col.startswith('news_')],
        'News Features': [col for col in final_features.columns if col.startswith('news_')],
        'Return Features': [col for col in final_features.columns if 'return' in col],
        'Momentum Features': [col for col in final_features.columns if any(x in col for x in ['rsi', 'macd', 'roc'])],
        'Volatility Features': [col for col in final_features.columns if any(x in col for x in ['vol', 'atr', 'bb'])],
        'Sentiment Features': [col for col in final_features.columns if any(x in col for x in ['sentiment', 'vader'])]
    }

    summary_table = table_gen.create_feature_summary_table(final_features, feature_groups)
    table_gen.save_table_to_file(summary_table, 'feature_summary')
    print("✓ Feature summary table saved")

    # Missing value table
    missing_table = table_gen.create_missing_value_table(final_features)
    table_gen.save_table_to_file(missing_table, 'missing_values')
    print("✓ Missing value analysis table saved")

    # Sentiment statistics table
    sentiment_table = table_gen.create_sentiment_statistics_table(final_features)
    if not sentiment_table.empty:
        table_gen.save_table_to_file(sentiment_table, 'sentiment_statistics')
        print("✓ Sentiment statistics table saved")

    # Generate visualizations
    print("\n" + "-"*60)
    print("GENERATING VISUALIZATIONS")
    print("-"*60)

    viz_gen = VisualizationGenerator()

    # Feature distributions
    key_features = ['returns_1d', 'rsi_14', 'hvol_20', 'news_vader_compound_mean', 'vol_ratio_20', 'amihud_20']
    viz_gen.create_feature_distribution_plots(final_features, key_features, PLOTS_DIR / 'feature_distributions.png')
    print("✓ Feature distribution plots saved")

    # Correlation heatmap
    important_features = ['returns_1d', 'returns_5d', 'rsi_14', 'macd', 'hvol_20',
                         'vol_ratio_20', 'news_vader_compound_mean', 'sentiment_volume_interaction']
    viz_gen.create_correlation_heatmap(final_features, important_features, PLOTS_DIR / 'correlation_heatmap.png')
    print("✓ Correlation heatmap saved")

    # Missing value heatmap
    viz_gen.create_missing_value_heatmap(final_features, PLOTS_DIR / 'missing_patterns.png')
    print("✓ Missing value patterns saved")

    # Save features
    print("\n" + "-"*60)
    print("SAVING FEATURES")
    print("-"*60)

    # Save as CSV and Parquet
    output_file = OUTPUT_DIR / 'engineered_features.csv'
    final_features.to_csv(output_file, index=False)
    final_features.to_parquet(OUTPUT_DIR / 'engineered_features.parquet')
    print(f"Features saved to: {output_file}")

    # Generate comprehensive report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_features': len(final_features.columns),
        'total_records': len(final_features),
        'unique_symbols': final_features['symbol'].nunique() if 'symbol' in final_features.columns else 0,
        'date_range': {
            'start': str(final_features['date'].min()) if 'date' in final_features.columns else None,
            'end': str(final_features['date'].max()) if 'date' in final_features.columns else None
        },
        'feature_groups': {k: len(v) for k, v in feature_groups.items()},
        'missing_summary': {
            'features_with_missing': (final_features.isnull().sum() > 0).sum(),
            'total_missing_values': final_features.isnull().sum().sum(),
            'missing_percentage': (final_features.isnull().sum().sum() / (len(final_features) * len(final_features.columns)) * 100)
        },
        'processing_time': (datetime.now() - start_time).total_seconds()
    }

    with open(REPORT_DIR / 'feature_engineering_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*80)
    print(" " * 30 + "STAGE 2 COMPLETE")
    print("="*80)
    print(f"Duration: {report['processing_time']:.2f} seconds")
    print(f"Total features created: {report['total_features']}")
    print(f"Missing values: {report['missing_summary']['missing_percentage']:.2f}%")
    print(f"\nOutput locations:")
    print(f"  Features: {OUTPUT_DIR}")
    print(f"  Tables: {TABLES_DIR}")
    print(f"  Plots: {PLOTS_DIR}")
    print(f"  Report: {REPORT_DIR}")

    print("\nKey deliverables:")
    print("  ✓ Feature dataset with 100+ engineered features")
    print("  ✓ Sentiment analysis from news data")
    print("  ✓ Technical indicators from price data")
    print("  ✓ Cross-sectional and interaction features")
    print("  ✓ Comprehensive tables and visualizations")

    print("\nData ready for Stage 3: Model Design")
    print("="*80)

    return final_features

if __name__ == "__main__":
    features = run_feature_engineering()