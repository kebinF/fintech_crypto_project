"""
Stage 2: Feature Engineering Pipeline
Advanced feature creation from structured and unstructured data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
import json
from sklearn.preprocessing import StandardScaler, RobustScaler
import talib
from scipy import stats

# Try to import sentiment analysis libraries
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        print("\nCreating price features...")
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
                df.loc[mask, 'bb_width'] = (upper - lower) / middle
                df.loc[mask, 'bb_position'] = (
                    (sym_data['close'] - lower) / (upper - lower)
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
                df.loc[mask, f'vol_ratio_{period}'] = sym_data['usd_volume'] / vol_ma

            # Volume-price correlation
            df.loc[mask, 'vol_price_corr'] = (
                sym_data['usd_volume'].rolling(20).corr(sym_data['close'])
            )

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
                    np.log1p(sym_data['usd_volume']) - np.log1p(rolling_mean)
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
        """Create cross-sectional features"""
        logger.info("Creating cross-sectional features...")

        # Rank features within each date
        rank_features = ['returns_1d', 'returns_5d', 'rsi_14', 'hvol_20', 'vol_ratio_20']

        for feature in rank_features:
            if feature in df.columns:
                df[f'{feature}_rank'] = (
                    df.groupby('date')[feature]
                    .rank(pct=True, method='average')
                )

                # Z-score normalization
                df[f'{feature}_zscore'] = (
                    df.groupby('date')[feature]
                    .transform(lambda x: (x - x.mean()) / (x.std() + 1e-10))
                )

        # Market-relative features
        df['excess_return'] = (
            df['returns_1d'] - df.groupby('date')['returns_1d'].transform('mean')
        )

        df['relative_volume'] = (
            df['usd_volume'] / df.groupby('date')['usd_volume'].transform('mean')
        )

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

        # Cryptocurrency entities for matching
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
            'hodl': 2.0, 'rally': 2.0, 'surge': 2.0, 'breakout': 2.0,
            'adoption': 2.0, 'institutional': 1.5,

            # Negative terms
            'dump': -2.5, 'crash': -3.0, 'bearish': -2.5, 'rekt': -3.0,
            'rugpull': -3.0, 'scam': -3.0, 'hack': -2.5,
            'fud': -2.0, 'panic': -2.5, 'plunge': -2.5,

            # Neutral/Technical
            'blockchain': 0.5, 'defi': 0.5, 'nft': 0.5
        }

        self.sia.lexicon.update(crypto_lexicon)

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all news-based features"""
        print("\nCreating news features...")

        # Ensure we have required columns
        if 'title' not in df.columns:
            df['title'] = ''
        if 'body' not in df.columns:
            df['body'] = ''

        # Combine title and body
        df['full_text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')

        # Basic text features
        df = self._create_text_stats(df)

        # Enhanced sentiment features
        df = self._create_sentiment_features(df)

        # Entity recognition
        df = self._create_entity_features(df)

        # Topic features
        df = self._create_topic_features(df)

        # Temporal features
        df = self._create_temporal_features(df)

        return df

    def _create_text_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic text statistics"""
        df['text_length'] = df['full_text'].str.len()
        df['word_count'] = df['full_text'].str.split().str.len()
        df['sentence_count'] = df['full_text'].str.count('[.!?]+')

        # Readability metrics
        df['exclamation_count'] = df['full_text'].str.count('!')
        df['question_count'] = df['full_text'].str.count('\\?')
        df['uppercase_ratio'] = df['full_text'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
        )

        return df

    def _create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced sentiment features"""

        if VADER_AVAILABLE and self.sia:
            # VADER sentiment
            vader_scores = df['full_text'].apply(lambda x: self.sia.polarity_scores(x))
            df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
            df['vader_positive'] = vader_scores.apply(lambda x: x['pos'])
            df['vader_negative'] = vader_scores.apply(lambda x: x['neg'])
            df['vader_neutral'] = vader_scores.apply(lambda x: x['neu'])
        else:
            # Use the simple sentiment from Stage 1
            df['vader_compound'] = df['positive'] * 2 - 1  # Convert 0-1 to -1 to 1
            df['vader_positive'] = df['positive']
            df['vader_negative'] = 1 - df['positive']
            df['vader_neutral'] = 0.5

        # Sentiment intensity
        df['emotion_intensity'] = np.abs(df['vader_compound'])

        # Sentiment consistency (if multiple sentences)
        def sent_volatility(text):
            if not text:
                return 0
            sentences = text.split('.')
            if len(sentences) < 2:
                return 0
            if VADER_AVAILABLE and self.sia:
                sentiments = [self.sia.polarity_scores(s)['compound'] for s in sentences if s.strip()]
            else:
                sentiments = [0.5] * len(sentences)  # Neutral if no VADER
            return np.std(sentiments) if len(sentiments) > 1 else 0

        df['sentiment_volatility'] = df['full_text'].apply(sent_volatility)

        return df

    def _create_entity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cryptocurrency entity features"""

        # Count mentions of each cryptocurrency
        for crypto, keywords in self.crypto_entities.items():
            pattern = '|'.join(keywords)
            df[f'{crypto}_mentions'] = df['full_text'].str.lower().str.count(pattern)

        # Total crypto mentions
        mention_cols = [col for col in df.columns if col.endswith('_mentions')]
        df['total_crypto_mentions'] = df[mention_cols].sum(axis=1)

        # Dominant cryptocurrency mentioned
        df['dominant_crypto'] = df[mention_cols].idxmax(axis=1).str.replace('_mentions', '')
        df.loc[df['total_crypto_mentions'] == 0, 'dominant_crypto'] = 'NONE'

        # Number of different cryptocurrencies mentioned
        df['crypto_diversity'] = (df[mention_cols] > 0).sum(axis=1)

        return df

    def _create_topic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create topic-based features"""

        # Keywords for different topics
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

        # Dominant topic
        topic_cols = [col for col in df.columns if col.startswith('topic_')]
        df['dominant_topic'] = df[topic_cols].idxmax(axis=1).str.replace('topic_', '')

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # News velocity (articles per day)
        df['date_only'] = df['date'].dt.date
        df['news_velocity'] = df.groupby('date_only')['id'].transform('count')

        return df

# ================== Feature Integration ==================
class FeatureIntegrator:
    """Integrate price and news features"""

    def integrate_features(self, price_df: pd.DataFrame,
                          news_df: pd.DataFrame) -> pd.DataFrame:
        """Merge and create interaction features"""
        print("\nIntegrating features...")

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

            # Fill missing news features with zeros
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

        # Get unique dominant cryptos
        unique_cryptos = news_df['dominant_crypto'].unique()
        unique_cryptos = [c for c in unique_cryptos if c != 'NONE']

        for symbol in unique_cryptos[:50]:  # Limit to top 50 for performance
            symbol_news = news_df[news_df['dominant_crypto'] == symbol].copy()

            if symbol_news.empty:
                continue

            # Ensure date is datetime
            symbol_news['date'] = pd.to_datetime(symbol_news['date'])

            # Daily aggregations
            daily_agg = symbol_news.groupby(symbol_news['date'].dt.date).agg({
                'vader_compound': ['mean', 'std'],
                'emotion_intensity': 'mean',
                'sentiment_volatility': 'mean',
                'word_count': 'mean',
                'id': 'count',  # Number of articles
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

        # Sentiment-volume interaction
        if 'news_vader_compound_mean' in df.columns:
            df['sentiment_volume_interaction'] = (
                df['news_vader_compound_mean'] * df.get('vol_ratio_20', 1)
            )

            # Sentiment-momentum alignment
            df['sentiment_momentum_alignment'] = (
                np.sign(df['news_vader_compound_mean']) == np.sign(df.get('returns_5d', 0))
            ).astype(int)

            # High sentiment with high volume
            if 'vol_ratio_20' in df.columns:
                df['high_sentiment_high_volume'] = (
                    (df['news_vader_compound_mean'] > df['news_vader_compound_mean'].quantile(0.8)) &
                    (df['vol_ratio_20'] > 1.5)
                ).astype(int)

        return df

    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for time series modeling"""

        lag_features = ['returns_1d', 'rsi_14', 'vol_ratio_20']

        # Add news features if available
        if 'news_vader_compound_mean' in df.columns:
            lag_features.append('news_vader_compound_mean')

        for feature in lag_features:
            if feature in df.columns:
                for lag in [1, 3, 5, 7]:
                    df[f'{feature}_lag{lag}'] = df.groupby('symbol')[feature].shift(lag)

        return df

# ================== Feature Quality Analysis ==================
class FeatureQualityAnalyzer:
    """Analyze and report on feature quality"""

    def analyze_features(self, df: pd.DataFrame) -> Dict:
        """Comprehensive feature quality analysis"""

        analysis = {
            'total_features': len(df.columns),
            'total_records': len(df),
            'unique_symbols': df['symbol'].nunique() if 'symbol' in df.columns else 0,
            'date_range': {
                'start': str(df['date'].min()) if 'date' in df.columns else None,
                'end': str(df['date'].max()) if 'date' in df.columns else None
            }
        }

        # Missing value analysis
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        analysis['missing_values'] = missing_pct[missing_pct > 0].to_dict()

        # Feature groups
        analysis['feature_groups'] = {
            'price_features': [col for col in df.columns if not col.startswith('news_')],
            'news_features': [col for col in df.columns if col.startswith('news_')],
            'return_features': [col for col in df.columns if 'return' in col],
            'momentum_features': [col for col in df.columns if any(x in col for x in ['rsi', 'macd', 'roc'])],
            'volatility_features': [col for col in df.columns if any(x in col for x in ['vol', 'atr', 'bb'])],
            'sentiment_features': [col for col in df.columns if any(x in col for x in ['sentiment', 'vader'])]
        }

        # Feature statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe()
            analysis['feature_statistics'] = stats_df.to_dict()

        return analysis

# ================== Main Pipeline ==================
def run_feature_engineering():
    """Main feature engineering pipeline"""

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

    # Integrate features
    print("\n" + "-"*60)
    print("FEATURE INTEGRATION")
    print("-"*60)
    integrator = FeatureIntegrator()
    final_features = integrator.integrate_features(price_features, news_features)
    print(f"Final dataset: {final_features.shape[0]} rows × {final_features.shape[1]} columns")

    # Analyze feature quality
    print("\n" + "-"*60)
    print("FEATURE QUALITY ANALYSIS")
    print("-"*60)
    analyzer = FeatureQualityAnalyzer()
    quality_report = analyzer.analyze_features(final_features)

    print(f"Total features created: {quality_report['total_features']}")
    print(f"Price features: {len(quality_report['feature_groups']['price_features'])}")
    print(f"News features: {len(quality_report['feature_groups']['news_features'])}")
    print(f"Sentiment features: {len(quality_report['feature_groups']['sentiment_features'])}")

    if quality_report['missing_values']:
        print(f"\nFeatures with missing values: {len(quality_report['missing_values'])}")
        for feat, pct in list(quality_report['missing_values'].items())[:5]:
            print(f"  - {feat}: {pct}%")

    # Save features
    print("\n" + "-"*60)
    print("SAVING FEATURES")
    print("-"*60)

    # Save as CSV
    output_file = OUTPUT_DIR / 'engineered_features.csv'
    final_features.to_csv(output_file, index=False)
    print(f"Features saved to: {output_file}")

    # Save as Parquet for better performance
    final_features.to_parquet(OUTPUT_DIR / 'engineered_features.parquet')
    print("Parquet file saved for faster loading")

    # Save feature report
    with open(OUTPUT_DIR / 'feature_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    print("Feature quality report saved")

    # Save news features separately for analysis
    news_features.to_csv(OUTPUT_DIR / 'news_features.csv', index=False)
    print("News features saved separately")

    # Print summary
    duration = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*80)
    print(" " * 30 + "FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nKey Statistics:")
    print(f"  - Total features: {quality_report['total_features']}")
    print(f"  - Records per symbol: {len(final_features) / final_features['symbol'].nunique():.0f}")
    print(f"  - Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")

    # Feature group breakdown
    print("\nFeature Breakdown:")
    for group, features in quality_report['feature_groups'].items():
        if features:
            print(f"  - {group}: {len(features)} features")

    print("\nData ready for Stage 3: Model Design")
    print("="*80)

    return final_features

if __name__ == "__main__":
    features = run_feature_engineering()