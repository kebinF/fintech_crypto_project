"""
Stage 1: Complete Data Collection Pipeline
Cryptocurrency price and news data collection with incremental updates
Date Range: 2021-08-01 to 2025-08-06
Target: 200 cryptocurrencies
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import sqlite3
from dataclasses import dataclass
import hashlib
import time

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
log_dir = PROJECT_ROOT / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
try:
    import config
    API_KEY = config.COINDESK_API_KEY
    START_DATE = config.START_DATE
    END_DATE = config.END_DATE
    TOP_N_COINS = config.TOP_N_COINS
    DB_PATH = PROJECT_ROOT / config.DB_PATH
except ImportError:
    # Default configuration
    API_KEY = "340031ae9d60a76b565ef5473187110a1982bfeb99bc1b6ee73545f7aa694446"
    START_DATE = "2021-08-01"
    END_DATE = "2025-08-06"
    TOP_N_COINS = 200
    DB_PATH = PROJECT_ROOT / "data" / "cache" / "crypto_cache.db"

# Ensure directories exist
for dir_path in ["data/cache", "data/processed", "data/raw", "logs"]:
    (PROJECT_ROOT / dir_path).mkdir(parents=True, exist_ok=True)

# ============== Data Classes ==============
@dataclass
class CollectionCheckpoint:
    """Checkpoint for resuming data collection"""
    timestamp: str
    symbols_completed: List[str]
    symbols_pending: List[str]
    news_last_date: Optional[str]
    price_last_date: Optional[str]
    total_price_records: int
    total_news_records: int
    errors: List[Dict]

@dataclass
class CollectionStats:
    """Statistics for collection tracking"""
    start_time: datetime
    api_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    price_records: int = 0
    news_records: int = 0

    @property
    def success_rate(self) -> float:
        if self.api_calls == 0:
            return 0.0
        return self.successful_calls / self.api_calls

# ============== Database Manager ==============
class DatabaseManager:
    """Database manager with incremental update support"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
        self.checkpoint_file = self.db_path.parent / 'collection_checkpoint.json'

    def _initialize_database(self):
        """Initialize database schema with optimized structure"""
        with sqlite3.connect(self.db_path) as conn:
            # Price data table with REAL types for large numbers
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    usd_volume REAL,
                    btc_volume REAL,
                    usd_volume_mil REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date)
                )
            """)

            # News data table with all required fields
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_data (
                    date TEXT NOT NULL,
                    id TEXT PRIMARY KEY,
                    published_on TEXT,
                    title TEXT,
                    body TEXT,
                    keywords TEXT,
                    lang TEXT,
                    positive REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Collection progress tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_type TEXT,
                    symbol TEXT,
                    date_start TEXT,
                    date_end TEXT,
                    records_collected INTEGER,
                    status TEXT,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price_date ON price_data(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price_symbol ON price_data(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_date ON news_data(date)")

            # Optimize database settings
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")

    def save_price_data_batch(self, df: pd.DataFrame, batch_size: int = 1000):
        """Save price data in batches with consistent date format"""
        if df.empty:
            return 0

        df = df.copy()

        # Ensure consistent date format (ISO format without time for price data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        df['created_at'] = datetime.now().isoformat()

        # Convert numeric columns to float to avoid overflow
        numeric_columns = ['open', 'high', 'low', 'close', 'usd_volume', 'btc_volume', 'usd_volume_mil']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

        total_saved = 0
        with sqlite3.connect(self.db_path) as conn:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]

                for _, row in batch.iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO price_data 
                        (symbol, date, open, high, low, close, usd_volume, btc_volume, usd_volume_mil, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (row['symbol'], row['date'], row['open'], row['high'],
                          row['low'], row['close'], row['usd_volume'], row['btc_volume'],
                          row['usd_volume_mil'], row['created_at']))

                conn.commit()
                total_saved += len(batch)

        return total_saved

    def save_news_data_batch(self, df: pd.DataFrame, batch_size: int = 500):
        """Save news data in batches with consistent date format"""
        if df.empty:
            return 0

        df = df.copy()

        # Ensure consistent date format (ISO format with time for news data)
        if 'date' in df.columns:
            # Convert to datetime first, then to consistent string format
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')

        df['created_at'] = datetime.now().isoformat()

        total_saved = 0
        with sqlite3.connect(self.db_path) as conn:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]

                for _, row in batch.iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO news_data 
                        (date, id, published_on, title, body, keywords, lang, positive, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (row['date'], row['id'], row['published_on'],
                          row['title'], row['body'], row['keywords'],
                          row['lang'], row['positive'], row['created_at']))

                conn.commit()
                total_saved += len(batch)

        return total_saved

    def load_price_data(self) -> pd.DataFrame:
        """Load all price data from database with robust date parsing"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM price_data ORDER BY symbol, date"
            df = pd.read_sql_query(query, conn)

            if not df.empty:
                # Robust date parsing that handles multiple formats
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

                # Remove any rows where date parsing failed
                if df['date'].isnull().any():
                    logger.warning(f"Removing {df['date'].isnull().sum()} rows with invalid dates")
                    df = df[df['date'].notna()]

                # Ensure numeric columns are float type
                numeric_cols = ['open', 'high', 'low', 'close', 'usd_volume', 'btc_volume', 'usd_volume_mil']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

    def load_news_data(self) -> pd.DataFrame:
        """Load all news data from database with robust date parsing"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM news_data ORDER BY date DESC"
            df = pd.read_sql_query(query, conn)

            if not df.empty:
                # Robust date parsing that handles multiple formats
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

                # Remove any rows where date parsing failed
                if df['date'].isnull().any():
                    logger.warning(f"Removing {df['date'].isnull().sum()} rows with invalid dates")
                    df = df[df['date'].notna()]

            return df

    def get_missing_symbols(self, symbols: List[str], start_date: str, end_date: str) -> List[str]:
        """Get symbols that need data collection"""
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ','.join(['?' for _ in symbols])
            cursor = conn.execute(f"""
                SELECT symbol, COUNT(*) as record_count
                FROM price_data 
                WHERE symbol IN ({placeholders}) 
                AND date >= ? AND date <= ?
                GROUP BY symbol
            """, symbols + [start_date, end_date])

            existing_data = dict(cursor.fetchall())

            # Calculate expected days
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            expected_days = (end_dt - start_dt).days + 1

            # Return symbols with insufficient data (less than 80% of expected days)
            missing_symbols = []
            for symbol in symbols:
                existing_count = existing_data.get(symbol, 0)
                if existing_count < (expected_days * 0.8):
                    missing_symbols.append(symbol)

            return missing_symbols

    def save_checkpoint(self, checkpoint: CollectionCheckpoint):
        """Save collection checkpoint for recovery"""
        checkpoint_dict = {
            'timestamp': checkpoint.timestamp,
            'symbols_completed': checkpoint.symbols_completed,
            'symbols_pending': checkpoint.symbols_pending,
            'news_last_date': checkpoint.news_last_date,
            'price_last_date': checkpoint.price_last_date,
            'total_price_records': checkpoint.total_price_records,
            'total_news_records': checkpoint.total_news_records,
            'errors': checkpoint.errors
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_dict, f, indent=2)

    def load_checkpoint(self) -> Optional[CollectionCheckpoint]:
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)

                checkpoint = CollectionCheckpoint(**data)
                logger.info(f"Checkpoint loaded: {len(checkpoint.symbols_completed)} symbols already completed")
                return checkpoint
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
        return None

    def clear_checkpoint(self):
        """Clear checkpoint after successful completion"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleared")

    def update_collection_progress(self, collection_type: str, symbol: str,
                                  start_date: str, end_date: str,
                                  records: int, status: str, error: str = None):
        """Update collection progress tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO collection_progress 
                (collection_type, symbol, date_start, date_end, records_collected, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (collection_type, symbol, start_date, end_date, records, status, error))
            conn.commit()

# ============== Price Data Collector ==============
class PriceDataCollector:
    """Cryptocurrency price data collector with checkpoint support"""

    def __init__(self, api_key: str, db_manager: DatabaseManager):
        self.api_key = api_key
        self.db = db_manager
        self.base_url = "https://data-api.coindesk.com"
        self.session = None
        self.semaphore = asyncio.Semaphore(20)
        self.stats = CollectionStats(start_time=datetime.now())

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=25, limit_per_host=20)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers={
                "authorization": f"Apikey {self.api_key}",
                "User-Agent": "CryptoPortfolioOptimizer/2.0"
            },
            connector=connector,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_top_cryptocurrencies(self, top_n: int = TOP_N_COINS) -> List[str]:
        """Fetch top N cryptocurrencies by market cap"""
        logger.info(f"Fetching top {top_n} cryptocurrencies")

        symbols = []
        pages_needed = (top_n + 99) // 100

        for page in range(1, pages_needed + 1):
            url = f"{self.base_url}/asset/v1/top/list"
            params = {
                'page': page,
                'page_size': 100,
                'sort_by': 'CIRCULATING_MKT_CAP_USD',
                'sort_direction': 'DESC'
            }

            async with self.semaphore:
                try:
                    async with self.session.get(url, params=params) as resp:
                        self.stats.api_calls += 1

                        if resp.status == 200:
                            data = await resp.json()
                            if "Data" in data and "LIST" in data["Data"]:
                                page_symbols = [coin["SYMBOL"] for coin in data["Data"]["LIST"]]
                                symbols.extend(page_symbols)
                                self.stats.successful_calls += 1

                                # Update progress
                                progress = (len(symbols) / top_n) * 100
                                print(f"\rFetching symbols: {progress:.1f}% ({len(symbols)}/{top_n})", end="", flush=True)
                        else:
                            self.stats.failed_calls += 1
                            logger.warning(f"Failed to fetch page {page}: HTTP {resp.status}")

                except Exception as e:
                    self.stats.failed_calls += 1
                    logger.error(f"Error fetching page {page}: {e}")

            await asyncio.sleep(0.1)

        print()  # New line after progress
        return symbols[:top_n]

    async def fetch_symbol_data(self, symbol: str, start_date: datetime,
                               end_date: datetime) -> pd.DataFrame:
        """Fetch price data for a single symbol"""
        days_needed = (end_date - start_date).days + 1

        url = f"{self.base_url}/index/cc/v1/historical/days"
        params = {
            'market': 'cadli',
            'instrument': f'{symbol}-USD',
            'limit': min(days_needed, 2000),
            'aggregate': 1
        }

        async with self.semaphore:
            try:
                async with self.session.get(url, params=params) as resp:
                    self.stats.api_calls += 1

                    if resp.status != 200:
                        self.stats.failed_calls += 1
                        return pd.DataFrame()

                    data = await resp.json()
                    if data.get('Response') == 'Error' or 'Data' not in data:
                        self.stats.failed_calls += 1
                        return pd.DataFrame()

                    df = pd.DataFrame(data['Data'])
                    if df.empty:
                        return df

                    # Process data
                    df['date'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
                    df['symbol'] = symbol
                    df['open'] = df['OPEN'].astype(float)
                    df['high'] = df['HIGH'].astype(float)
                    df['low'] = df['LOW'].astype(float)
                    df['close'] = df['CLOSE'].astype(float)
                    df['usd_volume'] = df.get('QUOTE_VOLUME', df.get('VOLUME', 0)).astype(float)
                    df['btc_volume'] = df.get('VOLUME', 0).astype(float)
                    df['usd_volume_mil'] = df['usd_volume'] / 1_000_000

                    # Filter by date range
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

                    # Select required columns
                    columns = ['symbol', 'date', 'open', 'high', 'low', 'close',
                              'usd_volume', 'btc_volume', 'usd_volume_mil']

                    self.stats.successful_calls += 1
                    return df[columns]

            except Exception as e:
                self.stats.failed_calls += 1
                logger.error(f"Error fetching {symbol}: {e}")
                return pd.DataFrame()

    async def collect_all_price_data(self, symbols: List[str],
                                    start_date: str, end_date: str) -> pd.DataFrame:
        """Collect price data for all symbols with checkpoint support"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Check for existing checkpoint
        checkpoint = self.db.load_checkpoint()

        if checkpoint:
            symbols_to_process = checkpoint.symbols_pending
            completed_symbols = checkpoint.symbols_completed
        else:
            # Check which symbols need data
            symbols_to_process = self.db.get_missing_symbols(symbols, start_date, end_date)
            completed_symbols = []

        if not symbols_to_process:
            logger.info("All price data already collected")
            return self.db.load_price_data()

        total_symbols = len(symbols_to_process) + len(completed_symbols)
        logger.info(f"Collecting data for {len(symbols_to_process)} symbols")

        # Process in batches
        batch_size = 10

        for i in range(0, len(symbols_to_process), batch_size):
            batch = symbols_to_process[i:i+batch_size]

            # Fetch batch data
            tasks = []
            for symbol in batch:
                task = self.fetch_symbol_data(symbol, start_dt, end_dt)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process and save results
            for symbol, result in zip(batch, results):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    saved = self.db.save_price_data_batch(result)
                    self.stats.price_records += saved

                    self.db.update_collection_progress(
                        'price', symbol, start_date, end_date,
                        len(result), 'success'
                    )
                else:
                    self.db.update_collection_progress(
                        'price', symbol, start_date, end_date,
                        0, 'failed', str(result) if isinstance(result, Exception) else 'No data'
                    )

                completed_symbols.append(symbol)

                # Update progress
                progress = (len(completed_symbols) / total_symbols) * 100
                success_rate = self.stats.success_rate * 100
                print(f"\rPrice collection: {progress:.1f}% | Success rate: {success_rate:.1f}% | Records: {self.stats.price_records:,}",
                      end="", flush=True)

            # Save checkpoint
            remaining_symbols = symbols_to_process[i+batch_size:]
            checkpoint = CollectionCheckpoint(
                timestamp=datetime.now().isoformat(),
                symbols_completed=completed_symbols,
                symbols_pending=remaining_symbols,
                news_last_date=None,
                price_last_date=end_date,
                total_price_records=self.stats.price_records,
                total_news_records=0,
                errors=[]
            )
            self.db.save_checkpoint(checkpoint)

            # Rate limiting
            await asyncio.sleep(0.2)

        print()  # New line after progress

        # Clear checkpoint after successful completion
        self.db.clear_checkpoint()

        return self.db.load_price_data()

# ============== News Data Collector ==============
class NewsDataCollector:
    """News data collector with optimized API usage"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.base_url = "https://data-api.coindesk.com/news/v1"
        self.session = None
        self.semaphore = asyncio.Semaphore(10)
        self.stats = CollectionStats(start_time=datetime.now())
        self.api_limit = 11000  # Free tier limit

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=15)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def calculate_sentiment_score(self, title: str, body: str) -> float:
        """Calculate simple sentiment score (0 to 1)"""
        text = f"{title} {body}".lower()

        # Sentiment keywords
        positive_words = [
            'bullish', 'surge', 'gain', 'rise', 'growth', 'positive', 'rally',
            'breakout', 'moon', 'pump', 'buy', 'uptrend', 'recovery', 'adoption',
            'success', 'milestone', 'breakthrough', 'strong', 'opportunity', 'profit',
            'innovation', 'upgrade', 'partnership', 'institutional', 'mainstream'
        ]

        negative_words = [
            'bearish', 'crash', 'drop', 'fall', 'decline', 'negative', 'dump',
            'breakdown', 'sell', 'downtrend', 'loss', 'risk', 'hack', 'scam',
            'failure', 'weak', 'warning', 'danger', 'fraud', 'regulatory', 'ban',
            'investigation', 'lawsuit', 'bubble', 'manipulation', 'crisis', 'fear'
        ]

        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        # Calculate score
        total = positive_count + negative_count
        if total == 0:
            return 0.5

        # Weighted score between 0.1 and 0.9
        return 0.1 + (positive_count / total * 0.8)

    async def fetch_news_batch(self, timestamp: int, limit: int = 100) -> List[Dict]:
        """Fetch a batch of news articles"""
        url = f"{self.base_url}/article/list"
        params = {
            'lang': 'EN',
            'to_ts': timestamp,
            'limit': limit
        }

        # Check API limit
        if self.stats.api_calls >= self.api_limit - 100:
            logger.warning(f"Approaching API limit ({self.stats.api_calls}/{self.api_limit})")
            return []

        async with self.semaphore:
            for attempt in range(3):  # Retry logic
                try:
                    async with self.session.get(url, params=params) as resp:
                        self.stats.api_calls += 1

                        if resp.status == 200:
                            data = await resp.json()
                            self.stats.successful_calls += 1
                            return self._parse_news_response(data)
                        elif resp.status == 429:  # Rate limited
                            wait_time = int(resp.headers.get('Retry-After', 60))
                            logger.warning(f"Rate limited, waiting {wait_time} seconds")
                            await asyncio.sleep(wait_time)
                        else:
                            self.stats.failed_calls += 1
                            logger.warning(f"API returned status {resp.status}")

                except asyncio.TimeoutError:
                    self.stats.failed_calls += 1
                    logger.warning(f"Request timeout (attempt {attempt + 1}/3)")
                except Exception as e:
                    self.stats.failed_calls += 1
                    logger.error(f"Error fetching news: {e}")

                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return []

    def _parse_news_response(self, data: Dict) -> List[Dict]:
        """Parse CoinDesk API response"""
        articles = []

        # CoinDesk uses uppercase field names
        if 'Data' in data and isinstance(data['Data'], list):
            for item in data['Data']:
                try:
                    article_id = str(item.get('ID', ''))
                    title = item.get('TITLE', '').strip()
                    body = item.get('BODY', '').strip()
                    published_on = item.get('PUBLISHED_ON', 0)

                    # Skip invalid articles
                    if not article_id or not title:
                        continue

                    # Use title as body if body is empty
                    if not body:
                        body = title

                    # Extract tags/keywords
                    tags = item.get('TAGS', [])
                    if isinstance(tags, list):
                        keywords = ', '.join(tags)
                    else:
                        keywords = str(tags)

                    articles.append({
                        'id': article_id,
                        'title': title[:500],
                        'body': body[:5000],
                        'published_on': str(published_on),
                        'keywords': keywords[:500],
                        'url': item.get('URL', ''),
                        'author': item.get('AUTHOR', ''),
                        'source': 'CoinDesk'
                    })

                except Exception as e:
                    logger.debug(f"Error parsing article: {e}")
                    continue

        return articles

    async def collect_news_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Collect news data with optimized strategy"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        logger.info(f"Collecting news from {start_date} to {end_date}")
        logger.info(f"API limit: {self.api_limit} calls")

        # Generate collection points - every 3 days, 2 times per day
        collection_points = []
        current_date = end_dt

        while current_date >= start_dt:
            # Collect at noon and evening
            for hour in [12, 20]:
                ts = int((current_date.replace(hour=hour)).timestamp())
                collection_points.append((current_date, ts))

            # Move to 3 days earlier
            current_date -= timedelta(days=3)

        total_points = len(collection_points)
        logger.info(f"Will collect from {total_points} time points")

        # Track unique articles
        seen_ids = set()
        all_articles = []
        successful_fetches = 0

        # Process in batches
        batch_size = 10

        for i in range(0, total_points, batch_size):
            batch = collection_points[i:i+batch_size]

            # Fetch batch data
            tasks = [self.fetch_news_batch(ts, limit=100) for _, ts in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for (date, _), result in zip(batch, results):
                if isinstance(result, list) and result:
                    successful_fetches += 1

                    for article in result:
                        article_id = article.get('id', '')

                        if article_id and article_id not in seen_ids:
                            seen_ids.add(article_id)

                            # Calculate sentiment
                            title = article.get('title', '')
                            body = article.get('body', '')
                            positive_score = self.calculate_sentiment_score(title, body)

                            # Prepare news item
                            news_item = {
                                'date': date.strftime('%Y-%m-%d %H:%M:%S'),
                                'id': article_id,
                                'published_on': article.get('published_on', ''),
                                'title': title,
                                'body': body,
                                'keywords': article.get('keywords', ''),
                                'lang': 'EN',
                                'positive': positive_score
                            }

                            all_articles.append(news_item)
                            self.stats.news_records += 1

            # Update progress
            progress = ((i + len(batch)) / total_points) * 100
            success_rate = (successful_fetches / max(i + len(batch), 1)) * 100
            print(f"\rNews collection: {progress:.1f}% | Success rate: {success_rate:.1f}% | Articles: {len(all_articles):,} | API calls: {self.stats.api_calls}",
                  end="", flush=True)

            # Save batch to database
            if len(all_articles) >= 1000:
                df_batch = pd.DataFrame(all_articles)
                self.db.save_news_data_batch(df_batch)
                all_articles = []  # Clear memory

            # Rate limiting
            await asyncio.sleep(0.5)

            # Check API limit
            if self.stats.api_calls >= self.api_limit - 100:
                logger.warning("API limit approaching, stopping collection")
                break

        print()  # New line after progress

        # Save remaining articles
        if all_articles:
            df = pd.DataFrame(all_articles)
            self.db.save_news_data_batch(df)

        return self.db.load_news_data()

# ============== Main Pipeline ==============
class DataCollectionPipeline:
    """Main data collection pipeline coordinator"""

    def __init__(self):
        self.db = DatabaseManager()
        self.start_time = datetime.now()

    async def run(self, start_date: str = START_DATE, end_date: str = END_DATE,
                  top_n: int = TOP_N_COINS) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Run complete data collection pipeline"""

        print(f"Period: {start_date} to {end_date}")
        print(f"Target: Top {top_n} cryptocurrencies")
        print(f"Database: {self.db.db_path}")
        print("="*80 + "\n")

        # Initialize statistics
        stats = {
            'start_time': self.start_time.isoformat(),
            'price_records': 0,
            'news_records': 0,
            'symbols_collected': 0,
            'total_api_calls': 0,
            'duration': 0
        }

        # Step 1: Price Data Collection
        print("STEP 1: PRICE DATA COLLECTION")
        print("-"*60)

        price_df = pd.DataFrame()
        async with PriceDataCollector(API_KEY, self.db) as collector:
            # Get top cryptocurrencies
            symbols = await collector.fetch_top_cryptocurrencies(top_n)
            stats['symbols_collected'] = len(symbols)
            print(f"Target symbols: {len(symbols)}")

            # Collect price data
            price_df = await collector.collect_all_price_data(symbols, start_date, end_date)
            stats['price_records'] = len(price_df)
            stats['price_api_calls'] = collector.stats.api_calls
            stats['price_success_rate'] = collector.stats.success_rate

        print(f"\nPrice collection complete: {len(price_df):,} records")

        # Step 2: News Data Collection
        print("\nSTEP 2: NEWS DATA COLLECTION")
        print("-"*60)

        news_df = pd.DataFrame()
        async with NewsDataCollector(self.db) as collector:
            news_df = await collector.collect_news_data(start_date, end_date)
            stats['news_records'] = len(news_df)
            stats['news_api_calls'] = collector.stats.api_calls
            stats['news_success_rate'] = collector.stats.success_rate

        print(f"\nNews collection complete: {len(news_df):,} articles")

        # Step 3: Export Data
        print("\nSTEP 3: EXPORTING DATA")
        print("-"*60)

        output_dir = PROJECT_ROOT / 'data' / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        price_file = output_dir / 'price_data.csv'
        price_df.to_csv(price_file, index=False)
        print(f"Price data saved: {price_file}")

        news_file = output_dir / 'news_data.csv'
        news_df.to_csv(news_file, index=False)
        print(f"News data saved: {news_file}")

        # Save as Parquet for better performance
        price_df.to_parquet(output_dir / 'price_data.parquet')
        news_df.to_parquet(output_dir / 'news_data.parquet')
        print("Parquet files saved for faster loading")

        # Calculate final statistics
        end_time = datetime.now()
        stats['duration'] = (end_time - self.start_time).total_seconds()
        stats['total_api_calls'] = stats.get('price_api_calls', 0) + stats.get('news_api_calls', 0)

        # Print summary
        print("\n" + "="*80)
        print(" " * 30 + "COLLECTION SUMMARY")
        print("="*80)
        print(f"Total Duration: {stats['duration']:.2f} seconds")
        print(f"Symbols Collected: {stats['symbols_collected']}")
        print(f"Price Records: {stats['price_records']:,}")
        print(f"News Articles: {stats['news_records']:,}")
        print(f"Total API Calls: {stats['total_api_calls']:,}")

        if not price_df.empty:
            print(f"\nPrice Data Quality:")
            print(f"  - Records per symbol: {len(price_df) / price_df['symbol'].nunique():.0f}")
            print(f"  - Missing values: {price_df.isnull().sum().sum()}")
            print(f"  - Date range: {price_df['date'].min()} to {price_df['date'].max()}")
            print(f"  - Success rate: {stats.get('price_success_rate', 0):.1%}")

        if not news_df.empty:
            # Calculate date statistics
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days_diff = (end_dt - start_dt).days + 1

            print(f"\nNews Data Quality:")
            print(f"  - Articles per day: {len(news_df) / days_diff:.1f}")
            print(f"  - Positive sentiment: {(news_df['positive'] > 0.5).mean():.1%}")
            print(f"  - Sentiment std: {news_df['positive'].std():.3f}")
            print(f"  - Success rate: {stats.get('news_success_rate', 0):.1%}")

        print("="*80 + "\n")

        # Save collection report
        report = {
            'statistics': stats,
            'price_summary': {
                'total_records': len(price_df),
                'unique_symbols': price_df['symbol'].nunique() if not price_df.empty else 0,
                'date_range': {
                    'start': str(price_df['date'].min()) if not price_df.empty else None,
                    'end': str(price_df['date'].max()) if not price_df.empty else None
                }
            },
            'news_summary': {
                'total_articles': len(news_df),
                'sentiment_mean': float(news_df['positive'].mean()) if not news_df.empty else 0,
                'sentiment_std': float(news_df['positive'].std()) if not news_df.empty else 0
            }
        }

        with open(output_dir / 'collection_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print("Collection report saved")
        print("\nData ready for Stage 2: Feature Engineering")

        return price_df, news_df, stats

# ============== Main Entry Point ==============
def run():
    """Entry point for module execution"""
    asyncio.run(main())

async def main():
    """Main execution function"""
    pipeline = DataCollectionPipeline()
    return await pipeline.run()

if __name__ == "__main__":
    run()