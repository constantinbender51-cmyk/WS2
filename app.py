import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import time
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import warnings
import os
import pickle
import json
warnings.filterwarnings('ignore')

class ETHDataProcessor:
    def __init__(self, data_dir: str = '/app/data/'):
        self.data_dir = data_dir
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _get_data_filename(self, start_date: str, end_date: str, timeframe: str) -> str:
        """Generate filename for cached data"""
        return os.path.join(self.data_dir, f'eth_ohlcv_{start_date}_{end_date}_{timeframe}.pkl')
    
    def _get_metadata_filename(self) -> str:
        """Get metadata filename"""
        return os.path.join(self.data_dir, 'data_metadata.json')
    
    def load_cached_data(self, start_date: str, end_date: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load cached OHLCV data if available"""
        filename = self._get_data_filename(start_date, end_date, timeframe)
        
        if os.path.exists(filename):
            print(f"Loading cached data from {filename}")
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                
                # Check if data is within requested date range
                if not data.empty:
                    data_start = data.index[0].strftime('%Y-%m-%d')
                    data_end = data.index[-1].strftime('%Y-%m-%d')
                    
                    # Check if cached data covers the requested period
                    requested_start = datetime.strptime(start_date, '%Y-%m-%d')
                    requested_end = datetime.strptime(end_date, '%Y-%m-%d')
                    actual_start = data.index[0]
                    actual_end = data.index[-1]
                    
                    if actual_start <= requested_start and actual_end >= requested_end:
                        print(f"Cached data covers {data_start} to {data_end}")
                        # Filter to exact requested range
                        mask = (data.index >= requested_start) & (data.index <= requested_end)
                        return data[mask]
                    else:
                        print(f"Cached data range ({data_start} to {data_end}) doesn't match request")
                        return None
                        
            except Exception as e:
                print(f"Error loading cached data: {e}")
                return None
        return None
    
    def save_data(self, data: pd.DataFrame, start_date: str, end_date: str, timeframe: str):
        """Save OHLCV data to cache"""
        filename = self._get_data_filename(start_date, end_date, timeframe)
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"Data saved to {filename}")
            
            # Update metadata
            self._update_metadata(filename, start_date, end_date, timeframe, len(data))
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def _update_metadata(self, filename: str, start_date: str, end_date: str, timeframe: str, num_rows: int):
        """Update metadata about saved data"""
        metadata_file = self._get_metadata_filename()
        metadata = {}
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except:
                metadata = {}
        
        key = f"{start_date}_{end_date}_{timeframe}"
        metadata[key] = {
            'filename': os.path.basename(filename),
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': timeframe,
            'num_rows': num_rows,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def fetch_ohlcv_data(self, start_date: str, end_date: str, timeframe: str = '30m', force_refresh: bool = False) -> pd.DataFrame:
        """Fetch OHLCV data from Binance or load from cache"""
        
        # Try to load cached data first
        if not force_refresh:
            cached_data = self.load_cached_data(start_date, end_date, timeframe)
            if cached_data is not None:
                return cached_data
        
        print(f"Fetching ETH/USDT data from {start_date} to {end_date} ({timeframe})")
        
        since = self.exchange.parse8601(start_date + ' 00:00:00')
        until = self.exchange.parse8601(end_date + ' 23:59:59')
        
        all_data = []
        current_since = since
        
        while current_since < until:
            try:
                ohlcv = self.exchange.fetch_ohlcv('ETH/USDT', timeframe, since=current_since, limit=1000)
                if not ohlcv:
                    break
                    
                all_data.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                
                print(f"Fetched {len(ohlcv)} candles, total: {len(all_data)}")
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error: {e}, retrying...")
                time.sleep(2)
        
        if not all_data:
            print("No data fetched from API. Using sample data for demonstration.")
            return self._create_sample_data(start_date, end_date, timeframe)
                
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Save to cache
        self.save_data(df, start_date, end_date, timeframe)
        
        return df
    
    def _create_sample_data(self, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
        """Create sample data when API fails"""
        print("Creating sample data for demonstration...")
        
        # Calculate number of periods
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if timeframe == '30m':
            freq = '30min'
        elif timeframe == '1h':
            freq = '1h'
        else:
            freq = '30min'
        
        dates = pd.date_range(start_dt, end_dt, freq=freq)
        
        # Create realistic-looking price data
        np.random.seed(42)
        base_price = 200
        returns = np.random.randn(len(dates)) * 0.02  # 2% daily volatility
        
        # Add some trend
        trend = np.linspace(0, 5, len(dates)) / 100  # 5% total trend
        
        prices = base_price * np.exp(np.cumsum(returns + trend))
        
        # Create OHLC data
        df = pd.DataFrame(index=dates)
        df['open'] = prices
        df['high'] = prices * (1 + np.random.random(len(dates)) * 0.02)
        df['low'] = prices * (1 - np.random.random(len(dates)) * 0.02)
        df['close'] = prices * (1 + np.random.randn(len(dates)) * 0.01)
        df['volume'] = np.random.lognormal(mean=10, sigma=1, size=len(dates))
        
        # Ensure high >= open, high >= close, low <= open, low <= close
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        # Save sample data
        self.save_data(df, start_date, end_date, timeframe)
        
        return df

class BucketSequenceGenerator:
    def __init__(self, k: float = 1.0):
        self.k = k
        
    def calculate_bucket_size(self, returns: pd.Series) -> float:
        """Calculate bucket size based on average absolute return"""
        avg_abs_return = returns.abs().mean()
        bucket_size = avg_abs_return * self.k
        return bucket_size
    
    def create_buckets(self, returns: pd.Series) -> Tuple[np.ndarray, float]:
        """Create bucket sequence from returns"""
        bucket_size = self.calculate_bucket_size(returns)
        
        # Create buckets: 1, 2, 3, ... based on return magnitude
        buckets = np.zeros(len(returns), dtype=int)
        
        for i, ret in enumerate(returns):
            abs_ret = abs(ret)
            if abs_ret == 0:
                bucket = 1
            else:
                bucket = int(np.ceil(abs_ret / bucket_size)) + 1
            buckets[i] = bucket
            
        return buckets, bucket_size
    
    def create_derivative_sequence(self, bucket_sequence: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Create derivative sequence (bucket - previous bucket) and mapping"""
        derivative = np.zeros(len(bucket_sequence), dtype=int)
        derivative[0] = 0  # First element is 0
        
        for i in range(1, len(bucket_sequence)):
            derivative[i] = bucket_sequence[i] - bucket_sequence[i-1]
        
        # Create mapping for negative numbers
        unique_values = np.unique(derivative)
        value_to_symbol = {}
        symbol_to_value = {}
        
        # Assign symbols: positive numbers keep their value, negatives get letters
        current_symbol = 0
        for value in unique_values:
            if value >= 0:
                symbol = str(value)
            else:
                # Use letters for negative values: -1 -> 'a', -2 -> 'b', etc.
                symbol = chr(ord('a') + abs(value) - 1)
            value_to_symbol[value] = symbol
            symbol_to_value[symbol] = value
            
        return derivative, {'value_to_symbol': value_to_symbol, 'symbol_to_value': symbol_to_value}
    
    def derivative_to_symbols(self, derivative_seq: np.ndarray, mapping: Dict) -> List[str]:
        """Convert derivative sequence to symbols"""
        symbols = []
        for value in derivative_seq:
            symbols.append(mapping['value_to_symbol'][value])
        return symbols
    
    def symbols_to_derivative(self, symbols: List[str], mapping: Dict) -> np.ndarray:
        """Convert symbols back to derivative sequence"""
        derivative = np.zeros(len(symbols), dtype=int)
        for i, symbol in enumerate(symbols):
            derivative[i] = mapping['symbol_to_value'][symbol]
        return derivative

class SubsequenceAnalyzer:
    def __init__(self, sequence: List[str], min_length: int = 2, max_length: int = 10):
        self.sequence = sequence
        self.min_length = min_length
        self.max_length = max_length
        self.subsequence_freq = self._calculate_subsequence_frequencies()
        self.completion_map = self._build_completion_map()
        
    def _calculate_subsequence_frequencies(self) -> Dict[str, int]:
        """Calculate frequencies of all subsequences"""
        # Convert sequence to string
        seq_str = ''.join(self.sequence)
        frequencies = {}
        
        for length in range(self.min_length, self.max_length + 1):
            for i in range(len(seq_str) - length + 1):
                substr = seq_str[i:i+length]
                frequencies[substr] = frequencies.get(substr, 0) + 1
                
        return frequencies
    
    def _build_completion_map(self) -> Dict[str, Dict[str, int]]:
        """Build map of incomplete sequences to possible completions"""
        completion_map = defaultdict(lambda: defaultdict(int))
        
        for subsequence, freq in self.subsequence_freq.items():
            for i in range(1, len(subsequence)):
                prefix = subsequence[:i]
                next_char = subsequence[i:i+1]
                completion_map[prefix][next_char] += freq
                
        return completion_map
    
    def get_completion(self, incomplete_seq: str) -> Tuple[str, int]:
        """Get most probable completion for incomplete sequence"""
        if incomplete_seq not in self.completion_map:
            return "", 0
            
        completions = self.completion_map[incomplete_seq]
        if not completions:
            return "", 0
            
        # Find completion with highest frequency
        best_completion = max(completions.items(), key=lambda x: x[1])
        return best_completion[0], best_completion[1]
    
    def get_completions(self, incomplete_seq: str, top_n: int = 3) -> List[Tuple[str, int]]:
        """Get top N completions for incomplete sequence"""
        if incomplete_seq not in self.completion_map:
            return []
            
        completions = self.completion_map[incomplete_seq]
        if not completions:
            return []
            
        # Get top N completions
        sorted_completions = sorted(completions.items(), key=lambda x: x[1], reverse=True)
        return sorted_completions[:top_n]

class BucketPredictor:
    def __init__(self, train_bucket_seq: np.ndarray, train_derivative_seq: np.ndarray, 
                 derivative_mapping: Dict):
        self.train_bucket_seq = train_bucket_seq.astype(int)
        self.train_derivative_seq = train_derivative_seq.astype(int)
        self.derivative_mapping = derivative_mapping
        
        # Convert sequences for analysis
        self.train_bucket_str = [str(x) for x in self.train_bucket_seq]
        self.train_derivative_symbols = [derivative_mapping['value_to_symbol'][x] 
                                         for x in self.train_derivative_seq]
        
    def predict_validation(self, val_bucket_seq: np.ndarray, val_derivative_seq: np.ndarray, 
                          k: float, seq_len: int, use_derivative: bool = True) -> Tuple[float, Dict]:
        """Test predictions on validation set with given parameters"""
        # Convert validation sequences
        val_bucket_str = [str(x) for x in val_bucket_seq.astype(int)]
        val_derivative_symbols = [self.derivative_mapping['value_to_symbol'][x] 
                                  for x in val_derivative_seq.astype(int)]
        
        # Create analyzers
        bucket_analyzer = SubsequenceAnalyzer(
            self.train_bucket_str, 
            min_length=seq_len, 
            max_length=seq_len + 3
        )
        
        derivative_analyzer = SubsequenceAnalyzer(
            self.train_derivative_symbols,
            min_length=seq_len,
            max_length=seq_len + 3
        )
        
        correct_predictions = 0
        total_predictions = 0
        predictions_log = []
        
        # Test predictions on validation sequence
        for i in range(len(val_bucket_seq) - seq_len):
            # Get incomplete sequence for buckets
            incomplete_bucket = ''.join(val_bucket_str[i:i+seq_len])
            
            # Get prediction for bucket sequence
            bucket_pred, bucket_freq = bucket_analyzer.get_completion(incomplete_bucket)
            
            if bucket_pred:
                # Convert prediction to compare with actual
                pred_value = int(bucket_pred)
                actual_value = val_bucket_seq[i+seq_len]
                
                is_correct = pred_value == actual_value
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                predictions_log.append({
                    'position': i,
                    'incomplete_seq': incomplete_bucket,
                    'prediction': pred_value,
                    'actual': int(actual_value),
                    'correct': is_correct,
                    'confidence': bucket_freq,
                    'type': 'bucket'
                })
            
            # Get prediction for derivative sequence if enabled
            if use_derivative and i < len(val_derivative_seq) - seq_len:
                incomplete_derivative = ''.join(val_derivative_symbols[i:i+seq_len])
                derivative_pred, derivative_freq = derivative_analyzer.get_completion(incomplete_derivative)
                
                if derivative_pred:
                    # Convert symbol back to value
                    pred_value = self.derivative_mapping['symbol_to_value'][derivative_pred]
                    actual_value = val_derivative_seq[i+seq_len]
                    
                    is_correct = pred_value == actual_value
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    predictions_log.append({
                        'position': i,
                        'incomplete_seq': incomplete_derivative,
                        'prediction': pred_value,
                        'actual': int(actual_value),
                        'correct': is_correct,
                        'confidence': derivative_freq,
                        'type': 'derivative'
                    })
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'predictions_log': predictions_log[:100]  # Keep only first 100 for memory
        }
        
        return accuracy, results

def main():
    # Initialize data processor
    processor = ETHDataProcessor(data_dir='/app/data/')
    
    print("=" * 60)
    print("ETH BUCKET SEQUENCE PREDICTION SYSTEM")
    print("=" * 60)
    
    # 1. Fetch and prepare data
    print("\nSTEP 1: Loading/Downloading OHLC data")
    print("-" * 40)
    
    # Check if data exists
    data_files = [f for f in os.listdir('/app/data/') if f.endswith('.pkl')]
    if data_files:
        print(f"Found {len(data_files)} cached data files")
        for file in data_files[:5]:  # Show first 5 files
            print(f"  {file}")
        if len(data_files) > 5:
            print(f"  ... and {len(data_files) - 5} more")
    
    # Fetch 30m data
    print("\nFetching 30m OHLC data for ETH/USDT (2020-2026)...")
    eth_data = processor.fetch_ohlcv_data('2020-01-01', '2026-01-01', '30m')
    
    print(f"\nTotal data points: {len(eth_data)}")
    print(f"Date range: {eth_data.index[0]} to {eth_data.index[-1]}")
    print(f"Columns: {list(eth_data.columns)}")
    
    # 2. Create two datasets: original and base 30 (starting at candle 2)
    print("\n" + "=" * 60)
    print("STEP 2: Creating datasets")
    print("=" * 60)
    
    # Original dataset
    original_data = eth_data.copy()
    
    # Base 30 dataset (starting at candle 2)
    base_30_data = eth_data.iloc[1:].copy().reset_index(drop=False)
    base_30_data.set_index('timestamp', inplace=True)
    
    print(f"Original dataset shape: {original_data.shape}")
    print(f"Base 30 dataset shape: {base_30_data.shape}")
    
    # 3. Resample both to 1h
    print("\n" + "=" * 60)
    print("STEP 3: Resampling to 1h")
    print("=" * 60)
    
    # Resample original data to 1h
    original_1h = original_data.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Resample base 30 data to 1h
    base_30_1h = base_30_data.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    print(f"Original 1h dataset shape: {original_1h.shape}")
    print(f"Base 30 1h dataset shape: {base_30_1h.shape}")
    
    # Split data into training (2020-2025) and validation (2025-2026)
    print("\n" + "=" * 60)
    print("STEP 4: Splitting data into training and validation")
    print("=" * 60)
    
    train_mask = original_1h.index < '2025-01-01'
    val_mask = (original_1h.index >= '2025-01-01') & (original_1h.index < '2026-01-01')
    
    train_data = original_1h[train_mask]
    val_data = original_1h[val_mask]
    
    print(f"Training data: {train_data.shape[0]} candles ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"Validation data: {val_data.shape[0]} candles ({val_data.index[0]} to {val_data.index[-1]})")
    
    # Calculate returns for training data
    train_returns = train_data['close'].pct_change().dropna()
    val_returns = val_data['close'].pct_change().dropna()
    
    print(f"\nTraining returns: mean={train_returns.mean():.6f}, std={train_returns.std():.6f}")
    print(f"Validation returns: mean={val_returns.mean():.6f}, std={val_returns.std():.6f}")
    
    # 4. Grid search for optimal k and sequence length
    print("\n" + "=" * 60)
    print("STEP 5: Grid search for optimal parameters")
    print("=" * 60)
    
    # Parameter grid
    k_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    seq_lengths = [2, 3, 4, 5, 6]
    
    best_accuracy = 0
    best_params = {}
    results = []
    
    # Generate validation sequences once (for efficiency)
    print("Generating validation bucket sequences...")
    val_bucket_sequences = {}
    val_derivative_sequences = {}
    val_mappings = {}
    
    for k in k_values:
        bucket_generator = BucketSequenceGenerator(k=k)
        
        # Create validation bucket sequence
        val_buckets, val_bucket_size = bucket_generator.create_buckets(val_returns)
        val_derivative, val_mapping = bucket_generator.create_derivative_sequence(val_buckets)
        
        val_bucket_sequences[k] = val_buckets
        val_derivative_sequences[k] = val_derivative
        val_mappings[k] = val_mapping
    
    # Grid search
    for k in k_values:
        bucket_generator = BucketSequenceGenerator(k=k)
        
        # Create bucket sequence for training data
        train_buckets, bucket_size = bucket_generator.create_buckets(train_returns)
        train_derivative, train_mapping = bucket_generator.create_derivative_sequence(train_buckets)
        
        # Get validation sequences for this k
        val_buckets = val_bucket_sequences[k]
        val_derivative = val_derivative_sequences[k]
        
        predictor = BucketPredictor(train_buckets, train_derivative, train_mapping)
        
        for seq_len in seq_lengths:
            # Skip if sequence length is too long
            if len(train_buckets) < seq_len * 10 or len(val_buckets) < seq_len * 2:
                continue
                
            try:
                accuracy, result_details = predictor.predict_validation(
                    val_buckets, 
                    val_derivative, 
                    k, 
                    seq_len,
                    use_derivative=True
                )
                
                results.append({
                    'k': k,
                    'seq_len': seq_len,
                    'accuracy': accuracy,
                    'bucket_size': bucket_size,
                    'correct': result_details['correct_predictions'],
                    'total': result_details['total_predictions']
                })
                
                print(f"k={k:.1f}, seq_len={seq_len}: accuracy={accuracy:.4f} ({result_details['correct_predictions']}/{result_details['total_predictions']})")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'k': k,
                        'seq_len': seq_len,
                        'accuracy': accuracy,
                        'bucket_size': bucket_size,
                        'correct': result_details['correct_predictions'],
                        'total': result_details['total_predictions'],
                        'details': result_details
                    }
                    
            except Exception as e:
                print(f"Error with k={k}, seq_len={seq_len}: {e}")
                continue
    
    # 5. Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if not results:
        print("No valid results found. Check your data and parameters.")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by accuracy
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print("\nTop 10 configurations:")
    print(results_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("WINNING CONFIGURATION")
    print("=" * 60)
    print(f"Best k value: {best_params['k']:.2f}")
    print(f"Best sequence length: {best_params['seq_len']}")
    print(f"Best accuracy: {best_params['accuracy']:.4%}")
    print(f"Correct predictions: {best_params['correct']}/{best_params['total']}")
    print(f"Bucket size: {best_params['bucket_size']:.6f}")
    
    # 6. Additional analysis
    print("\n" + "=" * 60)
    print("ADDITIONAL ANALYSIS")
    print("=" * 60)
    
    # Create bucket sequence with best k
    bucket_generator = BucketSequenceGenerator(k=best_params['k'])
    train_buckets, _ = bucket_generator.create_buckets(train_returns)
    train_derivative, train_mapping = bucket_generator.create_derivative_sequence(train_buckets)
    
    # Convert to symbols for analysis
    train_bucket_str = [str(x) for x in train_buckets]
    train_derivative_symbols = [train_mapping['value_to_symbol'][x] for x in train_derivative]
    
    # Analyze subsequence frequencies
    print("\nMost common bucket subsequences (length 3):")
    bucket_analyzer = SubsequenceAnalyzer(train_bucket_str, min_length=3, max_length=3)
    sorted_bucket_freq = sorted(bucket_analyzer.subsequence_freq.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
    
    for i, (seq, freq) in enumerate(sorted_bucket_freq[:10]):
        print(f"  {seq}: {freq} occurrences")
    
    print("\nMost common derivative subsequences (length 3):")
    derivative_analyzer = SubsequenceAnalyzer(train_derivative_symbols, min_length=3, max_length=3)
    sorted_derivative_freq = sorted(derivative_analyzer.subsequence_freq.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
    
    for i, (seq, freq) in enumerate(sorted_derivative_freq[:10]):
        print(f"  {seq}: {freq} occurrences")
    
    # Test some predictions
    print("\nSample predictions with best configuration:")
    
    # Get a sample from validation data
    if 'details' in best_params and best_params['details']['predictions_log']:
        sample_pred = best_params['details']['predictions_log'][0]
        print(f"Incomplete sequence: {sample_pred['incomplete_seq']}")
        print(f"Prediction: {sample_pred['prediction']}")
        print(f"Actual: {sample_pred['actual']}")
        print(f"Correct: {sample_pred['correct']}")
        print(f"Type: {sample_pred['type']}")
    
    # Save results
    results_file = os.path.join('/app/data/', 'prediction_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Save best configuration
    best_config_file = os.path.join('/app/data/', 'best_configuration.json')
    with open(best_config_file, 'w') as f:
        json.dump({
            'best_k': float(best_params['k']),
            'best_seq_len': int(best_params['seq_len']),
            'accuracy': float(best_params['accuracy']),
            'bucket_size': float(best_params['bucket_size']),
            'predictions': f"{best_params['correct']}/{best_params['total']}"
        }, f, indent=2)
    print(f"Best configuration saved to {best_config_file}")

if __name__ == "__main__":
    main()