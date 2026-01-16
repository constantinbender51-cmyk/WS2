import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import time
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class ETHDataProcessor:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
    def fetch_ohlcv_data(self, start_date: str, end_date: str, timeframe: str = '30m') -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
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
                
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
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
        
        # Create buckets: 0, bucket_size, 2*bucket_size, etc.
        buckets = np.zeros(len(returns))
        
        for i, ret in enumerate(returns):
            abs_ret = abs(ret)
            if abs_ret == 0:
                bucket = 1
            else:
                bucket = int(np.ceil(abs_ret / bucket_size)) + 1
            buckets[i] = bucket
            
        return buckets, bucket_size
    
    def create_derivative_sequence(self, bucket_sequence: np.ndarray) -> np.ndarray:
        """Create derivative sequence (bucket - previous bucket)"""
        derivative = np.zeros(len(bucket_sequence))
        derivative[0] = 0  # First element is 0
        
        for i in range(1, len(bucket_sequence)):
            derivative[i] = bucket_sequence[i] - bucket_sequence[i-1]
            
        return derivative

class SubsequenceAnalyzer:
    def __init__(self, sequence: np.ndarray, min_length: int = 2, max_length: int = 10):
        self.sequence = sequence
        self.min_length = min_length
        self.max_length = max_length
        self.subsequence_freq = self._calculate_subsequence_frequencies()
        self.completion_map = self._build_completion_map()
        
    def _calculate_subsequence_frequencies(self) -> Dict[str, int]:
        """Calculate frequencies of all subsequences"""
        seq_str = ''.join(str(int(x)) for x in self.sequence)
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

class BucketPredictor:
    def __init__(self, train_bucket_seq: np.ndarray, train_derivative_seq: np.ndarray):
        self.train_bucket_seq = train_bucket_seq
        self.train_derivative_seq = train_derivative_seq
        
    def predict_validation(self, k: float, seq_len: int) -> float:
        """Test predictions on validation set with given parameters"""
        # Create sequences for training
        train_bucket_str = ''.join(str(int(x)) for x in self.train_bucket_seq)
        train_derivative_str = ''.join(str(int(x)) for x in self.train_derivative_seq)
        
        # Analyze training sequences
        bucket_analyzer = SubsequenceAnalyzer(
            self.train_bucket_seq, 
            min_length=seq_len, 
            max_length=seq_len + 3
        )
        
        derivative_analyzer = SubsequenceAnalyzer(
            self.train_derivative_seq,
            min_length=seq_len,
            max_length=seq_len + 3
        )
        
        # For validation, we'll use a sliding window approach
        # In practice, you would have separate validation data
        # For this example, we'll use the last portion of training data
        val_size = len(self.train_bucket_seq) // 10
        val_seq = self.train_bucket_seq[-val_size:]
        val_derivative = self.train_derivative_seq[-val_size:]
        
        correct_predictions = 0
        total_predictions = 0
        
        # Test predictions on validation sequence
        for i in range(len(val_seq) - seq_len):
            # Get incomplete sequence
            incomplete_bucket = ''.join(str(int(x)) for x in val_seq[i:i+seq_len])
            incomplete_derivative = ''.join(str(int(x)) for x in val_derivative[i:i+seq_len])
            
            # Get predictions
            bucket_pred, _ = bucket_analyzer.get_completion(incomplete_bucket)
            derivative_pred, _ = derivative_analyzer.get_completion(incomplete_derivative)
            
            # Check if predictions are available
            if bucket_pred:
                # Convert prediction to compare with actual
                pred_value = int(bucket_pred)
                actual_value = int(val_seq[i+seq_len])
                
                if pred_value == actual_value:
                    correct_predictions += 1
                total_predictions += 1
                
            if derivative_pred and total_predictions < len(val_seq) - seq_len:
                # For derivative predictions
                pred_value = int(derivative_pred)
                actual_value = int(val_derivative[i+seq_len])
                
                if pred_value == actual_value:
                    correct_predictions += 0.5  # Half weight for derivative predictions
                total_predictions += 0.5
                
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy

def main():
    # Initialize data processor
    processor = ETHDataProcessor()
    
    # 1. Fetch and prepare data
    print("=" * 60)
    print("STEP 1: Fetching and preparing data")
    print("=" * 60)
    
    # Fetch 30m data
    eth_data = processor.fetch_ohlcv_data('2020-01-01', '2026-01-01', '30m')
    
    if eth_data.empty:
        print("No data fetched. Using sample data for demonstration.")
        # Create sample data for demonstration
        dates = pd.date_range('2020-01-01', '2026-01-01', freq='30min')
        np.random.seed(42)
        prices = np.random.lognormal(mean=0.0001, sigma=0.01, size=len(dates)).cumsum() + 200
        eth_data = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.random.random(len(dates)) * 0.02),
            'low': prices * (1 - np.random.random(len(dates)) * 0.02),
            'close': prices,
            'volume': np.random.random(len(dates)) * 10000
        }, index=dates)
    
    print(f"\nTotal data points: {len(eth_data)}")
    print(f"Date range: {eth_data.index[0]} to {eth_data.index[-1]}")
    
    # 2. Create two datasets: original and base 30 (starting at candle 2)
    print("\n" + "=" * 60)
    print("STEP 2: Creating datasets")
    print("=" * 60)
    
    # Original dataset
    original_data = eth_data.copy()
    
    # Base 30 dataset (starting at candle 2)
    base_30_data = eth_data.iloc[1:].copy().reset_index(drop=False)
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
    base_30_data.set_index('timestamp', inplace=True)
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
    
    # 4. Grid search for optimal k and sequence length
    print("\n" + "=" * 60)
    print("STEP 5: Grid search for optimal parameters")
    print("=" * 60)
    
    # Parameter grid
    k_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    seq_lengths = [2, 3, 4, 5, 6, 7, 8]
    
    best_accuracy = 0
    best_params = {}
    results = []
    
    bucket_generator = BucketSequenceGenerator()
    
    for k in k_values:
        bucket_generator.k = k
        
        # Create bucket sequence for training data
        train_buckets, bucket_size = bucket_generator.create_buckets(train_returns)
        train_derivative = bucket_generator.create_derivative_sequence(train_buckets)
        
        predictor = BucketPredictor(train_buckets, train_derivative)
        
        for seq_len in seq_lengths:
            # Skip if sequence length is too long for validation
            if len(train_buckets) < seq_len * 10:
                continue
                
            accuracy = predictor.predict_validation(k, seq_len)
            results.append({
                'k': k,
                'seq_len': seq_len,
                'accuracy': accuracy,
                'bucket_size': bucket_size
            })
            
            print(f"k={k:.1f}, seq_len={seq_len}: accuracy={accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'k': k,
                    'seq_len': seq_len,
                    'accuracy': accuracy,
                    'bucket_size': bucket_size
                }
    
    # 5. Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
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
    print(f"Bucket size: {best_params['bucket_size']:.6f}")
    
    # 6. Additional analysis
    print("\n" + "=" * 60)
    print("ADDITIONAL ANALYSIS")
    print("=" * 60)
    
    # Create bucket sequence with best k
    bucket_generator.k = best_params['k']
    train_buckets, _ = bucket_generator.create_buckets(train_returns)
    
    # Analyze subsequence frequencies
    analyzer = SubsequenceAnalyzer(train_buckets, min_length=2, max_length=5)
    
    print("\nMost common subsequences (length 3):")
    sorted_freq = sorted(analyzer.subsequence_freq.items(), 
                        key=lambda x: x[1], 
                        reverse=True)
    
    for i, (seq, freq) in enumerate(sorted_freq[:10]):
        if len(seq) == 3:
            print(f"  {seq}: {freq} occurrences")
    
    # Test some predictions
    print("\nSample predictions with best configuration:")
    sample_incomplete = ''.join(str(int(x)) for x in train_buckets[:best_params['seq_len']])
    completion, freq = analyzer.get_completion(sample_incomplete)
    
    if completion:
        print(f"Incomplete sequence: {sample_incomplete}")
        print(f"Most probable completion: {completion}")
        print(f"Frequency: {freq}")
    else:
        print("No completion found for sample sequence")

if __name__ == "__main__":
    main()