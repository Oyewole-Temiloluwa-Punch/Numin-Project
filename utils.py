"""
Complete utilities for Stock Market Pattern Clustering API
Built from the original Stock Market Dashboard utils
"""

import pandas as pd
import numpy as np
import json
import boto3
from typing import Dict, List, Tuple, Optional, Any
from statistics import mean
from datetime import datetime
from pathlib import Path

# Configuration constants
AWS_ACCESS_KEY_ID = "AKIARVSAETUO7JZ6FK5D"
AWS_SECRET_ACCESS_KEY = "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"
AWS_REGION = "eu-north-1"
BUCKET_NAME = "numin-cache-files"
DEFAULT_LOOKBACK = 1800
DEFAULT_PROJ_DAYS = 30
DEFAULT_REAL_BARS = 30
DEFAULT_WINDOW_SIZE = 50
DEFAULT_STEPS = 15
DEFAULT_GAP = 3

# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_data(path=None):
    """Load and process stock market data"""
    if path is None:
        # Use current directory
        path = Path(__file__).parent / "SPY Chart 2025-08-22-09-36.csv"
    
    df = pd.read_csv(path)
    df.columns = ["Date", "Open", "High", "Low", "Close"]
    # normalize dates to midnight to avoid equality bugs
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# ============================================================================
# AWS S3 UTILITIES
# ============================================================================

def get_s3_client():
    """Get configured S3 client"""
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

def save_cache_to_s3(results, filename):
    """Save pattern results to S3 cache"""
    s3 = get_s3_client()
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"cache/{filename}",
        Body=json.dumps(results, indent=4),
        ContentType="application/json"
    )
    return filename

def load_cache_from_s3(filename):
    """Load pattern results from S3 cache"""
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"cache/{filename}")
    return json.loads(obj["Body"].read().decode("utf-8"))

def cache_exists_in_s3(filename):
    """Check if cache file exists in S3"""
    try:
        s3 = get_s3_client()
        s3.head_object(Bucket=BUCKET_NAME, Key=f"cache/{filename}")
        return True
    except Exception:
        return False

# ============================================================================
# PATTERN UTILITIES
# ============================================================================

def is_repetitive_pattern(pattern: str) -> bool:
    """Check if a pattern is repetitive (all parts are the same)"""
    parts = pattern.split("/")
    return all(p == parts[0] for p in parts) if parts else False

def generate_pattern_at_index(df, idx, steps=15, gap=3):
    """Generate a pattern at a specific index"""
    patterns = []
    for i in range(steps):
        if idx - i - gap < 0:
            break

        barA = df.loc[idx - i - 1]
        barB = df.loc[idx - i - gap]
        midpoint_B = (barB["High"] + barB["Low"]) / 2

        pattern = []
        for col in ["Open", "High", "Low", "Close"]:
            valA, valB = barA[col], barB[col]
            if valA > valB and valA > midpoint_B:
                pattern.append("4")
            elif valA > valB and valA <= midpoint_B:
                pattern.append("3")
            elif valA < valB and valA < midpoint_B:
                pattern.append("1")
            elif valA < valB and valA >= midpoint_B:
                pattern.append("2")
            else:
                pattern.append("0")
        patterns.append("".join(pattern))
    return "/".join(patterns)

def find_pattern_occurrences(df, steps=15, gap=3, lookback=1800):
    """Find occurrences of a pattern in the dataframe"""
    latest_idx = df.index[-1]
    target_pattern = generate_pattern_at_index(df, latest_idx, steps=steps, gap=gap)

    matches = []
    start_idx = max(steps + gap, latest_idx - lookback)
    for idx in range(latest_idx - 1, start_idx - 1, -1):
        candidate_pattern = generate_pattern_at_index(df, idx, steps=steps, gap=gap)
        if candidate_pattern == target_pattern:
            matches.append(str(pd.to_datetime(df.loc[idx, "Date"]).date()))
    return target_pattern, matches

def generate_cache(df, current_date=None, lookback=DEFAULT_LOOKBACK):
    """Generate pattern cache for a given date"""
    results = {}

    if current_date:
        # normalize for robust equality
        current_date = pd.to_datetime(current_date).normalize()
        if current_date not in set(df["Date"]):
            # Instead of raising, return None so UI can show a friendly message
            return None
        latest_idx = df.index[df["Date"] == current_date][0]
    else:
        latest_idx = df.index[-1]
        current_date = df.loc[latest_idx, "Date"]

    for gap in range(2, 8):
        for steps in range(2, 8):
            target_pattern, matches = find_pattern_occurrences(
                df.iloc[: latest_idx + 1], steps=steps, gap=gap, lookback=lookback
            )

            if is_repetitive_pattern(target_pattern):
                continue

            results[f"gap_{gap}_steps_{steps}"] = {
                "target_pattern": target_pattern,
                "occurrences_count": len(matches),
                "occurrences": matches,
            }

    date_str = pd.to_datetime(current_date).strftime("%Y-%m-%d")
    filename = f"{date_str}_patterns_cache.json"
    save_cache_to_s3(results, filename)
    return filename

def analyze_current_day_pattern_outcomes(df, occurrence_dates, pattern_date: str, look_ahead=DEFAULT_PROJ_DAYS):
    """Analyze pattern outcomes for given occurrence dates"""
    df = df.copy()
    # work with normalized dates as strings for stable joins
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize().dt.strftime("%Y-%m-%d")
    occurrence_dates = [pd.to_datetime(d).normalize().strftime("%Y-%m-%d") for d in occurrence_dates]

    # Use the current directory for data file
    data_path = Path(__file__).parent / "SPY Chart 2025-08-22-09-36.csv"
    df_raw = pd.read_csv(data_path)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"]).dt.normalize()
    target_date = pd.to_datetime(pattern_date).normalize()
    base_price = float(df_raw.loc[df_raw["Date"] == target_date, "Close"].iloc[0])

    occurrence_indices = []
    for d in occurrence_dates:
        idx_list = df.index[df["Date"] == d].tolist()
        if idx_list:
            occurrence_indices.append(idx_list[0])

    result = {}
    for k in range(1, look_ahead + 1):
        deltas = []
        for idx in occurrence_indices:
            if idx + k < len(df):
                ref_close = df.loc[idx, "Close"]
                future_close = df.loc[idx + k, "Close"]
                deltas.append(future_close - ref_close)

        positives = [d for d in deltas if d > 0]
        negatives = [d for d in deltas if d < 0]
        flats = [d for d in deltas if d == 0]

        pos_mean = float(np.mean(positives)) if positives else 0.0
        neg_mean = float(np.mean(negatives)) if negatives else 0.0

        pos_max = float(max(positives)) if positives else 0.0
        neg_max = float(min(negatives)) if negatives else 0.0

        pos_min = float(min(positives)) if positives else 0.0
        neg_min = float(max(negatives)) if negatives else 0.0

        pos_count, neg_count, flat_count = len(positives), len(negatives), len(flats)
        total_count = pos_count + neg_count + flat_count

        if pos_count > neg_count:
            hi_prob_tgt = pos_mean
            probability = (pos_count / total_count) if total_count else 0.0
        elif neg_count > pos_count:
            hi_prob_tgt = neg_mean
            probability = (neg_count / total_count) if total_count else 0.0
        else:
            hi_prob_tgt, probability = 0.0, 0.0

        stats = {
            "days": k,
            "positive_range": pos_mean,
            "positive_count": pos_count,
            "positive_max": pos_max,
            "positive_min": pos_min,
            "negative_range": neg_mean,
            "negative_count": neg_count,
            "negative_max": neg_max,
            "negative_min": neg_min,
            "flat_count": flat_count,
            "total_count": total_count,
            "hi_prob_tgt": hi_prob_tgt,
            "positive_probability": probability if hi_prob_tgt >= 0 else 1 - probability,
            "negative_probability": 1 - probability if hi_prob_tgt >= 0 else probability,
            "dominant_probability": float(1 if hi_prob_tgt > 0 else -1 if hi_prob_tgt < 0 else 0),
            "base_price": base_price
        }
        # Probabilities
        if hi_prob_tgt < 0:
            negative_prob = probability
            positive_prob = 1 - probability
        else:
            positive_prob = probability
            negative_prob = 1 - probability
        stats["positive_probability"] = positive_prob
        stats["negative_probability"] = negative_prob

        # Dominance
        if positive_prob > negative_prob:
            dominant_prob = 1
        elif negative_prob > positive_prob:
            dominant_prob = -1
        else:
            dominant_prob = 0
        stats["dominant_probability"] = float(dominant_prob)

        # Variances
        stats["probability_variance"] = (
            (positive_prob / negative_prob) * dominant_prob if negative_prob != 0 else 0.0
        )

        denominator = (pos_mean - neg_mean)
        stats["initial_range_variance"] = (pos_max - neg_max) / denominator if denominator != 0 else 0.0

        denom2 = (pos_max - neg_max)
        stats["second_range_variance"] = (pos_mean - neg_mean) / denom2 if denom2 != 0 else 0.0

        if stats["second_range_variance"] != 0:
            stats["consolidated_variance"] = stats["initial_range_variance"] / stats["second_range_variance"]
        else:
            stats["consolidated_variance"] = 0.0

        if dominant_prob == 1:
            factoring_offset = abs((1 - positive_prob) * pos_mean * 0.5)
        elif dominant_prob == -1:
            factoring_offset = abs((1 - negative_prob) * neg_mean * 0.5)
        else:
            factoring_offset = 0.0
        stats["factoring_offset"] = factoring_offset
        stats["abnormal_factoring"] = factoring_offset * 0.382 if factoring_offset > 1 else factoring_offset

        # Midpoint & ranges
        stats["midpoint"] = (neg_mean + pos_mean) / 2.0

        range_width = pos_mean - neg_mean
        stats["range_width"] = range_width
        normalized_range = range_width - ((k / (30 + 1.0)) * range_width)
        stats["normalized_range"] = normalized_range

        if dominant_prob > 0:
            stats["normalized_range_direction"] = normalized_range * dominant_prob
        else:
            stats["normalized_range_direction"] = normalized_range * 0.3812

        result[str(k)] = stats

    return pd.DataFrame(result).T

# ============================================================================
# TECHNICAL UTILITIES
# ============================================================================

def calculate_rsi(data: pd.DataFrame, period: int = 14, source: str = 'Close') -> pd.DataFrame:
    """
    Calculate RSI (Relative Strength Index) using TradingView default parameters.
    Optimized version for better performance.
    """
    df = data.copy()
    
    # Get price values
    prices = df[source].values
    
    # Calculate price changes
    price_changes = np.diff(prices)
    price_changes = np.insert(price_changes, 0, 0)
    
    # Separate gains and losses
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    
    # Calculate average gains and losses
    df['Avg_Gain'] = pd.Series(gains).rolling(window=period, min_periods=1).mean()
    df['Avg_Loss'] = pd.Series(losses).rolling(window=period, min_periods=1).mean()
    
    # Calculate RS and RSI
    df['RS'] = df['Avg_Gain'] / df['Avg_Loss'].replace(0, np.inf)
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    
    # Clean up intermediate columns
    df = df.drop(['Avg_Gain', 'Avg_Loss', 'RS'], axis=1)
    
    return df

def group_days(dates: List[str], df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Group dates based on 7-day trend and RSI conditions.
    Optimized version for better performance.
    """
    try:
        # Calculate RSI for the dataframe
        df_with_rsi = calculate_rsi(df.reset_index(), period=14, source='Close')
        df_with_rsi = df_with_rsi.reset_index(drop=True)
        
        # Create date to index mapping
        date_to_index = {}
        for idx, row in df_with_rsi.iterrows():
            if pd.notna(row['Date']):
                # Convert to datetime if it's not already
                if isinstance(row['Date'], str):
                    try:
                        if '/' in row['Date']:
                            date_obj = datetime.strptime(row['Date'], '%m/%d/%Y')
                        else:
                            date_obj = datetime.strptime(row['Date'], '%Y-%m-%d')
                    except ValueError:
                        continue
                else:
                    date_obj = row['Date']
                
                date_to_index[date_obj] = idx
        
        # Parse input dates
        parsed_dates = []
        for date_str in dates:
            try:
                if '/' in date_str:
                    date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                else:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                parsed_dates.append((date_str, date_obj))
            except ValueError:
                continue
        
        # Initialize groups
        groups = {
            '7D+': [],
            '7D-': [],
            'Tech+': [],
            'Tech-': []
        }
        
        # Process each date
        for date_str, date_obj in parsed_dates:
            try:
                if date_obj not in date_to_index:
                    continue
                
                current_index = date_to_index[date_obj]
                current_row = df_with_rsi.iloc[current_index]
                current_rsi = current_row['RSI']
                
                if pd.isna(current_rsi):
                    continue
                
                # Calculate 7-day trend
                start_index = max(0, current_index - 7)
                if current_index - start_index < 5:
                    continue
                
                # Get past 7 days data
                past_slice = df_with_rsi.iloc[start_index:current_index]
                start_price = past_slice.iloc[0]['Close']
                end_price = past_slice.iloc[-1]['Close']
                trend = end_price - start_price
                
                # Group based on trend and RSI conditions
                if trend > 0:  # Uptrend
                    if 30 <= current_rsi <= 70:  # Healthy RSI
                        groups['7D+'].append(date_str)
                    elif current_rsi > 60:  # Overbought
                        groups['Tech+'].append(date_str)
                else:  # Downtrend
                    if 30 <= current_rsi <= 70:  # Healthy RSI
                        groups['7D-'].append(date_str)
                    elif current_rsi < 40:  # Oversold
                        groups['Tech-'].append(date_str)
                        
            except Exception as e:
                continue
        
        return groups
        
    except Exception as e:
        raise Exception(f"Error in group_days: {e}")

# ============================================================================
# CLUSTERING UTILITIES
# ============================================================================

def get_adaptive_cluster_mean(points: List[float], base_gap: float = 0.3, max_gap: float = 1.5) -> Optional[float]:
    """
    Find the mean of the largest cluster in points with an adaptive gap.
    
    Args:
        points: List of points to cluster
        base_gap: minimum allowed distance between points in a cluster
        max_gap: maximum allowed distance between points in a cluster
        
    Returns:
        Mean of the largest cluster, or None if no points
    """
    if not points:
        return None
    
    points_sorted = sorted(points)
    n_points = len(points_sorted)
    
    # Dynamically adjust gap based on the spread of points
    if n_points > 1:
        spread = points_sorted[-1] - points_sorted[0]
        cluster_gap = max(base_gap, min(spread / (n_points - 1), max_gap))
    else:
        cluster_gap = base_gap
    
    clusters = []
    current_cluster = [points_sorted[0]]
    
    for p in points_sorted[1:]:
        if abs(p - current_cluster[-1]) <= cluster_gap:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)
    
    # Find the largest cluster
    largest_cluster = max(clusters, key=lambda c: len(c), default=None)
    if largest_cluster is None:
        return None
    return mean(largest_cluster)

def cluster_projection_points(pos_prices_by_k: Dict[int, List[float]], 
                            neg_prices_by_k: Dict[int, List[float]],
                            base_gap: float = 0.2, max_gap: float = 1.5) -> Dict[int, float]:
    """
    Cluster projection points by combining positive and negative prices
    """
    all_prices_by_k = {}
    all_keys = sorted(set(pos_prices_by_k.keys()) | set(neg_prices_by_k.keys()))
    
    for k in all_keys:
        combined_points = pos_prices_by_k.get(k, []) + neg_prices_by_k.get(k, [])
        cluster_mean = get_adaptive_cluster_mean(combined_points, base_gap, max_gap)
        if cluster_mean is not None:
            all_prices_by_k[k] = cluster_mean
    
    return all_prices_by_k

def calculate_mean_aggregates(pos_prices_by_k: Dict[int, List[float]], 
                            neg_prices_by_k: Dict[int, List[float]]) -> Dict[str, Dict[int, float]]:
    """
    Calculate mean aggregates for positive and negative projections
    """
    aggregates = {'positive': {}, 'negative': {}}
    
    # Calculate positive means
    for k, prices in pos_prices_by_k.items():
        if prices:
            aggregates['positive'][k] = mean(prices)
    
    # Calculate negative means
    for k, prices in neg_prices_by_k.items():
        if prices:
            aggregates['negative'][k] = mean(prices)
    
    return aggregates

# ============================================================================
# PROJECTION UTILITIES
# ============================================================================

def get_selected_dates(df: pd.DataFrame, recent_bars: int) -> List[str]:
    """
    Get the list of selected dates for projection analysis
    """
    n_recent = int(recent_bars)
    return df["Date"].iloc[max(0, len(df) - n_recent):].dt.strftime("%Y-%m-%d").tolist()

def get_pattern_position_and_close(df: pd.DataFrame, sel_date: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Get the pattern position and close price for a selected date
    """
    try:
        sel_dt = pd.to_datetime(sel_date).normalize()
        hits = df.index[df["Date"].dt.normalize() == sel_dt].tolist()
        if not hits:
            return None, None
        
        pattern_label = hits[0]
        pattern_pos = df.index.get_loc(pattern_label)
        pattern_close = float(df["Close"].iloc[pattern_pos])
        return pattern_pos, pattern_close
    except Exception:
        return None, None

def process_pattern_projections(df: pd.DataFrame, sel_date: str, pattern_pos: int, pattern_close: float,
                              patterns_data: Dict, bar_count: str, max_overlay: int,
                              compute_outcomes_cached, DEFAULT_PROJ_DAYS: int) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Process pattern projections for a selected date
    """
    pos_prices_by_k = {}
    neg_prices_by_k = {}
    
    # Get matching patterns
    matching = [(k, v) for k, v in patterns_data.items() if "_steps_" in k and v.get("occurrences_count", 0) > 0]
    if bar_count != "All":
        matching = [(k, v) for k, v in matching if str(int(k.split("_steps_")[1])) == bar_count]
    matching.sort(key=lambda kv: kv[1].get("occurrences_count", 0), reverse=True)
    matching = matching[:max_overlay]
    
    # Process each matching pattern
    for key, val in matching:
        df_outcomes = compute_outcomes_cached(tuple(val.get("occurrences", [])), sel_date, DEFAULT_PROJ_DAYS)
        max_k_here = min(DEFAULT_PROJ_DAYS, len(df_outcomes))
        
        for k in range(1, max_k_here + 1):
            if str(k) not in df_outcomes.index:
                continue
                
            row = df_outcomes.loc[str(k)]
            dom = float(row.get("dominant_probability", 0.0))
            pos_rng = float(row.get("positive_range", 0.0) or 0.0)
            neg_rng = float(row.get("negative_range", 0.0) or 0.0)
            
            # Determine projection level and color
            if dom > 0:
                y_lvl = pattern_close + pos_rng
                pos_prices_by_k.setdefault(k, []).append(y_lvl)
            elif dom < 0:
                y_lvl = pattern_close + neg_rng
                neg_prices_by_k.setdefault(k, []).append(y_lvl)
            # Note: neutral projections (dom == 0) are not included in clustering
    
    return pos_prices_by_k, neg_prices_by_k

def get_patterns_data_cached(sel_date: str):
    """Get patterns data from cache or generate if needed"""
    filename = f"{sel_date}_patterns_cache.json"
    if not cache_exists_in_s3(filename):
        return None  # Will be handled by the calling function
    return load_cache_from_s3(filename)

def compute_outcomes_cached(occurrences: Tuple[str, ...], sel_date: str, look_ahead: int, df):
    """Compute pattern outcomes with caching"""
    return analyze_current_day_pattern_outcomes(df, list(occurrences), sel_date, look_ahead=look_ahead)
