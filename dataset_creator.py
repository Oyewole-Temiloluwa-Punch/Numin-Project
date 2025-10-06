import pandas as pd
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI
from typing import Dict, List
import uvicorn
import os as _os
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import LinearRegression

# Optional imports from pattern_mapper, with safe fallbacks for lints/runtime
try:
    from pattern_mapper import (
        _load_csv_data,
        calculate_rsi,
        calculate_hma,
        calculate_obv,
        calculate_macd,
        calculate_stochastic,
        calculate_ichimoku,
        analyze_pattern_with_regression_channel,
        _find_most_contextually_relevant_group,
    )
except Exception:  # provide defined names for lints and allow graceful runtime fallback
    _load_csv_data = None
    calculate_rsi = None
    calculate_hma = None
    calculate_obv = None
    calculate_macd = None
    calculate_stochastic = None
    calculate_ichimoku = None
    analyze_pattern_with_regression_channel = None
    _find_most_contextually_relevant_group = None

def _local_load_csv_data():
    base = pd.read_csv(DB_PATH)
    base["Date"] = pd.to_datetime(base["Date"]).dt.strftime('%m/%d/%Y')
    base = base.set_index("Date", drop=True)
    return base

def _local_find_most_contextually_relevant_group(counts: Dict[str, int]):
    # Simple argmax by count; tie-breaker by key name
    if not counts:
        return 'unknown', 0
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[0][0], items[0][1]

def _compute_stats_for_config(
    df_full: pd.DataFrame,
    df_full_dt: pd.DataFrame,
    date_to_idx: Dict[pd.Timestamp, int],
    date: str,
    gap: int,
    steps: int,
    lookback: int,
):
    try:
        # generate pattern for this date and config
        pattern, idx = generate_pattern_for_date(df_full, date, steps=steps, gap=gap)
    except Exception:
        return None

    if is_repetitive_pattern(pattern):
        return None

    matches = find_pattern_occurrences(
        df_full, pattern, idx, steps=steps, gap=gap, lookback=lookback
    )
    if not matches:
        return None

    # Build pattern_data with normalized keys
    look_ahead = 30

    pattern_data: Dict[str, Dict] = {}
    for start_date in matches:
        start_ts = pd.to_datetime(start_date)
        if start_ts not in date_to_idx:
            continue
        start_idx = date_to_idx[start_ts]
        pattern_day_close = (
            float(df_full_dt.loc[start_idx, "Close"]) if start_idx < len(df_full_dt) else None
        )
        grouped = []
        for k in range(1, look_ahead + 1):
            j = start_idx + k
            if j >= len(df_full_dt):
                break
            end_date = df_full_dt.loc[j, "Date"].strftime("%m/%d/%Y")
            close_val = float(df_full_dt.loc[j, "Close"])
            grouped.append({"end_date": end_date, "close": close_val})
        if pattern_day_close is None or not grouped:
            continue
        start_key = start_ts.strftime("%m/%d/%Y")
        pattern_data[start_key] = {
            "start_date": start_key,
            "pattern": pattern,
            "pattern_day_close": pattern_day_close,
            "grouped_data": grouped,
            "group_size": 1,
        }

    if not pattern_data:
        return None

    if pm_calculate_daily_statistics_for_groups is not None:
        grouped_stats = pm_calculate_daily_statistics_for_groups(
            pattern_date=date,
            pattern_data=pattern_data,
            pattern_name=pattern,
            pattern_dates=list(pattern_data.keys()),
        )
        df_stats = pd.DataFrame.from_dict(grouped_stats, orient="index")
    else:
        df_stats = pd.DataFrame.from_dict(
            {"info": {"note": "pattern_mapper grouped function unavailable"}}, orient="index"
        )

    sheet_name = f"g{gap}_s{steps}"[:31]
    return sheet_name, df_stats

# FastAPI app (define if not already defined elsewhere)
try:
    app  # type: ignore[name-defined]
except NameError:
    app = FastAPI()

# Try to import the grouped statistics function from pattern_mapper
try:
    from pattern_mapper import (
        calculate_daily_statistics_for_groups as pm_calculate_daily_statistics_for_groups,
    )
except Exception:
    pm_calculate_daily_statistics_for_groups = None


DB_PATH = r"C:\Users\User\Downloads\PUNCH\TubeGraph\db\daily_chart.csv"

SAVE_DIR = r"C:\Users\User\Downloads\PUNCH\TubeGraph\Train_excel"




def get_closes(pattern_date: str, rows: int = 30):
    df = pd.read_csv(DB_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    target_date = pd.to_datetime(pattern_date)
    if target_date not in df["Date"].values:
        raise ValueError(f"Date {pattern_date} not found in dataset")

    start_idx = df.index[df["Date"] == target_date][0]

    # :point_down: start from the *next day* after pattern_date
    closes = df.loc[start_idx + 1:start_idx + rows, ["Date", "Close"]]

    return {i + 1: row["Close"] for i, row in enumerate(closes.to_dict("records"))}


def analyze_pattern_outcomes(df, occurrence_dates, pattern_date: str, look_ahead=30):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    occurrence_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in occurrence_dates]

    # base price (pattern_date itself)
    df_raw = pd.read_csv(DB_PATH)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    target_date = pd.to_datetime(pattern_date)
    base_price = float(df_raw.loc[df_raw["Date"] == target_date, "Close"].iloc[0])

    # closes map for forward days
    close_map = get_closes(pattern_date, look_ahead)

    occurrence_indices = []
    for d in occurrence_dates:
        idx_list = df.index[df["Date"] == d].tolist()
        if idx_list:
            occurrence_indices.append(idx_list[0])

    result = {}
    total_days = look_ahead
    prev_close = 0
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

        pos_mean = np.mean(positives) if positives else 0
        neg_mean = np.mean(negatives) if negatives else 0

        pos_max = max(positives) if positives else 0
        pos_min = min(positives) if positives else 0
        neg_min = max(negatives) if negatives else 0
        neg_max = min(negatives) if negatives else 0

        pos_count, neg_count, flat_count = len(positives), len(negatives), len(flats)
        total_count = pos_count + neg_count + flat_count

        if pos_count > neg_count:
            hi_prob_tgt = pos_mean
            probability = (pos_count / total_count) if total_count else 0
        elif neg_count > pos_count:
            hi_prob_tgt = neg_mean
            probability = (neg_count / total_count) if total_count else 0
        else:
            hi_prob_tgt, probability = 0, 0

        stats = {}
        stats["days"] = k
        stats["positive_range"] = pos_mean
        stats["positive_count"] = pos_count
        stats["positive_max"] = pos_max
        stats["positive_min"] = pos_min
        stats["negative_range"] = neg_mean
        stats["negative_count"] = neg_count
        stats["negative_max"] = neg_max
        stats["negative_min"] = neg_min
        stats["flat_count"] = flat_count
        stats["total_count"] = total_count
        stats["hi_prob_tgt"] = hi_prob_tgt

        # ---- New Metrics ----
        if hi_prob_tgt < 0:
            negative_prob = probability
            positive_prob = 1 - probability
        else:
            positive_prob = probability
            negative_prob = 1 - probability
        stats["positive_probability"] = positive_prob
        stats["negative_probability"] = negative_prob

        if positive_prob > negative_prob:
            dominant_prob = 1
        elif negative_prob > positive_prob:
            dominant_prob = -1
        else:
            dominant_prob = 0
        stats["dominant_probability"] = dominant_prob

        stats["probability_variance"] = (
            (positive_prob / negative_prob) * dominant_prob if negative_prob != 0 else None
        )

        denominator = pos_mean - neg_mean
        stats["initial_range_variance"] = (pos_max - neg_max) / denominator if denominator != 0 else None

        denom2 = pos_max - neg_max
        stats["second_range_variance"] = (pos_mean - neg_mean) / denom2 if denom2 != 0 else None

        if stats["second_range_variance"] and stats["second_range_variance"] != 0:
            stats["consolidated_variance"] = stats["initial_range_variance"] / stats["second_range_variance"]
        else:
            stats["consolidated_variance"] = None

        if dominant_prob == 1:
            factoring_offset = abs(negative_prob * pos_mean * 0.5)
        elif dominant_prob == -1:
            factoring_offset = abs(positive_prob * neg_mean * 0.5)
        else:
            factoring_offset = 0
        stats["factoring_offset"] = factoring_offset
        stats["abnormal_factoring"] = factoring_offset * 0.382 if factoring_offset > 1 else factoring_offset

        stats["midpoint"] = (neg_mean + pos_mean) / 2

        range_width = pos_mean - neg_mean
        stats["range_width"] = range_width
        normalized_range = range_width - ((k / (total_days + 1)) * range_width)
        stats["normalized_range"] = normalized_range

        if dominant_prob > 0:
            stats["normalized_range_direction"] = normalized_range * dominant_prob
        else:
            stats["normalized_range_direction"] = normalized_range * 0.3812

        # base price fixed
        stats["base_price"] = base_price

        # close value from pattern_date forward
        stats["close"] = close_map.get(k)

        result[str(k)] = stats

    return pd.DataFrame(result).T




def calculate_daily_statistics_for_groups(pattern_date: str, pattern_data: Dict, pattern_name: str = None, pattern_dates: List[str] = None) -> Dict:
    """
    Calculate statistics across all pattern occurrences for each group.
    This is similar to calculate_daily_statistics but works with grouped data.
    
    Args:
        pattern_data (dict): Data returned from extract_future_dates_for_groups function
        pattern_name (str): The pattern name for regression analysis
        pattern_dates (list): List of pattern dates for regression analysis
    
    Returns:
        dict: Dictionary containing statistics for each group (1-30) 
    """
    if not pattern_data:
        return {}

                                  
    pattern_dates = pattern_dates or [data['start_date'] for data in pattern_data.values()]
    pattern_name = pattern_name or [data['pattern'] for data in pattern_data.values()]
    
                                                
    group_stats = {}
    
                                                      
    group_size = next(iter(pattern_data.values())).get('group_size', 1)
    
                                                                                
    regression_trend_result = {}

    if pattern_name and pattern_dates and analyze_pattern_with_regression_channel is not None:
        try:
            regression_trend_result = analyze_pattern_with_regression_channel(pattern_name, pattern_dates)
        except Exception as e:
            print(f"Warning: Could not get regression channel data: {e}")
    
                                
    try:
        base_df = (_load_csv_data() if _load_csv_data is not None else _local_load_csv_data()).copy()
                                           
                                                             
        rsi_df = calculate_rsi(base_df.reset_index(), period=14, source='Close')
        rsi_df['date_str'] = rsi_df['Date'].dt.strftime('%m/%d/%Y')
        rsi_df.set_index('date_str', inplace=True)
        base_df['RSI'] = rsi_df['RSI']
                 
        base_df = calculate_hma(base_df, window=20, source='Close')
                                
        base_df = calculate_obv(base_df)
                                                     
        base_df = calculate_macd(base_df)
        base_df = calculate_stochastic(base_df)
        base_df = calculate_ichimoku(base_df)
    except Exception:
        base_df = None

                                    
    prev_close_group = None
    for group_num in range(1, 31):
                                                                            
        group_differences = []
        group_closes = []
        group_channel_groups = []
                                      
        group_pos_hmas = []
        group_neg_hmas = []
        group_pos_rsis = []
        group_neg_rsis = []
        group_pos_macd = []
        group_neg_macd = []
        group_pos_stoch_k = []
        group_neg_stoch_k = []
        group_pos_tenkan = []
        group_neg_tenkan = []
        group_pos_kijun = []
        group_neg_kijun = []
        
        for start_date, data in pattern_data.items():
            grouped_data = data['grouped_data']
            if group_num <= len(grouped_data):
                group_data = grouped_data[group_num - 1]
                following_group_close = group_data['close']
                price_difference = following_group_close - data['pattern_day_close']
                group_differences.append(price_difference)
                group_closes.append(following_group_close)
                if base_df is not None:
                    try:
                        ref_idx_label = group_data['end_date']
                        hma_val = float(base_df.loc[ref_idx_label].get('HMA_20', np.nan))
                        rsi_val = float(base_df.loc[ref_idx_label].get('RSI', np.nan))
                        macd_val = float(base_df.loc[ref_idx_label].get('MACD', np.nan))
                        stoch_val = float(base_df.loc[ref_idx_label].get('STOCH_K', np.nan))
                        tenkan_val = float(base_df.loc[ref_idx_label].get('TENKAN', np.nan))
                        kijun_val = float(base_df.loc[ref_idx_label].get('KIJUN', np.nan))
                        if price_difference > 0:
                            if not np.isnan(hma_val):
                                group_pos_hmas.append(hma_val)
                            if not np.isnan(rsi_val):
                                group_pos_rsis.append(rsi_val)
                            if not np.isnan(macd_val):
                                group_pos_macd.append(macd_val)
                            if not np.isnan(stoch_val):
                                group_pos_stoch_k.append(stoch_val)
                            if not np.isnan(tenkan_val):
                                group_pos_tenkan.append(tenkan_val)
                            if not np.isnan(kijun_val):
                                group_pos_kijun.append(kijun_val)
                        elif price_difference < 0:
                            if not np.isnan(hma_val):
                                group_neg_hmas.append(hma_val)
                            if not np.isnan(rsi_val):
                                group_neg_rsis.append(rsi_val)
                            if not np.isnan(macd_val):
                                group_neg_macd.append(macd_val)
                            if not np.isnan(stoch_val):
                                group_neg_stoch_k.append(stoch_val)
                            if not np.isnan(tenkan_val):
                                group_neg_tenkan.append(tenkan_val)
                            if not np.isnan(kijun_val):
                                group_neg_kijun.append(kijun_val)
                    except Exception:
                        pass
                
                                                                         
                if start_date in regression_trend_result:
                    regression_data = regression_trend_result[start_date]
                    daily_positions = regression_data.get('daily_positions', [])

                                                            
                    group_start_day = (group_num - 1) * group_size + 1
                    group_end_day = group_num * group_size
                    
                                                                                                         
                    middle_day = min(group_start_day + (group_size // 2), group_end_day)
                    
                    if middle_day <= len(daily_positions):
                        day_position = daily_positions[middle_day - 1] if daily_positions else {}
                        channel_level = day_position.get('channel_level', 1)
                        direction = day_position.get('direction', 'within')
                        
                                                 
                        if direction == 'above':
                            if channel_level == 2:
                                channel_group = '2σ+'
                            elif channel_level == 3:
                                channel_group = '3σ+'
                            elif channel_level == 4:
                                channel_group = '4σ+'
                            else:
                                channel_group = 'beyondσ+'
                        elif direction == 'below':
                            if channel_level == 2:
                                channel_group = '2σ-'
                            elif channel_level == 3:
                                channel_group = '3σ-'
                            elif channel_level == 4:
                                channel_group = '4σ-'
                            else:
                                channel_group = 'beyondσ-'
                        else:
                            channel_group = 'within'
                        
                        group_channel_groups.append(channel_group)
                    else:
                        group_channel_groups.append('unknown')
                else:
                    group_channel_groups.append('unknown')
        
        if not group_differences:
            continue

                                                          
        differences_array = np.array(group_differences, dtype=np.float64)
        

                                                                              
        group_close_val = None
        try:
            spy_df = (_load_csv_data() if _load_csv_data is not None else _local_load_csv_data())
                                              
            if pattern_date in spy_df.index:
                pattern_date_idx = spy_df.index.get_loc(pattern_date)
                                                        
                group_start_day = (group_num - 1) * group_size + 1
                group_end_day = group_num * group_size
                
                                                                           
                future_date_idx = pattern_date_idx + group_end_day
                if future_date_idx < len(spy_df):
                    group_close_val = float(spy_df.iloc[future_date_idx]['Close'])
                    print(f"Group {group_num} close (day {group_end_day}): {group_close_val}")
                else:
                    # Use NaN if the future date is beyond available data
                    group_close_val = float('nan')
                    print(f"Group {group_num} close (day {group_end_day}): NaN - beyond available data")
            else:
                # Use NaN if pattern_date is not found in CSV
                group_close_val = float('nan')
                print(f"Group {group_num} close: NaN - pattern_date not found in CSV")
        except Exception as e:
            # Use NaN if any error occurs
            group_close_val = float('nan')
            print(f"Group {group_num} close: NaN - error getting group close: {e}")

                                                 
        positive_mask = differences_array > 0
        negative_mask = differences_array < 0
        flat_mask = np.abs(differences_array) <= 0.1
        
                                 
        positive_diffs = differences_array[positive_mask]
        negative_diffs = differences_array[negative_mask]
        flat_diffs = differences_array[flat_mask]

                                                     
        pos_count = len(positive_diffs)
        neg_count = len(negative_diffs)
        flat_count = len(flat_diffs)
        total_count = len(differences_array)

                                                    
        pos_range = float(np.mean(positive_diffs)) if pos_count > 0 else 0.0
        pos_max = float(np.max(positive_diffs)) if pos_count > 0 else 0.0
        pos_min = float(np.min(positive_diffs)) if pos_count > 0 else 0.0
        
        neg_range = float(np.mean(negative_diffs)) if neg_count > 0 else 0.0
        neg_max = float(np.min(negative_diffs)) if neg_count > 0 else 0.0
        neg_min = float(np.max(negative_diffs)) if neg_count > 0 else 0.0

                         
        denominator = (pos_range - neg_range)
        initial_range_variance = ((pos_max - neg_max) / denominator) if denominator != 0 else None
        denom2 = (pos_max - neg_max)
        second_range_variance = ((pos_range - neg_range) / denom2) if denom2 != 0 else None
        consolidated_variance = (initial_range_variance / second_range_variance) if (second_range_variance is not None and second_range_variance != 0) else None

                                                             
        if pos_count > neg_count:
            hi_prob_tgt = pos_range
        elif pos_count < neg_count:
            hi_prob_tgt = neg_range
        else:
            hi_prob_tgt = 0.0

        positive_probability = (pos_count / total_count) if total_count > 0 else 0.0
        negative_probability = (neg_count / total_count) if total_count > 0 else 0.0
        if positive_probability > negative_probability:
            dominant_probability = 1
        elif negative_probability > positive_probability:
            dominant_probability = -1
        else:
            dominant_probability = 0
        epsilon = 1e-9
        probability_variance = (positive_probability / (negative_probability if negative_probability > 0 else epsilon)) * dominant_probability
        
                                            
        channel_group_counts = {}
        most_occurred_group = 'unknown'
        most_occurred_count = 0
        
        if group_channel_groups:
            for group in group_channel_groups:
                if group != 'unknown':
                    channel_group_counts[group] = channel_group_counts.get(group, 0) + 1
            
                                      
            if channel_group_counts:
                if _find_most_contextually_relevant_group is not None:
                    most_occurred_group, most_occurred_count = _find_most_contextually_relevant_group(channel_group_counts)
                else:
                    most_occurred_group, most_occurred_count = _local_find_most_contextually_relevant_group(channel_group_counts)
        
                                                      
        day_int = group_num
        total_days = 30
        if dominant_probability == 1:
            factoring_offset = abs(negative_probability * pos_range * 0.5)
        elif dominant_probability == -1:
            factoring_offset = abs(positive_probability * neg_range * 0.5)
        else:
            factoring_offset = 0
        abnormal_factoring = (factoring_offset * 0.382) if factoring_offset > 1 else factoring_offset
        midpoint = (neg_range + pos_range) / 2
        range_width = pos_range - neg_range
        normalized_range = range_width - ((day_int / (total_days + 1)) * range_width)
        if dominant_probability > 0:
            normalized_range_direction = normalized_range * dominant_probability
        else:
            normalized_range_direction = normalized_range * 0.3812

                                                         
        if group_num == 1:
                                                                      
                                                                                     
            try:
                spy_df = (_load_csv_data() if _load_csv_data is not None else _local_load_csv_data())
                if pattern_date in spy_df.index:
                    pattern_day_close = float(spy_df.loc[pattern_date, 'Close'])
                else:
                                                                     
                    pattern_day_close = next(iter(pattern_data.values()))['pattern_day_close']
            except Exception as e:
                                                                 
                pattern_day_close = next(iter(pattern_data.values()))['pattern_day_close']
            
            projected1 = pattern_day_close
            projected2 = pattern_day_close
        else:
            if prev_close_group is not None:
                projected1 = (
                    prev_close_group
                    + (midpoint / 2)
                    + (
                        midpoint
                        * (initial_range_variance or 0)
                        * (consolidated_variance or 0)
                        * dominant_probability
                        * (probability_variance or 0)
                    )
                )
                if dominant_probability >= 0:
                    projected2 = (
                        prev_close_group
                        + (midpoint / 2)
                        + (
                            midpoint
                            * (initial_range_variance or 0)
                            * (consolidated_variance or 0)
                            * (abnormal_factoring or 0)
                            * dominant_probability
                            * dominant_probability
                        )
                    )
                else:
                    projected2 = (
                        prev_close_group
                        + (midpoint / 2)
                        + (
                            midpoint
                            * (initial_range_variance or 0)
                            * (consolidated_variance or 0)
                            * (abnormal_factoring or 0)
                            * dominant_probability
                        )
                    )
            else:
                projected1 = None
                projected2 = None

        if projected1 is not None:
            prev_close_group = projected1

                       
        group_stats[group_num] = {
            'positive_range': pos_range,
            'positive_count': pos_count,
            'positive_max': pos_max,
            'positive_min': pos_min,
            'negative_range': neg_range,
            'negative_count': neg_count,
            'negative_max': neg_max,
            'negative_min': neg_min,
            'flat_count': flat_count,
            'total_count': total_count,
            'hi_prob_tgt': hi_prob_tgt,
            'positive_probability': positive_probability,
            'negative_probability': negative_probability,
            'dominant_probability': dominant_probability,
            'probability_variance': probability_variance,
            'group_size': group_size,
            'total_days_represented': group_size * group_num,
            'channel_group_counts': channel_group_counts,
            'most_occurred_group': most_occurred_group,
            'most_occurred_count': most_occurred_count,
            'initial_range_variance': initial_range_variance,
            'second_range_variance': second_range_variance,
            'consolidated_variance': consolidated_variance,
            'close': group_close_val,
            'projected1': projected1,
            'projected2': projected2,
            'factoring_offset': factoring_offset,
            'abnormal_factoring': abnormal_factoring,
            'midpoint': midpoint,
            'range_width': range_width,
            'normalized_range': normalized_range,
            'normalized_range_direction': normalized_range_direction
        }
                                                 
        if base_df is not None:
            if dominant_probability == 1:
                avg_hma = float(np.mean(group_pos_hmas)) if group_pos_hmas else None
                avg_rsi = float(np.mean(group_pos_rsis)) if group_pos_rsis else None
                avg_macd = float(np.mean(group_pos_macd)) if group_pos_macd else None
                avg_stoch_k = float(np.mean(group_pos_stoch_k)) if group_pos_stoch_k else None
                avg_tenkan = float(np.mean(group_pos_tenkan)) if group_pos_tenkan else None
                avg_kijun = float(np.mean(group_pos_kijun)) if group_pos_kijun else None
            elif dominant_probability == -1:
                avg_hma = float(np.mean(group_neg_hmas)) if group_neg_hmas else None
                avg_rsi = float(np.mean(group_neg_rsis)) if group_neg_rsis else None
                avg_macd = float(np.mean(group_neg_macd)) if group_neg_macd else None
                avg_stoch_k = float(np.mean(group_neg_stoch_k)) if group_neg_stoch_k else None
                avg_tenkan = float(np.mean(group_neg_tenkan)) if group_neg_tenkan else None
                avg_kijun = float(np.mean(group_neg_kijun)) if group_neg_kijun else None
            else:
                avg_hma = None
                avg_rsi = None
                avg_macd = None
                avg_stoch_k = None
                avg_tenkan = None
                avg_kijun = None
            group_stats[group_num]['average_hma20'] = avg_hma
            group_stats[group_num]['average_rsi14'] = avg_rsi
            group_stats[group_num]['average_macd'] = avg_macd
            group_stats[group_num]['average_stoch_k'] = avg_stoch_k
            group_stats[group_num]['average_tenkan'] = avg_tenkan
            group_stats[group_num]['average_kijun'] = avg_kijun
    
    return group_stats



def generate_pattern_at_index(df, idx, steps=15, gap=3):
    patterns = []
    for i in range(steps):
        if idx - i - gap < 0:
            break

        barA = df.loc[idx - i - 1]
        barB = df.loc[idx - i - gap]
        midpoint_B = (barB['High'] + barB['Low']) / 2

        pattern = []
        for col in ['Open', 'High', 'Low', 'Close']:
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


def generate_pattern_for_date(df, date_str, steps=15, gap=3):
    """Generate a pattern for a specific date in the DataFrame."""
    df["Date"] = pd.to_datetime(df["Date"])
    target_date = pd.to_datetime(date_str)

    if target_date not in df["Date"].values:
        raise ValueError(f"Date {date_str} not found in dataset")

    idx = df.index[df["Date"] == target_date][0]
    return generate_pattern_at_index(df, idx, steps=steps, gap=gap), idx


def find_pattern_occurrences(df, target_pattern, target_idx, steps=15, gap=3, lookback=1800):
    """Look back in history to find all matching dates for the target pattern."""
    matches = []
    start_idx = max(steps + gap, target_idx - lookback)
    for idx in range(target_idx - 1, start_idx - 1, -1):
        candidate_pattern = generate_pattern_at_index(df, idx, steps=steps, gap=gap)
        if candidate_pattern == target_pattern:
            matches.append(str(df.loc[idx, "Date"].date()))  # return as date string
    return matches


def is_repetitive_pattern(pattern: str) -> bool:
    parts = pattern.split("/")
    return all(p == parts[0] for p in parts)


class BatchPatternRequest(BaseModel):
    steps: int = 8
    gap: int = 2
    lookback: int = 1800

def _process_one_date_job(date: str, lookback: int) -> str:
    # Ensure output directory exists in the worker
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    excel_path = Path(SAVE_DIR) / f"patterns_{date.replace('/', '-')}.xlsx"
    print(f"Processing date: {date} -> {excel_path}")

    # Load data fresh in the worker to avoid heavy pickling
    df_full = pd.read_csv(DB_PATH)
    df_full_dt = df_full.copy()
    df_full_dt["Date"] = pd.to_datetime(df_full_dt["Date"])  # ensure dtype
    date_to_idx = {df_full_dt.loc[i, "Date"]: i for i in range(len(df_full_dt))}

    configs = [(g, s) for g in range(2, 8) for s in range(2, 8)]
    results = []
    for (g, s) in configs:
        try:
            out = _compute_stats_for_config(
                df_full,
                df_full_dt,
                date_to_idx,
                date,
                g,
                s,
                lookback,
            )
            if out is not None:
                results.append(out)
        except Exception:
            continue

    writer = pd.ExcelWriter(excel_path, engine="xlsxwriter")
    for sheet_name, df_stats in results:
        try:
            print(f"  Writing sheet: {sheet_name}")
            df_stats.to_excel(writer, sheet_name=sheet_name)
        except Exception:
            continue
    writer.close()
    print(f"Finished writing {len(results)} sheets to: {excel_path}")
    return str(excel_path)


@app.post("/batch-december-patterns/")
def batch_december_patterns(req: BatchPatternRequest, start, end):
    df = pd.read_csv(DB_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    # restrict to the trading days window
    # mask = (df["Date"] >= "2025-07-03") & (df["Date"] <= "2025-08-03")
    mask = (df["Date"] >= start) & (df["Date"] <= end)
    df = df.loc[mask].reset_index(drop=True)

    dates_str = [d.strftime("%m/%d/%Y") for d in df["Date"].tolist()]

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # Determine a conservative number of workers to avoid CPU overload
    try:
        env_workers = int(_os.environ.get("BATCH_MAX_WORKERS", "0"))
    except Exception:
        env_workers = 0
    cpu_count = (_os.cpu_count() or 2)
    default_workers = max(1, cpu_count - 1)
    max_workers = env_workers if env_workers > 0 else default_workers

    saved_files: List[str] = []
    errors: List[str] = []

    # Parallelize safely across processes
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {
            executor.submit(_process_one_date_job, date_str, req.lookback): date_str
            for date_str in dates_str
        }
        for future in as_completed(future_to_date):
            date_str = future_to_date[future]
            try:
                path = future.result()
                saved_files.append(path)
            except Exception as e:
                errors.append(f"{date_str}: {e}")

    return {
        "message": "Batch processing complete",
        "saved_files": saved_files,
        "errors": errors,
        "max_workers": max_workers,
    }

if __name__ == "__main__":
    # Test the batch_december_patterns function
    from datetime import datetime
    
    # Create a test request
    test_request = BatchPatternRequest(
        steps=8,
        gap=2,
        lookback=1800
    )
    
    # Test with a small date range
    start_date = "2025-07-21"
    end_date = "2025-07-23"
    
    print(f"Testing batch_december_patterns with date range: {start_date} to {end_date}")
    print(f"Request parameters: steps={test_request.steps}, gap={test_request.gap}, lookback={test_request.lookback}")
    
    try:
        result = batch_december_patterns(test_request, start_date, end_date)
        print("Test completed successfully!")
        print(f"Results: {result}")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()



    from merge import read_excel_files

    result = read_excel_files()

    print(result)

