"""
Pattern analysis utilities for stock market data
"""

import pandas as pd
import numpy as np
from config import DEFAULT_LOOKBACK, DEFAULT_PROJ_DAYS
from .aws_utils import save_cache_to_s3


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

    from config import CSV_FILENAME, DATA_DIR
    from pathlib import Path
    
    db_path = Path(__file__).resolve().parent.parent / DATA_DIR / CSV_FILENAME
    df_raw = pd.read_csv(db_path)
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
