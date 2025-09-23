import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
import numpy as np
from datetime import datetime

# ==== Pattern Mapper Core ====

def is_repetitive_pattern(pattern: str) -> bool:
    parts = pattern.split("/")
    return all(p == parts[0] for p in parts)

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

def find_pattern_occurrences(df, steps=15, gap=3, lookback=1800): 
    latest_idx = df.index[-1]
    target_pattern = generate_pattern_at_index(df, latest_idx, steps=steps, gap=gap)

    matches = []
    start_idx = max(steps + gap, latest_idx - lookback)
    for idx in range(latest_idx - 1, start_idx - 1, -1):
        candidate_pattern = generate_pattern_at_index(df, idx, steps=steps, gap=gap)
        if candidate_pattern == target_pattern:
            matches.append(str(df.loc[idx, "Date"].date()))  # return as str

    return target_pattern, matches

def generate_cache(df, current_date=None, lookback=1800):
    results = {}

    # Pick index of the current_date
    if current_date:
        current_date = pd.to_datetime(current_date)
        if current_date not in df["Date"].values:
            raise ValueError(f"Date {current_date} not found in dataset")
        latest_idx = df.index[df["Date"] == current_date][0]
    else:
        latest_idx = df.index[-1]
        current_date = df.loc[latest_idx, "Date"]

    # Build patterns
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
                "occurrences": matches
            }

    # Save to file named after chosen current_date
    date_str = pd.to_datetime(current_date).strftime("%Y-%m-%d")
    cache_file = Path(__file__).resolve().parent / "app" / "cache" / f"{date_str}_patterns_cache.json"
    with open(cache_file, "w") as f:
        json.dump(results, f, indent=4)

    return cache_file

def analyze_current_day_pattern_outcomes(df, occurrence_dates, pattern_date: str, look_ahead=30):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    occurrence_dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in occurrence_dates]

    df_raw = pd.read_csv(DB_PATH)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    target_date = pd.to_datetime(pattern_date)
    base_price = float(df_raw.loc[df_raw["Date"] == target_date, "Close"].iloc[0])

    occurrence_indices = []
    for d in occurrence_dates:
        idx_list = df.index[df["Date"] == d].tolist()
        if idx_list:
            occurrence_indices.append(idx_list[0])

    result = {}
    total_days = look_ahead

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
        pos_min = float(min(positives)) if positives else 0.0
        neg_min = float(max(negatives)) if negatives else 0.0
        neg_max = float(min(negatives)) if negatives else 0.0

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
        stats["dominant_probability"] = float(dominant_prob)

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

        stats["midpoint"] = (neg_mean + pos_mean) / 2.0
        range_width = pos_mean - neg_mean
        stats["range_width"] = range_width
        normalized_range = range_width - ((k / (total_days + 1.0)) * range_width)
        stats["normalized_range"] = normalized_range

        if dominant_prob > 0:
            stats["normalized_range_direction"] = normalized_range * dominant_prob
        else:
            stats["normalized_range_direction"] = normalized_range * 0.3812

        stats["base_price"] = base_price
        result[str(k)] = stats

    return pd.DataFrame(result).T


# ==== Paths ====
DB_PATH = Path(__file__).resolve().parent / "SPY Chart 2025-08-22-09-36.csv"

st.set_page_config(page_title="SPY Daily Candlestick Chart with Pattern Mapper", layout="wide")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = ["Date", "Open", "High", "Low", "Close"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

# ==== Load OHLC Data ====
df = load_data(DB_PATH)

st.title("📈 SPY Daily Candlestick Chart with Pattern Mapper")

# ==== Candlestick Chart ====
st.subheader("Candlestick Chart")

# User input for window size
window_size = st.number_input("Days to show:", min_value=5, max_value=200, value=50, step=5)

# Moving Average toggle + inputs
use_ma = st.checkbox("Show Moving Average Crossover")
short_len = st.number_input("Short MA Length", min_value=5, max_value=50, value=20, step=1)
long_len = st.number_input("Long MA Length", min_value=10, max_value=200, value=50, step=1)

# Default window = last N days
max_index = len(df) - 1
start_idx = max_index - window_size + 1 if max_index >= window_size else 0
end_idx = max_index

window_df = df.iloc[start_idx:end_idx + 1]
ymin, ymax = window_df["Low"].min(), window_df["High"].max()

fig = go.Figure(
    data=[
        go.Candlestick(
            x=df["Date"].dt.strftime("%Y-%m-%d"),  # all data
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="SPY",
        )
    ]
)

# Add Moving Averages + signals if enabled
if use_ma:
    df["ShortMA"] = df["Close"].rolling(short_len).mean()
    df["LongMA"] = df["Close"].rolling(long_len).mean()

    fig.add_trace(go.Scatter(
        x=df["Date"].dt.strftime("%Y-%m-%d"),
        y=df["ShortMA"],
        mode="lines",
        line=dict(color="teal", width=2),
        name=f"Short MA ({short_len})"
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"].dt.strftime("%Y-%m-%d"),
        y=df["LongMA"],
        mode="lines",
        line=dict(color="orange", width=2),
        name=f"Long MA ({long_len})"
    ))

    # Buy/Sell signals
    df["BuySignal"] = (df["ShortMA"] > df["LongMA"]) & (df["ShortMA"].shift(1) <= df["LongMA"].shift(1))
    df["SellSignal"] = (df["ShortMA"] < df["LongMA"]) & (df["ShortMA"].shift(1) >= df["LongMA"].shift(1))

    buys = df[df["BuySignal"]]
    sells = df[df["SellSignal"]]

    fig.add_trace(go.Scatter(
        x=buys["Date"].dt.strftime("%Y-%m-%d"),
        y=buys["Low"] * 0.995,
        mode="markers+text",
        marker=dict(color="green", symbol="triangle-up", size=12),
        text=["BUY"] * len(buys),
        textposition="bottom center",
        name="Buy Signal"
    ))

    fig.add_trace(go.Scatter(
        x=sells["Date"].dt.strftime("%Y-%m-%d"),
        y=sells["High"] * 1.005,
        mode="markers+text",
        marker=dict(color="red", symbol="triangle-down", size=12),
        text=["SELL"] * len(sells),
        textposition="top center",
        name="Sell Signal"
    ))

fig.update_layout(
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    title="SPY Candlestick Chart",
    xaxis_title="Trading Days",
    yaxis_title="Price (USD)",
    height=700,
    xaxis=dict(type="category"),  # no gaps for weekends
)

# Set initial view to last N days
fig.update_xaxes(range=[start_idx, end_idx])
fig.update_yaxes(range=[ymin * 0.995, ymax * 1.005])

st.plotly_chart(fig, use_container_width=True)

# ==== Pattern Mapper Section ====
st.header("🧩 Pattern Mapper")

# Default = last date in file
last_date = df["Date"].iloc[-1]
pattern_date = st.date_input(
    "Select date for pattern cache & analysis:",
    value=last_date
).strftime("%Y-%m-%d")

# Cache file path depends on selected pattern_date
cache_file = Path(__file__).resolve().parent / f"{pattern_date}_patterns_cache.json"

# Generate cache if not exists
if not cache_file.exists():
    with st.spinner(f"Generating pattern cache for {pattern_date}..."):
        cache_file = generate_cache(df, current_date=pattern_date, lookback=700)
        st.success(f"✅ Cache generated: {cache_file.name}")
else:
    st.info(f"Using existing cache file: {cache_file.name}")

# Load patterns
with open(cache_file, "r") as f:
    patterns_data = json.load(f)

if st.button("📊 Analyze Patterns"):
    for key, val in patterns_data.items():
        if val["occurrences_count"] > 0:
            df_outcomes = analyze_current_day_pattern_outcomes(
                df, val["occurrences"], pattern_date, look_ahead=30
            )
            with st.expander(f"{key} → {val['target_pattern']}"):
                st.write(f"Occurrences Count: **{val['occurrences_count']}**")
                st.write("Occurrences Dates:", val["occurrences"])

                st.dataframe(df_outcomes.T)  # 🔄 Transposed table

