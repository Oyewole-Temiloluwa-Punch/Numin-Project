import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
import numpy as np
import boto3

# ==== AWS S3 Config (hardcoded as requested) ====
AWS_ACCESS_KEY_ID = "AKIARVSAETUO7JZ6FK5D"
AWS_SECRET_ACCESS_KEY = "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"
AWS_REGION = "eu-north-1"
BUCKET_NAME = "numin-cache-files"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def save_cache_to_s3(results, filename):
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"cache/{filename}",
        Body=json.dumps(results, indent=4),
        ContentType="application/json"
    )
    return filename

def load_cache_from_s3(filename):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"cache/{filename}")
    return json.loads(obj["Body"].read().decode("utf-8"))

def cache_exists_in_s3(filename):
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key=f"cache/{filename}")
        return True
    except Exception:
        return False


# ==== Pattern Mapper Core ====

def is_repetitive_pattern(pattern: str) -> bool:
    parts = pattern.split("/")
    return all(p == parts[0] for p in parts) if parts else False

def generate_pattern_at_index(df, idx, steps=15, gap=3):
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
    latest_idx = df.index[-1]
    target_pattern = generate_pattern_at_index(df, latest_idx, steps=steps, gap=gap)

    matches = []
    start_idx = max(steps + gap, latest_idx - lookback)
    for idx in range(latest_idx - 1, start_idx - 1, -1):
        candidate_pattern = generate_pattern_at_index(df, idx, steps=steps, gap=gap)
        if candidate_pattern == target_pattern:
            matches.append(str(df.loc[idx, "Date"].date()))
    return target_pattern, matches

def generate_cache(df, current_date=None, lookback=1800):
    results = {}

    if current_date:
        current_date = pd.to_datetime(current_date)
        if current_date not in df["Date"].values:
            raise ValueError(f"Date {current_date} not found in dataset")
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
            "negative_range": neg_mean,
            "negative_count": neg_count,
            "negative_max": neg_max,
            "flat_count": flat_count,
            "total_count": total_count,
            "hi_prob_tgt": hi_prob_tgt,
            "positive_probability": probability if hi_prob_tgt >= 0 else 1 - probability,
            "negative_probability": 1 - probability if hi_prob_tgt >= 0 else probability,
            "dominant_probability": float(1 if hi_prob_tgt > 0 else -1 if hi_prob_tgt < 0 else 0),
            "base_price": base_price
        }

        result[str(k)] = stats

    return pd.DataFrame(result).T


# ==== Paths / Load Data ====
DB_PATH = Path(__file__).resolve().parent / "app" / "db" / "SPY Chart 2025-08-22-09-36.csv"

st.set_page_config(page_title="SPY Pattern Mapper — Per-Pattern Projections", layout="wide")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = ["Date", "Open", "High", "Low", "Close"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = load_data(DB_PATH)

st.title("📈 SPY — Candles then Per-Pattern Projections")


# ==== TOP: Pure Candlestick Chart (no projections) ====
st.subheader("Candlestick Chart")

# Window size control
window_size = st.number_input("Days to show:", min_value=5, max_value=200, value=50, step=5)

# Build full-history trace, then set initial x-range & y-range to the window
fig_top = go.Figure(
    data=[
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="SPY",
        )
    ]
)

# Initial x-range (last N days)
max_index = len(df) - 1
start_idx = max(0, max_index - int(window_size) + 1)
end_idx = max_index

fig_top.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    height=600,
    xaxis_rangeslider_visible=False,
    showlegend=False,
)

# Set initial x-range by date (you can still pan/zoom across full history)
fig_top.update_xaxes(range=[df["Date"].iloc[start_idx], df["Date"].iloc[end_idx]])

# Auto-scale Y to the visible window only
window_slice = df.iloc[start_idx:end_idx + 1]
w_low = float(window_slice["Low"].min())
w_high = float(window_slice["High"].max())
span = max(1e-6, w_high - w_low)
pad = 0.05 * span
fig_top.update_yaxes(range=[w_low - pad, w_high + pad])

st.plotly_chart(fig_top, use_container_width=True, key="top_candles_chart")


# ==== Projection chart builder (per pattern) with window-based auto-scaling ====
def build_projection_chart(
    df_full: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    look_ahead: int = 30,
    real_candles: int = 30
) -> go.Figure:
    """
    Single chart per pattern:
      - Full history candlesticks (pan back is possible).
      - Initial x-view focuses on last `real_candles` bars + `look_ahead` projections.
      - For k=1..look_ahead, draw dashed horizontal segment at:
          last_close + positive_max[k] if dom=+1 (yellow)
          last_close + negative_max[k] if dom=-1 (red)
          last_close if dom=0 (gray)
      - No baseline line across last_close.
      - Auto-scale Y to the initial visible window (last `real_candles` + projections).
    """
    df_full = df_full.copy()
    x_candles = list(range(len(df_full)))
    last_idx = x_candles[-1]
    last_close = float(df_full["Close"].iloc[-1])

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x_candles,
        open=df_full["Open"],
        high=df_full["High"],
        low=df_full["Low"],
        close=df_full["Close"],
        name="SPY"
    ))

    proj_days = min(look_ahead, len(outcomes_df))
    proj_levels = []
    for k in range(1, proj_days + 1):
        row = outcomes_df.loc[str(k)] if str(k) in outcomes_df.index else None
        if row is None:
            dom = 0.0
            pos_max = 0.0
            neg_max = 0.0
        else:
            dom = float(row.get("dominant_probability", 0.0))
            pos_max = row.get("positive_max", 0.0)
            neg_max = row.get("negative_max", 0.0)
            pos_max = float(pos_max) if pd.notna(pos_max) else 0.0
            neg_max = float(neg_max) if pd.notna(neg_max) else 0.0

        if dom > 0:
            y_lvl = last_close + pos_max
            color = "yellow"
        elif dom < 0:
            y_lvl = last_close + neg_max
            color = "red"
        else:
            y_lvl = last_close
            color = "gray"

        proj_levels.append(y_lvl)

        x_center = last_idx + k
        fig.add_trace(go.Scatter(
            x=[x_center - 0.35, x_center + 0.35],
            y=[y_lvl, y_lvl],
            mode="lines",
            line=dict(dash="dash", width=2, color=color),
            name=f"P+{k}",
            showlegend=False
        ))

    # Initial x-window: last 30 real + projections
    start_view = max(0, last_idx - real_candles + 1)
    end_view = last_idx + proj_days + 1
    fig.update_xaxes(range=[start_view - 1, end_view])

    # Auto-scale Y to that window only
    window_slice = df_full.iloc[start_view:last_idx + 1]
    if not window_slice.empty:
        window_low = float(window_slice["Low"].min())
        window_high = float(window_slice["High"].max())
    else:
        window_low = float(df_full["Low"].iloc[-1])
        window_high = float(df_full["High"].iloc[-1])

    if proj_levels:
        window_low = min(window_low, min(proj_levels))
        window_high = max(window_high, max(proj_levels))

    span = max(1e-6, window_high - window_low)
    pad = 0.05 * span
    fig.update_yaxes(range=[window_low - pad, window_high + pad])

    fig.update_layout(
        template="plotly_dark",
        height=520,
        title="Projection from Last Close (dominance-based)",
        xaxis_title="Bars (history + projected slots)",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# ==== Pattern Mapper UI ====
st.header("🧩 Pattern Mapper")
last_date = df["Date"].iloc[-1]
pattern_date = st.date_input("Select date for pattern cache & analysis:", value=last_date).strftime("%Y-%m-%d")

filename = f"{pattern_date}_patterns_cache.json"

if not cache_exists_in_s3(filename):
    with st.spinner(f"Generating pattern cache for {pattern_date}..."):
        filename = generate_cache(df, current_date=pattern_date, lookback=1800)
        st.success(f"✅ Cache generated in S3: {filename}")
else:
    st.info(f"Using existing cache file from S3: {filename}")

patterns_data = load_cache_from_s3(filename)

# Fixed config per your request
PROJ_DAYS = 30
REAL_BARS = 30

if st.button("📊 Analyze Patterns"):
    for key, val in patterns_data.items():
        if val.get("occurrences_count", 0) <= 0:
            continue

        df_outcomes = analyze_current_day_pattern_outcomes(
            df, val["occurrences"], pattern_date, look_ahead=PROJ_DAYS
        )

        with st.expander(f"{key} → {val['target_pattern']}  (occurrences: {val['occurrences_count']})", expanded=False):
            st.write("Occurrences Dates:", val["occurrences"])
            st.dataframe(df_outcomes.T)

            fig_proj = build_projection_chart(
                df_full=df,
                outcomes_df=df_outcomes,
                look_ahead=PROJ_DAYS,
                real_candles=REAL_BARS
            )
            # Unique key per expander chart prevents duplicate-element error
            st.plotly_chart(fig_proj, use_container_width=True, key=f"proj_chart_{key}")
