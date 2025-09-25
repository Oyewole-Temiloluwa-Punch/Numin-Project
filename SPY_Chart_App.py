import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
import numpy as np
import boto3
import requests

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
            matches.append(str(pd.to_datetime(df.loc[idx, "Date"]).date()))
    return target_pattern, matches

def generate_cache(df, current_date=None, lookback=1800):
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

def analyze_current_day_pattern_outcomes(df, occurrence_dates, pattern_date: str, look_ahead=30):
    df = df.copy()
    # work with normalized dates as strings for stable joins
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize().dt.strftime("%Y-%m-%d")
    occurrence_dates = [pd.to_datetime(d).normalize().strftime("%Y-%m-%d") for d in occurrence_dates]

    df_raw = pd.read_csv(DB_PATH)
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
DB_PATH = Path(__file__).resolve().parent / "SPY Chart 2025-08-22-09-36.csv"

st.set_page_config(page_title="SPY Pattern Mapper — Per-Pattern Projections", layout="wide")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = ["Date", "Open", "High", "Low", "Close"]
    # normalize dates to midnight to avoid equality bugs
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = load_data(DB_PATH)

st.title("📈 SPY — Pattern Analysis Dashboard")


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


# ==== Projection chart builder (per pattern) — ANCHOR AT SELECTED DATE ====
def build_projection_chart(
    df_full: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    pattern_date: str,
    look_ahead: int = 30,
    real_candles: int = 30
) -> go.Figure:
    """
    Single chart per pattern:
      - Full history candlesticks (pan back is possible).
      - Projections anchor at the SELECTED date, not the latest bar.
      - Initial x-view focuses on last `real_candles` BEFORE the selected date + `look_ahead` projected trading days.
      - For k=1..look_ahead, draw dashed horizontal segment at:
          pattern_close + positive_max[k] if dom=+1 (yellow)
          pattern_close + negative_max[k] if dom=-1 (red)
          pattern_close if dom=0 (gray)
      - Auto-scale Y to that window only.
    """
    df_full = df_full.copy()

    # normalize selected date, find its index
    pattern_date_dt = pd.to_datetime(pattern_date).normalize()
    hits = df_full.index[df_full["Date"] == pattern_date_dt].tolist()
    if not hits:
        # safety: if UI bypassed our earlier check
        raise ValueError(f"Selected date {pattern_date} not found in dataset.")
    pattern_idx = hits[0]
    pattern_close = float(df_full["Close"].iloc[pattern_idx])

    fig = go.Figure()

    # Add all candlesticks (full history) to allow panning anywhere
    fig.add_trace(go.Candlestick(
        x=df_full["Date"],
        open=df_full["Open"],
        high=df_full["High"],
        low=df_full["Low"],
        close=df_full["Close"],
        name="SPY"
    ))

    # Optional: thin marker for selected date (unobtrusive)
    fig.add_vline(
        x=pattern_date_dt,
        line_width=1,
        line_dash="dot",
        line_color="white"
    )

    # Build projections on actual trading dates following the selected date
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
            y_lvl = pattern_close + pos_max
            color = "yellow"
        elif dom < 0:
            y_lvl = pattern_close + neg_max
            color = "red"
        else:
            y_lvl = pattern_close
            color = "gray"

        proj_levels.append(y_lvl)

        # P+K placed on actual trading timestamps if available, else calendar fallback
        if pattern_idx + k < len(df_full):
            proj_dt = pd.to_datetime(df_full["Date"].iloc[pattern_idx + k])
        else:
            proj_dt = pattern_date_dt + pd.Timedelta(days=k)

        # Draw a short horizontal dashed segment centered at proj_dt (use real timestamps, not strings)
        dtw = pd.Timedelta(hours=6)  # dash width ~ half-day
        fig.add_trace(go.Scatter(
            x=[proj_dt - dtw, proj_dt + dtw],
            y=[y_lvl, y_lvl],
            mode="lines",
            line=dict(dash="dash", width=3, color=color),
            name=f"P+{k}",
            showlegend=False
        ))

    # Initial x-window: last `real_candles` BEFORE selected date + projected trading dates after
    start_view = max(0, pattern_idx - real_candles + 1)
    start_date = df_full["Date"].iloc[start_view]
    if pattern_idx + proj_days < len(df_full):
        end_date = df_full["Date"].iloc[pattern_idx + proj_days]
    else:
        end_date = pattern_date_dt + pd.Timedelta(days=proj_days + 1)

    fig.update_xaxes(range=[start_date, end_date])

    # Auto-scale Y to that window only
    window_slice = df_full.iloc[start_view:pattern_idx + 1]
    if not window_slice.empty:
        window_low = float(window_slice["Low"].min())
        window_high = float(window_slice["High"].max())
    else:
        window_low = float(df_full["Low"].iloc[pattern_idx])
        window_high = float(df_full["High"].iloc[pattern_idx])

    if proj_levels:
        window_low = min(window_low, min(proj_levels))
        window_high = max(window_high, max(proj_levels))

    span = max(1e-6, window_high - window_low)
    pad = 0.05 * span
    fig.update_yaxes(range=[window_low - pad, window_high + pad])

    fig.update_layout(
        template="plotly_dark",
        height=520,
        title="Projection from Selected Date (dominance-based)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# ==== Pattern Analysis ====
st.header("🧩 Pattern Analysis")
last_date = df["Date"].iloc[-1]
pattern_date = st.date_input(
    "Select date for pattern analysis:",
    value=last_date
).strftime("%Y-%m-%d")

# API endpoint for cycle patterns
CYCLE_PATTERN_API = "https://cycle-pattern-api-yr4x.onrender.com/api/v1/find-prior-days-with-same-pattern"

def fetch_cycle_pattern_data(date_str):
    """Fetch cycle pattern data from external API"""
    try:
        # Convert date to MM/DD/YYYY format for the API
        date_obj = pd.to_datetime(date_str)
        api_date = date_obj.strftime("%m/%d/%Y")
        
        payload = {"date": api_date}
        response = requests.post(CYCLE_PATTERN_API, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ Error fetching cycle pattern data: {str(e)}")
        return None

# Fixed config
PROJ_DAYS = 30
REAL_BARS = 30

if st.button("📊 Analyze Patterns"):
    # ==== Bar Patterns ====
    st.subheader("🧩 Bar Patterns")
    
    filename = f"{pattern_date}_patterns_cache.json"

    if not cache_exists_in_s3(filename):
        with st.spinner(f"Generating pattern cache for {pattern_date}..."):
            filename = generate_cache(df, current_date=pattern_date, lookback=1800)
            if filename is None:
                st.error(f"❌ The selected date ({pattern_date}) was not a trading day.")
                st.stop()
            else:
                st.success(f"✅ Cache generated in S3: {filename}")
    else:
        st.info(f"Using existing cache file from S3: {filename}")

    patterns_data = load_cache_from_s3(filename)

    for key, val in patterns_data.items():
        if val.get("occurrences_count", 0) <= 0:
            continue

        df_outcomes = analyze_current_day_pattern_outcomes(
            df, val["occurrences"], pattern_date, look_ahead=PROJ_DAYS
        )

        with st.expander(f"{key} → {val['target_pattern']}  (occurrences: {val['occurrences_count']})", expanded=False):
            st.dataframe(df_outcomes.T)

            fig_proj = build_projection_chart(
                df_full=df,
                outcomes_df=df_outcomes,
                pattern_date=pattern_date,
                look_ahead=PROJ_DAYS,
                real_candles=REAL_BARS
            )
            st.plotly_chart(fig_proj, use_container_width=True, key=f"proj_chart_{key}")

    # ==== Cycle Patterns ====
    st.subheader("🔄 Cycle Patterns")
    
    with st.spinner(f"Fetching cycle pattern data for {pattern_date}..."):
        cycle_data = fetch_cycle_pattern_data(pattern_date)
        
        if cycle_data:
            st.success(f"✅ Cycle pattern data retrieved successfully!")
            
            # Show pattern code
            st.write(f"**Pattern Code:** {cycle_data.get('target_pattern', 'N/A')}")
            st.write(f"**Pattern Name:** {cycle_data.get('target_pattern_name', 'N/A')}")
            
            matching_dates = cycle_data.get("matching_prior_dates", [])
            
            if matching_dates:
                # Analyze outcomes for cycle patterns
                df_outcomes = analyze_current_day_pattern_outcomes(
                    df, matching_dates, pattern_date, look_ahead=PROJ_DAYS
                )
                
                # Display the outcomes table
                st.dataframe(df_outcomes.T)
                
                # Create projection chart for cycle patterns
                fig_cycle_proj = build_projection_chart(
                    df_full=df,
                    outcomes_df=df_outcomes,
                    pattern_date=pattern_date,
                    look_ahead=PROJ_DAYS,
                    real_candles=REAL_BARS
                )
                
                # Update chart title for cycle patterns
                fig_cycle_proj.update_layout(
                    title=f"Cycle Pattern Projection: {cycle_data.get('target_pattern_name', 'Unknown Pattern')}"
                )
                
                st.plotly_chart(fig_cycle_proj, use_container_width=True, key="cycle_pattern_chart")
                
            else:
                st.warning("⚠️ No matching prior dates found for this cycle pattern.")
        else:
            st.error("❌ Failed to retrieve cycle pattern data.")
