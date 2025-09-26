import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
import numpy as np
import boto3
import requests
from urllib.parse import quote
from urllib.parse import quote

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

df["Index"] = df.index

st.title("📈 SPY — Pattern Analysis Dashboard")


# ==== TOP: Pure Candlestick Chart (no projections) ====
st.subheader("Candlestick Chart")

# Window size control
window_size = st.number_input("Days to show:", min_value=5, max_value=200, value=50, step=5)

# Build full-history trace, then set initial x-range & y-range to the window
fig_top = go.Figure(
    data=[
        go.Candlestick(
            x=df["Index"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="SPY",
            hoverinfo="text",
            hovertext=[
                f"Date: {date}<br>Open: {op}<br>High: {hi}<br>Low: {lo}<br>Close: {cl}" 
                for date, op, hi, lo, cl in zip(
                    df["Date"].dt.strftime("%Y-%m-%d"),
                    df["Open"],
                    df["High"],
                    df["Low"],
                    df["Close"]
                )
            ],
        )
    ]
)

# Initial x-range (last N days)
max_index = len(df) - 1
start_idx = max(0, max_index - int(window_size) + 1)
end_idx = max_index

fig_top.update_layout(
    template="plotly_dark",
    xaxis_title="Trading Days",
    yaxis_title="Price (USD)",
    height=600,
    xaxis_rangeslider_visible=False,
    showlegend=False,
)

# Set initial x-range by index (continuous, no gaps)
fig_top.update_xaxes(range=[start_idx, end_idx])

# Auto-scale Y to the visible window only
window_slice = df.iloc[start_idx:end_idx + 1]
w_low = float(window_slice["Low"].min())
w_high = float(window_slice["High"].max())
span = max(1e-6, w_high - w_low)
pad = 0.05 * span
fig_top.update_yaxes(range=[w_low - pad, w_high + pad])

# Map index ticks back to dates for readability
tick_step = max(1, (end_idx - start_idx) // 10)
tick_vals = list(range(start_idx, end_idx + 1, tick_step))
tick_text = [df["Date"].iloc[i].strftime("%Y-%m-%d") for i in tick_vals]
fig_top.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=45,
    )
)

st.plotly_chart(fig_top, use_container_width=True, key="top_candles_chart")


# ==== Projection chart builder (per pattern) — ANCHOR AT SELECTED DATE ====
def build_projection_chart(
    df_full: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    pattern_date: str,
    look_ahead: int = 30,
    real_candles: int = 30,
    highlights: dict | None = None
) -> go.Figure:
    """
    Single chart per pattern using index-based plotting to eliminate gaps:
      - Uses index for x-axis to create continuous chart without gaps
      - Maps dates back to actual trading dates for display
      - Projections anchor at the SELECTED date, not the latest bar.
      - Initial x-view focuses on last `real_candles` BEFORE the selected date + `look_ahead` projected trading days.
      - For k=1..look_ahead, draw dashed horizontal segment at:
          pattern_close + positive_max[k] if dom=+1 (blue)
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

    # Create index-based x-axis for continuous plotting
    df_full['Index'] = df_full.index
    
    # Add all candlesticks using index for x-axis (no gaps)
    fig.add_trace(go.Candlestick(
        x=df_full["Index"],
        open=df_full["Open"],
        high=df_full["High"],
        low=df_full["Low"],
        close=df_full["Close"],
        name="SPY",
        hoverinfo="text",
        hovertext=[f"Date: {date}<br>Open: {op}<br>High: {hi}<br>Low: {lo}<br>Close: {cl}"
                   for date, op, hi, lo, cl in zip(
                       df_full["Date"].dt.strftime("%Y-%m-%d"),
                       df_full["Open"],
                       df_full["High"],
                       df_full["Low"],
                       df_full["Close"]
                   )]
    ))

    # Optional: thin marker for selected date (unobtrusive)
    fig.add_vline(
        x=pattern_idx,
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
            pos_max = row.get("positive_range", 0.0)
            neg_max = row.get("negative_range", 0.0)
            pos_max = float(pos_max) if pd.notna(pos_max) else 0.0
            neg_max = float(neg_max) if pd.notna(neg_max) else 0.0

        if dom > 0:
            y_lvl = pattern_close + pos_max
            color = "blue"
        elif dom < 0:
            y_lvl = pattern_close + neg_max
            color = "red"
        else:
            y_lvl = pattern_close
            color = "gray"

        proj_levels.append(y_lvl)

        # P+K placed on continuous index positions (no jump at end of history)
        proj_idx = pattern_idx + k

        # Draw a short horizontal dashed segment centered at proj_idx
        dtw = 0.3  # dash width in index units
        fig.add_trace(go.Scatter(
            x=[proj_idx - dtw, proj_idx + dtw],
            y=[y_lvl, y_lvl],
            mode="lines",
            line=dict(dash="dash", width=3, color=color),
            name=f"P+{k}",
            showlegend=False,
            hovertemplate=f"<b>Projection Day {k}</b><br>" +
                         f"Price Level: %{{y:.2f}}<br>" +
                         f"<extra></extra>"
        ))

    # Add highlight points at D5/D10/.../D30 if provided
    if highlights:
        for k, entry in highlights.items():
            dom = entry.get("dom", 0)
            val = float(entry.get("value", 0.0) or 0.0)
            proj_idx = pattern_idx + int(k)
            # compute y: base on pattern_close plus range effect
            if dom == 0:
                y_pt = pattern_close
                marker_color = "#FFFFFF"  # white
            elif dom > 0:
                y_pt = pattern_close + val
                marker_color = "#00E5FF"  # bright cyan
            else:
                y_pt = pattern_close + val
                marker_color = "#FF00FF"  # magenta

            fig.add_trace(go.Scatter(
                x=[proj_idx],
                y=[y_pt],
                mode="markers",
                marker=dict(size=10, color=marker_color, symbol="star", line=dict(color="#000", width=1)),
                name=f"D{k} highlight",
                showlegend=False,
                hovertemplate=f"<b>Highlight D{k}</b><br>Dom: {dom}<br>Value: %{{y:.2f}}<extra></extra>"
        ))

    # Initial x-window: last `real_candles` BEFORE selected date + projected trading dates after
    start_view = max(0, pattern_idx - real_candles + 1)
    # Calculate end index to include projections seamlessly
    if pattern_idx + proj_days < len(df_full):
        end_idx = pattern_idx + proj_days
    else:
        # Continue projections from the last real data point
        end_idx = len(df_full) - 1 + proj_days

    fig.update_xaxes(range=[start_view, end_idx])

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

    # Create custom tick labels that show actual dates (no duplication beyond history)
    tick_vals = []
    tick_text = []
    step = max(1, (end_idx - start_view) // 10)
    for i in range(start_view, min(end_idx + 1, len(df_full)), step):
        tick_vals.append(i)
        tick_text.append(df_full["Date"].iloc[i].strftime("%Y-%m-%d"))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        title="Projection from Selected Date (dominance-based)",
        xaxis_title="Trading Days",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=45
        )
    )

    return fig


# ==== Pattern outcome highlights (D5, D10, ..., D30) ====
def compute_stepwise_highlights(outcomes_df: pd.DataFrame, steps: list[int] | None = None) -> dict:
    """Compute highlight values at D5, D10, ..., D30 based on dominant_probability.

    Rules per checkpoint k:
      - if dom > 0 → pick max(positive_range[1..k])
      - if dom < 0 → pick min(negative_max[1..k]) (most negative)
      - if dom == 0 → 0.0
    Returns {k: {"dom": int, "value": float}} for k in steps intersect available days.
    """
    if steps is None:
        steps = [5, 10, 15, 20, 25, 30]

    if outcomes_df is None or outcomes_df.empty:
        return {}

    # outcomes_df index are strings of day numbers; ensure safe access
    # Build a numeric-day-indexed view
    try:
        days_numeric = [int(str(i)) for i in outcomes_df.index]
        df_num = outcomes_df.copy()
        df_num["_day"] = days_numeric
        df_num = df_num.sort_values("_day")
    except Exception:
        # fallback to original
        df_num = outcomes_df.copy()
        df_num["_day"] = range(1, len(df_num) + 1)

    highlights: dict[int, dict] = {}
    for k in steps:
        subset = df_num[df_num["_day"] <= k]
        if subset.empty:
            continue
        # dom at checkpoint k (use exact row if present, else last available before k)
        dom_row = df_num[df_num["_day"] == k]
        if dom_row.empty:
            dom_row = subset.tail(1)
        dom_val = float(dom_row.get("dominant_probability", pd.Series([0.0])).iloc[0])
        dom = 1 if dom_val > 0 else (-1 if dom_val < 0 else 0)

        value: float
        pos_max_val: float = 0.0
        neg_max_val: float = 0.0
        
        if dom > 0:
            series = pd.to_numeric(subset.get("positive_range", pd.Series([0.0])), errors="coerce").fillna(0.0)
            value = float(series.max()) if not series.empty else 0.0
        elif dom < 0:
            series = pd.to_numeric(subset.get("negative_range", pd.Series([0.0])), errors="coerce").fillna(0.0)
            # choose most negative (minimum)
            value = float(series.min()) if not series.empty else 0.0
        else:
            value = 0.0

        # Always compute pos_max and neg_max for display
        pos_series = pd.to_numeric(subset.get("positive_max", pd.Series([0.0])), errors="coerce").fillna(0.0)
        neg_series = pd.to_numeric(subset.get("negative_max", pd.Series([0.0])), errors="coerce").fillna(0.0)
        pos_max_val = float(pos_series.max()) if not pos_series.empty else 0.0
        neg_max_val = float(neg_series.min()) if not neg_series.empty else 0.0

        highlights[k] = {"dom": dom, "value": value, "pos_max": pos_max_val, "neg_max": neg_max_val}

    return highlights


def render_highlights(highlights: dict):
    """Render highlights in a row of compact metrics above the table."""
    if not highlights:
        return
    ks = sorted(highlights.keys())
    cols = st.columns(len(ks))
    for idx, k in enumerate(ks):
        entry = highlights[k]
        dom = entry.get("dom", 0)
        val = entry.get("value", 0.0)
        pos_max = entry.get("pos_max", 0.0)
        neg_max = entry.get("neg_max", 0.0)
        arrow = "↑" if dom > 0 else ("↓" if dom < 0 else "→")
        label = f"D{k}"
        value_txt = f"{arrow} {val:.2f}"
        cols[idx].metric(label, value_txt)
        
        # Display pos_max and neg_max below each metric
        with cols[idx]:
            st.caption(f"Highest Pos Max: {pos_max:.2f}")
            st.caption(f"Highest Neg Max: {neg_max:.2f}")

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

# Secondary patterns API
SECONDARY_API_BASE = "http://52.53.169.196:8002/api/patterns"

def fetch_secondary_patterns(date_str: str) -> dict | None:
    try:
        # API expects MM/DD/YYYY in query string
        api_date = pd.to_datetime(date_str).strftime("%m/%d/%Y")
        url = f"{SECONDARY_API_BASE}?end_date={quote(api_date)}&download=false"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            st.error(f"❌ Secondary API error {resp.status_code}: {resp.text}")
            return None
        return resp.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch secondary patterns: {e}")
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
            # Highlights row
            highlights = compute_stepwise_highlights(df_outcomes)
            render_highlights(highlights)
            st.dataframe(df_outcomes.T)

            fig_proj = build_projection_chart(
                df_full=df,
                outcomes_df=df_outcomes,
                pattern_date=pattern_date,
                look_ahead=PROJ_DAYS,
                real_candles=REAL_BARS,
                highlights=highlights
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
            
            # Highlights row
            highlights = compute_stepwise_highlights(df_outcomes)
            render_highlights(highlights)
            # Display the outcomes table
            st.dataframe(df_outcomes.T)
            
            # Create projection chart for cycle patterns
            fig_cycle_proj = build_projection_chart(
                df_full=df,
                outcomes_df=df_outcomes,
                pattern_date=pattern_date,
                look_ahead=PROJ_DAYS,
                real_candles=REAL_BARS,
                highlights=highlights
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

# ==== Secondary Patterns Section (auto-run) ====
st.subheader("🧪 Secondary Patterns")
with st.spinner(f"Fetching secondary patterns for {pattern_date}..."):
    sec_data = fetch_secondary_patterns(pattern_date)
    
    if not sec_data or "windows" not in sec_data:
        st.warning("No secondary patterns available for this date.")
    else:
        windows = sec_data.get("windows", {})
        # iterate through windows 3..7
        for win in [3, 4, 5, 6, 7]:
            wkey = str(win)
            if wkey not in windows:
                continue

            win_obj = windows[wkey]
            passed = win_obj.get("passed_day", {})
            counts = passed.get("prior_match_counts", [])
            if not counts:
                continue

            # pick the top pattern by count
            top = max(counts, key=lambda x: x.get("count", 0))
            pattern_id = top.get("pattern")
            match_dates = top.get("dates", [])

            st.markdown(f"**Window {win}** — Pattern: `{pattern_id}` · Matches: {len(match_dates)}")

            if not match_dates:
                st.info("No matches to analyze.")
                continue

            # Analyze using our existing outcomes computation
            df_outcomes = analyze_current_day_pattern_outcomes(
                df, match_dates, pattern_date, look_ahead=PROJ_DAYS
            )

            # Highlights row + table
            highlights = compute_stepwise_highlights(df_outcomes)
            render_highlights(highlights)
            st.dataframe(df_outcomes.T)

            # Chart
            fig_sec = build_projection_chart(
                df_full=df,
                outcomes_df=df_outcomes,
                pattern_date=pattern_date,
                look_ahead=PROJ_DAYS,
                real_candles=REAL_BARS,
                highlights=highlights
            )
            fig_sec.update_layout(title=f"Secondary Pattern Projection — Window {win} · Pattern {pattern_id}")
            st.plotly_chart(fig_sec, use_container_width=True, key=f"secondary_chart_{win}")
