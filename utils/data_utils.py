"""
Data loading and processing utilities
"""

import pandas as pd
import streamlit as st
from pathlib import Path
from config import DATA_DIR, CSV_FILENAME


@st.cache_data
def load_data(path=None):
    """Load and process stock market data"""
    if path is None:
        # Use default path from config
        path = Path(__file__).resolve().parent.parent / DATA_DIR / CSV_FILENAME
    
    df = pd.read_csv(path)
    df.columns = ["Date", "Open", "High", "Low", "Close"]
    # normalize dates to midnight to avoid equality bugs
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df
