"""
API utilities for fetching external pattern data
"""

import pandas as pd
import requests
import streamlit as st
from urllib.parse import quote
from config import CYCLE_PATTERN_API, SECONDARY_API_BASE


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


def fetch_secondary_patterns(date_str: str) -> dict | None:
    """Fetch secondary patterns from external API"""
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
