"""
Technical Analysis Utilities
Handles RSI calculation and market condition grouping for pattern analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from sklearn.linear_model import LinearRegression


def calculate_rsi(data: pd.DataFrame, period: int = 14, source: str = 'Close') -> pd.DataFrame:
    """
    Calculate RSI (Relative Strength Index) using TradingView default parameters.
    Optimized version for better performance.
    
    Args:
        data (DataFrame): DataFrame with OHLC data
        period (int): RSI period (default: 14)
        source (str): Price source to use (default: 'Close')
    
    Returns:
        DataFrame: Original data with RSI column added
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
    
    Args:
        dates: List of date strings to group
        df: DataFrame with stock data (must have Date, Close columns)
        
    Returns:
        dict: Dictionary with groups: '7d_plus', '7d_minus', 'tech_plus', 'tech_minus'
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


def format_group_display(groups: Dict[str, List[str]]) -> str:
    """
    Format the groups dictionary for display in Streamlit
    
    Args:
        groups: Dictionary with grouped dates
        
    Returns:
        str: Formatted string for display
    """
    if not any(groups.values()):
        return "No dates could be grouped (insufficient data or invalid dates)"
    
    display_text = "**Market Condition Groups:**\n\n"
    
    group_descriptions = {
        '7D+': '🟢 **7D+ (7-Day Uptrend + Healthy RSI 30-70)**: Patterns in healthy uptrends',
        '7D-': '🔴 **7D- (7-Day Downtrend + Healthy RSI 30-70)**: Patterns in healthy downtrends', 
        'Tech+': '🟡 **Tech+ (7-Day Uptrend + Overbought RSI >60)**: Patterns in overbought conditions',
        'Tech-': '🟠 **Tech- (7-Day Downtrend + Oversold RSI <40)**: Patterns in oversold conditions'
    }
    
    for group_key, description in group_descriptions.items():
        dates = groups.get(group_key, [])
        if dates:
            display_text += f"{description}\n"
            display_text += f"Count: {len(dates)} dates\n"
            # Show first few dates as examples
            if len(dates) <= 5:
                display_text += f"Dates: {', '.join(dates)}\n\n"
            else:
                display_text += f"Dates: {', '.join(dates[:3])}... (+{len(dates)-3} more)\n\n"
    
    return display_text


def get_group_summary_stats(groups: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Get summary statistics for the groups
    
    Args:
        groups: Dictionary with grouped dates
        
    Returns:
        dict: Summary statistics
    """
    total_dates = sum(len(dates) for dates in groups.values())
    
    return {
        'total_dates': total_dates,
        '7D+_count': len(groups.get('7D+', [])),
        '7D-_count': len(groups.get('7D-', [])),
        'Tech+_count': len(groups.get('Tech+', [])),
        'Tech-_count': len(groups.get('Tech-', [])),
        'healthy_trends': len(groups.get('7D+', [])) + len(groups.get('7D-', [])),
        'extreme_conditions': len(groups.get('Tech+', [])) + len(groups.get('Tech-', []))
    }


def calculate_regression_channel(df: pd.DataFrame, pattern_date: str, lookback_days: int = 20, future_days: int = 30) -> Dict:
    """
    Calculate linear regression channel for a pattern date.
    
    Args:
        df: DataFrame with stock data
        pattern_date: Pattern date in 'YYYY-MM-DD' format
        lookback_days: Number of days to look back for regression (default: 20)
        future_days: Number of days to project into future (default: 30)
    
    Returns:
        dict: Dictionary containing regression channel data
    """
    try:
        # Convert pattern date to datetime
        pattern_dt = pd.to_datetime(pattern_date)
        
        # Find pattern date index
        pattern_idx = df.index[df["Date"].dt.normalize() == pattern_dt.normalize()]
        if len(pattern_idx) == 0:
            return None
        
        pattern_idx = pattern_idx[0]
        pattern_pos = df.index.get_loc(pattern_idx)
        
        # Get lookback data (20 days before pattern date)
        start_idx = max(0, pattern_pos - lookback_days + 1)
        lookback_data = df.iloc[start_idx:pattern_pos + 1]
        
        if len(lookback_data) < 5:
            return None
        
        # Prepare data for regression
        y = lookback_data["Close"].values
        x = np.arange(len(y)).reshape(-1, 1)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(x, y)
        trend = model.predict(x)
        
        # Calculate standard deviation of residuals
        residuals = y - trend
        std = np.std(residuals)
        
        # Create multiple channel bands (1σ, 2σ, 3σ, 4σ)
        bands = {}
        for level in range(1, 5):  # 1σ, 2σ, 3σ, 4σ
            multiplier = level
            bands[f'upper_{level}'] = trend + (std * multiplier)
            bands[f'lower_{level}'] = trend - (std * multiplier)
        
        # Project future trend and bands
        future_x = np.arange(len(y), len(y) + future_days).reshape(-1, 1)
        future_trend = model.predict(future_x)
        
        future_bands = {}
        for level in range(1, 5):  # 1σ, 2σ, 3σ, 4σ
            multiplier = level
            future_bands[f'upper_{level}'] = future_trend + (std * multiplier)
            future_bands[f'lower_{level}'] = future_trend - (std * multiplier)
        
        return {
            'pattern_date': pattern_date,
            'pattern_day_close': float(lookback_data.iloc[-1]['Close']),
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'r_squared': float(model.score(x, y)),
            'std_deviation': float(std),
            'lookback_trend': trend.tolist(),
            'future_trend': future_trend.tolist(),
            'lookback_dates': lookback_data["Date"].dt.strftime('%Y-%m-%d').tolist(),
            'pattern_position': pattern_pos,
            # Multiple bands for lookback period
            'lookback_upper_1': bands['upper_1'].tolist(),
            'lookback_lower_1': bands['lower_1'].tolist(),
            'lookback_upper_2': bands['upper_2'].tolist(),
            'lookback_lower_2': bands['lower_2'].tolist(),
            'lookback_upper_3': bands['upper_3'].tolist(),
            'lookback_lower_3': bands['lower_3'].tolist(),
            'lookback_upper_4': bands['upper_4'].tolist(),
            'lookback_lower_4': bands['lower_4'].tolist(),
            # Multiple bands for future period
            'future_upper_1': future_bands['upper_1'].tolist(),
            'future_lower_1': future_bands['lower_1'].tolist(),
            'future_upper_2': future_bands['upper_2'].tolist(),
            'future_lower_2': future_bands['lower_2'].tolist(),
            'future_upper_3': future_bands['upper_3'].tolist(),
            'future_lower_3': future_bands['lower_3'].tolist(),
            'future_upper_4': future_bands['upper_4'].tolist(),
            'future_lower_4': future_bands['lower_4'].tolist(),
        }
    except Exception as e:
        return None
