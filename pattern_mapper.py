import openpyxl
from openpyxl.utils import get_column_letter, column_index_from_string
import pandas as pd
import numpy as np
from datetime import datetime
import json
import csv
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional

class CalculationTable:
    """
    A table structure for mathematical calculations with columns 1-30 and specified metric rows.
    """
    
    def __init__(self):
        self.row_names = [
            "Pos Range", "Pos Count", "Pos Max", "Pos Min",
            "Neg Range", "Neg Count", "Neg Max", "Neg Min",
            "Flat Count", "Total Count", "Hi-Prob TGT", "Probability"
        ]
        
        self.table = {}
        self.initialize_table()
    
    def initialize_table(self):
        """Initialize the table with empty values for all cells."""
        for row_name in self.row_names:
            self.table[row_name] = {}
            for col_num in range(1, 31):
                self.table[row_name][col_num] = 0.0
    
    def set_value(self, row_name, column, value):
        """
        Set a value in the table.
        
        Args:
            row_name (str): Name of the row (e.g., "Pos Count", "Probability")
            column (int): Column number (1-30)
            value (float/int): Value to set
        """
        if row_name not in self.row_names:
            raise ValueError(f"Invalid row name: {row_name}")
        if column < 1 or column > 30:
            raise ValueError(f"Column must be between 1 and 30, got: {column}")
        
        self.table[row_name][column] = float(value)
    
    def get_value(self, row_name, column):
        """
        Get a value from the table.
        
        Args:
            row_name (str): Name of the row
            column (int): Column number (1-30)
        
        Returns:
            float: The value at the specified position
        """
        if row_name not in self.row_names:
            raise ValueError(f"Invalid row name: {row_name}")
        if column < 1 or column > 30:
            raise ValueError(f"Column must be between 1 and 30, got: {column}")
        
        return self.table[row_name][column]
    
    def get_column_data(self, column):
        """
        Get all values for a specific column.
        
        Args:
            column (int): Column number (1-30)
        
        Returns:
            dict: Dictionary with row names as keys and values as values
        """
        if column < 1 or column > 30:
            raise ValueError(f"Column must be between 1 and 30, got: {column}")
        
        return {row_name: self.table[row_name][column] for row_name in self.row_names}
    
    def get_row_data(self, row_name):
        """
        Get all values for a specific row.
        
        Args:
            row_name (str): Name of the row
        
        Returns:
            dict: Dictionary with column numbers as keys and values as values
        """
        if row_name not in self.row_names:
            raise ValueError(f"Invalid row name: {row_name}")
        
        return self.table[row_name]
    
    def calculate_column_statistics(self, column):
        """
        Calculate basic statistics for a specific column.
        
        Args:
            column (int): Column number (1-30)
        
        Returns:
            dict: Dictionary with statistics
        """
        data = self.get_column_data(column)
        
        
        non_zero_values = [v for v in data.values() if v != 0]
        
        if not non_zero_values:
            return {
                'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                'count': 0, 'non_zero_count': 0
            }
        
        return {
            'mean': np.mean(non_zero_values),
            'std': np.std(non_zero_values),
            'min': np.min(non_zero_values),
            'max': np.max(non_zero_values),
            'count': len(data),
            'non_zero_count': len(non_zero_values)
        }
    
    def save_to_excel(self, filename="calculation_table.xlsx"):
        """
        Save the table to an Excel file.
        
        Args:
            filename (str): Name of the Excel file to save
        """
        df_data = {}
        for row_name in self.row_names:
            df_data[row_name] = [self.table[row_name][col] for col in range(1, 31)]
        
        df = pd.DataFrame(df_data, index=range(1, 31))
        df.index.name = 'Column'
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Calculation Table')
            
            workbook = writer.book
            worksheet = writer.sheets['Calculation Table']
            
            for col_num in range(1, 31):
                col_letter = get_column_letter(col_num + 1) 
                worksheet[f'{col_letter}1'] = f'Col {col_num}'
        
        print(f"Table saved to {filename}")
    
    def save_to_json(self, filename="calculation_table.json"):
        """
        Save the table to a JSON file.
        
        Args:
            filename (str): Name of the JSON file to save
        """
        with open(filename, 'w') as f:
            json.dump(self.table, f, indent=2)
        
        print(f"Table saved to {filename}")
    
    def load_from_json(self, filename="calculation_table.json"):
        """
        Load the table from a JSON file.
        
        Args:
            filename (str): Name of the JSON file to load
        """
        try:
            with open(filename, 'r') as f:
                self.table = json.load(f)
            print(f"Table loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found. Creating new table.")
            self.initialize_table()
    
    def display_table(self, max_columns=10):
        """
        Display the table in a formatted way.
        
        Args:
            max_columns (int): Maximum number of columns to display
        """
        print("\n" + "="*80)
        print("CALCULATION TABLE")
        print("="*80)
        
        header = f"{'Row Name':<15}"
        for col in range(1, min(max_columns + 1, 31)):
            header += f"{'Col ' + str(col):<10}"
        print(header)
        print("-" * (15 + max_columns * 10))
        
        for row_name in self.row_names:
            row_str = f"{row_name:<15}"
            for col in range(1, min(max_columns + 1, 31)):
                value = self.table[row_name][col]
                row_str += f"{value:<10.2f}"
            print(row_str)
        
        if max_columns < 30:
            print(f"\n... showing first {max_columns} columns. Use display_table(max_columns=30) to see all columns.")
    
    def clear_table(self):
        """Clear all values in the table (set to 0)."""
        self.initialize_table()
        print("Table cleared.")
    
    def copy_column(self, source_column, target_column):
        """
        Copy data from one column to another.
        
        Args:
            source_column (int): Source column number (1-30)
            target_column (int): Target column number (1-30)
        """
        if source_column < 1 or source_column > 30 or target_column < 1 or target_column > 30:
            raise ValueError("Column numbers must be between 1 and 30")
        
        source_data = self.get_column_data(source_column)
        for row_name, value in source_data.items():
            self.set_value(row_name, target_column, value)
        
        print(f"Column {source_column} copied to column {target_column}")
    
    def sum_columns(self, columns, target_column):
        """
        Sum multiple columns and store the result in a target column.
        
        Args:
            columns (list): List of column numbers to sum
            target_column (int): Target column number for the result
        """
        if target_column < 1 or target_column > 30:
            raise ValueError("Target column must be between 1 and 30")
        
        for row_name in self.row_names:
            total = sum(self.get_value(row_name, col) for col in columns)
            self.set_value(row_name, target_column, total)
        
        print(f"Sum of columns {columns} stored in column {target_column}")

_csv_data_cache = None
_csv_data_cache_timestamp = None


def _load_csv_data() -> pd.DataFrame:
    """
    Load CSV data with caching for performance.
    
    Returns:
        pd.DataFrame: DataFrame containing the CSV data
    """
    global _csv_data_cache, _csv_data_cache_timestamp
    
    # csv_file = os.path.join("db", "daily_chart.csv")

    csv_file = "C:/Users/User/Downloads/PUNCH/pattern_mapper_api/db/daily_chart.csv"

    if _csv_data_cache is not None:
        try:
            current_timestamp = os.path.getmtime(csv_file)
            if _csv_data_cache_timestamp == current_timestamp:
                return _csv_data_cache
        except OSError:
            pass
    
    try:

        df = pd.read_csv(csv_file)

        df['Date'] = pd.to_datetime(df['Date'])

        numeric_columns = ['Open', 'High', 'Low', 'Close']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['date_str'] = df['Date'].dt.strftime('%m/%d/%Y')
        df.set_index('date_str', inplace=True)

        _csv_data_cache = df
        _csv_data_cache_timestamp = os.path.getmtime(csv_file)
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")


def _parse_dates_batch(dates: List[str]) -> List[datetime]:
    """
    Parse dates in batch for better performance.
    
    Args:
        dates: List of date strings
        
    Returns:
        List of datetime objects
    """
    date_objects = []
    for date_str in dates:
        try:
            if '/' in date_str:
                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
            else:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            date_objects.append(date_obj)
        except ValueError:
            continue
    return date_objects



def group_days(dates: List[str]) -> Dict[str, List[str]]:
    """
    Group dates based on 7-day trend and RSI conditions.
    Optimized version for better performance.
    
    Args:
        dates: List of date strings to group
        
    Returns:
        dict: Dictionary with groups: '7d_plus', '7d_minus', 'tech_plus', 'tech_minus'
    """
    try:

        spy_df = _load_csv_data()
        df_with_rsi = calculate_rsi(spy_df.reset_index(), period=14, source='Close')
        
        df_with_rsi = df_with_rsi.reset_index(drop=True)
        date_to_index = {row['Date']: idx for idx, row in df_with_rsi.iterrows()}
        
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
        
        groups = {
            '7d_plus': [],
            '7d_minus': [],
            'tech_plus': [],
            'tech_minus': []
        }
        
        for date_str, date_obj in parsed_dates:
            try:

                if date_obj not in date_to_index:
                    continue
                
                current_index = date_to_index[date_obj]
                current_row = df_with_rsi.iloc[current_index]
                current_rsi = current_row['RSI']
                
                if pd.isna(current_rsi):
                    continue
                
                                                         
                start_index = max(0, current_index - 7)
                if current_index - start_index < 5:                                
                    continue
                
                                                             
                past_slice = df_with_rsi.iloc[start_index:current_index]
                start_price = past_slice.iloc[0]['Close']
                end_price = past_slice.iloc[-1]['Close']
                trend = end_price - start_price
                
                                                                 
                if trend > 0:           
                    if 30 <= current_rsi <= 70:                
                        groups['7d_plus'].append(date_str)
                    elif current_rsi > 60:                   
                        groups['tech_plus'].append(date_str)
                else:             
                    if 30 <= current_rsi <= 70:                
                        groups['7d_minus'].append(date_str)
                    elif current_rsi < 40:                 
                        groups['tech_minus'].append(date_str)
                        
            except Exception as e:
                continue
        
        return groups
        
    except Exception as e:
        raise Exception(f"Error in group_days: {e}")


def calculate_regression_channel_for_pattern_day(pattern_date, lookback_days=20, csv_file_path="db/daily_chart.csv"):
    """
    Calculate linear regression channel for 20 days before a pattern day.
    
    Args:
        pattern_date (str): Pattern date in 'MM/DD/YYYY' format
        lookback_days (int): Number of days to look back for regression (default: 20)
        csv_file_path (str): Path to the CSV file containing stock data
    
    Returns:
        dict: Dictionary containing regression channel data
    """
    try:

        csv_file_path = "C:/Users/User/Downloads/PUNCH/pattern_mapper_api/db/daily_chart.csv"


                       
        df = pd.read_csv(csv_file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df = df.sort_index()
        
                                          
        if '/' in pattern_date:
            pattern_dt = pd.to_datetime(pattern_date, format='%m/%d/%Y')
        else:
            pattern_dt = pd.to_datetime(pattern_date)
        
                                    
        pattern_idx = df.index.get_loc(pattern_dt)
        
                                                   
        start_idx = max(0, pattern_idx - lookback_days + 1)
        
                               
        lookback_data = df.iloc[start_idx:pattern_idx + 1]
        
        if len(lookback_data) < 5:                                                  
            return None
        
                                     
        y = lookback_data["Close"].values
        x = np.arange(len(y)).reshape(-1, 1)
        
                               
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x, y)
        trend = model.predict(x)
        
                                                   
        residuals = y - trend
        std = np.std(residuals)
        
                                                            
        upper_band = trend + std
        lower_band = trend - std
        
                                                                      
        future_x = np.arange(len(y), len(y) + 30).reshape(-1, 1)
        future_trend = model.predict(future_x)
        future_upper = future_trend + std
        future_lower = future_trend - std
        
        return {
            'pattern_date': pattern_date,
            'pattern_day_close': float(lookback_data.iloc[-1]['Close']),
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'r_squared': float(model.score(x, y)),
            'std_deviation': float(std),
            'lookback_trend': trend.tolist(),
            'lookback_upper': upper_band.tolist(),
            'lookback_lower': lower_band.tolist(),
            'future_trend': future_trend.tolist(),
            'future_upper': future_upper.tolist(),
            'future_lower': future_lower.tolist(),
            'lookback_dates': lookback_data.index.strftime('%m/%d/%Y').tolist()
        }
        
    except Exception as e:
        print(f"Error calculating regression channel for {pattern_date}: {e}")
        return None

def check_price_position_relative_to_channel(price, upper_band, lower_band, tolerance=0.001):
    """
    Check if a price is above, below, or within the regression channel.
    
    Args:
        price (float): Current price
        upper_band (float): Upper band value
        lower_band (float): Lower band value
        tolerance (float): Tolerance for considering price "within" the channel
    
    Returns:
        str: 'above', 'below', or 'within'
    """
    if price > upper_band + tolerance:
        return 'above'
    elif price < lower_band - tolerance:
        return 'below'
    else:
        return 'within'

def check_standard_deviation_channel(price, trend_value, std_deviation, max_channels=5):
    """
    Check which standard deviation channel a price is in.
    
    Args:
        price (float): Current price
        trend_value (float): Regression trend value
        std_deviation (float): Standard deviation of the regression
        max_channels (int): Maximum number of channels to check (default: 5)
    
    Returns:
        dict: Dictionary containing channel information
            {
                'channel_level': int,  # 1, 2, 3, 4, 5, etc.
                'direction': str,      # 'above' or 'below'
                'distance_from_trend': float,
                'distance_from_nearest_band': float,
                'is_extreme': bool     # True if beyond max_channels
            }
    """
    distance_from_trend = price - trend_value
    abs_distance = abs(distance_from_trend)
    
                                            
    channel_level = int(abs_distance / std_deviation) + 1
    
                         
    direction = 'above' if distance_from_trend > 0 else 'below'
    
                                            
    is_extreme = channel_level > max_channels
    
                                              
    if direction == 'above':
        nearest_band = trend_value + (channel_level * std_deviation)
        distance_from_nearest_band = price - nearest_band
    else:
        nearest_band = trend_value - (channel_level * std_deviation)
        distance_from_nearest_band = nearest_band - price
    
    return {
        'channel_level': min(channel_level, max_channels),
        'direction': direction,
        'distance_from_trend': distance_from_trend,
        'distance_from_nearest_band': distance_from_nearest_band,
        'is_extreme': is_extreme,
        'nearest_band_value': nearest_band
    }

def check_price_position_with_channel_levels(price, upper_band, lower_band, trend_value, std_deviation, tolerance=0.001, max_channels=5):
    """
    Enhanced function that checks price position and provides detailed channel information.
    
    Args:
        price (float): Current price
        upper_band (float): Upper band value (±1σ)
        lower_band (float): Lower band value (±1σ)
        trend_value (float): Regression trend value
        std_deviation (float): Standard deviation of the regression
        tolerance (float): Tolerance for considering price "within" the channel
        max_channels (int): Maximum number of channels to check
    
    Returns:
        dict: Dictionary containing position and channel information
    """
                                           
    if price > upper_band + tolerance:
        position = 'above'
    elif price < lower_band - tolerance:
        position = 'below'
    else:
        position = 'within'
    
                                                  
    if position == 'within':
        return {
            'position': 'within',
            'channel_level': 1,
            'direction': 'within',
            'distance_from_trend': price - trend_value,
            'distance_from_nearest_band': 0,
            'is_extreme': False,
            'nearest_band_value': None
        }
    
                                                              
    channel_info = check_standard_deviation_channel(price, trend_value, std_deviation, max_channels)
    
    return {
        'position': position,
        'channel_level': channel_info['channel_level'],
        'direction': channel_info['direction'],
        'distance_from_trend': channel_info['distance_from_trend'],
        'distance_from_nearest_band': channel_info['distance_from_nearest_band'],
        'is_extreme': channel_info['is_extreme'],
        'nearest_band_value': channel_info['nearest_band_value']
    }

def analyze_pattern_with_regression_channel(pattern: str, dates: List[str]) -> Dict:
    """
    Analyze pattern days with regression channels and track price positions for 30 days.
    
    Args:
        pattern (str): The pattern name (e.g., "3-5-6", "4-4-6")
        dates (list): List of dates when the pattern occurred
    
    Returns:
        dict: Dictionary containing analysis results for each pattern occurrence
    """
    try:
                        
        spy_df = _load_csv_data()
        
                                              
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
        
        if not parsed_dates:
            return {}
        
                                              
        date_to_index = {date: idx for idx, date in enumerate(spy_df.index)}
        
        pattern_analysis = {}
        
        for date_str, date_obj in parsed_dates:
            date_str_formatted = date_obj.strftime('%m/%d/%Y')
            
                                                               
            channel_data = calculate_regression_channel_for_pattern_day(date_str_formatted)
            
            if channel_data is None:
                continue
            
                                               
            if date_str_formatted not in date_to_index:
                continue

            pattern_idx = date_to_index[date_str_formatted]
            pattern_row = spy_df.iloc[pattern_idx]
            pattern_day_close = pattern_row['Close']

                                                              
            future_start_idx = pattern_idx + 1
            future_end_idx = min(future_start_idx + 30, len(spy_df))
            
            if future_start_idx >= len(spy_df):
                continue

                                                             
            future_slice = spy_df.iloc[future_start_idx:future_end_idx]
            
            if len(future_slice) == 0:
                continue

                                                                            
            daily_positions = []
            for i, (idx, row) in enumerate(future_slice.iterrows()):
                day_close = float(row['Close'])
                
                                                                           
                if i < len(channel_data['future_trend']):
                    future_trend = channel_data['future_trend'][i]
                    future_upper = channel_data['future_upper'][i]
                    future_lower = channel_data['future_lower'][i]
                    
                                                                    
                    position_info = check_price_position_with_channel_levels(
                        day_close, future_upper, future_lower, 
                        future_trend, channel_data['std_deviation']
                    )
                    
                    daily_positions.append({
                        'date': idx,
                        'day_number': i + 1,
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': day_close,
                        'regression_trend': future_trend,
                        'upper_band': future_upper,
                        'lower_band': future_lower,
                        'position': position_info['position'],
                        'channel_level': position_info['channel_level'],
                        'direction': position_info['direction'],
                        'distance_from_trend': position_info['distance_from_trend'],
                        'distance_from_upper': day_close - future_upper,
                        'distance_from_lower': day_close - future_lower,
                        'distance_from_nearest_band': position_info['distance_from_nearest_band'],
                        'is_extreme': position_info['is_extreme'],
                        'nearest_band_value': position_info['nearest_band_value']
                    })
            
            pattern_analysis[date_str_formatted] = {
                'pattern': pattern,
                'start_date': date_str_formatted,
                'pattern_day_close': pattern_day_close,
                'regression_channel': channel_data,
                'daily_positions': daily_positions
            }
        
        return pattern_analysis
        
    except Exception as e:
        raise Exception(f"Error in analyze_pattern_with_regression_channel: {e}")

def calculate_channel_statistics(pattern_analysis: Dict) -> Dict:
    """
    Calculate statistics about price positions relative to regression channels.
    Enhanced version that includes channel level statistics.
    
    Args:
        pattern_analysis (dict): Data returned from analyze_pattern_with_regression_channel function
    
    Returns:
        dict: Dictionary containing statistics for each day (1-30)
    """
    if not pattern_analysis:
        return {}
    
    daily_stats = {}
    
                      
    for day_num in range(1, 31):
        day_positions = []
        day_distances = []
        day_channel_levels = []
        day_extreme_counts = []
        
        for start_date, data in pattern_analysis.items():
            daily_positions = data['daily_positions']
            if day_num <= len(daily_positions):
                day_data = daily_positions[day_num - 1]
                day_positions.append(day_data['position'])
                day_distances.append(day_data['distance_from_trend'])
                day_channel_levels.append(day_data['channel_level'])
                day_extreme_counts.append(day_data['is_extreme'])
        
        if not day_positions:
            continue
        
                         
        above_count = day_positions.count('above')
        below_count = day_positions.count('below')
        within_count = day_positions.count('within')
        total_count = len(day_positions)
        
                              
        channel_level_counts = {}
        for level in day_channel_levels:
            channel_level_counts[level] = channel_level_counts.get(level, 0) + 1
        
                                           
        extreme_count = sum(day_extreme_counts)
        
                                     
        distances_array = np.array(day_distances)
        avg_distance = float(np.mean(distances_array))
        std_distance = float(np.std(distances_array))
        
                                     
        if above_count > below_count and above_count > within_count:
            dominant_position = 'above'
            dominant_probability = above_count / total_count
        elif below_count > above_count and below_count > within_count:
            dominant_position = 'below'
            dominant_probability = below_count / total_count
        elif within_count > above_count and within_count > below_count:
            dominant_position = 'within'
            dominant_probability = within_count / total_count
        else:
                                                  
            max_count = max(above_count, below_count, within_count)
            if above_count == max_count:
                dominant_position = 'above'
                dominant_probability = above_count / total_count
            elif below_count == max_count:
                dominant_position = 'below'
                dominant_probability = below_count / total_count
            else:
                dominant_position = 'within'
                dominant_probability = within_count / total_count
        
                                        
        most_common_channel = max(channel_level_counts.items(), key=lambda x: x[1])[0] if channel_level_counts else 1
        
        daily_stats[day_num] = {
            'above_count': above_count,
            'below_count': below_count,
            'within_count': within_count,
            'total_count': total_count,
            'above_probability': above_count / total_count if total_count > 0 else 0,
            'below_probability': below_count / total_count if total_count > 0 else 0,
            'within_probability': within_count / total_count if total_count > 0 else 0,
            'dominant_position': dominant_position,
            'dominant_probability': dominant_probability,
            'avg_distance_from_trend': avg_distance,
            'std_distance_from_trend': std_distance,
            'channel_level_counts': channel_level_counts,
            'most_common_channel': most_common_channel,
            'extreme_count': extreme_count,
            'extreme_probability': extreme_count / total_count if total_count > 0 else 0
        }
    
    return daily_stats

def analyze_trend_with_deviation(start_date, end_date, csv_file_path="db/daily_chart.csv", show_plot=True):
    """
    Analyze linear regression trend with ±1 standard deviation bands for a given date range.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format (e.g., '2025-08-01')
        end_date (str): End date in 'YYYY-MM-DD' format (e.g., '2025-08-21')
        csv_file_path (str): Path to the CSV file containing stock data
        show_plot (bool): Whether to display the plot (default: True)
    
    Returns:
        dict: Dictionary containing the analysis results including trend data and statistics
    """

    csv_file_path = "C:/Users/User/Downloads/PUNCH/pattern_mapper_api/db/daily_chart.csv"
                   
    df = pd.read_csv(csv_file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    
                                              
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
                                             
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    df_filtered = df.loc[mask]
    
    if len(df_filtered) == 0:
        print(f"No data found for the date range {start_date} to {end_date}")
        return None
    
                         
    y = df_filtered["Close"].values
    x = np.arange(len(y)).reshape(-1, 1)              
    
                           
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x, y)
    trend = model.predict(x)
    
                                     
    residuals = y - trend
    std = np.std(residuals)
    
                                              
    upper = trend + std
    lower = trend - std
    
                                     
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(x, y)
    pearson_r = np.sqrt(r_squared) if r_squared >= 0 else -np.sqrt(abs(r_squared))
    
                               
    results = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'pearson_r': pearson_r,
        'std_deviation': std,
        'data_points': len(y),
        'start_date': start_date,
        'end_date': end_date,
        'trend_data': trend,
        'upper_band': upper,
        'lower_band': lower,
        'close_prices': y,
        'dates': df_filtered.index
    }
    
    return results

def calculate_rsi(data, period=14, source='Close'):
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
    
                                       
    prices = df[source].values
    
                                              
    price_changes = np.diff(prices)
    price_changes = np.insert(price_changes, 0, 0)                         
    
                                                      
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    
                                                                      
    df['Avg_Gain'] = pd.Series(gains).rolling(window=period, min_periods=1).mean()
    df['Avg_Loss'] = pd.Series(losses).rolling(window=period, min_periods=1).mean()
    
                                                                       
    df['RS'] = df['Avg_Gain'] / df['Avg_Loss'].replace(0, np.inf)
    
                                               
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    
                                   
    df = df.drop(['Avg_Gain', 'Avg_Loss', 'RS'], axis=1)
    
    return df



def calculate_hma(df: pd.DataFrame, window: int = 20, source: str = 'Close') -> pd.DataFrame:
    """
    Calculate Hull Moving Average and add as a column 'HMA_{window}'.
    HMA reduces lag and provides smoother signals than traditional moving averages.
    """
    hma_col = f'HMA_{window}'
    out = df.copy()
    
    # Calculate WMA with half period
    half_period = int(window / 2)
    sqrt_period = int(np.sqrt(window))
    
    # First WMA
    wma1 = out[source].rolling(window=half_period, min_periods=1).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    )
    
    # Second WMA with full period
    wma2 = out[source].rolling(window=window, min_periods=1).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    )
    
    # Raw HMA
    raw_hma = 2 * wma1 - wma2
    
    # Final WMA with sqrt period
    out[hma_col] = raw_hma.rolling(window=sqrt_period, min_periods=1).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    )
    
    return out


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate On-Balance Volume and add as a column 'OBV'.
    Requires columns: 'Close', 'Volume'.
    """
    out = df.copy()
    if 'Volume' not in out.columns:
                                                                       
        out['OBV'] = np.nan
        return out
    close = out['Close'].values
    volume = pd.to_numeric(out['Volume'], errors='coerce').fillna(0).values
    price_diff = np.diff(close, prepend=close[0])
    direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
    obv = np.cumsum(direction * volume)
    out['OBV'] = obv
    return out


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, source: str = 'Close') -> pd.DataFrame:
    """
    Calculate MACD (12,26,9 by default). Adds: 'MACD', 'MACD_signal', 'MACD_hist'.
    """
    out = df.copy()
    ema_fast = out[source].ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = out[source].ewm(span=slow, adjust=False, min_periods=1).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=1).mean()
    out['MACD'] = macd
    out['MACD_signal'] = signal_line
    out['MACD_hist'] = macd - signal_line
    return out


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator. Adds: 'STOCH_K', 'STOCH_D'.
    Requires 'High', 'Low', 'Close'.
    """
    out = df.copy()
    lowest_low = out['Low'].rolling(window=k_period, min_periods=1).min()
    highest_high = out['High'].rolling(window=k_period, min_periods=1).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    out['STOCH_K'] = ((out['Close'] - lowest_low) / denom) * 100
    out['STOCH_D'] = out['STOCH_K'].rolling(window=d_period, min_periods=1).mean()
    return out


def calculate_ichimoku(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52) -> pd.DataFrame:
    """
    Calculate Ichimoku Cloud indicators. Adds: 'TENKAN', 'KIJUN', 'SENKOU_A', 'SENKOU_B', 'CHIKOU'.
    Requires 'High', 'Low', 'Close'.
    
    Args:
        df: DataFrame with OHLC data
        tenkan_period: Tenkan-sen period (default: 9)
        kijun_period: Kijun-sen period (default: 26)
        senkou_b_period: Senkou Span B period (default: 52)
    
    Returns:
        DataFrame with Ichimoku indicators added
    """
    out = df.copy()
    
    # Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2 over 9 periods
    tenkan_high = out['High'].rolling(window=tenkan_period, min_periods=1).max()
    tenkan_low = out['Low'].rolling(window=tenkan_period, min_periods=1).min()
    out['TENKAN'] = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (Highest High + Lowest Low) / 2 over 26 periods
    kijun_high = out['High'].rolling(window=kijun_period, min_periods=1).max()
    kijun_low = out['Low'].rolling(window=kijun_period, min_periods=1).min()
    out['KIJUN'] = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods forward
    out['SENKOU_A'] = ((out['TENKAN'] + out['KIJUN']) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B): (Highest High + Lowest Low) / 2 over 52 periods, shifted 26 periods forward
    senkou_b_high = out['High'].rolling(window=senkou_b_period, min_periods=1).max()
    senkou_b_low = out['Low'].rolling(window=senkou_b_period, min_periods=1).min()
    out['SENKOU_B'] = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span): Close price shifted 26 periods backward
    out['CHIKOU'] = out['Close'].shift(-kijun_period)
    
    return out


def extract_future_dates(pattern: str, dates: List[str]) -> Dict:
    """
    Extract the next 30 days of SPY data for each date in the pattern.
    Optimized version using pandas and numpy for better performance.
    
    Args:
        pattern (str): The pattern name (e.g., "3-5-6", "4-4-6")
        dates (list): List of dates when the pattern occurred
    
    Returns:
        dict: Dictionary containing the extracted data for each date
    """
    try:
                        
        spy_df = _load_csv_data()
        
                                              
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
        
        if not parsed_dates:
            return {}
        
                                              
        date_to_index = {date: idx for idx, date in enumerate(spy_df.index)}
        
        pattern_data = {}
        
                               
        for date_str, date_obj in parsed_dates:
            date_str_formatted = date_obj.strftime('%m/%d/%Y')
            
                                               
            if date_str_formatted not in date_to_index:
                continue

            pattern_idx = date_to_index[date_str_formatted]
            pattern_row = spy_df.iloc[pattern_idx]
            pattern_day_close = pattern_row['Close']
            pattern_date = pattern_row['Date']

                                                              
            future_start_idx = pattern_idx + 1
            future_end_idx = min(future_start_idx + 30, len(spy_df))
            
            if future_start_idx >= len(spy_df):
                continue

                                                             
            future_slice = spy_df.iloc[future_start_idx:future_end_idx]
            
            if len(future_slice) == 0:
                continue

                                                
            next_30_days = []
            for i, (idx, row) in enumerate(future_slice.iterrows()):
                next_30_days.append({
                    'date': idx,                                
                    'day_number': i + 1,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close'])
                })
            
            pattern_data[date_str_formatted] = {
                'pattern': pattern,
                'start_date': date_str_formatted,
                'pattern_day_close': pattern_day_close,
                'next_30_days': next_30_days
            }
        
        return pattern_data
        
    except Exception as e:
        raise Exception(f"Error in extract_future_dates: {e}")




def extract_future_dates_for_groups(pattern: str, dates: List[str]) -> Dict:
    """
    Extract the next 30 days of SPY data for each date in the pattern.
    Optimized version using pandas and numpy for better performance.
    
    Args:
        pattern (str): The pattern name (e.g., "3-5-6", "4-4-6")
        dates (list): List of dates when the pattern occurred
    
    Returns:
        dict: Dictionary containing the extracted data for each date
    """
    try:
                        
        spy_df = _load_csv_data()
        
                                              
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
        
        if not parsed_dates:
            return {}
        
                                              
        date_to_index = {date: idx for idx, date in enumerate(spy_df.index)}
        
        pattern_data = {}
        
                               
        for date_str, date_obj in parsed_dates:
            date_str_formatted = date_obj.strftime('%m/%d/%Y')
            
                                               
            if date_str_formatted not in date_to_index:
                continue

            pattern_idx = date_to_index[date_str_formatted]
            pattern_row = spy_df.iloc[pattern_idx]
            pattern_day_close = pattern_row['Close']
            pattern_date = pattern_row['Date']

                                                              
            future_start_idx = pattern_idx + 1
            future_end_idx = min(future_start_idx + 30, len(spy_df))
            
            if future_start_idx >= len(spy_df):
                continue

                                                             
            future_slice = spy_df.iloc[future_start_idx:future_end_idx]
            
            if len(future_slice) == 0:
                continue

                                                
            next_30_days = []
            for i, (idx, row) in enumerate(future_slice.iterrows()):
                next_30_days.append({
                    'date': idx,                                
                    'day_number': i + 1,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close'])
                })
            
            pattern_data[date_str_formatted] = {
                'pattern': pattern,
                'start_date': date_str_formatted,
                'pattern_day_close': pattern_day_close,
                'next_30_days': next_30_days
            }
        
        return pattern_data
        
    except Exception as e:
        raise Exception(f"Error in extract_future_dates: {e}")


def extract_future_dates_for_groups(pattern: str, dates: List[str], group_size: int = 1) -> Dict:
    """
    Extract the next 30 groups of SPY data for each date in the pattern.
    Each group combines 'group_size' daily candles into one.
    
    Args:
        pattern (str): The pattern name (e.g., "3-5-6", "4-4-6")
        dates (list): List of dates when the pattern occurred
        group_size (int): Number of daily candles to combine into one group
    
    Returns:
        dict: Dictionary containing the extracted grouped data for each date
    """
    try:
                        
        spy_df = _load_csv_data()
        
                                              
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
        
        if not parsed_dates:
            return {}
        
                                              
        date_to_index = {date: idx for idx, date in enumerate(spy_df.index)}
        
        pattern_data = {}
        
                               
        for date_str, date_obj in parsed_dates:
            date_str_formatted = date_obj.strftime('%m/%d/%Y')
            
                                               
            if date_str_formatted not in date_to_index:
                continue

            pattern_idx = date_to_index[date_str_formatted]
            pattern_row = spy_df.iloc[pattern_idx]
            pattern_day_close = pattern_row['Close']

                                                                                 
            total_days_needed = 30 * group_size
            future_start_idx = pattern_idx + 1
            future_end_idx = min(future_start_idx + total_days_needed, len(spy_df))
            
            if future_start_idx >= len(spy_df):
                continue

                                                             
            future_slice = spy_df.iloc[future_start_idx:future_end_idx]
            
            if len(future_slice) < group_size:
                continue

                            
            grouped_data = []
            for group_num in range(30):                     
                start_idx = group_num * group_size
                end_idx = start_idx + group_size
                
                                                             
                if end_idx > len(future_slice):
                    break
                
                group_slice = future_slice.iloc[start_idx:end_idx]
                
                                             
                group_open = float(group_slice.iloc[0]['Open'])                    
                group_high = float(group_slice['High'].max())                   
                group_low = float(group_slice['Low'].min())                   
                group_close = float(group_slice.iloc[-1]['Close'])                   
                group_start_date = group_slice.index[0]
                group_end_date = group_slice.index[-1]
                
                grouped_data.append({
                    'group_number': group_num + 1,
                    'start_date': group_start_date,
                    'end_date': group_end_date,
                    'days_in_group': len(group_slice),
                    'open': group_open,
                    'high': group_high,
                    'low': group_low,
                    'close': group_close
                })
            
            pattern_data[date_str_formatted] = {
                'pattern': pattern,
                'start_date': date_str_formatted,
                'pattern_day_close': pattern_day_close,
                'group_size': group_size,
                'grouped_data': grouped_data
            }
        
        return pattern_data
        
    except Exception as e:
        raise Exception(f"Error in extract_future_dates_for_groups: {e}")


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

    if pattern_name and pattern_dates:
        try:
            regression_trend_result = analyze_pattern_with_regression_channel(pattern_name, pattern_dates)
        except Exception as e:
            print(f"Warning: Could not get regression channel data: {e}")
    
                                
    try:
        base_df = _load_csv_data().copy()
                                           
                                                             
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
                        day_position = daily_positions[middle_day - 1]
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
            spy_df = _load_csv_data()
                                              
            if pattern_date in spy_df.index:
                pattern_date_idx = spy_df.index.get_loc(pattern_date)
                                                        
                group_start_day = (group_num - 1) * group_size + 1
                group_end_day = group_num * group_size
                
                                                                           
                future_date_idx = pattern_date_idx + group_end_day
                if future_date_idx < len(spy_df):
                    group_close_val = float(spy_df.iloc[future_date_idx]['Close'])
                    print(f"Group {group_num} close (day {group_end_day}): {group_close_val}")
                else:
                                                                                       
                    group_close_val = float(spy_df.iloc[-1]['Close'])
                    print(f"Group {group_num} close (last available): {group_close_val}")
            else:
                print("Did not find pattern_date in the csv file FALLBACK")
                                                
                if group_num == 30:
                    group_close_val = float(np.mean(group_closes)) if group_closes else None
                else:
                    first_pattern_data = next(iter(pattern_data.values()))
                    if len(first_pattern_data['grouped_data']) >= 30:
                        group_close_val = first_pattern_data['grouped_data'][29]['close']
                    else:
                        group_close_val = float(np.mean(group_closes)) if group_closes else None
        except Exception as e:
            print(f"Error getting group close: {e}")
                                            
            if group_num == 30:
                group_close_val = float(np.mean(group_closes)) if group_closes else None
            else:
                first_pattern_data = next(iter(pattern_data.values()))
                if len(first_pattern_data['grouped_data']) >= 30:
                    group_close_val = first_pattern_data['grouped_data'][29]['close']
                else:
                    group_close_val = float(np.mean(group_closes)) if group_closes else None

                                                 
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
                most_occurred_group, most_occurred_count = _find_most_contextually_relevant_group(channel_group_counts)
        
                                                      
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
                spy_df = _load_csv_data()
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


def calculate_daily_statistics(pattern_date: str, pattern_data: Dict, pattern_name: str = None, pattern_dates: List[str] = None) -> Dict:
    """
    Calculate statistics across all pattern occurrences for each day.
    Enhanced version that groups channel results from regression analysis.
    
    Args:
        pattern_date (str): The specific pattern date passed to the API call
        pattern_data (dict): Data returned from extract_future_dates function
        pattern_name (str): The pattern name for regression analysis
        pattern_dates (list): List of pattern dates for regression analysis
    
    Returns:
        dict: Dictionary containing statistics for each day (1-30) with channel grouping
    """
    if not pattern_data:
        return {}


    pattern_dates = pattern_dates or [data['start_date'] for data in pattern_data.values()]
    pattern_name = pattern_name or [data['pattern'] for data in pattern_data.values()]

    
                                                
    daily_stats = {}
    
                                                          
    pattern_closes = np.array([data['pattern_day_close'] for data in pattern_data.values()])

                                                                                
    regression_trend_result = {}

    if pattern_name and pattern_dates:
        try:
            regression_trend_result = analyze_pattern_with_regression_channel(pattern_name, pattern_dates)
        except Exception as e:
            print(f"Warning: Could not get regression channel data: {e}")
    
                                
    try:
        base_df = _load_csv_data().copy()
        rsi_df = calculate_rsi(base_df.reset_index(), period=14, source='Close')
        rsi_df['date_str'] = rsi_df['Date'].dt.strftime('%m/%d/%Y')
        rsi_df.set_index('date_str', inplace=True)
        base_df['RSI'] = rsi_df['RSI']
        base_df = calculate_hma(base_df, window=20, source='Close')
        base_df = calculate_obv(base_df)
        base_df = calculate_macd(base_df)
        base_df = calculate_stochastic(base_df)
        base_df = calculate_ichimoku(base_df)
        base_df = calculate_ichimoku(base_df)
    except Exception:
        base_df = None

                                  
    prev_close_day = None
    for day_num in range(1, 31):
                                                                          
        day_differences = []
        day_closes = []
        day_channel_groups = []
                                 
        day_indicator_hmas = []
        day_indicator_rsis = []
        day_indicator_macd = []
        day_indicator_stoch_k = []
        day_indicator_tenkan = []
        day_indicator_kijun = []
        
        for start_date, data in pattern_data.items():
            days_data = data['next_30_days']
            if day_num <= len(days_data):
                day_data = days_data[day_num - 1]
                following_day_close = day_data['close']
                price_difference = following_day_close - data['pattern_day_close']
                day_differences.append(price_difference)
                day_closes.append(following_day_close)
                
                                                                         
                if start_date in regression_trend_result:
                    regression_data = regression_trend_result[start_date]
                    daily_positions = regression_data.get('daily_positions', [])

                    if day_num <= len(daily_positions):
                        day_position = daily_positions[day_num - 1]
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
                        
                        day_channel_groups.append(channel_group)
                    else:
                        day_channel_groups.append('unknown')
                else:
                    day_channel_groups.append('unknown')
        
        if not day_differences:
            continue

                                                          
        differences_array = np.array(day_differences, dtype=np.float64)
        
                                                                     
        day_close_val = None
        try:
            spy_df = _load_csv_data()
                                              
            if pattern_date in spy_df.index:
                pattern_date_idx = spy_df.index.get_loc(pattern_date)
                                                                                  
                future_date_idx = pattern_date_idx + day_num
                if future_date_idx < len(spy_df):
                    day_close_val = float(spy_df.iloc[future_date_idx]['Close'])
                    print(f"Day {day_num} close: {day_close_val}")
                                                                                       
                    if base_df is not None:
                        try:
                            ref_idx_label = spy_df.index[future_date_idx]
                            hma_val = float(base_df.loc[ref_idx_label].get('HMA_20', np.nan))
                            rsi_val = float(base_df.loc[ref_idx_label].get('RSI', np.nan))
                            if not np.isnan(hma_val):
                                day_indicator_hmas.append(hma_val)
                            if not np.isnan(rsi_val):
                                day_indicator_rsis.append(rsi_val)
                            macd_val = float(base_df.loc[ref_idx_label].get('MACD', np.nan))
                            stoch_val = float(base_df.loc[ref_idx_label].get('STOCH_K', np.nan))
                            if not np.isnan(macd_val):
                                day_indicator_macd.append(macd_val)
                            if not np.isnan(stoch_val):
                                day_indicator_stoch_k.append(stoch_val)
                            tenkan_val = float(base_df.loc[ref_idx_label].get('TENKAN', np.nan))
                            kijun_val = float(base_df.loc[ref_idx_label].get('KIJUN', np.nan))
                            if not np.isnan(tenkan_val):
                                day_indicator_tenkan.append(tenkan_val)
                            if not np.isnan(kijun_val):
                                day_indicator_kijun.append(kijun_val)
                        except Exception:
                            pass
                else:
                                                                                       
                    day_close_val = float(spy_df.iloc[-1]['Close'])
                    print(f"Day {day_num} close (last available): {day_close_val}")
            else:
                                                
                print("Did not find pattern_date in the csv file FALLBACK")
                if day_num == 30:
                    day_close_val = float(np.mean(day_closes)) if day_closes else None
                else:
                    first_pattern_data = next(iter(pattern_data.values()))
                    if len(first_pattern_data['next_30_days']) >= 30:
                        day_close_val = first_pattern_data['next_30_days'][29]['close']
                    else:
                        day_close_val = float(np.mean(day_closes)) if day_closes else None
        except Exception as e:
            print(f"Error getting day close: {e}")
                                            
            if day_num == 30:
                day_close_val = float(np.mean(day_closes)) if day_closes else None
            else:
                first_pattern_data = next(iter(pattern_data.values()))
                if len(first_pattern_data['next_30_days']) >= 30:
                    day_close_val = first_pattern_data['next_30_days'][29]['close']
                else:
                    day_close_val = float(np.mean(day_closes)) if day_closes else None

                                                 
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
        
        if day_channel_groups:
            for group in day_channel_groups:
                if group != 'unknown':
                    channel_group_counts[group] = channel_group_counts.get(group, 0) + 1
            
                                      
            if channel_group_counts:
                most_occurred_group, most_occurred_count = _find_most_contextually_relevant_group(channel_group_counts)
        
                                                      
        day_int = day_num
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

                                                         
        if day_num == 1:
                                                                    
                                                                                     
            try:
                spy_df = _load_csv_data()
                if pattern_date in spy_df.index:
                    pattern_day_close = float(spy_df.loc[pattern_date, 'Close'])
                    print(f"pattern_day_close: {pattern_day_close}")
                else:
                                                                     
                    pattern_day_close = next(iter(pattern_data.values()))['pattern_day_close']
                    print("Did not find pattern_date in the csv file")
                    print(f"pattern_day_close: {pattern_day_close}")
            except Exception as e:
                                                                 
                pattern_day_close = next(iter(pattern_data.values()))['pattern_day_close']
            
            projected1 = pattern_day_close
            projected2 = pattern_day_close
        else:
            if prev_close_day is not None:
                projected1 = (
                    prev_close_day
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
                        prev_close_day
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
                        prev_close_day
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
            prev_close_day = projected1

                       
        daily_stats[day_num] = {
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
            'channel_group_counts': channel_group_counts,
            'most_occurred_group': most_occurred_group,
            'most_occurred_count': most_occurred_count,
            'initial_range_variance': initial_range_variance,
            'second_range_variance': second_range_variance,
            'consolidated_variance': consolidated_variance,
            'close': day_close_val,
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
            avg_hma = float(np.mean(day_indicator_hmas)) if day_indicator_hmas else None
            avg_rsi = float(np.mean(day_indicator_rsis)) if day_indicator_rsis else None
            avg_macd = float(np.mean(day_indicator_macd)) if day_indicator_macd else None
            avg_stoch_k = float(np.mean(day_indicator_stoch_k)) if day_indicator_stoch_k else None
            avg_tenkan = float(np.mean(day_indicator_tenkan)) if day_indicator_tenkan else None
            avg_kijun = float(np.mean(day_indicator_kijun)) if day_indicator_kijun else None
            daily_stats[day_num]['average_hma20'] = avg_hma
            daily_stats[day_num]['average_rsi14'] = avg_rsi
            daily_stats[day_num]['average_macd'] = avg_macd
            daily_stats[day_num]['average_stoch_k'] = avg_stoch_k
            daily_stats[day_num]['average_tenkan'] = avg_tenkan
            daily_stats[day_num]['average_kijun'] = avg_kijun
    
    return daily_stats


def _get_channel_group_numeric_value(channel_group: str) -> int:
    """
    Get the numeric value of a channel group for comparison.
    
    Args:
        channel_group (str): Channel group string (e.g., "2σ+", "4σ-", "beyondσ+")
    
    Returns:
        int: Numeric value representing the sigma level
    """
    if channel_group == 'within':
        return 0
    elif channel_group == 'beyondσ+' or channel_group == 'beyondσ-':
        return 5                      
    else:
                                                                  
        try:
            return int(channel_group.split('σ')[0])
        except (ValueError, IndexError):
            return 0


def _find_most_contextually_relevant_group(channel_group_counts: Dict[str, int]) -> Tuple[str, int]:
    """
    Find the most contextually relevant channel group when there are ties.
    Considers the numerical closeness of sigma levels rather than random selection.
    
    Args:
        channel_group_counts (dict): Dictionary with channel groups as keys and counts as values
    
    Returns:
        tuple: (most_relevant_group, count)
    """
    if not channel_group_counts:
        return 'unknown', 0
    
                            
    max_count = max(channel_group_counts.values())
    
                                           
    max_groups = [group for group, count in channel_group_counts.items() if count == max_count]
    
    if len(max_groups) == 1:
                                         
        return max_groups[0], max_count
    
                                                 
                                                                                  
    positive_groups = [g for g in max_groups if g.endswith('+')]
    negative_groups = [g for g in max_groups if g.endswith('-')]
    
    if positive_groups and negative_groups:
                                                                                
        pos_values = [_get_channel_group_numeric_value(g) for g in positive_groups]
        neg_values = [_get_channel_group_numeric_value(g) for g in negative_groups]
        
                                                                   
        pos_mean = np.mean(pos_values)
        neg_mean = np.mean(neg_values)
        
        pos_closest = positive_groups[np.argmin(np.abs(np.array(pos_values) - pos_mean))]
        neg_closest = negative_groups[np.argmin(np.abs(np.array(neg_values) - neg_mean))]
        
                                                                              
        if _get_channel_group_numeric_value(pos_closest) <= _get_channel_group_numeric_value(neg_closest):
            return pos_closest, max_count
        else:
            return neg_closest, max_count
    
    elif positive_groups:
                                                                         
        pos_values = [_get_channel_group_numeric_value(g) for g in positive_groups]
        pos_mean = np.mean(pos_values)
        most_relevant = positive_groups[np.argmin(np.abs(np.array(pos_values) - pos_mean))]
        return most_relevant, max_count
    
    elif negative_groups:
                                                                         
        neg_values = [_get_channel_group_numeric_value(g) for g in negative_groups]
        neg_mean = np.mean(neg_values)
        most_relevant = negative_groups[np.argmin(np.abs(np.array(neg_values) - neg_mean))]
        return most_relevant, max_count
    
    else:
                                                                                 
        return max_groups[0], max_count


def test_grouped_vs_daily_analysis():
    """
    Test function to demonstrate the difference between grouped and daily analysis.
    """
    print("Testing Grouped vs Daily Analysis")
    print("=" * 50)
    
                                                                                   
    test_dates = ["01/15/2024", "02/20/2024", "03/18/2024"]
    pattern_name = "test-pattern"
    
    print(f"Test Pattern: {pattern_name}")
    print(f"Test Dates: {test_dates}")
    print()
    
                                                      
    print("1. STANDARD DAILY ANALYSIS (group_size = 1)")
    print("-" * 40)
    try:
        daily_data = extract_future_dates(pattern_name, test_dates)
        if daily_data:
            daily_stats = calculate_daily_statistics(test_dates[0], daily_data, pattern_name, test_dates)
            sample_day = daily_stats.get(1, {})
            print(f"Day 1 statistics (sample): {sample_day}")
            print(f"Total periods analyzed: 30 days")
        else:
            print("No data available for daily analysis")
    except Exception as e:
        print(f"Error in daily analysis: {e}")
    
    print()
    
                                                     
    print("2. 3-DAY GROUPED ANALYSIS (group_size = 3)")
    print("-" * 40)
    try:
        grouped_data = extract_future_dates_for_groups(pattern_name, test_dates, group_size=3)
        if grouped_data:
            group_stats = calculate_daily_statistics_for_groups(grouped_data, pattern_name, test_dates)
            sample_group = group_stats.get(1, {})
            print(f"Group 1 statistics (sample): {sample_group}")
            print(f"Total periods analyzed: 30 groups (90 trading days)")
            print(f"Each group combines 3 daily candles into 1 OHLC group")
        else:
            print("No data available for grouped analysis")
    except Exception as e:
        print(f"Error in grouped analysis: {e}")
    
    print()
    
                                                      
    print("3. 5-DAY GROUPED ANALYSIS (group_size = 5)")
    print("-" * 40)
    try:
        grouped_data_5 = extract_future_dates_for_groups(pattern_name, test_dates, group_size=5)
        if grouped_data_5:
            group_stats_5 = calculate_daily_statistics_for_groups(grouped_data_5, pattern_name, test_dates)
            sample_group_5 = group_stats_5.get(1, {})
            print(f"Group 1 statistics (sample): {sample_group_5}")
            print(f"Total periods analyzed: 30 groups (150 trading days)")
            print(f"Each group combines 5 daily candles into 1 OHLC group")
        else:
            print("No data available for 5-day grouped analysis")
    except Exception as e:
        print(f"Error in 5-day grouped analysis: {e}")
    
    print()
    print("SUMMARY:")
    print("- group_size = 1: Analyze 30 individual trading days")
    print("- group_size = 3: Analyze 30 groups of 3-day periods (90 trading days total)")
    print("- group_size = 5: Analyze 30 groups of 5-day periods (150 trading days total)")
    print("- Higher group_size = Longer-term trend analysis")


