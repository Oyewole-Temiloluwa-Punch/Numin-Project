"""
FastAPI application for Stock Market Pattern Clustering
Complete implementation with all utilities from the original project
"""

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Tuple
from statistics import mean
import uvicorn
from pathlib import Path
import sys

# Import all utilities
from utils import (
    load_data,
    generate_cache,
    analyze_current_day_pattern_outcomes,
    cache_exists_in_s3,
    load_cache_from_s3,
    get_adaptive_cluster_mean,
    cluster_projection_points,
    calculate_mean_aggregates,
    get_selected_dates,
    get_pattern_position_and_close,
    process_pattern_projections,
    get_patterns_data_cached,
    compute_outcomes_cached,
    group_days,
    DEFAULT_PROJ_DAYS
)

# Data file is now in the current directory
CSV_FILENAME = "SPY Chart 2025-08-22-09-36.csv"

# Initialize FastAPI app
app = FastAPI(
    title="Stock Market Pattern Clustering API",
    description="API for clustering stock market pattern projections using adaptive clustering algorithm",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data loading
df = None

def load_stock_data():
    """Load stock data from CSV file"""
    global df
    try:
        # Data file is now in the current directory
        data_path = Path(__file__).parent / CSV_FILENAME
        df = load_data(data_path)
        df["Index"] = df.index
        return True
    except Exception as e:
        print(f"Failed to load stock data: {str(e)}")
        return False

# Load data on startup
if not load_stock_data():
    print("Warning: Could not load stock data. API may not work correctly.")

def get_adaptive_cluster_mean(points: List[float], base_gap: float = 0.3, max_gap: float = 1.5) -> Optional[float]:
    """
    Find the mean of the largest cluster in points with an adaptive gap.
    - base_gap: minimum allowed distance between points in a cluster
    - max_gap: maximum allowed distance between points in a cluster
    """
    if not points:
        return None
    
    points_sorted = sorted(points)
    n_points = len(points_sorted)
    
    # Dynamically adjust gap based on the spread of points
    if n_points > 1:
        spread = points_sorted[-1] - points_sorted[0]
        cluster_gap = max(base_gap, min(spread / (n_points - 1), max_gap))
    else:
        cluster_gap = base_gap
    
    clusters = []
    current_cluster = [points_sorted[0]]
    
    for p in points_sorted[1:]:
        if abs(p - current_cluster[-1]) <= cluster_gap:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)
    
    # Find the largest cluster
    largest_cluster = max(clusters, key=lambda c: len(c), default=None)
    if largest_cluster is None:
        return None
    return mean(largest_cluster)

def get_patterns_data_cached(sel_date: str):
    """Get patterns data from cache or generate if needed"""
    filename = f"{sel_date}_patterns_cache.json"
    if not cache_exists_in_s3(filename):
        gen_name = generate_cache(df, current_date=sel_date)
        if gen_name is None:
            return None
        filename = gen_name
    return load_cache_from_s3(filename)

def compute_outcomes_cached_wrapper(occurrences: Tuple[str, ...], sel_date: str, look_ahead: int):
    """Wrapper for compute_outcomes_cached with df parameter"""
    return compute_outcomes_cached(occurrences, sel_date, look_ahead, df)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Stock Market Pattern Clustering API",
        "version": "1.0.0",
        "description": "API for clustering stock market pattern projections",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "data_loaded": df is not None
    }

@app.get("/data/stock")
async def get_stock_data(
    limit: Optional[int] = Query(None, description="Limit number of records returned"),
    start_date: Optional[date] = Query(None, description="Start date for data filtering"),
    end_date: Optional[date] = Query(None, description="End date for data filtering")
):
    """Get stock market data with optional filtering"""
    if df is None:
        raise HTTPException(status_code=500, detail="Stock data not loaded")
    
    try:
        data = df.copy()
        
        # Apply date filtering
        if start_date:
            data = data[data["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data["Date"] <= pd.to_datetime(end_date)]
        
        # Apply limit
        if limit:
            data = data.tail(limit)
        
        return {
            "success": True,
            "data": data[["Date", "Open", "High", "Low", "Close"]].to_dict('records'),
            "total_records": len(data),
            "message": "Stock data retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stock data: {str(e)}")

@app.get("/data/stock/latest")
async def get_latest_stock_data():
    """Get the latest stock data entry"""
    if df is None:
        raise HTTPException(status_code=500, detail="Stock data not loaded")
    
    try:
        latest = df.iloc[-1]
        return {
            "success": True,
            "data": {
                "date": latest["Date"].strftime("%Y-%m-%d"),
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "close": float(latest["Close"])
            },
            "message": "Latest stock data retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving latest data: {str(e)}")

@app.get("/clustering/projections")
async def get_clustered_projections(
    category: str = Query("All", description="Category filter: All, 7D+, 7D-, Tech+, Tech-"),
    bar_count_filter: str = Query("All", description="Filter by number of bars: All, 2, 3, 4, 5, 6, 7"),
    recent_bars: int = Query(6, description="Number of recent bars to project from"),
    max_overlay: int = Query(3, description="Maximum overlay patterns per date"),
    base_gap: float = Query(0.2, description="Base gap for clustering algorithm"),
    max_gap: float = Query(1.5, description="Maximum gap for clustering algorithm"),
    look_ahead: int = Query(30, description="Number of days to look ahead for projections")
):
    """
    Get clustered pattern projections based on the adaptive clustering algorithm from app2.py
    
    This endpoint implements the exact clustering logic used in the Streamlit app:
    1. Gets pattern data for recent dates
    2. Computes pattern outcomes and projections
    3. Separates positive and negative projections
    4. Applies adaptive clustering to find the largest cluster
    5. Returns aggregated cluster lines for days 1-30
    
    Parameters:
    - category: Market condition filter (All, 7D+, 7D-, Tech+, Tech-)
    - bar_count_filter: Number of bars in pattern (All, 2-7)
    - recent_bars: Number of recent bars to analyze
    - max_overlay: Maximum patterns per date
    - base_gap/max_gap: Clustering algorithm parameters
    - look_ahead: Projection days (1-30)
    """
    if df is None:
        raise HTTPException(status_code=500, detail="Stock data not loaded")
    
    try:
        # Validate category parameter
        valid_categories = ["All", "7D+", "7D-", "Tech+", "Tech-"]
        if category not in valid_categories:
            raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of: {valid_categories}")
        
        # Validate bar_count_filter parameter
        valid_bar_counts = ["All", "2", "3", "4", "5", "6", "7"]
        if bar_count_filter not in valid_bar_counts:
            raise HTTPException(status_code=400, detail=f"Invalid bar_count_filter. Must be one of: {valid_bar_counts}")
        
        # Initialize variables (matching app2.py logic)
        pos_prices_by_k = {}
        neg_prices_by_k = {}
        last_idx = len(df) - 1
        
        # Get recent dates for projection
        sel_dates = get_selected_dates(df, recent_bars)
        
        # Process each selected date
        for sel_date in sel_dates:
            # Get pattern position and close
            pattern_pos, pattern_close = get_pattern_position_and_close(df, sel_date)
            if pattern_pos is None or pattern_close is None:
                continue
            
            # Get patterns data
            patterns_data = get_patterns_data_cached(sel_date)
            if not patterns_data:
                continue
            
            # Filter patterns by category if not 'All'
            if category != 'All':
                # Get all occurrence dates for this date
                all_occurrences = []
                for key, val in patterns_data.items():
                    if val.get("occurrences_count", 0) > 0:
                        all_occurrences.extend(val.get("occurrences", []))
                
                # Group the dates and filter by category
                try:
                    groups = group_days(all_occurrences, df)
                    category_dates = groups.get(category, [])
                    
                    # Filter patterns_data to only include patterns that have occurrences in the selected category
                    filtered_patterns_data = {}
                    for key, val in patterns_data.items():
                        if val.get("occurrences_count", 0) > 0:
                            # Check if any of this pattern's occurrences are in the selected category
                            pattern_occurrences = val.get("occurrences", [])
                            if any(date in category_dates for date in pattern_occurrences):
                                # Filter the occurrences to only include those in the selected category
                                filtered_occurrences = [date for date in pattern_occurrences if date in category_dates]
                                filtered_patterns_data[key] = {
                                    **val,
                                    "occurrences": filtered_occurrences,
                                    "occurrences_count": len(filtered_occurrences)
                                }
                    
                    patterns_data = filtered_patterns_data
                except Exception as e:
                    # If grouping fails, continue with all patterns
                    pass
            
            # Process projections
            pos_prices, neg_prices = process_pattern_projections(
                df, sel_date, pattern_pos, pattern_close, patterns_data, bar_count_filter, max_overlay,
                compute_outcomes_cached_wrapper, look_ahead
            )
            
            # Merge with global collections
            for k, prices in pos_prices.items():
                pos_prices_by_k.setdefault(k, []).extend(prices)
            for k, prices in neg_prices.items():
                neg_prices_by_k.setdefault(k, []).extend(prices)
        
        # Apply adaptive clustering (matching app2.py logic)
        all_prices_by_k = cluster_projection_points(pos_prices_by_k, neg_prices_by_k, base_gap, max_gap)
        
        # Create simple mean aggregated lines (matching app2.py logic)
        aggregated_positive = []
        aggregated_negative = []
        combined_aggregated = []
        
        if pos_prices_by_k:
            for k in sorted(pos_prices_by_k.keys()):
                avg_price = mean(pos_prices_by_k[k])
                aggregated_positive.append({
                    "day": k,
                    "price": avg_price,
                    "type": "positive",
                    "count": len(pos_prices_by_k[k])
                })
        
        if neg_prices_by_k:
            for k in sorted(neg_prices_by_k.keys()):
                avg_price = mean(neg_prices_by_k[k])
                aggregated_negative.append({
                    "day": k,
                    "price": avg_price,
                    "type": "negative",
                    "count": len(neg_prices_by_k[k])
                })
        
        # Combined aggregated using adaptive clustering
        for k in sorted(all_prices_by_k.keys()):
            combined_aggregated.append({
                "day": k,
                "price": all_prices_by_k[k],
                "type": "combined",
                "cluster_size": len(pos_prices_by_k.get(k, [])) + len(neg_prices_by_k.get(k, []))
            })
        
        # Organize projections by day
        projections_by_day = {}
        
        # Get all days that have any projections
        all_days = set()
        for proj in aggregated_positive:
            all_days.add(proj["day"])
        for proj in aggregated_negative:
            all_days.add(proj["day"])
        for proj in combined_aggregated:
            all_days.add(proj["day"])
        
        # Create projections organized by day
        for day in sorted(all_days):
            positive_price = next((p["price"] for p in aggregated_positive if p["day"] == day), None)
            negative_price = next((p["price"] for p in aggregated_negative if p["day"] == day), None)
            cluster_price = next((p["price"] for p in combined_aggregated if p["day"] == day), None)
            
            projections_by_day[f"day_{day}"] = {
                "positive": positive_price,
                "negative": negative_price,
                "cluster": cluster_price
            }
        
        # Calculate sectioned data (rolling window analysis)
        sectioned_data = {}
        
        # Get base price (use the latest pattern date's close price)
        if sel_dates:
            latest_date = sel_dates[-1]
            pattern_pos, pattern_close = get_pattern_position_and_close(df, latest_date)
            base_price = pattern_close if pattern_close else 0
        else:
            base_price = 0
        
        # Create rolling window analysis for different periods
        window_periods = [5, 10, 15, 20, 25, 30]
        
        for window in window_periods:
            # Get projections for this window (days 1 to window)
            window_positive_prices = []
            window_negative_prices = []
            window_cluster_prices = []
            
            for day in range(1, window + 1):
                day_key = f"day_{day}"
                if day_key in projections_by_day:
                    day_data = projections_by_day[day_key]
                    if day_data["positive"] is not None:
                        window_positive_prices.append(day_data["positive"])
                    if day_data["negative"] is not None:
                        window_negative_prices.append(day_data["negative"])
                    if day_data["cluster"] is not None:
                        window_cluster_prices.append(day_data["cluster"])
            
            # Calculate metrics for this window
            highest_positive = max(window_positive_prices) if window_positive_prices else None
            lowest_negative = min(window_negative_prices) if window_negative_prices else None
            
            # Get the cluster price for the last day in the window
            last_day_cluster = None
            if window_cluster_prices:
                # Find the cluster price for the last available day in the window
                for day in range(window, 0, -1):
                    day_key = f"day_{day}"
                    if day_key in projections_by_day and projections_by_day[day_key]["cluster"] is not None:
                        last_day_cluster = projections_by_day[day_key]["cluster"]
                        break
            
            # Calculate cluster price change from base price
            cluster_change = None
            if last_day_cluster is not None and base_price > 0:
                cluster_change = last_day_cluster - base_price
            
            sectioned_data[f"window_{window}"] = {
                "highest_positive": highest_positive,
                "lowest_negative": lowest_negative,
                "cluster_change_from_base": cluster_change
            }
        
        # Return only the essential values
        return {
            "success": True,
            "pattern_dates": sel_dates,
            "projections": projections_by_day,
            "sectioned_data": sectioned_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating clustered projections: {str(e)}")

@app.get("/clustering/adaptive-cluster")
async def adaptive_cluster_points(
    points: str = Query(..., description="Comma-separated list of points to cluster"),
    base_gap: float = Query(0.3, description="Base gap for clustering algorithm"),
    max_gap: float = Query(1.5, description="Maximum gap for clustering algorithm")
):
    """
    Apply adaptive clustering algorithm to a set of points
    
    This endpoint allows you to test the clustering algorithm with custom data points.
    """
    try:
        # Parse points from comma-separated string
        point_list = [float(p.strip()) for p in points.split(",")]
        
        # Apply clustering
        cluster_mean = get_adaptive_cluster_mean(point_list, base_gap, max_gap)
        
        return {
            "success": True,
            "input_points": point_list,
            "cluster_mean": cluster_mean,
            "parameters": {
                "base_gap": base_gap,
                "max_gap": max_gap
            },
            "message": "Clustering completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in clustering: {str(e)}")

@app.get("/clustering/statistics")
async def get_clustering_statistics():
    """Get statistics about the clustering algorithm and data"""
    if df is None:
        raise HTTPException(status_code=500, detail="Stock data not loaded")
    
    try:
        # Get latest date for analysis
        latest_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")
        
        # Get some basic statistics
        total_records = len(df)
        date_range = {
            "start": df["Date"].iloc[0].strftime("%Y-%m-%d"),
            "end": df["Date"].iloc[-1].strftime("%Y-%m-%d")
        }
        
        return {
            "success": True,
            "data_statistics": {
                "total_records": total_records,
                "date_range": date_range,
                "latest_date": latest_date
            },
            "clustering_algorithm": {
                "name": "Adaptive Clustering",
                "description": "Finds the mean of the largest cluster with adaptive gap",
                "default_base_gap": 0.3,
                "default_max_gap": 1.5
            },
            "message": "Statistics retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "Value Error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "File Not Found",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
