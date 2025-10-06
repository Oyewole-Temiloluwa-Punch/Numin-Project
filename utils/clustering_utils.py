"""
Clustering Utilities
Handles clustering algorithms for projection aggregation
"""

from statistics import mean
from typing import List, Optional, Dict, Any


def get_adaptive_cluster_mean(points: List[float], base_gap: float = 0.3, max_gap: float = 1.5) -> Optional[float]:
    """
    Find the mean of the largest cluster in points with an adaptive gap.
    
    Args:
        points: List of points to cluster
        base_gap: minimum allowed distance between points in a cluster
        max_gap: maximum allowed distance between points in a cluster
        
    Returns:
        Mean of the largest cluster, or None if no points
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


def cluster_projection_points(pos_prices_by_k: Dict[int, List[float]], 
                            neg_prices_by_k: Dict[int, List[float]],
                            base_gap: float = 0.2, max_gap: float = 1.5) -> Dict[int, float]:
    """
    Cluster projection points by combining positive and negative prices
    
    Args:
        pos_prices_by_k: Dictionary mapping projection day to positive price points
        neg_prices_by_k: Dictionary mapping projection day to negative price points
        base_gap: Base gap for clustering
        max_gap: Maximum gap for clustering
        
    Returns:
        Dictionary mapping projection day to clustered mean price
    """
    all_prices_by_k = {}
    all_keys = sorted(set(pos_prices_by_k.keys()) | set(neg_prices_by_k.keys()))
    
    for k in all_keys:
        combined_points = pos_prices_by_k.get(k, []) + neg_prices_by_k.get(k, [])
        cluster_mean = get_adaptive_cluster_mean(combined_points, base_gap, max_gap)
        if cluster_mean is not None:
            all_prices_by_k[k] = cluster_mean
    
    return all_prices_by_k


def calculate_mean_aggregates(pos_prices_by_k: Dict[int, List[float]], 
                            neg_prices_by_k: Dict[int, List[float]]) -> Dict[str, Dict[int, float]]:
    """
    Calculate mean aggregates for positive and negative projections
    
    Args:
        pos_prices_by_k: Dictionary mapping projection day to positive price points
        neg_prices_by_k: Dictionary mapping projection day to negative price points
        
    Returns:
        Dictionary with 'positive' and 'negative' mean aggregates
    """
    aggregates = {'positive': {}, 'negative': {}}
    
    # Calculate positive means
    for k, prices in pos_prices_by_k.items():
        if prices:
            aggregates['positive'][k] = mean(prices)
    
    # Calculate negative means
    for k, prices in neg_prices_by_k.items():
        if prices:
            aggregates['negative'][k] = mean(prices)
    
    return aggregates


def prepare_aggregate_line_data(aggregates: Dict[int, float], last_idx: int) -> tuple:
    """
    Prepare x and y data for aggregate line plotting
    
    Args:
        aggregates: Dictionary mapping projection day to mean price
        last_idx: Last index of the data
        
    Returns:
        Tuple of (x_values, y_values)
    """
    x_values = [last_idx + k for k in sorted(aggregates.keys())]
    y_values = [aggregates[k] for k in sorted(aggregates.keys())]
    return x_values, y_values
