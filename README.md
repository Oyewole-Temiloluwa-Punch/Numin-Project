# Stock Market Pattern Clustering API

A focused FastAPI application that implements the adaptive clustering algorithm from the Stock Market Dashboard's app2.py. This API specifically handles the clustering logic for pattern projections.

## Features

- **Adaptive Clustering Algorithm**: Implements the exact clustering logic from app2.py
- **Pattern Projection Clustering**: Groups and clusters stock market pattern projections
- **Custom Clustering**: Test the clustering algorithm with custom data points
- **Statistics**: Get information about the clustering algorithm and data

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the CSV data file `SPY Chart 2025-08-22-09-36.csv` is in the current directory.

## Running the API

```bash
python main.py
```

The API will be available at:
- **API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Base URL**: http://localhost:8000

## API Endpoints

### Core Endpoints

#### `GET /`
- **Description**: Root endpoint with API information
- **Response**: Basic API information and version

#### `GET /health`
- **Description**: Health check endpoint
- **Response**: API health status and data loading status

### Clustering Endpoints

#### `GET /clustering/projections`
- **Description**: Get clustered pattern projections using the adaptive clustering algorithm
- **Parameters**:
  - `category` (optional, default: "All"): Market condition filter (All, 7D+, 7D-, Tech+, Tech-)
  - `bar_count_filter` (optional, default: "All"): Filter by number of bars (All, 2, 3, 4, 5, 6, 7)
  - `recent_bars` (optional, default: 6): Number of recent bars to project from
  - `max_overlay` (optional, default: 3): Maximum overlay patterns per date
  - `base_gap` (optional, default: 0.2): Base gap for clustering algorithm
  - `max_gap` (optional, default: 1.5): Maximum gap for clustering algorithm
  - `look_ahead` (optional, default: 30): Number of days to look ahead for projections

- **Response**: Simplified clustering data with pattern dates, projections organized by day, and rolling window analysis

#### `GET /clustering/adaptive-cluster`
- **Description**: Apply adaptive clustering algorithm to custom data points
- **Parameters**:
  - `points` (required): Comma-separated list of points to cluster
  - `base_gap` (optional, default: 0.3): Base gap for clustering algorithm
  - `max_gap` (optional, default: 1.5): Maximum gap for clustering algorithm

- **Response**: Cluster mean and clustering details

#### `GET /clustering/statistics`
- **Description**: Get statistics about the clustering algorithm and data
- **Response**: Data statistics and clustering algorithm information

## Clustering Algorithm

The API implements the adaptive clustering algorithm from app2.py:

```python
def get_adaptive_cluster_mean(points, base_gap=0.3, max_gap=1.5):
    """
    Find the mean of the largest cluster in points with an adaptive gap.
    - base_gap: minimum allowed distance between points in a cluster
    - max_gap: maximum allowed distance between points in a cluster
    """
```

### How it works:

1. **Sort Points**: Points are sorted in ascending order
2. **Adaptive Gap Calculation**: The gap between points is dynamically adjusted based on the spread
3. **Cluster Formation**: Points are grouped into clusters based on the adaptive gap
4. **Largest Cluster Selection**: The algorithm finds the largest cluster
5. **Mean Calculation**: Returns the mean of the largest cluster

## Example Usage

### Get Clustered Projections

```bash
# Get all projections
curl "http://localhost:8000/clustering/projections?category=All&bar_count_filter=All&recent_bars=6&max_overlay=3"

# Get projections for 7D+ category with 5-bar patterns
curl "http://localhost:8000/clustering/projections?category=7D%2B&bar_count_filter=5&recent_bars=6&max_overlay=3"

# Get projections for Tech- category with custom clustering parameters
curl "http://localhost:8000/clustering/projections?category=Tech-&bar_count_filter=All&recent_bars=6&max_overlay=3&base_gap=0.3&max_gap=2.0"
```

### Test Clustering Algorithm

```bash
curl "http://localhost:8000/clustering/adaptive-cluster?points=1.0,1.2,1.1,2.5,2.6,2.4,5.0&base_gap=0.3&max_gap=1.5"
```

### Get Statistics

```bash
curl "http://localhost:8000/clustering/statistics"
```

## Response Format

### Clustered Projections Response

```json
{
  "pattern_dates": ["2024-01-15", "2024-01-16", "2024-01-17"],
  "projections": {
    "day_1": {
      "positive": 450.83,
      "negative": 448.17,
      "cluster": 449.5
    },
    "day_2": {
      "positive": 452.1,
      "negative": 446.9,
      "cluster": 449.5
    },
    "day_3": {
      "positive": 453.2,
      "negative": 445.3,
      "cluster": 449.25
    }
  },
  "sectioned_data": {
    "window_5": {
      "highest_positive": 453.2,
      "lowest_negative": 445.3,
      "cluster_change_from_base": -0.25,
      "base_price": 449.5,
      "last_day_cluster_price": 449.25,
      "days_analyzed": 5
    },
    "window_10": {
      "highest_positive": 455.1,
      "lowest_negative": 443.8,
      "cluster_change_from_base": 1.2,
      "base_price": 449.5,
      "last_day_cluster_price": 450.7,
      "days_analyzed": 10
    },
    "window_15": {
      "highest_positive": 456.8,
      "lowest_negative": 442.1,
      "cluster_change_from_base": 2.1,
      "base_price": 449.5,
      "last_day_cluster_price": 451.6,
      "days_analyzed": 15
    }
  }
}
```

## Integration with Original Project

This API directly uses the clustering logic from `app2.py`:

- **Data Loading**: Uses the same data loading functions from the original project
- **Pattern Analysis**: Leverages the existing pattern analysis utilities
- **Clustering Algorithm**: Implements the exact `get_adaptive_cluster_mean` function
- **Cache Management**: Uses the same S3 cache system for pattern data
- **Data File**: Uses the CSV file in the current directory

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input parameters
- **404 Not Found**: Requested resources not found
- **500 Internal Server Error**: Server-side errors

Error responses follow this format:
```json
{
  "success": false,
  "error": "Error Type",
  "message": "Detailed error message",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Dependencies

- **FastAPI**: Modern, fast web framework for building APIs
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Uvicorn**: ASGI server for running the API
- **Boto3**: AWS SDK for S3 cache operations

## License

This project is part of the Stock Market Dashboard system.
