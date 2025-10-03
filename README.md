# SPY Pattern Analysis Dashboard

A comprehensive stock market pattern analysis dashboard built with Streamlit for analyzing SPY (SPDR S&P 500 ETF Trust) patterns and generating trading projections.

## Project Structure

```
Stock Market Dashboard/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/                          # Data directory
│   └── SPY Chart 2025-08-22-09-36.csv  # Stock market data
└── utils/                         # Utility modules
    ├── __init__.py                # Package initialization
    ├── aws_utils.py               # AWS S3 utilities
    ├── api_utils.py               # External API utilities
    ├── chart_utils.py             # Chart and visualization utilities
    ├── data_utils.py              # Data loading utilities
    └── pattern_utils.py           # Pattern analysis utilities
```

## Features

- **Interactive Candlestick Charts**: Visualize SPY price data with customizable time windows
- **Pattern Analysis**: Identify and analyze recurring price patterns
- **Projection Charts**: Generate future price projections based on historical patterns
- **AWS S3 Caching**: Efficient caching of pattern analysis results
- **External API Integration**: Support for cycle patterns and secondary pattern analysis
- **Real-time Analysis**: Dynamic pattern matching and outcome analysis

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure your SPY data CSV file is placed in the `data/` directory
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to the provided local URL (typically `http://localhost:8501`)

## Configuration

The application uses several configuration parameters defined in `config.py`:

- **AWS S3 Settings**: For caching pattern analysis results
- **API Endpoints**: For external pattern analysis services
- **Analysis Parameters**: Default values for lookback periods, projection days, etc.
- **Data Configuration**: File paths and data settings

## Key Components

### Main Application (`app.py`)
- Streamlit interface and user interactions
- Chart rendering and data visualization
- Pattern analysis orchestration

### Utility Modules (`utils/`)

#### `aws_utils.py`
- S3 client configuration and management
- Cache saving, loading, and existence checking

#### `pattern_utils.py`
- Core pattern analysis algorithms
- Pattern generation and matching
- Outcome analysis and statistics

#### `chart_utils.py`
- Interactive chart creation using Plotly
- Projection visualization
- Highlight rendering and metrics display

#### `data_utils.py`
- Data loading and preprocessing
- CSV file handling with caching

#### `api_utils.py`
- External API integration
- Cycle pattern and secondary pattern fetching

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive charts and visualizations
- **boto3**: AWS SDK for S3 operations
- **numpy**: Numerical computations
- **requests**: HTTP library for API calls

## Data Format

The application expects CSV files with the following columns:
- Date
- Open
- High
- Low
- Close

## Notes

- The application includes placeholder CSV data - replace with your actual SPY data
- AWS credentials are configured in `config.py` - ensure they have proper S3 permissions
- External API endpoints are optional and can be enabled by uncommenting relevant sections in `app.py`
- Pattern analysis results are cached in S3 for performance optimization

