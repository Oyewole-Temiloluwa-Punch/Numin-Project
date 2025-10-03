"""
Configuration settings for the Stock Market Dashboard
"""

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = "AKIARVSAETUO7JZ6FK5D"
AWS_SECRET_ACCESS_KEY = "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"
AWS_REGION = "eu-north-1"
BUCKET_NAME = "numin-cache-files"

# API Configuration
CYCLE_PATTERN_API = "https://cycle-pattern-api-yr4x.onrender.com/api/v1/find-prior-days-with-same-pattern"
SECONDARY_API_BASE = "http://52.53.169.196:8002/api/patterns"

# Analysis Parameters
DEFAULT_LOOKBACK = 1800
DEFAULT_PROJ_DAYS = 30
DEFAULT_REAL_BARS = 30
DEFAULT_WINDOW_SIZE = 50
DEFAULT_STEPS = 15
DEFAULT_GAP = 3

# Data Configuration
CSV_FILENAME = "SPY Chart 2025-08-22-09-36.csv"
CSV_FILENAME_SEPTEMBER = "SPY Chart September.csv"
CSV_FILENAME_AUGUST = "SPY Chart August.csv"
DATA_DIR = "data"
