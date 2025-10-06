"""
Startup script for the Stock Market Pattern Clustering API
"""

import uvicorn
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install them using:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_data_file():
    """Check if the required data file exists"""
    # Check if the data file exists in the current directory
    data_file = Path(__file__).parent / "SPY Chart 2025-08-22-09-36.csv"
    
    if data_file.exists():
        print(f"✅ Data file found: {data_file}")
        return True
    else:
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure the CSV file 'SPY Chart 2025-08-22-09-36.csv' is in the current directory.")
        return False

def main():
    """Main startup function"""
    print("=" * 60)
    print("Stock Market Pattern Clustering API Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data file
    if not check_data_file():
        print("\n⚠️  Warning: Data file not found. The API may not work correctly.")
        response = input("Do you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("\n🚀 Starting the Clustering API server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("📚 ReDoc Documentation will be available at: http://localhost:8000/redoc")
    print("🔗 API Base URL: http://localhost:8000")
    print("\n🎯 Main endpoint: /clustering/projections")
    print("🧪 Test endpoint: /clustering/adaptive-cluster")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Start the server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
