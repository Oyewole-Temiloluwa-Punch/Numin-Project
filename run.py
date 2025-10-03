"""
Simple script to run the Streamlit application
"""

import subprocess
import sys

def main():
    """Run the Streamlit application"""
    try:
        # Check if streamlit is installed
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed")

    # Run the streamlit app using subprocess
    print("🚀 Starting Streamlit application...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
