# 📈 SPY Daily Candlestick Chart with Pattern Mapper

This project is a **Streamlit-based web application** that visualizes SPY daily candlestick data and applies a **pattern-matching algorithm** to detect and analyze historical price movement patterns.  
It integrates with **AWS S3** for storing and retrieving pattern cache files.

---

## 🚀 Features

- Interactive **candlestick chart** of SPY with Plotly.  
- Option to display **Moving Average (MA) crossover signals** (Buy/Sell).  
- **Pattern Mapper**:
  - Detects repeating OHLC patterns using configurable steps & gaps.
  - Stores pattern results as JSON **cache files in AWS S3**.
  - Reuses existing cache files to save time.
  - Analyzes historical outcomes of detected patterns.
- Streamlit interface with expandable pattern analysis tables.

---

## 🛠 Requirements

Install dependencies with pip:

````bash
pip install -r requirements.txt
````

---

## ▶️ Run the App
Run the Streamlit app locally:
````bash
streamlit run SPY_Chart_App.py
````
