# 📈 SPY Daily Candlestick Chart + Pattern Mapper (Streamlit)

An interactive Streamlit app that:

- Plots a **daily candlestick chart** of SPY with optional **moving averages** and **buy/sell crossover** markers.
- Computes a **Pattern Mapper** for a selected date:
  - Generates symbolic OHLC patterns (configurable `gap` and `steps`)
  - Finds historical **occurrences** of that exact pattern
  - Computes **forward-day outcome statistics** for up to 30 days
- **Caches** pattern-computation results as JSON to **AWS S3** and reuses them on subsequent runs.

---

## 🚀 Quick Start

### 1) Clone & enter the project

```bash
git clone <your-repo-url>
cd <your-repo-folder>
