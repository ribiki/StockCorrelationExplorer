# Stock Correlation Explorer 📈

<img width="1494" alt="image" src="https://github.com/user-attachments/assets/6c7e87a7-3908-4dfd-9a71-887ae9827ba2" />

## ✨ Features:
1. Automated Data Pipeline:
  a. Processes zipped CSV files to optimized Parquet format.
  b. Handles missing data and integrated trading calendar.

2. High-Performance Correlation Engine
   a. Incremental sliding window updates (not recomputing from scratch).
   b. Memory-mapped storage for instant access to years of data.
4. Interactive Explorer GUI
   a. Daily correlation matrices
   b. Pair-stock correlation history
   c. Price performance metrics Optimized Architecture
5. Faster implementations with memory efficiencies

## Project layout

```text
.
├── app/ ← Streamlit GUI
│ └── run_gui_app.py
├── data/
│ ├── raw/ ← put your ZIP(s) here e.g. stock_data.zip
│ └── processed/ ← auto-generated parquet + mmap files
├── src/
│ ├── file_loader.py ← CSV → tidy → price matrix
│ ├── correlation_engine.py ← imcremental rolling-window correlation calculations
│ └── … ← (future helpers could be added)
├── utils/
│ └── config.py ← all vairables can be defined here
├── tests/ ← unittest suite
│ ├── test_file_loader.py
│ └── test_correlation_engine.py
└── run.py ← end-to-end pipeline entry-point

```

## ⚙️ Installation
```bash

# Clone repository
git clone https://github.com/ribiki/StockCorrelationExplorer.git
cd StockCorrelationExplorer

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage
1. To run the full pipeline (processing data + GUI)
```bash
python run.py 
```
2. To luanch the GUI directly
```bash
streamlit run app/run_gui_app.py
```
## ✍🏻 Configuration
Modify utils/config.py for:
```python
class CONFIG:
    MARKET_CALENDAR = "NYSE"           # Trading calendar (NYSE, NASDAQ, etc.)
    CORR_WINDOW = 20                   # Rolling window size (days)
    MAX_STOCK_MISSING_PCT = 0.05       # Max % missing data per stock
    PROCESSED_DATA_DIR = "data/processed"  # Output directory
    RAW_ZIP = Path("data/raw/stock_data.zip") # Input .zip file containing .csv files

```

## 🩺 Running the tests
python -m unittest discover -s tests -v
This command covers loader shape/gap-fill logic, sparse-ticker handling, error paths, and numerical agreement of the correlation engine with a Pandas/NumPy reference.


## 💡 Performance notes

| Resource             | Approx. size calculation | Example (500 days, 400–500 tickers, 90-day window) |
|----------------------|--------------------------|----------------------------------------------------|
| **Rolling buffer**   | `window × tickers × 4 bytes` (float32) | `90 × 500 × 4 B` → **≈ 180 MB RAM** |
| **Correlation mmap** | `num_corr_days × num_pairs × 2 bytes` (float16) | `411 × (400×399/2) × 2 B` → **≈ 37 MB disk** |
| **Read-only usage**  | – | Open `CorrelationEngine` *without* `num_tickers`/`window_size`; no RAM buffer is allocated. |

## 📋 Todo
1. Option to remove .bfill() so sparse tickers are really dropped
2. To add more interactive visualisations on the GUI
3. Calendar-aware index (pandas_market_calendars)
4. Dockerfile + GitHub Actions CI
5. CLI sub-commands (stock-corr load|corr|gui)













