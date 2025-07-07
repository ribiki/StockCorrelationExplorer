# Stock Correlation Explorer 📈

## This is a mini app (Link: http://localhost:8501/)

<img width="1494" alt="image" src="https://github.com/user-attachments/assets/51901d33-d839-4afc-84cf-a0d0dadd7d4d" />

## 📖 Overview
A Python application that:
1. Processes raw stock price data (CSV/ZIP) into a cleaned price matrix
2. Computes rolling-window correlations between all stock pairs
3. Provides an interactive GUI to explore correlations and performance statistics

### 💡 Key features:
1. Memory-efficient correlation calculations using memory-mapped files
2. Streamlit-based interactive visualization
3. Automatic handling of missing data and date alignment
4. Performance metrics (returns, volatility) alongside correlations

---

## 1. 🛠️ Components:

🛠️ Components

```text
.
├── app/ ← Streamlit GUI for interactive exploration
│ └── run_gui_app.py
├── data/
│ ├── raw/ ← put your ZIP(s) here e.g. stock_data.zip
│ └── processed/ ← auto-generated parquet + mmap files
├── src/
│ ├── file_loader.py ← Data loading and preprocessing utilities: CSV → cleanup → price matrix
│ ├── correlation_engine.py ← High-performance rolling correlation calculator
│ └── … ← (future helpers)
├── utils/
│ └── config.py ← Configuration constants
├── tests/ ← unittest suite
│ ├── test_file_loader.py
│ └── test_correlation_engine.py
└── **run.py** ← Main pipeline (data processing → correlation calculation → GUI launch)

```


---

## 2 Approach

| Step | Script | Key ideas |
|------|--------|-----------|
| **Extract & clean** | `src/file_loader.build_price_matrix` | *Pivot only the dates that actually occur* in the CSV rows, forward- & backward-fill small gaps, drop tickers with too many missing values, write one Parquet file. |
| **Rolling correlations** | `src/correlation_engine.CorrelationEngine` | Maintains sliding sums in float64, computes Pearson correlations in a Numba kernel, stores each upper-triangular matrix in an on-disk mmap (`float16` ≈ 2 bytes per value). |
| **GUI** | `app/run_gui_app.py` | Loads the Parquet(s), lets you pick date range + two tickers, shows correlation coefficient + basic performance metrics. |
| **Orchestration** | `run.py` | Calls the loader, streams returns through the engine, writes `data/processed/corr_mmap.bin`, then launches Streamlit. |

---

## 3 Prerequisites

* ***Python 3.9 – 3.12**
* ***pip / venv** (or conda/mamba)
* Tested with the stack below. If you use newer NumPy you may need the Numba
  nightly (or pin NumPy as shown):

```text
numpy==1.26.0
pandas==2.1.0
numba==0.58.1
streamlit==1.28.0
plotly==5.18.0
pyarrow==14.0.1
tqdm==4.66.1
```

## 4 Installation

Clone the repository, create a virtual-environment, and install the dependencies.
git clone https://github.com/<your-user>/StockCorrelationExplorer.git
cd StockCorrelationExplorer

python -m venv .venv        # create virtual env
source .venv/bin/activate   # Windows: .venv\Scripts\activate

## install requirements (Pin NumPy for Numba compatibility)
pip install -r requirements.txt
## Please note:
pip install 'numpy<2.3' pandas numba pyarrow streamlit tqdm


## 5 Running the full pipeline

1. Drop your ZIP(s)** into `data/raw/`
2. Each CSV inside the archive must contain these three columns (one trading-day file per CSV):
```text
Ticker,Date,Price
AAA,2024-01-02,10.23
BBB,2024-01-02,20.11
```
2. Launch the end-to-end script:
```text
python run.py
```
The script will:
1. Extract & clean the raw files → write.
2. Save price_matrix_YYYYMMDD_YYYYMMDD.parquet to data/processed/.
3. Compute rolling correlations using the window length defined by CONFIG.CORR_WINDOW (default 20 trading days).
5. Start the Streamlit GUI at http://localhost:8501.
6. Explore correlations interactively in your browser.


## 6 Configuration
All configs live in utils/config.py.
```text
MAX_STOCK_MISSING_PCT = 0.35         # drop ticker if >35 % still NaN
CORR_WINDOW            = 90          # rolling window length (trading days)
PROCESSED_DATA_DIR     = Path("data/processed")
RAW_ZIP                = Path("data/raw/stock_data.zip")
```
## 7 GUI features:
1. Date Range Selection: Choose any subset of available data
2. Automatic validation of overlapping trading days
3. Ticker Pair Analysis:
   a. Correlation coefficient calculation
   b. Total return comparison
   c. Volatility metrics (daily/monthly/annualized)
4. Data Quality Indicators
5. Missing data warnings
6. Gap detection between trading days


## 8 Running the tests
python -m unittest discover -s tests -v
This command covers loader shape/gap-fill logic, sparse-ticker handling, error paths, and numerical agreement of the correlation engine with a Pandas/NumPy reference.


## 9 Performance notes

| Resource             | Approx. size calculation | Example (500 days, 400–500 tickers, 90-day window) |
|----------------------|--------------------------|----------------------------------------------------|
| **Rolling buffer**   | `window × tickers × 4 bytes` (float32) | `90 × 500 × 4 B` → **≈ 180 MB RAM** |
| **Correlation mmap** | `num_corr_days × num_pairs × 2 bytes` (float16) | `411 × (400×399/2) × 2 B` → **≈ 37 MB disk** |
| **Read-only usage**  | – | Open `CorrelationEngine` *without* `num_tickers`/`window_size`; no RAM buffer is allocated. |

## 10 Todo
1. Option to remove .bfill() so sparse tickers are really dropped
2. To add more interactive visualisations on the GUI
3. Calendar-aware index (pandas_market_calendars)
4. Dockerfile + GitHub Actions CI
5. CLI sub-commands (stock-corr load|corr|gui)


## 11 GitHub
git remote add origin git@github.com:ribiki/StockCorrelationExplorer.git
git branch -M main          # rename master → main (optional)
git push -u origin main













