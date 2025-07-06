# Stock Correlation Explorer 📈

This is a mini app that:

1. **Ingests** daily price CSVs packed inside one or many ZIP archives,
2. **Builds** a clean `[date × ticker]` price matrix (Parquet),
3. **Streams** daily returns through a blazing-fast Numba engine to create a memory-mapped store of rolling correlation matrices,
4. **Visualises** user can evaluate any two tickers in a Streamlit GUI.

Everything is pure-Python, no database required.

---

## 1. Project layout

```text
.
├── app/ ← Streamlit GUI
│ └── run_gui_app.py
├── data/
│ ├── raw/ ← put your ZIP(s) here e.g. stock_data.zip
│ └── processed/ ← auto-generated parquet + mmap files
├── src/
│ ├── file_loader.py ← CSV → tidy → price matrix
│ ├── correlation_engine.py ← rolling-window Pearson correlations
│ └── … ← (future helpers)
├── utils/
│ └── config.py ← single source of truth for all tunables
├── tests/ ← unittest suite
│ ├── test_file_loader.py
│ └── test_correlation_engine.py
└── run.py ← end-to-end pipeline entry-point

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

* **Python 3.9 – 3.12**
* **pip / venv** (or conda/mamba)
* Tested with the stack below. If you use newer NumPy you may need the Numba
  nightly (or pin NumPy as shown):

```text
numpy    >=2.2,<2.3
pandas   >=2.2
numba    >=0.58
pyarrow  >=15
streamlit>=1.32


# clone the repo
git clone https://github.com/ribiki/StockCorrelationExplorer.git
cd StockCorrelationExplorer

# create & activate venv
python -m venv .venv
source .venv/bin/activate       # (Windows: .venv\Scripts\activate)

# install dependencies
pip install -r requirements.txt
# or pin NumPy for Numba:
pip install 'numpy<2.3' pyarrow pandas numba streamlit tqdm

Running the full pipeline
Drop your raw ZIP(s) into data/raw/ (e.g. stock_data.zip with one CSV
per trading day, columns Ticker,Date,Price).

```
## 4 Installation
Clone the repository, create a virtual-environment, and install the dependencies.
git clone https://github.com/<your-user>/StockCorrelationExplorer.git
cd StockCorrelationExplorer

python -m venv .venv        # create virtual env
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install deps (pin NumPy for Numba compatibility)
pip install -r requirements.txt
# or explicit:
pip install 'numpy<2.3' pandas numba pyarrow streamlit tqdm


## 5 Running the full pipeline

1. **Drop your ZIP(s)** into `data/raw/` – for example:


Each CSV inside the archive must contain these three columns  
(one trading-day file per CSV):

```text
Ticker,Date,Price
AAA,2024-01-02,10.23
BBB,2024-01-02,20.11
```
2. Launch the end-to-end script:

python run.py

The script will:

1. Extract & clean the raw files → write.
2. Save price_matrix_YYYYMMDD_YYYYMMDD.parquet to data/processed/.
3. Compute rolling correlations using the window length defined by CONFIG.CORR_WINDOW (default 20 trading days).
5. Start the Streamlit GUI at http://localhost:8501.
6. Explore correlations interactively in your browser.


## Configuration
All configs live in utils/config.py.
```text
MAX_STOCK_MISSING_PCT = 0.35         # drop ticker if >35 % still NaN
CORR_WINDOW            = 90          # rolling window length (trading days)
PROCESSED_DATA_DIR     = Path("data/processed")
RAW_ZIP                = Path("data/raw/stock_data.zip")
```


## Running the tests
python -m unittest discover -s tests -v
This command covers loader shape/gap-fill logic, sparse-ticker handling, error paths, and numerical agreement of the correlation engine with a Pandas/NumPy reference.


## 8 Performance notes

| Resource             | Approx. size calculation | Example (500 days, 400–500 tickers, 90-day window) |
|----------------------|--------------------------|----------------------------------------------------|
| **Rolling buffer**   | `window × tickers × 4 bytes` (float32) | `90 × 500 × 4 B` → **≈ 180 MB RAM** |
| **Correlation mmap** | `num_corr_days × num_pairs × 2 bytes` (float16) | `411 × (400×399/2) × 2 B` → **≈ 37 MB disk** |
| **Read-only usage**  | – | Open `CorrelationEngine` *without* `num_tickers`/`window_size`; no RAM buffer is allocated. |

## 9 Todo
1. Option to remove .bfill() so sparse tickers are really dropped
2. To add more interactive visualisations on the GUI
3. Calendar-aware index (pandas_market_calendars)
4. Dockerfile + GitHub Actions CI
5. CLI sub-commands (stock-corr load|corr|gui)


## 10 GitHub
git remote add origin git@github.com:ribiki/StockCorrelationExplorer.git
git branch -M main          # rename master → main (optional)
git push -u origin main













