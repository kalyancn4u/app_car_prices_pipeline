# 🗃️ Data-Format Benchmarks — CSV vs Parquet vs Feather

**Measured, not estimated** — produced by `car_pricing.data.benchmark_formats`
on the real cleaned dataset (19,820 × 17), reproducible in
[`notebooks/03_data_format_benchmarks.ipynb`](../notebooks/03_data_format_benchmarks.ipynb).
Read time is the fastest of 5 pandas reads on one machine; absolute numbers vary
by hardware, but the **ranking is stable**.

| Format | Size | vs CSV | Write | Read | Human-readable? | Excel double-click? | Keeps dtypes? |
| :----- | ---: | -----: | ----: | ---: | :-------------- | :------------------ | :------------ |
| **CSV (raw)** | 1,535 KB | 100 % | 99 ms | 23 ms | ✅ plain text | ✅ yes | ❌ all text |
| **CSV + gzip** (`.csv.gz`) | 290 KB | 19 % | 300 ms | 26 ms | ⚠️ after unzip | ❌ archive tool | ❌ |
| **CSV + xz / LZMA** (`.csv.xz`) | 163 KB | 11 % | 602 ms | 32 ms | ⚠️ after unzip | ❌ | ❌ |
| **CSV + bzip2** (`.csv.bz2`) | 151 KB | 10 % | 209 ms | 70 ms | ⚠️ after unzip | ❌ | ❌ |
| **Parquet (snappy)** | 274 KB | 18 % | 25 ms | **8 ms** | ❌ binary | ❌ needs tools | ✅ yes |
| **Parquet (zstd)** | 227 KB | 14 % | 22 ms | **8 ms** | ❌ binary | ❌ | ✅ yes |
| **Feather / Arrow** | 867 KB | 57 % | **10 ms** | **5 ms** | ❌ binary | ❌ | ✅ yes |

## How to read it

- **Smallest on disk:** `CSV + bzip2` (10 %), but the **slowest to read** (~3× a
  raw CSV) *and* slow to write — good only for cold archival you rarely open.
- **Best low-friction shrink for a git repo:** **`CSV + gzip`** — 5× smaller than
  raw CSV, reads just as fast, one command (`gzip -9 file.csv`), and pandas reads
  it directly with no code change. (This is what the sibling `app_car_prices_flask`
  ships.)
- **Best for real pipelines / larger data:** **Parquet** — nearly as small as gzip
  **and ~3–4× faster to read** than CSV, because it's *columnar* and stores each
  column's dtype (no "is this a number or a string?" guessing on load). Needs
  `pyarrow`. The analytics default.
- **Fastest read + write, size no object:** **Feather/Arrow** — near-instant, but
  ~3× larger than Parquet; ideal for short-lived local hand-offs between steps.
- **Most universal & readable:** **raw CSV** — opens anywhere in Excel/Notepad with
  no library; the price is size and lost type info.

## Why it opens in Excel (and why that's a trap)

On Windows a `.csv` is associated with **Excel**, so double-clicking opens it
there. But Excel silently rewrites CSVs — strips leading zeros (`007`→`7`), turns
long numbers into scientific notation (`9.19E+09`), reads `3-4` as a date, and can
change the encoding on save. **Never double-click a data CSV, edit, and Save.**
To inspect safely: a text editor, `head file.csv`, or `pd.read_csv(..., nrows=5)`.
A `.csv.gz` won't open in Excel at all — its extension is `.gz`, so Windows hands
it to an archive tool; peek with `zcat f.csv.gz | head` or `pd.read_csv("f.csv.gz")`.

## Decision for this project

`data/raw/` ships the dataset as a **gzip-compressed CSV** (`.csv.gz`, ~81%
smaller) — the low-friction size win: `pandas.read_csv` decompresses it
transparently, so no code changes and the file stays a plain CSV once read. The
pipeline can still cache a **Parquet** copy via `data.write_dataframe(...)` when a
fast, type-safe reload is wanted during iterative modelling. If the dataset ever
grew to millions of rows, Parquet would become the primary storage format.

> 🧭 **Rules of thumb:** *human sharing / tiny files* → **CSV**; *shrink a CSV in
> a repo with zero friction* → **CSV + gzip**; *a real analytics workflow or big
> data* → **Parquet**; *max read speed for temp files* → **Feather**. Avoid
> **pickle** for datasets — fast, but it runs arbitrary code on load (a security
> risk) and breaks across library versions.
