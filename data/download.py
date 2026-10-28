"""Download the datasets the course uses into data/raw/.

Run: python data/download.py

Notebooks also call `ensure_online_retail()` themselves, so running this
script is a convenience, not a requirement.
"""
from pathlib import Path
import zipfile

import requests

RAW = Path(__file__).parent / "raw"
RAW.mkdir(exist_ok=True)

ONLINE_RETAIL_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ONLINE_RETAIL_XLSX = RAW / "online_retail_II.xlsx"


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def ensure_online_retail() -> Path:
    """Return the path to online_retail_II.xlsx, downloading it if needed.

    Source: Online Retail II, UCI Machine Learning Repository (CC BY 4.0).
    Two sheets (2009-2010 and 2010-2011) of real transactions from a
    UK-based online retailer. This is the "ShopWave" data in the course.
    """
    if ONLINE_RETAIL_XLSX.exists():
        return ONLINE_RETAIL_XLSX
    zip_path = RAW / "online_retail_ii.zip"
    _download(ONLINE_RETAIL_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".xlsx"):
                z.extract(name, RAW)
                (RAW / name).rename(ONLINE_RETAIL_XLSX)
    zip_path.unlink()
    print(f"Saved {ONLINE_RETAIL_XLSX}")
    return ONLINE_RETAIL_XLSX


def load_retail(sheet: str = "Year 2010-2011"):
    """Return the raw transactions as a DataFrame, fast after the first call.

    Reading the Excel sheet takes 10-60 seconds, so the first call saves a
    Parquet copy next to it; later calls read that copy in about a second.
    """
    import pandas as pd

    cache = RAW / f"online_retail_{sheet.split()[-1]}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_excel(ensure_online_retail(), sheet_name=sheet)
    # Excel stores some codes and descriptions as numbers; make them all text
    for col in ["Invoice", "StockCode", "Description"]:
        df[col] = df[col].astype("string")
    df.to_parquet(cache, index=False)
    return df


def load_reviews(n: int = 10_000, seed: int = 42):
    """Return n Amazon product reviews (half positive, half negative) as a DataFrame.

    Source: amazon_polarity (Zhang, Zhao and LeCun, 2015; Apache 2.0), test split:
    400,000 reviews rated 1-2 stars (label 0, negative) or 4-5 stars (label 1,
    positive). ShopWave has no review data of its own, so these stand in for it.
    Columns: title, content, text (title + content), label.
    The first call downloads one 117 MB file; the sample is cached in data/raw/.
    """
    import pandas as pd

    cache = RAW / f"reviews_{n}_{seed}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("fancyzhx/amazon_polarity",
                           "amazon_polarity/test-00000-of-00001.parquet", repo_type="dataset")
    full = pd.read_parquet(path)
    df = (full.groupby("label").sample(n // 2, random_state=seed)
              .sample(frac=1, random_state=seed).reset_index(drop=True))
    title = df["title"].str.strip()
    sep = title.str[-1:].isin([".", "!", "?"]).map({True: " ", False: ". "})
    df["text"] = title + sep + df["content"].str.strip()
    df = df[["title", "content", "text", "label"]]
    df.to_parquet(cache, index=False)
    return df


if __name__ == "__main__":
    ensure_online_retail()
    for sheet in ["Year 2010-2011", "Year 2009-2010"]:
        print(sheet, load_retail(sheet).shape)
    print("Done.")
