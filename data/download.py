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


if __name__ == "__main__":
    ensure_online_retail()
    print("Done.")
