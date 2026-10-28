"""Download the raw data into data/raw/.

Run from the project root:  python src/download_data.py

Anyone cloning your repository must be able to recreate your data with this
one command. Never commit the data itself.
"""
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

URL = "https://TODO/your-dataset.csv"     # TODO: a direct download link
DEST = RAW / "TODO.csv"                   # TODO: file name


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print("already downloaded:", DEST)
        return
    print("downloading", URL)
    urllib.request.urlretrieve(URL, DEST)
    print("saved", DEST)


if __name__ == "__main__":
    main()
