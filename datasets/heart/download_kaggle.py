"""Download the requested Kaggle Heart Transplant Survival Dataset.
Requires internet access and, for private/rate-limited Kaggle resources, Kaggle credentials.
"""
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
URL = "https://www.kaggle.com/api/v1/datasets/download/ayyappanmarimuthu/heart-transplant-survival-dataset"
ZIP = ROOT / "heart-transplant-survival-dataset.zip"

def main():
    print("Downloading requested Kaggle dataset...")
    urllib.request.urlretrieve(URL, ZIP)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(ROOT)
    print("Extracted to", ROOT)
    print("The LifeLink-AI pipeline will automatically detect CSV files in this folder.")

if __name__ == '__main__':
    main()
