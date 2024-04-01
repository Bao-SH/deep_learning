import os
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()


def download_and_extract_data(competition_slug, download_dir, extract_dir):
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    # Download the dataset
    api.competition_download_files(competition_slug, path=download_dir)
    # Extract the dataset
    with ZipFile(f"{download_dir}/{competition_slug}.zip", "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Dataset downloaded and extracted successfully.")
    os.remove(f"{download_dir}/{competition_slug}.zip")
