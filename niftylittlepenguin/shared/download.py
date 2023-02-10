import os
import urllib.request

class Downloader:

    def check_exists(self, path: str) -> bool:
        if os.path.exists(path):
            return True
        else:
            return False

    def download(self, url: str, path: str):
        """ Download the file from `url` and save it locally under `path`: """
        if self.check_exists(path):
            print("Data already exists. Skipping download.")
            return
        with urllib.request.urlopen(url) as response, open(path, 'w', encoding="utf-8") as out_file:
            print(f"Downloading data from '{url}' to '{path}'.")
            data = response.read().decode("utf-8")
            out_file.write(data)
