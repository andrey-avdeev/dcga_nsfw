import os
from concurrent.futures.thread import ThreadPoolExecutor

import requests

from settings import WORKERS


class ScrapeService:
    def __init__(self,
                 root_dir: str):
        self._root_dir = root_dir

    def scrape(self):
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            for dirpath, dirnames, filenames in os.walk(self._root_dir):
                if len(filenames) > 0:
                    for filename in filenames:
                        suffix = filename.split(".")[-1]
                        if suffix != "txt":
                            continue

                        path = os.path.join(dirpath, filename)
                        with open(path, 'r') as f:
                            lines = f.readlines()
                            for url in lines:
                                executor.submit(self._download_image, url.strip(), dirpath)

    def _download_image(self, url: str, folder: str):
        file_name = url.split("/")[-1]
        file_path = os.path.join(folder, file_name)
        if os.path.exists(file_path):
            print(f"SKIP: {url} -> {file_path}")
            return

        try:
            download_file = requests.get(url, timeout=3)
            status = download_file.status_code
            if status == 200:
                with open(file_path, 'wb') as outfile:
                    outfile.write(download_file.content)

                print(f"OK: {url} -> {file_path}")
            else:
                print(f"ERROR_I: {url} -> {file_path} ", download_file.status_code)
        except Exception as e:
            print(f"ERROR_O: {url} -> {file_path} ", e)
