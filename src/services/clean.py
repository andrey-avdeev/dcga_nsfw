import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

from settings import IMAGE_SUFFIXES, WORKERS


class CleanService:
    def __init__(self,
                 root_dir: str,
                 save_suffixes: List[str] = IMAGE_SUFFIXES):
        self._root_dir = root_dir
        self._save_suffixes = save_suffixes

    def scan(self):
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            for path, subdirs, filenames in os.walk(self._root_dir):
                for filename in filenames:
                    filepath = os.path.join(path, filename)
                    executor.submit(self._clean, filepath)

    def _clean(self, filepath: str):
        suffix = filepath.split(".")[-1]
        if suffix not in self._save_suffixes:
            print(f'remove={filepath}')
            # os.remove(filepath)
