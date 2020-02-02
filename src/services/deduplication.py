import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Dict

from settings import WORKERS, IMAGE_SUFFIXES
from utils import load_and_hash
import pickle


class DeduplicationService:
    def __init__(self,
                 root_dir: str,
                 image_suffixes: List[str] = IMAGE_SUFFIXES,
                 index_filename: str = "image_index.pkl"):
        self._root_dir = root_dir
        self._image_suffixes = image_suffixes
        self._image_index: Dict[str, List[Dict]] = {}
        self._index_filename = index_filename

    def scan(self):
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            for path, subdirs, filenames in os.walk(self._root_dir):
                for filename in filenames:
                    filepath = os.path.join(path, filename)
                    executor.submit(self._calc_hash, filepath)

    def _calc_hash(self, filepath: str):
        suffix = filepath.split(".")[-1]
        if suffix not in self._image_suffixes:
            raise ValueError(f"not an image: {filepath}")
        else:
            hash_, meta = load_and_hash(filepath)
            if hash_ not in self._image_index:
                print(f"original={filepath}")
                self._image_index[hash_] = [meta]
            else:
                original_filepath = self._image_index[hash_][0]["filepath"]
                print(f"duplication:original={original_filepath}\nduplicate={filepath}")
                self._image_index[hash_].append(meta)

    def save(self):
        checkpoint = {
            "image_index": self._image_index,
        }

        with open(self._index_filename, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self._index_filename, 'rb') as f:
            checkpoint = pickle.load(f)
            self._image_index = checkpoint["image_index"]
