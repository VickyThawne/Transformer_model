import tiktoken
from pathlib import Path
import numpy as np

class DataFeeder:
    def __init__(self, file_path, training_ratio=0.75):
        self.file_path = file_path
        self.training_ratio = training_ratio
        self.enc = tiktoken.get_encoding("gpt2")

    def _split_data(self, text):
        n = int(self.training_ratio * len(text))
        train_data = text[:n]
        val_data = text[n:]
        return train_data, val_data

    def _encode_data(self, data):
        return self.enc.encode_ordinary(data)

    def _write_to_bin(self, name, data):
        ids = np.array(data, dtype=np.uint16)
        return ids.tofile(f'data/{name}.bin')

    def preprocess_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        train_data, val_data = self._split_data(text)
        print(f"train has {len(train_data):,} tokens")
        print(f"val has {len(val_data):,} tokens")

        train_ids = self._encode_data(train_data)
        val_ids = self._encode_data(val_data)

        self._write_to_bin("train", train_ids)
        self._write_to_bin("val", val_ids)

        return train_ids, val_ids

if __name__ == "__main__":
    sherlocks_diary_path = Path('data/sherlocks_diary.txt')
    data_feeder = DataFeeder(sherlocks_diary_path)
    train_data, val_data = data_feeder.preprocess_data()
