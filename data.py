"""Đọc dữ liệu VietHTR-Line, chia tập và xây từ điển ký tự.

Kiến thức dùng:
- Chương 1: chia dữ liệu train / validation / test.
- Chương 8 (Encoder-Decoder): tokenization, token <bos>/<eos>, pad chuỗi về cùng độ dài.
- Chương 3.2 (Softmax Regression): mỗi ký tự là một lớp; mô hình dự đoán phân phối softmax trên từ điển.
"""
import json
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


# ---------------------------------------------------------------------------
# Đọc Parquet
# ---------------------------------------------------------------------------
def parquet_files(data_dir, split):
    """Trả về danh sách file Parquet của split 'train' hoặc 'test'."""
    pattern = "train-*.parquet" if split == "train" else "test-*.parquet"
    root = Path(data_dir)
    files = sorted(root.glob(pattern)) or sorted(root.glob("data/" + pattern))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy {pattern} trong {root} (hoặc {root}/data)")
    return files


def normalize_text(text):
    """Chuẩn Unicode NFC để mỗi chữ có dấu chỉ có một cách mã hóa."""
    return unicodedata.normalize("NFC", text or "")


def load_split(data_dir, split, limit=0):
    """Đọc toàn bộ một split vào RAM: (list bytes ảnh JPEG, list nhãn văn bản).

    Toàn bộ train khoảng 1,3 GB bytes ảnh, vừa với RAM của Kaggle.
    limit > 0 chỉ đọc limit dòng đầu (dùng khi chạy thử).
    """
    images, texts = [], []
    for path in parquet_files(data_dir, split):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["image", "text"], batch_size=512):
            image_col = batch.column("image").field("bytes").to_pylist()
            text_col = batch.column("text").to_pylist()
            for image_bytes, text in zip(image_col, text_col):
                images.append(image_bytes)
                texts.append(normalize_text(text))
                if limit and len(images) >= limit:
                    return images, texts
    return images, texts


def train_val_indices(n, val_ratio, seed):
    """Hoán vị ngẫu nhiên có seed, lấy val_ratio đầu làm validation."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = int(round(n * val_ratio))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def load_all_splits(cfg, need_train=True, need_test=True):
    """Trả về dict {'train': (images, texts), 'val': ..., 'test': ...}."""
    splits = {}
    if need_train:
        images, texts = load_split(cfg.data_dir, "train")
        train_idx, val_idx = train_val_indices(len(images), cfg.val_ratio, cfg.seed)
        if cfg.limit_train:
            train_idx = train_idx[: cfg.limit_train]
        if cfg.limit_val:
            val_idx = val_idx[: cfg.limit_val]
        splits["train"] = ([images[i] for i in train_idx], [texts[i] for i in train_idx])
        splits["val"] = ([images[i] for i in val_idx], [texts[i] for i in val_idx])
    if need_test:
        splits["test"] = load_split(cfg.data_dir, "test", cfg.limit_test)
    return splits


# ---------------------------------------------------------------------------
# Từ điển ký tự
# ---------------------------------------------------------------------------
class Vocab:
    def __init__(self, chars):
        self.itos = SPECIAL_TOKENS + list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, texts):
        chars = sorted({ch for text in texts for ch in text})
        return cls(chars)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.itos[len(SPECIAL_TOKENS):], ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def encode(self, text, max_len):
        """'abc' -> [<bos>, a, b, c, <eos>, <pad>, ...], tổng độ dài max_len + 2."""
        ids = [self.stoi.get(ch, UNK) for ch in text[:max_len]]
        ids = [BOS] + ids + [EOS]
        return ids + [PAD] * (max_len + 2 - len(ids))

    def decode(self, ids):
        """Dừng ở <eos> đầu tiên, bỏ các token đặc biệt."""
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i >= len(SPECIAL_TOKENS):
                out.append(self.itos[i])
        return "".join(out)


def get_vocab(cfg, train_texts=None):
    """Đọc vocab.json nếu đã có, nếu chưa thì xây từ nhãn train và lưu lại."""
    if cfg.vocab_path.exists():
        return Vocab.load(cfg.vocab_path)
    if train_texts is None:
        raise FileNotFoundError(f"Chưa có {cfg.vocab_path}; hãy chạy train.py trước.")
    vocab = Vocab.build(train_texts)
    vocab.save(cfg.vocab_path)
    return vocab


def encode_texts(vocab, texts, max_len):
    return np.array([vocab.encode(t, max_len) for t in texts], dtype=np.int32)
