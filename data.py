import json
import unicodedata
from pathlib import Path

import keras
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf

from config import IMG_H, IMG_W, MAX_LEN, SEED, VAL_RATIO

# Token đặc biệt (Chương 8)
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def load_split(data_dir, split, limit=0):
    """Đọc ảnh (bytes JPEG) và nhãn của tập 'train' hoặc 'test' từ các file Parquet."""
    root = Path(data_dir)
    files = sorted(root.glob(f"{split}-*.parquet")) or sorted(root.glob(f"data/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy {split}-*.parquet trong {root}")
    images, texts = [], []
    for path in files:
        table = pq.read_table(path, columns=["image", "text"])
        images += table.column("image").combine_chunks().field("bytes").to_pylist()
        # Chuẩn NFC để mỗi chữ có dấu chỉ có một cách mã hóa
        texts += [unicodedata.normalize("NFC", t or "") for t in table.column("text").to_pylist()]
        if limit and len(images) >= limit:
            return images[:limit], texts[:limit]
    return images, texts


def split_train_val(n):
    """Chương 1: lấy ngẫu nhiên 10% tập train chính thức làm validation."""
    order = np.random.default_rng(SEED).permutation(n)
    n_val = int(round(n * VAL_RATIO))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def load_train_val(data_dir, limit=0):
    images, texts = load_split(data_dir, "train")
    train_idx, val_idx = split_train_val(len(images))
    if limit:
        train_idx, val_idx = train_idx[:limit], val_idx[:limit]
    train = ([images[i] for i in train_idx], [texts[i] for i in train_idx])
    val = ([images[i] for i in val_idx], [texts[i] for i in val_idx])
    return train, val


# Từ điển ký tự: vocab là list, vị trí trong list chính là id của ký tự
def build_vocab(texts, path):
    """Có vocab.json rồi thì dùng lại (để không làm hỏng trọng số đã train), chưa có thì tạo từ nhãn train."""
    if Path(path).exists():
        return load_vocab(path)
    vocab = SPECIAL + sorted(set("".join(texts)))
    Path(path).write_text(json.dumps(vocab[len(SPECIAL):], ensure_ascii=False), encoding="utf-8")
    return vocab


def load_vocab(path):
    return SPECIAL + json.loads(Path(path).read_text(encoding="utf-8"))


def encode_texts(texts, vocab):
    """'abc' -> [<bos>, a, b, c, <eos>, <pad>, ...], mỗi dòng dài MAX_LEN + 2."""
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    rows = []
    for text in texts:
        ids = [BOS] + [char_to_id.get(ch, UNK) for ch in text[:MAX_LEN]] + [EOS]
        rows.append(ids + [PAD] * (MAX_LEN + 2 - len(ids)))
    return np.array(rows, dtype=np.int32)


def decode_ids(ids, vocab):
    """Đổi dãy id về chuỗi, dừng ở <eos> đầu tiên."""
    chars = []
    for i in ids:
        if i == EOS:
            break
        if i >= len(SPECIAL):
            chars.append(vocab[i])
    return "".join(chars)


# Tiền xử lý ảnh (Chương 5.1) và tăng cường dữ liệu (Chương 5.3, không lật ảnh vì chữ sẽ ngược)
rotate = keras.layers.RandomRotation(2 / 360, fill_mode="constant", seed=SEED, dtype="float32")


def load_image(image_bytes, augment=False):
    image = tf.io.decode_image(image_bytes, channels=1, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if augment:
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
    image = 1.0 - image  # đảo màu: nền = 0, nên pad bằng 0 cũng là pad nền

    # Resize về cao 64, giữ tỉ lệ; ảnh quá dài thì nén ngang cho vừa IMG_W
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    new_w = w * IMG_H / h
    if augment:
        new_w = new_w * tf.random.uniform([], 0.8, 1.2)  # co giãn ngang
    new_w = tf.cast(tf.clip_by_value(tf.round(new_w), 1.0, IMG_W), tf.int32)
    image = tf.image.resize(image, [IMG_H, new_w], antialias=True)

    if augment:
        # Mỗi phép chỉ áp dụng cho khoảng 50% số ảnh
        if tf.random.uniform([]) < 0.5:
            image = rotate(image, training=True)
        if tf.random.uniform([]) < 0.5:
            image = image + tf.random.normal(tf.shape(image), stddev=0.03)
        image = tf.clip_by_value(image, 0.0, 1.0)

    image = tf.image.pad_to_bounding_box(image, 0, 0, IMG_H, IMG_W)
    image.set_shape([IMG_H, IMG_W, 1])
    return image


def make_dataset(images, labels, batch_size, training=False):
    """Dataset cho model.fit: ((ảnh, đầu vào decoder), nhãn đích).

    Teacher forcing (Chương 8): đầu vào decoder = nhãn bỏ token cuối, đích = nhãn bỏ <bos>.
    """
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(len(images), seed=SEED)
    ds = ds.map(lambda img, y: ((load_image(img, training), y[:-1]), y[1:]),
                num_parallel_calls=tf.data.AUTOTUNE)
    if not training:
        ds = ds.cache()  # val không augment nên giữ lại trong RAM sau epoch đầu
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_image_dataset(images, batch_size):
    """Dataset chỉ có ảnh, dùng khi giải mã."""
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
