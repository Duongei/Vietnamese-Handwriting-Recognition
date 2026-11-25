"""Tiền xử lý ảnh và tạo pipeline tf.data.

Kiến thức dùng:
- Chương 5.1 (CNN): ảnh là tensor H x W x C; padding để mọi ảnh cùng kích thước.
- Chương 5.3 (CNN techniques): data augmentation (xoay, co giãn, nhiễu, độ sáng) giúp giảm overfitting.
  Không lật ảnh vì chữ sẽ bị ngược.
"""
import math

import tensorflow as tf

from data import PAD


def _rotate(image, max_degrees):
    """Xoay nhẹ quanh tâm ảnh bằng phép biến đổi affine (nền = 0 = giấy trắng sau khi đảo màu)."""
    angle = tf.random.uniform([], -max_degrees, max_degrees) * math.pi / 180.0
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cx, cy = w / 2.0, h / 2.0
    cos, sin = tf.cos(angle), tf.sin(angle)
    transform = tf.stack([cos, sin, cx - cos * cx - sin * cy,
                          -sin, cos, cy + sin * cx - cos * cy, 0.0, 0.0])
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis], transforms=transform[tf.newaxis],
        output_shape=tf.shape(image)[:2], fill_value=0.0,
        interpolation="BILINEAR", fill_mode="CONSTANT")
    return out[0]


def load_image(image_bytes, img_height, img_width, augment=False):
    """bytes JPEG -> tensor (img_height, img_width, 1), mực = 1, nền = 0.

    1. Đọc ảnh xám, đưa về [0, 1].
    2. (train) đổi độ sáng / tương phản.
    3. Đảo màu để nền bằng 0, nhờ đó pad bằng 0 chính là pad nền trắng.
    4. Resize cao img_height, giữ tỉ lệ; ảnh quá dài thì nén ngang cho vừa img_width.
    5. (train) co giãn ngang 0,8-1,2 lần, xoay ±2 độ, thêm nhiễu Gauss.
    6. Pad bên phải tới img_width.
    """
    image = tf.io.decode_image(image_bytes, channels=1, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if augment:
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
    image = 1.0 - image

    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    new_w = w * img_height / h
    if augment:
        new_w = new_w * tf.random.uniform([], 0.8, 1.2)
    new_w = tf.cast(tf.clip_by_value(tf.round(new_w), 1.0, float(img_width)), tf.int32)
    image = tf.image.resize(image, [img_height, new_w], antialias=True)

    if augment:
        image = tf.cond(tf.random.uniform([]) < 0.5, lambda: _rotate(image, 2.0), lambda: image)
        image = tf.cond(tf.random.uniform([]) < 0.5,
                        lambda: image + tf.random.normal(tf.shape(image), stddev=0.03),
                        lambda: image)
        image = tf.clip_by_value(image, 0.0, 1.0)

    image = tf.image.pad_to_bounding_box(image, 0, 0, img_height, img_width)
    image.set_shape([img_height, img_width, 1])
    return image


def make_train_dataset(images, labels, cfg, training):
    """Dataset cho model.fit: ((ảnh, đầu vào decoder), nhãn đích).

    labels: mảng (N, max_text_len + 2) = <bos> c1 ... cn <eos> <pad>...
    Teacher forcing (Chương 8): đầu vào decoder = labels[:, :-1], đích = labels[:, 1:].
    """
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(len(images), seed=cfg.seed, reshuffle_each_iteration=True)
    augment = training and cfg.augment

    def _map(image_bytes, label):
        image = load_image(image_bytes, cfg.img_height, cfg.img_width, augment)
        return (image, label[:-1]), label[1:]

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)


def make_image_dataset(images, cfg, batch_size=None):
    """Dataset chỉ có ảnh (không augmentation) để suy luận."""
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(lambda b: load_image(b, cfg.img_height, cfg.img_width, False),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size or cfg.batch_size).prefetch(tf.data.AUTOTUNE)


__all__ = ["load_image", "make_train_dataset", "make_image_dataset", "PAD"]
