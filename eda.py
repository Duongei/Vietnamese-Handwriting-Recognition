"""Phân tích dữ liệu (EDA) cho mục 3.1 của báo cáo.

Ví dụ:
    python eda.py --data-dir /kaggle/input/viethtr-line/data --output-dir outputs

Kết quả lưu ở <output-dir>/eda/:
    eda_summary.json     số ảnh, số mẫu train/val/test, kích thước ảnh, độ dài nhãn, số ký tự
    text_length.png      phân bố số ký tự mỗi dòng theo tập
    image_size.png       phân bố chiều rộng, chiều cao, tỉ lệ rộng/cao
    top_chars.png        30 ký tự xuất hiện nhiều nhất
    samples.png          ảnh mẫu kèm nhãn
"""
import argparse
import io
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import add_common_args, config_from_args
from data import load_split, train_val_indices


def describe(series):
    return {k: round(float(v), 2) for k, v in series.describe(percentiles=[.05, .5, .95]).items()}


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    cfg = config_from_args(parser.parse_args())
    out = cfg.run_dir.parent / "eda"
    out.mkdir(parents=True, exist_ok=True)

    train_imgs, train_texts = load_split(cfg.data_dir, "train")
    test_imgs, test_texts = load_split(cfg.data_dir, "test")
    train_idx, val_idx = train_val_indices(len(train_imgs), cfg.val_ratio, cfg.seed)
    split = np.array(["train"] * len(train_imgs), dtype=object)
    split[val_idx] = "val"

    records = []
    for name, imgs, texts, labels in (("train", train_imgs, train_texts, split),
                                      ("test", test_imgs, test_texts, ["test"] * len(test_imgs))):
        for b, t, s in zip(imgs, texts, labels):
            with Image.open(io.BytesIO(b)) as im:
                w, h, fmt, mode = im.width, im.height, im.format, im.mode
            records.append({"split": s, "width": w, "height": h, "format": fmt, "mode": mode,
                            "chars": len(t), "words": len(t.split())})
    df = pd.DataFrame(records)
    df["aspect"] = df["width"] / df["height"]
    df["width_at_h64"] = df["aspect"] * cfg.img_height

    char_counts = Counter(ch for t in train_texts for ch in t)
    test_chars = {ch for t in test_texts for ch in t}
    summary = {
        "dataset": "https://huggingface.co/datasets/5CD-AI/VietHTR-Line",
        "total_images": len(df),
        "samples": df["split"].value_counts().to_dict(),
        "image_formats": df["format"].value_counts().to_dict(),
        "image_modes": df["mode"].value_counts().to_dict(),
        "width_px": describe(df["width"]),
        "height_px": describe(df["height"]),
        "aspect_ratio": describe(df["aspect"]),
        "share_wider_than_input": round(float((df["width_at_h64"] > cfg.img_width).mean()), 4),
        "chars_per_line": {s: describe(g["chars"]) for s, g in df.groupby("split")},
        "words_per_line": {s: describe(g["words"]) for s, g in df.groupby("split")},
        "distinct_chars_train": len(char_counts),
        "distinct_chars_test": len(test_chars),
        "test_chars_missing_from_train": sorted(test_chars - set(char_counts)),
        "unique_transcripts_train": len(set(train_texts)),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Độ dài nhãn
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in ("train", "val", "test"):
        ax.hist(df.loc[df["split"] == s, "chars"], bins=50, alpha=.5, density=True, label=s)
    ax.set_xlabel("Số ký tự mỗi dòng"); ax.set_ylabel("Mật độ"); ax.legend(); ax.set_title("Độ dài nhãn")
    fig.tight_layout(); fig.savefig(out / "text_length.png", dpi=150); plt.close(fig)

    # Kích thước ảnh
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, label in zip(axes, ("width", "height", "aspect"), ("Rộng (px)", "Cao (px)", "Tỉ lệ rộng/cao")):
        ax.hist(df[col], bins=60, color="tab:blue")
        ax.set_xlabel(label); ax.set_ylabel("Số ảnh")
    axes[2].axvline(cfg.img_width / cfg.img_height, color="tab:red", linestyle="--",
                    label=f"giới hạn {cfg.img_width}/{cfg.img_height}")
    axes[2].legend()
    fig.tight_layout(); fig.savefig(out / "image_size.png", dpi=150); plt.close(fig)

    # Ký tự phổ biến
    top = char_counts.most_common(30)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(top)), [c for _, c in top])
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(["␣" if ch == " " else ch for ch, _ in top])
    ax.set_title("30 ký tự xuất hiện nhiều nhất trong train"); ax.set_ylabel("Số lần")
    fig.tight_layout(); fig.savefig(out / "top_chars.png", dpi=150); plt.close(fig)

    # Ảnh mẫu
    rng = np.random.default_rng(cfg.seed)
    picks = rng.choice(len(train_imgs), size=6, replace=False)
    fig, axes = plt.subplots(6, 1, figsize=(12, 9))
    for ax, i in zip(axes, picks):
        ax.imshow(Image.open(io.BytesIO(train_imgs[i])).convert("L"), cmap="gray", aspect="auto")
        ax.set_title(train_texts[i], fontsize=9, loc="left"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "samples.png", dpi=150); plt.close(fig)
    print(f"Đã lưu EDA vào {out}")


if __name__ == "__main__":
    main()
