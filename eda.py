import io
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import IMG_H, IMG_W, SEED, get_parser
from data import load_split, split_train_val


def stats(series):
    return series.describe(percentiles=[0.05, 0.5, 0.95]).round(2).to_dict()


def main():
    args = get_parser().parse_args()
    out = Path(args.output_dir) / "eda"
    out.mkdir(parents=True, exist_ok=True)

    train_imgs, train_texts = load_split(args.data_dir, "train")
    test_imgs, test_texts = load_split(args.data_dir, "test")
    _, val_idx = split_train_val(len(train_imgs))
    names = np.array(["train"] * len(train_imgs) + ["test"] * len(test_imgs), dtype=object)
    names[val_idx] = "val"

    rows = []
    for name, image, text in zip(names, train_imgs + test_imgs, train_texts + test_texts):
        with Image.open(io.BytesIO(image)) as im:
            rows.append({"split": name, "width": im.width, "height": im.height,
                         "chars": len(text), "words": len(text.split())})
    df = pd.DataFrame(rows)
    df["aspect"] = df["width"] / df["height"]

    char_counts = Counter("".join(train_texts))
    test_chars = set("".join(test_texts))
    summary = {
        "total_images": len(df),
        "samples": df["split"].value_counts().to_dict(),
        "width_px": stats(df["width"]),
        "height_px": stats(df["height"]),
        "aspect_ratio": stats(df["aspect"]),
        "share_wider_than_input": round(float((df["aspect"] * IMG_H > IMG_W).mean()), 4),
        "chars_per_line": {s: stats(g["chars"]) for s, g in df.groupby("split")},
        "words_per_line": {s: stats(g["words"]) for s, g in df.groupby("split")},
        "distinct_chars_train": len(char_counts),
        "test_chars_missing_from_train": sorted(test_chars - set(char_counts)),
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Độ dài nhãn
    plt.figure(figsize=(8, 4))
    for s in ["train", "val", "test"]:
        plt.hist(df.loc[df["split"] == s, "chars"], bins=50, alpha=0.5, density=True, label=s)
    plt.xlabel("Số ký tự mỗi dòng"); plt.ylabel("Mật độ"); plt.title("Độ dài nhãn"); plt.legend()
    plt.tight_layout(); plt.savefig(out / "text_length.png", dpi=150); plt.close()

    # Kích thước ảnh
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, label in zip(axes, ["width", "height", "aspect"], ["Rộng (px)", "Cao (px)", "Tỉ lệ rộng/cao"]):
        ax.hist(df[col], bins=60)
        ax.set_xlabel(label); ax.set_ylabel("Số ảnh")
    axes[2].axvline(IMG_W / IMG_H, color="red", linestyle="--", label=f"giới hạn {IMG_W}/{IMG_H}")
    axes[2].legend()
    fig.tight_layout(); fig.savefig(out / "image_size.png", dpi=150); plt.close(fig)

    # 30 ký tự phổ biến nhất
    top = char_counts.most_common(30)
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(top)), [n for _, n in top])
    plt.xticks(range(len(top)), ["space" if ch == " " else ch for ch, _ in top], rotation=90)
    plt.title("30 ký tự xuất hiện nhiều nhất trong train"); plt.ylabel("Số lần")
    plt.tight_layout(); plt.savefig(out / "top_chars.png", dpi=150); plt.close()

    # Ảnh mẫu
    picks = np.random.default_rng(SEED).choice(len(train_imgs), size=6, replace=False)
    fig, axes = plt.subplots(6, 1, figsize=(12, 9))
    for ax, i in zip(axes, picks):
        ax.imshow(Image.open(io.BytesIO(train_imgs[i])).convert("L"), cmap="gray", aspect="auto")
        ax.set_title(train_texts[i], fontsize=9, loc="left"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "samples.png", dpi=150); plt.close(fig)
    print("Đã lưu EDA vào", out)


if __name__ == "__main__":
    main()
