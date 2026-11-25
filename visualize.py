"""Vẽ hình cho báo cáo từ một mô hình đã huấn luyện.

Ví dụ:
    python visualize.py --data-dir /kaggle/input/viethtr-line/data --variant bahdanau --num-samples 6

Hình lưu ở <output-dir>/<variant>/figures/:
    learning_curve.png       loss train/val và CER validation theo epoch
    predictions.png          ảnh test kèm nhãn và dự đoán
    attention_<i>.png        heatmap attention (Chương 9): mỗi hàng là một ký tự được sinh,
                             mỗi cột là một vị trí ngang của ảnh (1 vị trí = 4 pixel)
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import add_common_args, config_from_args
from data import EOS, load_split
from inference import greedy_decode
from evaluate import load_trained
from preprocessing import load_image


def plot_learning_curve(run_dir, out_dir):
    path = run_dir / "history.csv"
    if not path.exists():
        print("Không có history.csv, bỏ qua learning curve")
        return
    hist = pd.read_csv(path)
    epochs = hist["epoch"] + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, hist["loss"], marker="o", label="train loss")
    axes[0].plot(epochs, hist["val_loss"], marker="o", label="validation loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(epochs, hist["val_cer"], marker="o", color="tab:red", label="validation CER")
    if "val_wer" in hist:
        axes[1].plot(epochs, hist["val_wer"], marker="o", color="tab:purple", label="validation WER")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Tỉ lệ lỗi"); axes[1].legend(); axes[1].grid(alpha=.3)
    fig.suptitle(f"Learning curve - {run_dir.name}")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=150)
    plt.close(fig)


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--num-samples", type=int, default=6)
    args = parser.parse_args()
    cfg, vocab, encoder_model, decoder_model = load_trained(config_from_args(args))
    out_dir = cfg.run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curve(cfg.run_dir, out_dir)

    images, texts = load_split(cfg.data_dir, "test", limit=args.num_samples)
    batch = tf.stack([load_image(b, cfg.img_height, cfg.img_width) for b in images])
    ids, attention = greedy_decode(encoder_model, decoder_model, batch, cfg.max_decode_len, return_attention=True)
    preds = [vocab.decode(row) for row in ids]

    # Ảnh + nhãn + dự đoán
    fig, axes = plt.subplots(len(images), 1, figsize=(12, 1.6 * len(images)))
    for ax, img, ref, pred in zip(np.atleast_1d(axes), batch.numpy(), texts, preds):
        ax.imshow(1 - img[..., 0], cmap="gray", aspect="auto")
        ax.set_title(f"Nhãn: {ref}\nĐoán: {pred}", fontsize=9, loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "predictions.png", dpi=150)
    plt.close(fig)

    # Heatmap attention
    if attention is None:
        print("Biến thể 'none' không có attention, bỏ qua heatmap")
    else:
        steps = attention.shape[2]
        for i in range(len(images)):
            row = ids[i]
            n = int(np.argmax(row == EOS)) if (row == EOS).any() else len(row)
            n = max(min(n, 60), 1)  # tối đa 60 ký tự đầu cho hình dễ đọc
            chars = [vocab.itos[t] if t > 3 else "?" for t in row[:n]]
            fig, (ax_img, ax_att) = plt.subplots(2, 1, figsize=(12, 2 + 0.18 * n),
                                                 gridspec_kw={"height_ratios": [1, max(n / 10, 1.5)]}, sharex=True)
            ax_img.imshow(1 - batch.numpy()[i, ..., 0], cmap="gray", aspect="auto", extent=[0, steps, 1, 0])
            ax_img.set_yticks([]); ax_img.set_title(f"Dự đoán: {preds[i]}", fontsize=9, loc="left")
            ax_att.imshow(attention[i, :n], cmap="viridis", aspect="auto", extent=[0, steps, n, 0])
            ax_att.set_yticks(np.arange(n) + 0.5)
            ax_att.set_yticklabels([c if c != " " else "␣" for c in chars], fontsize=7)
            ax_att.set_xlabel("Vị trí ngang trên ảnh (bước encoder)")
            ax_att.set_ylabel("Ký tự sinh ra")
            fig.tight_layout()
            fig.savefig(out_dir / f"attention_{i}.png", dpi=150)
            plt.close(fig)
    print(f"Đã lưu hình vào {out_dir}")


if __name__ == "__main__":
    main()
