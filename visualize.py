from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import MAX_DECODE_LEN, get_parser, run_dir
from data import EOS, decode_ids, load_image, load_split, load_vocab
from model import build_models, greedy_decode


def plot_learning_curve(folder, out):
    hist = pd.read_csv(folder / "history.csv")
    epochs = hist["epoch"] + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, hist["loss"], marker="o", label="train loss")
    axes[0].plot(epochs, hist["val_loss"], marker="o", label="val loss")
    axes[0].set_ylabel("Cross-entropy")
    axes[1].plot(epochs, hist["val_cer"], marker="o", color="tab:red", label="val CER")
    axes[1].plot(epochs, hist["val_wer"], marker="o", color="tab:purple", label="val WER")
    axes[1].set_ylabel("Tỉ lệ lỗi")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=150); plt.close(fig)


def main():
    parser = get_parser()
    parser.add_argument("--num-samples", type=int, default=6)
    args = parser.parse_args()
    folder = run_dir(args)
    out = folder / "figures"
    out.mkdir(exist_ok=True)
    plot_learning_curve(folder, out)

    vocab = load_vocab(Path(args.output_dir) / "vocab.json")
    train_model, encoder_model, decoder_model = build_models(len(vocab), args.variant)
    train_model.load_weights(folder / "best.weights.h5")

    images, texts = load_split(args.data_dir, "test", args.num_samples)
    batch = tf.stack([load_image(b) for b in images])
    ids, attention = greedy_decode(encoder_model, decoder_model, batch, MAX_DECODE_LEN)
    preds = [decode_ids(row, vocab) for row in ids]
    pictures = 1 - batch.numpy()[..., 0]  # đảo màu lại để chữ đen nền trắng

    # Ảnh test kèm nhãn và dự đoán
    fig, axes = plt.subplots(len(images), 1, figsize=(12, 1.6 * len(images)))
    for ax, picture, ref, pred in zip(axes, pictures, texts, preds):
        ax.imshow(picture, cmap="gray", aspect="auto")
        ax.set_title(f"Nhãn: {ref}\nĐoán: {pred}", fontsize=9, loc="left")
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "predictions.png", dpi=150); plt.close(fig)

    # Heatmap attention (Chương 9): hàng là ký tự sinh ra, cột là vị trí ngang trên ảnh
    if attention is None:
        print("Biến thể none không có attention, bỏ qua heatmap")
        return
    steps = attention.shape[2]
    for i in range(len(images)):
        row = list(ids[i])
        n = min(row.index(EOS) if EOS in row else len(row), 60)  # tối đa 60 ký tự cho dễ nhìn
        n = max(n, 1)
        chars = ["space" if vocab[t] == " " else vocab[t] for t in row[:n]]
        fig, (ax_img, ax_att) = plt.subplots(2, 1, figsize=(12, 2 + 0.18 * n), sharex=True,
                                             gridspec_kw={"height_ratios": [1, max(n / 10, 1.5)]})
        ax_img.imshow(pictures[i], cmap="gray", aspect="auto", extent=[0, steps, 1, 0])
        ax_img.set_yticks([]); ax_img.set_title(f"Dự đoán: {preds[i]}", fontsize=9, loc="left")
        ax_att.imshow(attention[i, :n], cmap="viridis", aspect="auto", extent=[0, steps, n, 0])
        ax_att.set_yticks(np.arange(n) + 0.5)
        ax_att.set_yticklabels(chars, fontsize=7)
        ax_att.set_xlabel("Vị trí ngang trên ảnh (bước encoder)"); ax_att.set_ylabel("Ký tự sinh ra")
        fig.tight_layout(); fig.savefig(out / f"attention_{i}.png", dpi=150); plt.close(fig)
    print("Đã lưu hình vào", out)


if __name__ == "__main__":
    main()
