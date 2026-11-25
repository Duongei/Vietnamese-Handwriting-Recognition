"""Huấn luyện một biến thể mô hình.

Ví dụ:
    python train.py --data-dir /kaggle/input/viethtr-line/data --variant bahdanau

Kết quả lưu ở <output-dir>/<variant>/:
    best.weights.h5     trọng số có CER validation thấp nhất
    history.csv         loss, val_loss, val_cer, learning rate theo epoch (để vẽ learning curve)
    config.json         cấu hình đã dùng
    model_summary.txt   kiến trúc và số tham số
    train_info.json     số mẫu, thời gian huấn luyện, CER tốt nhất

Kiến thức dùng:
- Chương 2.2: mini-batch gradient descent (Adam), learning rate.
- Chương 3.2: cross-entropy trên softmax.
- Chương 4.3: regularization (dropout), BatchNorm; dừng sớm khi validation không cải thiện.
- Chương 7: gradient clipping (clipnorm) chống bùng nổ gradient trong RNN.
- Callbacks EarlyStopping + ReduceLROnPlateau lấy theo DL_pollution.ipynb của lecture.
"""
import argparse
import json
import time

import keras
import numpy as np

from config import add_common_args, config_from_args
from data import encode_texts, get_vocab, load_all_splits
from inference import predict_texts
from metrics import compute_metrics
from model import build_models, compile_model
from preprocessing import make_image_dataset, make_train_dataset


class ValidationCER(keras.callbacks.Callback):
    """Sau mỗi epoch, giải mã greedy một phần tập validation và ghi 'val_cer' vào logs."""

    def __init__(self, encoder_model, decoder_model, images, texts, vocab, cfg):
        super().__init__()
        self.encoder_model, self.decoder_model = encoder_model, decoder_model
        self.texts, self.vocab, self.cfg = texts, vocab, cfg
        self.image_ds = make_image_dataset(images, cfg, batch_size=max(cfg.batch_size, 64))

    def on_epoch_end(self, epoch, logs=None):
        preds = predict_texts(self.encoder_model, self.decoder_model, self.image_ds, self.vocab,
                              self.cfg.max_decode_len)
        m = compute_metrics(self.texts, preds)
        if logs is not None:
            logs["val_cer"] = m["cer"]
            logs["val_wer"] = m["wer"]
        print(f"\n  val_cer={m['cer']:.4f}  val_wer={m['wer']:.4f}  line_acc={m['line_accuracy']:.4f}"
              f"\n  ví dụ | nhãn : {self.texts[0]}\n        | đoán : {preds[0]}")


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    cfg = config_from_args(parser.parse_args())
    keras.utils.set_random_seed(cfg.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg.run_dir / "config.json")

    print("Đọc dữ liệu ...")
    splits = load_all_splits(cfg, need_train=True, need_test=False)
    (train_imgs, train_texts), (val_imgs, val_texts) = splits["train"], splits["val"]
    print(f"Train: {len(train_imgs)} | Validation: {len(val_imgs)}")

    vocab = get_vocab(cfg, train_texts)
    print(f"Từ điển: {len(vocab)} token (gồm 4 token đặc biệt)")
    too_long = sum(len(t) > cfg.max_text_len for t in train_texts)
    if too_long:
        print(f"Cảnh báo: {too_long} nhãn train dài hơn {cfg.max_text_len} ký tự sẽ bị cắt")

    train_ds = make_train_dataset(train_imgs, encode_texts(vocab, train_texts, cfg.max_text_len), cfg, True)
    val_ds = make_train_dataset(val_imgs, encode_texts(vocab, val_texts, cfg.max_text_len), cfg, False)

    train_model, encoder_model, decoder_model = build_models(cfg, len(vocab))
    compile_model(train_model, cfg)
    lines = []
    train_model.summary(print_fn=lambda s, **kw: lines.append(s))
    (cfg.run_dir / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Số tham số: {train_model.count_params():,}")

    n_cer = cfg.cer_eval_samples or len(val_imgs)
    callbacks = [
        # Phải đứng đầu để các callback sau đọc được 'val_cer'
        ValidationCER(encoder_model, decoder_model, val_imgs[:n_cer], val_texts[:n_cer], vocab, cfg),
        keras.callbacks.ModelCheckpoint(str(cfg.run_dir / "best.weights.h5"), monitor="val_cer", mode="min",
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_cer", mode="min", patience=cfg.patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=cfg.lr_patience,
                                          min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(str(cfg.run_dir / "history.csv")),
    ]

    start = time.time()
    history = train_model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks,
                              verbose=2)  # 1 dòng log mỗi epoch, gọn khi chạy trên Kaggle
    minutes = (time.time() - start) / 60

    val_cer = history.history.get("val_cer", [float("nan")])
    info = {
        "variant": cfg.variant,
        "train_samples": len(train_imgs),
        "val_samples": len(val_imgs),
        "vocab_size": len(vocab),
        "params": int(train_model.count_params()),
        "epochs_run": len(val_cer),
        "best_epoch": int(np.nanargmin(val_cer)) + 1,
        "best_val_cer_subset": float(np.nanmin(val_cer)),
        "train_minutes": round(minutes, 1),
    }
    (cfg.run_dir / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
