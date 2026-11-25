import json
import time
from pathlib import Path

import keras
import tensorflow as tf

from config import (CER_SAMPLES, CLIPNORM, LEARNING_RATE, LR_PATIENCE, MAX_DECODE_LEN, PATIENCE, SEED,
                    get_parser, run_dir)
from data import build_vocab, encode_texts, load_train_val, load_vocab, make_dataset, make_image_dataset
from model import build_models, masked_accuracy, masked_loss, predict_texts
from test import save_results, scores


class ValCER(keras.callbacks.Callback):
    """Sau mỗi epoch, giải mã greedy CER_SAMPLES ảnh val và ghi 'val_cer' vào logs.

    Chỉ lấy một phần tập val vì giải mã từng ký tự chậm hơn nhiều so với tính loss.
    """

    def __init__(self, encoder_model, decoder_model, images, texts, vocab):
        super().__init__()
        self.encoder_model, self.decoder_model = encoder_model, decoder_model
        self.texts, self.vocab = texts, vocab
        self.image_ds = make_image_dataset(images, 64).cache()

    def on_epoch_end(self, epoch, logs=None):
        preds = predict_texts(self.encoder_model, self.decoder_model, self.image_ds, self.vocab, MAX_DECODE_LEN)
        result = scores(self.texts, preds)
        logs["val_cer"], logs["val_wer"] = result["cer"], result["wer"]
        print(f"  val_cer={result['cer']:.4f}  val_wer={result['wer']:.4f}")
        print(f"  nhãn: {self.texts[0]}\n  đoán: {preds[0]}")


def main():
    parser = get_parser()
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true", help="Dùng float16 trên GPU, nhanh hơn trên T4")
    args = parser.parse_args()
    folder = run_dir(args)
    keras.utils.set_random_seed(SEED)
    if args.mixed_precision and tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")

    (train_imgs, train_texts), (val_imgs, val_texts) = load_train_val(args.data_dir, args.limit)
    print("Số ảnh train / val:", len(train_imgs), len(val_imgs))
    # Có từ điển rồi thì dùng lại để không làm hỏng trọng số đã train, chưa có thì mới tạo
    vocab_path = Path(args.output_dir) / "vocab.json"
    if vocab_path.exists():
        vocab = load_vocab(vocab_path)
    else:
        vocab = build_vocab(train_texts, vocab_path)
    print("Số token trong từ điển:", len(vocab))

    train_ds = make_dataset(train_imgs, encode_texts(train_texts, vocab), args.batch_size,
                            training=True, augment=not args.no_augment)
    val_ds = make_dataset(val_imgs, encode_texts(val_texts, vocab), args.batch_size)

    # Chương 2.2, 7: Adam, clipnorm chống bùng nổ gradient trong RNN
    train_model, encoder_model, decoder_model = build_models(len(vocab), args.variant)
    train_model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE, clipnorm=CLIPNORM),
                        loss=masked_loss, metrics=[masked_accuracy])
    train_model.summary()

    # Chương 4.3 và DL_pollution.ipynb: dừng sớm, giảm learning rate khi val không cải thiện
    weights_path = str(folder / "best.weights.h5")
    callbacks = [
        ValCER(encoder_model, decoder_model, val_imgs[:CER_SAMPLES], val_texts[:CER_SAMPLES], vocab),
        keras.callbacks.ModelCheckpoint(weights_path, monitor="val_cer", mode="min",
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_cer", mode="min", patience=PATIENCE, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=LR_PATIENCE,
                                          min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(str(folder / "history.csv")),
    ]
    start = time.time()
    history = train_model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                              callbacks=callbacks, verbose=2)

    val_cer = history.history["val_cer"]
    info = {"train_samples": len(train_imgs), "val_samples": len(val_imgs), "vocab_size": len(vocab),
            "params": train_model.count_params(), "batch_size": args.batch_size,
            "epochs_run": len(val_cer), "best_epoch": val_cer.index(min(val_cer)) + 1,
            "train_minutes": round((time.time() - start) / 60, 1)}
    (folder / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(info)

    # Đánh giá trên toàn bộ tập val với trọng số tốt nhất
    train_model.load_weights(weights_path)
    preds = predict_texts(encoder_model, decoder_model, make_image_dataset(val_imgs, 64), vocab, MAX_DECODE_LEN)
    save_results(folder, "val", val_texts, preds)


if __name__ == "__main__":
    main()
