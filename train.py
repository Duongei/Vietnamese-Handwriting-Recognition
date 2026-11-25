from pathlib import Path

import keras

from config import CER_SAMPLES, CLIPNORM, LEARNING_RATE, LR_PATIENCE, MAX_DECODE_LEN, PATIENCE, SEED, get_parser, run_dir
from data import build_vocab, encode_texts, load_train_val, make_dataset, make_image_dataset
from model import build_models, masked_accuracy, masked_loss, predict_texts
from test import save_results, scores


def main():
    parser = get_parser()
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--mixed-precision", action="store_true", help="Dùng float16 trên GPU, nhanh hơn trên T4")
    args = parser.parse_args()
    folder = run_dir(args)
    keras.utils.set_random_seed(SEED)
    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    # 1. Dữ liệu
    (train_imgs, train_texts), (val_imgs, val_texts) = load_train_val(args.data_dir, args.limit)
    vocab = build_vocab(train_texts, Path(args.output_dir) / "vocab.json")
    print("Số ảnh train / val:", len(train_imgs), len(val_imgs), "| số token:", len(vocab))
    train_ds = make_dataset(train_imgs, encode_texts(train_texts, vocab), args.batch_size, training=True)
    val_ds = make_dataset(val_imgs, encode_texts(val_texts, vocab), args.batch_size)

    # 2. Mô hình. Chương 2.2, 7: Adam, clipnorm chống bùng nổ gradient trong RNN
    model, encoder, decoder = build_models(len(vocab), args.variant)
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE, clipnorm=CLIPNORM),
                  loss=masked_loss, metrics=[masked_accuracy])
    model.summary()

    # 3. Train. Sau mỗi epoch tính CER trên 512 ảnh val (giải mã cả tập val thì quá chậm),
    # rồi chọn trọng số theo CER. Dừng sớm và giảm learning rate theo DL_pollution.ipynb (Chương 4.3).
    cer_images = make_image_dataset(val_imgs[:CER_SAMPLES], 64).cache()

    def val_cer(epoch, logs):
        preds = predict_texts(encoder, decoder, cer_images, vocab, MAX_DECODE_LEN)
        result = scores(val_texts[:CER_SAMPLES], preds)
        logs["val_cer"], logs["val_wer"] = result["cer"], result["wer"]
        print(f"  val_cer={result['cer']:.4f}  nhãn: {val_texts[0]}  |  đoán: {preds[0]}")

    weights = str(folder / "best.weights.h5")
    callbacks = [
        keras.callbacks.LambdaCallback(on_epoch_end=val_cer),  # phải đứng đầu để các callback sau thấy val_cer
        keras.callbacks.ModelCheckpoint(weights, monitor="val_cer", mode="min", save_best_only=True,
                                        save_weights_only=True),
        keras.callbacks.EarlyStopping(monitor="val_cer", mode="min", patience=PATIENCE),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=LR_PATIENCE, min_lr=1e-6),
        keras.callbacks.CSVLogger(str(folder / "history.csv")),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)

    # 4. Chấm toàn bộ tập val với trọng số tốt nhất
    model.load_weights(weights)
    preds = predict_texts(encoder, decoder, make_image_dataset(val_imgs, 64), vocab, MAX_DECODE_LEN)
    save_results(folder, "val", val_texts, preds)


if __name__ == "__main__":
    main()
