"""Đánh giá một mô hình đã huấn luyện trên tập validation hoặc test.

Ví dụ:
    python evaluate.py --data-dir /kaggle/input/viethtr-line/data --variant bahdanau --split test
    python evaluate.py ... --split test --beam-width 3          # beam search (Chương 8)

Cấu hình mô hình được đọc lại từ <output-dir>/<variant>/config.json nên không cần nhắc lại siêu tham số.
Kết quả lưu ở <output-dir>/<variant>/:
    <split>_metrics[_beamK].json       CER, WER, line accuracy, CER không dấu, CER theo nhóm độ dài
    <split>_predictions[_beamK].csv    nhãn, dự đoán, CER từng dòng

Chỉ chạy --split test một lần ở cuối; chọn mô hình/siêu tham số bằng validation.
"""
import argparse
import json

import pandas as pd

from config import Config, add_common_args, config_from_args
from data import get_vocab, load_all_splits
from inference import predict_texts
from metrics import compute_metrics, per_line_cer
from model import build_models
from preprocessing import make_image_dataset

# Các trường lấy từ dòng lệnh; mọi trường khác lấy từ config.json lúc train
RUNTIME_FIELDS = ("data_dir", "output_dir", "variant", "batch_size", "limit_val", "limit_test", "max_decode_len")


def load_trained(cli_cfg):
    cfg = Config.load(cli_cfg.run_dir / "config.json")
    for name in RUNTIME_FIELDS:
        setattr(cfg, name, getattr(cli_cfg, name))
    vocab = get_vocab(cfg)
    _, encoder_model, decoder_model = models = build_models(cfg, len(vocab))
    models[0].load_weights(str(cfg.run_dir / "best.weights.h5"))
    return cfg, vocab, encoder_model, decoder_model


def length_buckets(df):
    bins = [0, 20, 40, 60, 80, 1000]
    labels = ["1-20", "21-40", "41-60", "61-80", ">80"]
    df = df.assign(length_group=pd.cut(df["ref_len"], bins=bins, labels=labels))
    out = {}
    for group, part in df.groupby("length_group", observed=True):
        out[str(group)] = {"lines": int(len(part)),
                           "cer": float(compute_metrics(part["reference"].tolist(), part["prediction"].tolist())["cer"])}
    return out


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--beam-width", type=int, default=1, help="1 = greedy")
    args = parser.parse_args()
    cfg, vocab, encoder_model, decoder_model = load_trained(config_from_args(args))

    splits = load_all_splits(cfg, need_train=args.split == "val", need_test=args.split == "test")
    images, texts = splits[args.split]
    print(f"Đánh giá {cfg.variant} trên {args.split}: {len(images)} dòng, beam_width={args.beam_width}")

    image_ds = make_image_dataset(images, cfg, batch_size=max(cfg.batch_size, 64))
    preds = predict_texts(encoder_model, decoder_model, image_ds, vocab, cfg.max_decode_len, args.beam_width)

    df = pd.DataFrame({"reference": texts, "prediction": preds})
    df["ref_len"] = df["reference"].str.len()
    df["cer"] = [per_line_cer(r, p) for r, p in zip(texts, preds)]
    metrics = compute_metrics(texts, preds)
    metrics.update({"variant": cfg.variant, "split": args.split, "beam_width": args.beam_width,
                    "cer_by_length": length_buckets(df)})

    suffix = f"_beam{args.beam_width}" if args.beam_width > 1 else ""
    df.to_csv(cfg.run_dir / f"{args.split}_predictions{suffix}.csv", index=False, encoding="utf-8-sig")
    (cfg.run_dir / f"{args.split}_metrics{suffix}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
