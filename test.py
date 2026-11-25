import json
import unicodedata
from pathlib import Path

import pandas as pd

from config import MAX_DECODE_LEN, get_parser, run_dir
from data import load_split, load_vocab, make_image_dataset
from model import build_models, predict_texts


def edit_distance(ref, hyp):
    """Khoảng cách Levenshtein: số phép thay, xóa, chèn ít nhất để biến hyp thành ref."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def remove_diacritics(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def scores(refs, preds):
    """CER, WER (càng thấp càng tốt), độ chính xác cả dòng, CER khi bỏ dấu tiếng Việt."""
    def error_rate(pairs):
        return sum(edit_distance(r, p) for r, p in pairs) / max(sum(len(r) for r, _ in pairs), 1)

    return {
        "cer": error_rate(list(zip(refs, preds))),
        "wer": error_rate([(r.split(), p.split()) for r, p in zip(refs, preds)]),
        "line_accuracy": sum(r == p for r, p in zip(refs, preds)) / max(len(refs), 1),
        "cer_no_diacritics": error_rate([(remove_diacritics(r), remove_diacritics(p)) for r, p in zip(refs, preds)]),
        "lines": len(refs),
    }


def save_results(folder, split, refs, preds, suffix=""):
    """Lưu <split>_metrics<suffix>.json và <split>_predictions<suffix>.csv."""
    result = scores(refs, preds)
    df = pd.DataFrame({"reference": refs, "prediction": preds})
    df["cer"] = [edit_distance(r, p) / max(len(r), 1) for r, p in zip(refs, preds)]
    df.to_csv(Path(folder) / f"{split}_predictions{suffix}.csv", index=False, encoding="utf-8-sig")
    (Path(folder) / f"{split}_metrics{suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(split + suffix, json.dumps(result, indent=2))
    return result


def main():
    parser = get_parser()
    parser.add_argument("--beam-width", type=int, default=3, help="1 = chỉ chạy greedy")
    args = parser.parse_args()
    folder = run_dir(args)

    vocab = load_vocab(Path(args.output_dir) / "vocab.json")
    train_model, encoder_model, decoder_model = build_models(len(vocab), args.variant)
    train_model.load_weights(folder / "best.weights.h5")

    images, texts = load_split(args.data_dir, "test", args.limit)
    image_ds = make_image_dataset(images, 64)
    print("Số ảnh test:", len(images))

    preds = predict_texts(encoder_model, decoder_model, image_ds, vocab, MAX_DECODE_LEN)
    save_results(folder, "test", texts, preds)
    if args.beam_width > 1:
        preds = predict_texts(encoder_model, decoder_model, image_ds, vocab, MAX_DECODE_LEN, args.beam_width)
        save_results(folder, "test", texts, preds, f"_beam{args.beam_width}")


if __name__ == "__main__":
    main()
