import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import VARIANTS


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    # Gom kết quả các biến thể đã chạy thành một bảng (mục 3.4 báo cáo)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    root = Path(args.output_dir)

    rows = []
    for variant in VARIANTS:
        folder = root / variant
        if not folder.exists():
            continue
        row = {"variant": variant}
        for name, file in [("val", "val_metrics.json"), ("test", "test_metrics.json"),
                           ("test_beam3", "test_metrics_beam3.json")]:
            metrics = read_json(folder / file)
            for key in ["cer", "wer", "line_accuracy", "cer_no_diacritics"]:
                row[f"{name}_{key}"] = metrics.get(key)
        rows.append(row)
    if not rows:
        print("Chưa có kết quả nào trong", root)
        return

    df = pd.DataFrame(rows)
    df.to_csv(root / "comparison.csv", index=False)
    # Bảng markdown để dán vào báo cáo
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.round(4).fillna("").values:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df.to_string(index=False))

    done = df.dropna(subset=["test_cer"])
    plt.figure(figsize=(6, 3.5))
    plt.bar(done["variant"], done["test_cer"])
    for x, y in zip(done["variant"], done["test_cer"]):
        plt.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("CER trên test (thấp hơn là tốt hơn)")
    plt.title("So sánh các biến thể attention")
    plt.tight_layout(); plt.savefig(root / "comparison_cer.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
