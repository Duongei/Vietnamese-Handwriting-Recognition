"""Cấu hình dùng chung cho Task 3 (VietHTR-Line: ảnh dòng chữ viết tay -> chuỗi ký tự).

Mọi script đều gọi `add_common_args` để có cùng tham số dòng lệnh, rồi
`config_from_args` để lấy một đối tượng Config.
"""
import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

VARIANTS = ("none", "bahdanau", "luong", "multihead")


@dataclass
class Config:
    # Đường dẫn
    data_dir: str = "data"            # thư mục chứa train-*.parquet và test-*.parquet
    output_dir: str = "outputs"       # nơi lưu vocab, checkpoint, kết quả

    # Chia tập (Chương 1): test chính thức giữ nguyên, tách validation từ train
    val_ratio: float = 0.10
    seed: int = 42

    # Ảnh (Chương 5.1): xám, cao 64, pad về rộng cố định
    img_height: int = 64
    img_width: int = 1024

    # Nhãn (Chương 8): số ký tự tối đa của một dòng khi huấn luyện
    max_text_len: int = 160
    max_decode_len: int = 200          # giới hạn khi suy luận (câu test dài tới 166 ký tự)

    # Mô hình (Chương 5.2, 7, 8, 9)
    variant: str = "bahdanau"         # none | bahdanau | luong | multihead
    embed_dim: int = 128
    enc_units: int = 128               # mỗi chiều của BiLSTM -> chuỗi encoder 256 chiều
    dec_units: int = 256               # = 2 * enc_units để nhận trạng thái cuối của BiLSTM
    num_heads: int = 4                 # cho biến thể multihead
    dropout: float = 0.2

    # Huấn luyện (Chương 2.2, 4.3, 7)
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    clipnorm: float = 5.0
    patience: int = 5                  # EarlyStopping
    lr_patience: int = 2               # ReduceLROnPlateau
    augment: bool = True
    cer_eval_samples: int = 512        # số ảnh validation dùng để tính CER sau mỗi epoch

    # Giới hạn số mẫu (chỉ để chạy thử nhanh); 0 = dùng hết
    limit_train: int = 0
    limit_val: int = 0
    limit_test: int = 0

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.variant

    @property
    def vocab_path(self) -> Path:
        return Path(self.output_dir) / "vocab.json"

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Thêm mọi trường của Config thành tham số --ten-truong."""
    defaults = Config()
    for f in fields(Config):
        name = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            parser.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=default,
                                help=f"(mặc định: {default})")
        elif f.name == "variant":
            parser.add_argument(name, choices=VARIANTS, default=default)
        else:
            parser.add_argument(name, type=type(default), default=default, help=f"(mặc định: {default})")
    return parser


def config_from_args(args) -> Config:
    return Config(**{f.name: getattr(args, f.name) for f in fields(Config)})
