import argparse
from pathlib import Path

# Ảnh (Chương 5.1): ảnh xám, cao 64, pad bên phải cho đủ rộng 1024
IMG_H, IMG_W = 64, 1024

# Nhãn (Chương 8): số ký tự tối đa khi train và khi giải mã
MAX_LEN = 160
MAX_DECODE_LEN = 200

# Mô hình (Chương 5.2, 7, 8, 9)
VARIANTS = ["none", "bahdanau", "luong", "multihead"]   # kiểu attention của decoder
NUM_HEADS = 4       # cho biến thể multihead
EMBED_DIM = 128
ENC_UNITS = 128   # mỗi chiều của BiLSTM
DEC_UNITS = 256   # = 2 * ENC_UNITS để nhận trạng thái cuối của BiLSTM
DROPOUT = 0.2

# Huấn luyện (Chương 2.2, 4.3, 7)
VAL_RATIO = 0.1
SEED = 42
LEARNING_RATE = 1e-3
CLIPNORM = 5.0
PATIENCE = 5        # EarlyStopping
LR_PATIENCE = 2     # ReduceLROnPlateau
CER_SAMPLES = 512   # số ảnh val dùng để tính CER sau mỗi epoch


def get_parser():
    """Tham số dòng lệnh dùng chung cho mọi script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Thư mục chứa train-*.parquet và test-*.parquet")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--variant", choices=VARIANTS, default="bahdanau")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Chỉ lấy N mẫu mỗi tập để chạy thử, 0 = dùng hết")
    return parser


def run_dir(args):
    # Kết quả mỗi biến thể nằm trong outputs/<variant>, từ điển dùng chung ở outputs/vocab.json
    path = Path(args.output_dir) / args.variant
    path.mkdir(parents=True, exist_ok=True)
    return path
