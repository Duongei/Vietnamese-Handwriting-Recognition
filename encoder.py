"""Encoder: CNN trích đặc trưng ảnh + BiLSTM đọc chuỗi đặc trưng theo chiều ngang.

Kiến thức dùng:
- Chương 5.1 / 5.2: Conv 3x3 + MaxPool theo khối kiểu VGG.
- Chương 4.3: BatchNorm, ReLU, Dropout.
- Chương 7: LSTM, Bidirectional RNN.
- Chương 8: trạng thái cuối của encoder (h, c) dùng làm trạng thái đầu của decoder.
"""
import keras
from keras import layers

# (số filter, số lớp conv, kích thước pool) cho mỗi khối CNN.
# Hai khối đầu pool (2,2); hai khối sau pool (2,1) chỉ giảm chiều cao để giữ độ phân giải ngang.
CNN_BLOCKS = [(64, 1, (2, 2)), (128, 1, (2, 2)), (256, 2, (2, 1)), (512, 2, (2, 1))]


def _cnn_encoder(images, cfg):
    x = images
    for filters, n_conv, pool in CNN_BLOCKS:
        for _ in range(n_conv):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=pool)(x)
    # x: (batch, H/16, W/4, 512). Mỗi cột theo chiều ngang thành một "bước thời gian".
    h_out, w_out, c_out = x.shape[1], x.shape[2], x.shape[3]
    x = layers.Permute((2, 1, 3))(x)                       # (batch, W/4, H/16, 512)
    x = layers.Reshape((w_out, h_out * c_out))(x)          # (batch, W/4, H/16 * 512)
    x = layers.Dense(cfg.dec_units, activation="relu")(x)
    return layers.Dropout(cfg.dropout)(x)


def build_encoder(cfg):
    """Trả về (đầu vào ảnh, chuỗi encoder, h0, c0, encoder_model)."""
    images = keras.Input((cfg.img_height, cfg.img_width, 1), name="image")
    features = _cnn_encoder(images, cfg)
    enc_seq, fh, fc, bh, bc = layers.Bidirectional(
        layers.LSTM(cfg.enc_units, return_sequences=True, return_state=True), name="encoder_bilstm"
    )(features)
    state_h = layers.Concatenate(name="init_h")([fh, bh])
    state_c = layers.Concatenate(name="init_c")([fc, bc])
    encoder_model = keras.Model(images, [enc_seq, state_h, state_c], name="encoder")
    return images, enc_seq, state_h, state_c, encoder_model
