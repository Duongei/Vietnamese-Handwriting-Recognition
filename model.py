"""Mô hình Image-to-Sequence: ghép Encoder (encoder.py) với Decoder (decoder.py), cùng loss và compile.

Kiến thức dùng:
- Chương 8: Encoder-Decoder; trạng thái đầu của decoder lấy từ trạng thái cuối của encoder.
- Chương 3.2: loss cross-entropy trên từ điển ký tự.
- Chương 2.2 / 4.3: Adam, gradient clipping.

build_models() trả về 3 mô hình dùng chung trọng số:
    train_model   : (ảnh, chuỗi vào decoder) -> xác suất ký tự từng bước   (dùng cho model.fit)
    encoder_model : ảnh -> (chuỗi encoder, h0, c0)
    decoder_model : (ký tự trước, chuỗi encoder, h, c) -> (xác suất, h, c, trọng số attention)
"""
import keras
from keras import ops

from data import PAD
from decoder import build_decoder
from encoder import build_encoder


def build_models(cfg, vocab_size):
    if cfg.dec_units != 2 * cfg.enc_units:
        raise ValueError("dec_units phải bằng 2 * enc_units")

    # Thứ tự tạo lớp (encoder trước, decoder sau) giữ nguyên để load được trọng số đã train.
    images, enc_seq, state_h, state_c, encoder_model = build_encoder(cfg)
    decode = build_decoder(cfg, vocab_size)

    # --------------------------- Mô hình train ---------------------------
    dec_tokens = keras.Input((cfg.max_text_len + 1,), dtype="int32", name="decoder_tokens")
    probs, _, _, _ = decode(dec_tokens, enc_seq, state_h, state_c)
    train_model = keras.Model([images, dec_tokens], probs, name=f"htr_{cfg.variant}")

    # ------------------- Mô hình suy luận từng bước -------------------
    step_token = keras.Input((1,), dtype="int32", name="prev_token")
    step_enc = keras.Input((enc_seq.shape[1], enc_seq.shape[2]), name="encoder_seq")
    step_h = keras.Input((cfg.dec_units,), name="h")
    step_c = keras.Input((cfg.dec_units,), name="c")
    step_probs, new_h, new_c, step_scores = decode(step_token, step_enc, step_h, step_c)
    outputs = [step_probs, new_h, new_c]
    if step_scores is not None:
        outputs.append(step_scores)
    decoder_model = keras.Model([step_token, step_enc, step_h, step_c], outputs, name="decoder_step")

    return train_model, encoder_model, decoder_model


# ---------------------------------------------------------------------------
# Loss và độ chính xác bỏ qua <pad> (Chương 3.2: cross-entropy)
# ---------------------------------------------------------------------------
def masked_loss(y_true, y_pred):
    y_true = ops.cast(y_true, "int32")
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = ops.cast(ops.not_equal(y_true, PAD), loss.dtype)
    return ops.sum(loss * mask) / ops.maximum(ops.sum(mask), 1.0)


def masked_accuracy(y_true, y_pred):
    y_true = ops.cast(y_true, "int32")
    pred = ops.cast(ops.argmax(y_pred, axis=-1), "int32")
    mask = ops.cast(ops.not_equal(y_true, PAD), "float32")
    correct = ops.cast(ops.equal(y_true, pred), "float32") * mask
    return ops.sum(correct) / ops.maximum(ops.sum(mask), 1.0)


def compile_model(train_model, cfg):
    optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate, clipnorm=cfg.clipnorm)
    train_model.compile(optimizer=optimizer, loss=masked_loss, metrics=[masked_accuracy])
    return train_model
