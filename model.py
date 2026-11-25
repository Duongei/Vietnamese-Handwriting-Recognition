import keras
import numpy as np
import tensorflow as tf
from keras import layers, ops

from config import DEC_UNITS, DROPOUT, EMBED_DIM, ENC_UNITS, IMG_H, IMG_W, MAX_LEN, NUM_HEADS
from data import BOS, EOS, PAD, decode_ids

# Lưu ý: giữ nguyên thứ tự tạo lớp và tên lớp để nạp được best.weights.h5 đã train 22 epoch.


def build_models(vocab_size, variant="bahdanau"):
    """Trả về 3 mô hình dùng chung trọng số:
    train_model   : (ảnh, chuỗi vào decoder) -> xác suất ký tự, dùng cho model.fit
    encoder_model : ảnh -> (chuỗi encoder, h0, c0)
    decoder_model : (ký tự trước, chuỗi encoder, h, c) -> (xác suất, h, c[, attention]), giải mã từng bước

    variant (Chương 9):
    none      : không attention, chỉ dùng trạng thái cuối của encoder (Seq2Seq Chương 8)
    bahdanau  : attention cộng, score = v^T tanh(W_q s_t + W_k h_i), mô hình chính
    luong     : attention tích vô hướng, giống LSTM_attention.py
    multihead : MultiHeadAttention 4 head, giống LSTM_MultiHeadAttention.py
    """
    # Encoder, Chương 5.1-5.2: 4 khối Conv 3x3 + BatchNorm + ReLU + MaxPool kiểu VGG.
    # Hai khối sau chỉ pool theo chiều cao để giữ độ phân giải ngang.
    images = keras.Input((IMG_H, IMG_W, 1), name="image")
    x = images
    for filters, n_conv, pool in [(64, 1, (2, 2)), (128, 1, (2, 2)), (256, 2, (2, 1)), (512, 2, (2, 1))]:
        for _ in range(n_conv):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool)(x)

    # (batch, 4, 256, 512) -> (batch, 256, 4 * 512): mỗi cột theo chiều ngang là một bước thời gian
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3]))(x)
    x = layers.Dense(DEC_UNITS, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)

    # Chương 7-8: BiLSTM đọc chuỗi đặc trưng, trạng thái cuối hai chiều ghép lại làm trạng thái đầu của decoder
    enc_seq, fh, fc, bh, bc = layers.Bidirectional(
        layers.LSTM(ENC_UNITS, return_sequences=True, return_state=True), name="encoder_bilstm")(x)
    state_h = layers.Concatenate(name="init_h")([fh, bh])
    state_c = layers.Concatenate(name="init_c")([fc, bc])
    encoder_model = keras.Model(images, [enc_seq, state_h, state_c], name="encoder")

    # Decoder, Chương 8: LSTM sinh từng ký tự
    embedding = layers.Embedding(vocab_size, EMBED_DIM, name="char_embedding")
    decoder_lstm = layers.LSTM(DEC_UNITS, return_sequences=True, return_state=True, name="decoder_lstm")
    if variant == "bahdanau":
        w_q = layers.Dense(DEC_UNITS, use_bias=False, name="attention_Wq")
        w_k = layers.Dense(DEC_UNITS, use_bias=False, name="attention_Wk")
        attention = layers.AdditiveAttention(name="attention")
    elif variant == "luong":
        attention = layers.Attention(name="attention")
    elif variant == "multihead":
        attention = layers.MultiHeadAttention(NUM_HEADS, DEC_UNITS // NUM_HEADS, name="attention")
    combine = layers.Dense(DEC_UNITS, activation="tanh", name="combine")
    dropout = layers.Dropout(DROPOUT)
    # Chương 3.2: softmax trên từ điển; để float32 cho ổn định khi bật mixed precision
    classifier = layers.Dense(vocab_size, activation="softmax", name="char_probs", dtype="float32")

    def decode(tokens, enc_seq, h, c):
        dec_seq, h, c = decoder_lstm(embedding(tokens), initial_state=[h, c])
        if variant == "none":
            return [classifier(dropout(dec_seq)), h, c]
        if variant == "bahdanau":
            # query = W_q s_t, value = h_i, key = W_k h_i
            context, scores = attention([w_q(dec_seq), enc_seq, w_k(enc_seq)], return_attention_scores=True)
        elif variant == "luong":
            context, scores = attention([dec_seq, enc_seq], return_attention_scores=True)
        else:
            context, scores = attention(dec_seq, enc_seq, return_attention_scores=True)
            scores = ops.mean(scores, axis=1)  # trung bình các head
        mixed = combine(layers.Concatenate()([dec_seq, context]))
        return [classifier(dropout(mixed)), h, c, scores]

    dec_tokens = keras.Input((MAX_LEN + 1,), dtype="int32", name="decoder_tokens")
    probs = decode(dec_tokens, enc_seq, state_h, state_c)[0]
    train_model = keras.Model([images, dec_tokens], probs, name=f"htr_{variant}")

    step_token = keras.Input((1,), dtype="int32", name="prev_token")
    step_enc = keras.Input(enc_seq.shape[1:], name="encoder_seq")
    step_h = keras.Input((DEC_UNITS,), name="h")
    step_c = keras.Input((DEC_UNITS,), name="c")
    decoder_model = keras.Model([step_token, step_enc, step_h, step_c],
                                decode(step_token, step_enc, step_h, step_c), name="decoder_step")
    return train_model, encoder_model, decoder_model


def masked_loss(y_true, y_pred):
    """Cross-entropy (Chương 3.2), bỏ qua các vị trí <pad>."""
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = ops.cast(ops.not_equal(y_true, PAD), loss.dtype)
    return ops.sum(loss * mask) / ops.maximum(ops.sum(mask), 1.0)


def masked_accuracy(y_true, y_pred):
    y_true = ops.cast(y_true, "int32")
    pred = ops.cast(ops.argmax(y_pred, axis=-1), "int32")
    mask = ops.cast(ops.not_equal(y_true, PAD), "float32")
    correct = ops.cast(ops.equal(y_true, pred), "float32") * mask
    return ops.sum(correct) / ops.maximum(ops.sum(mask), 1.0)


# Giải mã (Chương 8: greedy và beam search)
def greedy_decode(encoder_model, decoder_model, images, max_len):
    """Mỗi bước chọn ký tự có xác suất cao nhất, chạy cả batch.

    Trả về (id (B, bước), attention (B, bước, T)); attention là None với biến thể none.
    """
    enc_seq, h, c = encoder_model(images, training=False)
    tokens = tf.fill([images.shape[0], 1], BOS)
    finished = np.zeros(images.shape[0], dtype=bool)
    ids, attentions = [], []
    for _ in range(max_len):
        outputs = decoder_model([tokens, enc_seq, h, c], training=False)
        probs, h, c = outputs[:3]
        tokens = tf.argmax(probs, axis=-1, output_type=tf.int32)  # (B, 1)
        ids.append(tokens.numpy()[:, 0])
        if len(outputs) == 4:
            attentions.append(outputs[3].numpy()[:, 0])
        finished |= ids[-1] == EOS
        if finished.all():
            break
    return np.stack(ids, axis=1), (np.stack(attentions, axis=1) if attentions else None)


def beam_search(encoder_model, decoder_model, image, max_len, beam_width=3, alpha=0.7):
    """Giữ beam_width chuỗi tốt nhất ở mỗi bước, chạy từng ảnh (1, H, W, 1).

    Điểm = tổng log P / độ dài^alpha để không thiên vị chuỗi ngắn.
    """
    enc_seq, h, c = encoder_model(image, training=False)
    beams = [([], 0.0, h, c)]  # (id đã sinh, tổng log P, h, c)
    done = []
    for _ in range(max_len):
        tokens = tf.constant([[ids[-1] if ids else BOS] for ids, _, _, _ in beams])
        enc = tf.repeat(enc_seq, len(beams), axis=0)
        hs = tf.concat([b[2] for b in beams], axis=0)
        cs = tf.concat([b[3] for b in beams], axis=0)
        probs, new_h, new_c = decoder_model([tokens, enc, hs, cs], training=False)[:3]
        log_probs = np.log(probs.numpy()[:, 0] + 1e-12)

        candidates = []
        for i, (ids, score, _, _) in enumerate(beams):
            for tok in np.argsort(log_probs[i])[-beam_width:]:
                candidates.append((ids + [int(tok)], score + float(log_probs[i, tok]), i))
        candidates.sort(key=lambda cand: cand[1], reverse=True)

        beams = []
        for ids, score, i in candidates[:beam_width]:
            if ids[-1] == EOS:
                done.append((ids, score / len(ids) ** alpha))
            else:
                beams.append((ids, score, new_h[i:i + 1], new_c[i:i + 1]))
        if len(done) >= beam_width or not beams:
            break

    done += [(ids, score / len(ids) ** alpha) for ids, score, _, _ in beams]
    return max(done, key=lambda d: d[1])[0]


def predict_texts(encoder_model, decoder_model, image_ds, vocab, max_len, beam_width=1):
    """Dự đoán văn bản cho cả dataset ảnh. beam_width = 1 là greedy."""
    texts = []
    for images in image_ds:
        if beam_width == 1:
            ids, _ = greedy_decode(encoder_model, decoder_model, images, max_len)
            texts += [decode_ids(row, vocab) for row in ids]
        else:
            for i in range(images.shape[0]):
                ids = beam_search(encoder_model, decoder_model, images[i:i + 1], max_len, beam_width)
                texts.append(decode_ids(ids, vocab))
    return texts
