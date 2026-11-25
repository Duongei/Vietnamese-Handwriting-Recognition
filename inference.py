"""Suy luận: sinh chuỗi ký tự từ ảnh (Chương 8 - Prediction và Beam Search).

greedy_decode: mỗi bước chọn ký tự có xác suất lớn nhất, chạy theo batch.
beam_search  : giữ k chuỗi có log-xác suất cao nhất ở mỗi bước, chạy từng ảnh.
"""
import numpy as np
import tensorflow as tf

from data import BOS, EOS


def _step(decoder_model, tokens, enc_seq, h, c):
    outputs = decoder_model([tokens, enc_seq, h, c], training=False)
    probs, h, c = outputs[0], outputs[1], outputs[2]
    scores = outputs[3] if len(outputs) > 3 else None
    return probs[:, 0, :], h, c, (scores[:, 0, :] if scores is not None else None)


def greedy_decode(encoder_model, decoder_model, images, max_len, return_attention=False):
    """images: tensor (B, H, W, 1). Trả về danh sách dãy id (và attention (B, steps, T) nếu cần)."""
    enc_seq, h, c = encoder_model(images, training=False)
    batch = int(images.shape[0])
    tokens = tf.fill([batch, 1], BOS)
    outputs, attentions = [], []
    finished = np.zeros(batch, dtype=bool)
    for _ in range(max_len):
        probs, h, c, scores = _step(decoder_model, tokens, enc_seq, h, c)
        next_ids = tf.argmax(probs, axis=-1, output_type=tf.int32)
        outputs.append(next_ids.numpy())
        if return_attention and scores is not None:
            attentions.append(scores.numpy())
        finished |= outputs[-1] == EOS
        if finished.all():
            break
        tokens = next_ids[:, tf.newaxis]
    sequences = np.stack(outputs, axis=1)
    if return_attention:
        att = np.stack(attentions, axis=1) if attentions else None
        return sequences, att
    return sequences


def beam_search(encoder_model, decoder_model, image, max_len, beam_width=3, length_norm=0.7):
    """image: tensor (1, H, W, 1). Trả về dãy id tốt nhất.

    Điểm của một chuỗi = tổng log P / (độ dài ** length_norm), để không ưu tiên quá mức chuỗi ngắn.
    """
    enc_seq, h, c = encoder_model(image, training=False)
    beams = [([], 0.0, h, c)]           # (các id đã sinh, tổng log-prob, h, c)
    finished = []
    for _ in range(max_len):
        if not beams:
            break
        tokens = tf.constant([[b[0][-1] if b[0] else BOS] for b in beams], dtype=tf.int32)
        enc = tf.repeat(enc_seq, len(beams), axis=0)
        hs = tf.concat([b[2] for b in beams], axis=0)
        cs = tf.concat([b[3] for b in beams], axis=0)
        probs, new_h, new_c, _ = _step(decoder_model, tokens, enc, hs, cs)
        log_probs = np.log(probs.numpy() + 1e-12)
        candidates = []
        for i, (ids, score, _, _) in enumerate(beams):
            for tok in np.argsort(log_probs[i])[-beam_width:]:
                candidates.append((ids + [int(tok)], score + float(log_probs[i, tok]), i))
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = []
        for ids, score, i in candidates[:beam_width]:
            if ids[-1] == EOS:
                finished.append((ids, score / (len(ids) ** length_norm)))
            else:
                beams.append((ids, score, new_h[i:i + 1], new_c[i:i + 1]))
        if len(finished) >= beam_width:
            break
    finished += [(b[0], b[1] / (max(len(b[0]), 1) ** length_norm)) for b in beams]
    return max(finished, key=lambda x: x[1])[0]


def predict_texts(encoder_model, decoder_model, image_ds, vocab, max_len, beam_width=1):
    """Dự đoán văn bản cho cả dataset ảnh (tf.data đã batch)."""
    texts = []
    for images in image_ds:
        if beam_width > 1:
            for i in range(int(images.shape[0])):
                ids = beam_search(encoder_model, decoder_model, images[i:i + 1], max_len, beam_width)
                texts.append(vocab.decode(ids))
        else:
            for ids in greedy_decode(encoder_model, decoder_model, images, max_len):
                texts.append(vocab.decode(ids))
    return texts
