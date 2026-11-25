"""Decoder: LSTM sinh từng ký tự (+ Attention), dùng chung trọng số cho train và suy luận.

Kiến thức dùng:
- Chương 8: decoder sinh từng ký tự, đầu vào bước t là ký tự bước t-1 (teacher forcing khi huấn luyện).
- Chương 9: Attention. Bốn biến thể để so sánh:
    none      : không attention, decoder chỉ dựa vào trạng thái truyền từ encoder (Seq2Seq Chương 8)
    bahdanau  : attention cộng v^T tanh(W_q s + W_k h), Dense + keras.layers.AdditiveAttention
    luong     : attention tích vô hướng, keras.layers.Attention (giống LSTM_attention.py)
    multihead : keras.layers.MultiHeadAttention (giống LSTM_MultiHeadAttention.py)
- Chương 3.2: lớp ra Dense + softmax trên từ điển ký tự.
"""
from keras import layers, ops


def build_decoder(cfg, vocab_size):
    """Tạo các lớp decoder một lần và trả về hàm decode(tokens, enc_seq, h, c).

    Gọi decode nhiều lần (mô hình train và mô hình suy luận từng bước) sẽ dùng chung trọng số.
    """
    embedding = layers.Embedding(vocab_size, cfg.embed_dim, name="char_embedding")
    decoder_lstm = layers.LSTM(cfg.dec_units, return_sequences=True, return_state=True, name="decoder_lstm")
    if cfg.variant == "bahdanau":
        # score(s_t, h_i) = v^T tanh(W_q s_t + W_k h_i): W_q, W_k là hai lớp Dense không bias,
        # v là vector scale học được bên trong AdditiveAttention.
        query_proj = layers.Dense(cfg.dec_units, use_bias=False, name="attention_Wq")
        key_proj = layers.Dense(cfg.dec_units, use_bias=False, name="attention_Wk")
        attention = layers.AdditiveAttention(name="attention")
    elif cfg.variant == "luong":
        attention = layers.Attention(name="attention")
    elif cfg.variant == "multihead":
        attention = layers.MultiHeadAttention(num_heads=cfg.num_heads, key_dim=cfg.dec_units // cfg.num_heads,
                                              name="attention")
    else:
        attention = None
    combine = layers.Dense(cfg.dec_units, activation="tanh", name="combine")
    dropout = layers.Dropout(cfg.dropout)
    classifier = layers.Dense(vocab_size, activation="softmax", name="char_probs")

    def decode(tokens, enc_seq, h, c):
        """Chạy decoder trên một chuỗi ký tự vào; trả về (xác suất, h, c, trọng số attention)."""
        x = embedding(tokens)
        dec_seq, h, c = decoder_lstm(x, initial_state=[h, c])
        scores = None
        if attention is None:
            mixed = dec_seq
        else:
            if cfg.variant == "multihead":
                context, scores = attention(query=dec_seq, value=enc_seq, key=enc_seq,
                                            return_attention_scores=True)
                scores = ops.mean(scores, axis=1)  # trung bình các head -> (batch, Lq, Lv)
            elif cfg.variant == "bahdanau":
                # [query, value, key]: query = W_q s_t, key = W_k h_i, value = h_i
                context, scores = attention([query_proj(dec_seq), enc_seq, key_proj(enc_seq)],
                                            return_attention_scores=True)
            else:
                # Luong (dot): score = s_t^T h_i; query = trạng thái decoder, value = key = chuỗi encoder
                context, scores = attention([dec_seq, enc_seq], return_attention_scores=True)
            mixed = combine(layers.Concatenate()([dec_seq, context]))
        probs = classifier(dropout(mixed))
        return probs, h, c, scores

    return decode
