"""Chỉ số đánh giá cho nhận dạng chữ (thay cho R², NSE, MAE, RMSE vốn dành cho giá trị số liên tục).

- CER  = (S + D + I) / N trên ký tự   (S: thay, D: xóa, I: chèn, N: số ký tự của nhãn); càng thấp càng tốt
- WER  = như CER nhưng trên từ (tách theo khoảng trắng)
- Line accuracy = tỉ lệ dòng đoán đúng hoàn toàn
- CER không dấu = CER sau khi bỏ dấu tiếng Việt ở cả nhãn và dự đoán;
  CER - CER không dấu ≈ phần lỗi do dấu thanh / dấu mũ.
"""
import unicodedata


def edit_distance(ref, hyp):
    """Khoảng cách Levenshtein giữa hai dãy (chuỗi ký tự hoặc danh sách từ)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def remove_diacritics(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn"))


def compute_metrics(references, predictions):
    char_err = char_tot = word_err = word_tot = exact = 0
    plain_err = plain_tot = 0
    for ref, hyp in zip(references, predictions):
        char_err += edit_distance(ref, hyp)
        char_tot += len(ref)
        word_err += edit_distance(ref.split(), hyp.split())
        word_tot += len(ref.split())
        exact += ref == hyp
        ref_plain, hyp_plain = remove_diacritics(ref), remove_diacritics(hyp)
        plain_err += edit_distance(ref_plain, hyp_plain)
        plain_tot += len(ref_plain)
    n = max(len(references), 1)
    return {
        "cer": char_err / max(char_tot, 1),
        "wer": word_err / max(word_tot, 1),
        "line_accuracy": exact / n,
        "cer_no_diacritics": plain_err / max(plain_tot, 1),
        "lines": len(references),
    }


def per_line_cer(ref, hyp):
    return edit_distance(ref, hyp) / max(len(ref), 1)


if __name__ == "__main__":
    # Kiểm tra nhanh bằng ví dụ tính tay
    assert edit_distance("kitten", "sitting") == 3
    m = compute_metrics(["Tôi là người Việt Nam"], ["Toi là người Việt Nam"])
    assert abs(m["cer"] - 1 / 21) < 1e-9, m          # 1 ký tự sai / 21 ký tự
    assert abs(m["wer"] - 1 / 5) < 1e-9, m           # 1 từ sai / 5 từ
    assert m["cer_no_diacritics"] == 0.0, m          # lỗi chỉ nằm ở dấu
    assert m["line_accuracy"] == 0.0
    print("metrics.py: các phép kiểm tra đều đúng", m)
