# Task 3 – Nhận dạng dòng chữ viết tay tiếng Việt (Image-to-Sequence)

Đầu vào là ảnh một dòng chữ viết tay. Đầu ra là chuỗi ký tự Unicode của dòng đó.
Dataset dùng là [5CD-AI/VietHTR-Line](https://huggingface.co/datasets/5CD-AI/VietHTR-Line), gồm 59.247 ảnh train và 1.000 ảnh test, giấy phép CC BY-NC-ND 4.0.

Mô hình gồm CNN Encoder (khối kiểu VGG), BiLSTM, LSTM Decoder và Attention, viết bằng Keras (TensorFlow 2).
Cách viết theo code mẫu của lecture (`LSTM_attention.py`, `LSTM_MultiHeadAttention.py`, `DL_pollution.ipynb`).

## Cấu trúc file

| File | Vai trò | Kiến thức lecture |
| --- | --- | --- |
| `config.py` | Đường dẫn, siêu tham số, tham số dòng lệnh dùng chung | – |
| `data.py` | Đọc Parquet, chuẩn NFC, chia train/val, từ điển ký tự, mã hóa nhãn | Ch.1, Ch.8 (token, `<bos>`/`<eos>`), Ch.3.2 |
| `preprocessing.py` | Resize/pad ảnh, augmentation, pipeline `tf.data` | Ch.5.1, Ch.5.3 |
| `eda.py` | Thống kê và hình cho mục 3.1 của báo cáo | Ch.1 |
| `encoder.py` | Encoder: CNN kiểu VGG + BiLSTM, trả về chuỗi đặc trưng và trạng thái đầu cho decoder | Ch.4.3, 5.1, 5.2, 7, 8 |
| `decoder.py` | Decoder: LSTM sinh từng ký tự + 4 biến thể attention, lớp softmax | Ch.3.2, 7, 8, 9 |
| `model.py` | Ghép encoder-decoder thành mô hình train và mô hình suy luận từng bước, loss, compile | Ch.2.2, 3.2, 4.3, 8 |
| `metrics.py` | CER, WER, line accuracy, CER không dấu | – |
| `inference.py` | Suy luận: giải mã greedy và beam search | Ch.8 |
| `train.py` | Huấn luyện với EarlyStopping, ReduceLROnPlateau, checkpoint, history | Ch.2.2, 3.2, 4.3, 7 |
| `evaluate.py` | Chấm validation/test, lưu metrics và dự đoán | – |
| `visualize.py` | Learning curve, ảnh dự đoán, heatmap attention | Ch.9 |

Mô hình chính là `--variant bahdanau` (Attention cộng, Chương 9). Nhóm train bản này trước.
Code vẫn giữ tham số `--variant` để chạy thêm biến thể khác khi cần so sánh:

| `--variant` | Decoder dùng gì | Vai trò |
| --- | --- | --- |
| `bahdanau` | Attention cộng: v·tanh(W_q·s + W_k·h) | **Mô hình chính, train mặc định** |
| `none` | Chỉ trạng thái cuối của encoder (Seq2Seq, Ch.8) | Tùy chọn: train nếu còn quota GPU để thấy tác dụng của attention |
| `luong` | Attention tích vô hướng s·h, giống `LSTM_attention.py` | Không bắt buộc |
| `multihead` | `MultiHeadAttention` 4 head, giống `LSTM_MultiHeadAttention.py` | Không bắt buộc |

## Luồng chạy

```
eda.py ──► train.py ──► evaluate.py (val) ──► evaluate.py (test) ──► visualize.py
           (bahdanau)
   tùy chọn: lặp train/evaluate/visualize với --variant none
```

1. **Cài đặt:** `pip install -r requirements.txt`.
2. **Chạy thử (khoảng vài phút):** để kiểm tra đường dẫn và môi trường. Kết quả ghi vào thư mục riêng `outputs_smoke` để không lẫn với lần chạy thật.
   ```bash
   python train.py --data-dir DATA --output-dir outputs_smoke --limit-train 256 --limit-val 64 --epochs 1
   ```
3. **EDA:** `python eda.py --data-dir DATA` tạo `outputs/eda/` (số liệu và hình cho mục 3.1).
4. **Huấn luyện mô hình chính:** `python train.py --data-dir DATA --variant bahdanau --epochs 22` (mô hình nộp được train 22 epoch; mặc định trong `config.py` là 30) tạo `outputs/bahdanau/best.weights.h5`, `history.csv`, `train_info.json` và `model_summary.txt`.
5. **Đánh giá trên validation:** `python evaluate.py --data-dir DATA --variant bahdanau --split val`
6. **Đánh giá trên test (chạy một lần ở cuối):**
   ```bash
   python evaluate.py --data-dir DATA --variant bahdanau --split test
   python evaluate.py --data-dir DATA --variant bahdanau --split test --beam-width 3
   ```
7. **Vẽ hình:** `python visualize.py --data-dir DATA --variant bahdanau` tạo `outputs/bahdanau/figures/`.
8. **(Tùy chọn) Bản không attention:** nếu còn quota GPU, lặp lại bước 4–7 với `--variant none` để so sánh có và không có attention.

Ở đây `DATA` là thư mục chứa các file `train-*.parquet` và `test-*.parquet`, hoặc thư mục cha có thư mục con `data/`.
Mọi siêu tham số trong `config.py` đều đổi được bằng dòng lệnh, ví dụ `--batch-size 64 --epochs 40 --augment false`.
Xem đủ danh sách bằng `python train.py --help`.

## Chỉ số đánh giá

Ký tự là nhãn rời rạc, không phải số liên tục, nên R², NSE, MAE và RMSE không có ý nghĩa với bài toán này.
Project dùng các chỉ số chuẩn của nhận dạng chữ:

- **CER** = (S + D + I) / N tính trên ký tự, với S, D, I là số ký tự bị thay, xóa, chèn và N là số ký tự của nhãn. Đây là chỉ số chính, càng thấp càng tốt. Mô hình được chọn theo CER validation.
- **WER**: cách tính như CER nhưng trên từ.
- **Line accuracy**: tỉ lệ dòng đoán đúng hoàn toàn.
- **CER không dấu**: CER sau khi bỏ dấu tiếng Việt. Hiệu CER − CER không dấu cho biết phần lỗi do dấu thanh và dấu mũ.

## Kết quả

Mô hình `bahdanau` (22 epoch) trên tập test, beam search width 3: CER 4,95%, WER 12,8%, line accuracy 37,4%.

## Chia dữ liệu

- Test giữ nguyên tập chính thức gồm 1.000 ảnh.
- Từ 59.247 ảnh train chính thức, lấy ngẫu nhiên 10% (seed 42) làm validation: 53.322 ảnh train và 5.925 ảnh validation.
- Từ điển ký tự xây từ nhãn của tập train và lưu ở `outputs/vocab.json`. Mọi biến thể dùng chung từ điển này.
