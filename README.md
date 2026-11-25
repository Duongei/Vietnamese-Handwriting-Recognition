# Task 3 – Nhận dạng dòng chữ viết tay tiếng Việt (Image-to-Sequence)

Đầu vào là ảnh một dòng chữ viết tay, đầu ra là chuỗi ký tự của dòng đó.
Dataset: [5CD-AI/VietHTR-Line](https://huggingface.co/datasets/5CD-AI/VietHTR-Line), 59.247 ảnh train và 1.000 ảnh test, giấy phép CC BY-NC-ND 4.0.

Mô hình: CNN kiểu VGG + BiLSTM làm encoder, LSTM + Bahdanau attention làm decoder, viết bằng Keras (TensorFlow 2).

## Các file

| File | Nội dung | Kiến thức lecture |
| --- | --- | --- |
| `config.py` | Siêu tham số, tham số dòng lệnh dùng chung | – |
| `data.py` | Đọc Parquet, chia train/val, từ điển ký tự, tiền xử lý và tăng cường ảnh, `tf.data` | Ch.1, 3.2, 5.1, 5.3, 8 |
| `model.py` | Encoder, decoder, 4 kiểu attention, loss, giải mã greedy và beam search | Ch.2.2, 3.2, 4.3, 5.2, 7, 8, 9 |
| `train.py` | Huấn luyện, lưu trọng số tốt nhất theo CER val, chấm toàn bộ tập val | Ch.2.2, 4.3, 7 |
| `test.py` | Chấm tập test: CER, WER, line accuracy, CER không dấu (greedy và beam 3) | Ch.8 |
| `eda.py` | Thống kê và hình cho mục 3.1 của báo cáo | Ch.1 |
| `visualize.py` | Learning curve, ảnh dự đoán, heatmap attention | Ch.9 |
| `compare.py` | Gom kết quả các biến thể thành bảng và biểu đồ CER | – |

## Biến thể attention

Chọn bằng `--variant` (mặc định `bahdanau`). Kết quả mỗi biến thể nằm trong `outputs/<variant>/`.

| `--variant` | Decoder dùng gì | Vai trò |
| --- | --- | --- |
| `bahdanau` | Attention cộng: v·tanh(W_q·s + W_k·h) | Mô hình chính, đã train 22 epoch |
| `none` | Chỉ trạng thái cuối của encoder (Seq2Seq, Ch.8) | Tùy chọn, để thấy tác dụng của attention |
| `luong` | Attention tích vô hướng, giống `LSTM_attention.py` | Tùy chọn |
| `multihead` | `MultiHeadAttention` 4 head, giống `LSTM_MultiHeadAttention.py` | Tùy chọn |

## Thứ tự chạy

```
eda.py → train.py → test.py → visualize.py → compare.py
```

`DATA` là thư mục chứa `train-*.parquet` và `test-*.parquet` (hoặc thư mục cha có thư mục con `data/`).

1. Cài đặt: `pip install -r requirements.txt`
2. Chạy thử vài phút để kiểm tra môi trường (ghi vào thư mục riêng để không đè kết quả thật):
   `python train.py --data-dir DATA --output-dir outputs_smoke --limit 256 --epochs 1`
3. EDA: `python eda.py --data-dir DATA` → `outputs/eda/`
4. Train: `python train.py --data-dir DATA` → `outputs/vocab.json`, `outputs/bahdanau/best.weights.h5`, `history.csv`, `val_metrics.json`, `val_predictions.csv`
5. Test (chạy một lần ở cuối): `python test.py --data-dir DATA` → `test_metrics.json`, `test_metrics_beam3.json` và file dự đoán tương ứng
6. Vẽ hình: `python visualize.py --data-dir DATA` → `outputs/bahdanau/figures/`

7. (Tùy chọn) Biến thể khác: lặp bước 4 đến 6 với `--variant none` (hoặc `luong`, `multihead`), rồi `python compare.py` → `outputs/comparison.csv`, `comparison.md`, `comparison_cer.png`

Đã có `outputs/` (trọng số 22 epoch) thì bỏ qua bước 4, chạy thẳng bước 5 và 6.

## Cấu hình train và tăng tốc

Mô hình đã nộp được train 22 epoch, batch 32, không bật mixed precision. Đây cũng là mặc định của `train.py`.
Nếu train lại trên GPU và muốn nhanh hơn:

```
python train.py --data-dir DATA --batch-size 64 --mixed-precision
```

Cách này nhanh hơn trên T4 nhưng kết quả có thể lệch đôi chút so với bản đã nộp.
Sau mỗi epoch, CER chỉ được tính trên 512 ảnh val (`CER_SAMPLES` trong `config.py`) vì giải mã từng ký tự rất chậm.
Tập val không tăng cường nên được cache trong RAM (khoảng 1,5 GB) sau epoch đầu.

## Chỉ số đánh giá

Ký tự là nhãn rời rạc nên R², NSE, MAE, RMSE không có ý nghĩa với bài toán này. Project dùng:

- **CER** = (S + D + I) / N trên ký tự (thay, xóa, chèn so với số ký tự của nhãn). Chỉ số chính, càng thấp càng tốt.
- **WER**: như CER nhưng trên từ.
- **Line accuracy**: tỉ lệ dòng đoán đúng hoàn toàn.
- **CER không dấu**: CER sau khi bỏ dấu tiếng Việt, cho biết phần lỗi do dấu.

## Kết quả

Trên tập test, beam search width 3: CER 4,95%, WER 12,8%, line accuracy 37,4%.

## Chia dữ liệu

- Test giữ nguyên 1.000 ảnh chính thức.
- Lấy ngẫu nhiên 10% tập train chính thức (seed 42) làm validation: 53.322 ảnh train, 5.925 ảnh val.
- Từ điển ký tự xây từ nhãn train, lưu ở `outputs/vocab.json`.
