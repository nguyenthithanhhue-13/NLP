# Nghiên cứu và thực nghiệm mô hình CLIP trên bộ dữ liệu CIFAR-100

**Đề tài:** *Learning Transferable Visual Models From Natural Language Supervision* (Radford và cộng sự, ICML 2021)  
**Nhóm thực hiện:** Nhóm 12 — Trần Minh Đạt, Nguyễn Thị Thanh Huệ, Vũ Minh Thư  
**Giảng viên hướng dẫn:** ThS. Nguyễn Văn Sơn

## 1. Giới thiệu

Đề tài tìm hiểu mô hình **CLIP (Contrastive Language–Image Pre-training)** và đánh giá khả năng chuyển giao biểu diễn ảnh–văn bản sang bài toán phân loại ảnh CIFAR-100. Nhóm sử dụng mô hình **CLIP ViT-B/32 đã được huấn luyện trước**, giữ cố định các bộ mã hóa và so sánh ba cách phân loại: **zero-shot**, **Logistic Regression** và **bộ phân loại tuyến tính tối ưu bằng AdamW**. Nhóm **không huấn luyện lại CLIP từ đầu**.

Nội dung trình bày dựa trên slide của Nhóm 12 và notebook thực nghiệm được cung cấp. Số liệu được ghi theo **kết quả đã lưu trong notebook của phiên chạy cuối** và đối chiếu với slide; **chưa chạy lại toàn bộ thực nghiệm độc lập trong lần biên tập GitHub này**.

## 2. Mục tiêu nghiên cứu

1. Trình bày nguyên lý học tương phản và cơ chế phân loại zero-shot của CLIP.
2. Đánh giá ảnh hưởng của tên lớp, một prompt và ensemble 18 prompt.
3. So sánh kết quả zero-shot với hai phương pháp linear probe trên cùng đặc trưng ảnh CLIP.
4. Phân tích Top-1, Top-5, Macro-F1, nhóm độ khó A/B/C, 20 superclass, sai số và đường cong huấn luyện.
5. Thảo luận giới hạn của thiết kế thực nghiệm và khả năng tái lập.

## 3. Thiết lập thực nghiệm

| Thành phần | Thiết lập được báo cáo |
|---|---|
| Dữ liệu | CIFAR-100, 100 lớp chi tiết, 20 superclass |
| Chia dữ liệu | 40.000 train / 10.000 validation / 10.000 test; chia phân tầng, seed 42 |
| Backbone | CLIP ViT-B/32 (OpenAI), đóng băng |
| Đặc trưng ảnh | Vector 512 chiều, chuẩn hóa L2 |
| Zero-shot | Tên lớp; một prompt; ensemble 18 prompt |
| Logistic Regression | L-BFGS; chọn hệ số C trên validation |
| AdamW linear probe | Lớp tuyến tính 512 → 100; learning rate 0,003; weight decay 0,001; tối đa 100 epoch |
| Đánh giá | Top-1, Top-5, Macro-F1; đối chứng Dummy |

**Lưu ý:** Việc chọn prompt dựa vào nhãn validation là bước lựa chọn cấu hình có giám sát; không nên mô tả toàn bộ quy trình chọn prompt là zero-shot hoàn toàn không dùng nhãn.

## 4. Kết quả trên tập Test

Kết quả **được lưu trong đầu ra ô thực nghiệm cuối của notebook** (10.000 ảnh Test) và **trùng với bảng trên slide trang 18**:

| Phương pháp | Top-1 (%) | Top-5 (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| Zero-shot (ensemble 18 prompt) | 65,12 | 89,24 | 64,31 |
| Logistic Regression | **79,39** | **96,16** | **79,43** |
| AdamW linear probe | 78,84 | 96,04 | 78,86 |

Logistic Regression cao hơn zero-shot **14,27 điểm phần trăm Top-1** (tương ứng 1.427 ảnh dự đoán đúng nhiều hơn trên 10.000 ảnh Test). Chênh lệch Top-1 giữa Logistic Regression và AdamW là **0,55 điểm phần trăm**. Đây là so sánh trong **phiên chạy được lưu**, chưa đủ cơ sở để khẳng định một phương pháp luôn vượt trội ở các seed và cấu hình khác.

Các bảng nguồn và ghi chú đối chiếu: [Kết quả Test](ket_qua/ket_qua_test_phien_cuoi.csv), [Validation](ket_qua/ket_qua_validation_prompt.csv), [Kiểm tra nguồn số liệu](tai_lieu/kiem_tra_nguon_so_lieu.md).

## 5. Cấu trúc thư mục

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── tai_lieu/
│   ├── kiem_tra_nguon_so_lieu.md
│   └── ghi_chu_va_noi_dung_cu/     # tài liệu gốc, có thể thuộc phiên chạy trước
├── ket_qua/
│   ├── ket_qua_test_phien_cuoi.csv
│   ├── ket_qua_validation_prompt.csv
│   ├── ket_qua_nhom_do_kho_test.csv
│   ├── ket_qua_superclass_logistic_regression.csv
│   └── luu_tru_phien_cu/           # kết quả cũ, không dùng làm kết quả phiên cuối
├── ma_nguon/
│   └── HUONG_DAN_NOTEBOOK.md
└── bai_bao_va_slide/
    └── TAI_LIEU_THAM_KHAO.md
```

## 6. Chạy lại và kiểm chứng

1. Chuẩn bị Python, Jupyter/Google Colab và cài các gói trong `requirements.txt`.
2. Cài CLIP gốc: `pip install git+https://github.com/openai/CLIP.git`.
3. Sử dụng notebook gốc `CLIP_CIFAR100_BAN_NOP_DA_CHINH_LOGIC.ipynb` (không sửa mã nguồn khi chuyển vào repository).
4. Kiểm tra cách chia dữ liệu, seed, prompt, tham số, checkpoint và đầu ra trước khi đánh giá Test.
5. Đối chiếu bảng sinh ra từ notebook với các CSV ở `ket_qua/`.


## 7. Hạn chế

- Mới đánh giá một backbone CLIP trên một bộ dữ liệu.
- AdamW mới có một seed và một cấu hình learning rate; checkpoint tốt nhất ở epoch 100, chưa đủ bằng chứng kết luận hội tụ.
- Lựa chọn prompt có sử dụng nhãn validation.
- Theo slide kết luận, lịch sử phiên chạy trước chưa được lưu đầy đủ; chưa thể xác nhận tuyệt đối tính độc lập của Test trong toàn bộ quá trình thử nghiệm.
- Những con số được đối chiếu với đầu ra lưu trong notebook **không tương đương với kiểm chứng tái lập độc lập**.

## 8. Tài liệu tham khảo

1. Radford, A. và cộng sự (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. https://arxiv.org/abs/2103.00020
2. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. https://www.cs.toronto.edu/~kriz/cifar.html
3. Mã nguồn CLIP của OpenAI: https://github.com/openai/CLIP
