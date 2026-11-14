# Nghiên cứu và thực nghiệm mô hình CLIP trên bộ dữ liệu CIFAR-100

**Đề tài:** *Learning Transferable Visual Models From Natural Language Supervision* (Radford và cộng sự, ICML 2021)  
**Nhóm thực hiện:** Nhóm 12 — Trần Minh Đạt, Nguyễn Thị Thanh Huệ, Vũ Minh Thư  
**Giảng viên hướng dẫn:** ThS. Nguyễn Văn Sơn

## 1. Giới thiệu

Đề tài tìm hiểu mô hình **CLIP (Contrastive Language–Image Pre-training)** và đánh giá khả năng chuyển giao biểu diễn ảnh–văn bản sang bài toán phân loại ảnh CIFAR-100. Nhóm sử dụng mô hình **CLIP ViT-B/32 đã được huấn luyện trước**, giữ cố định các bộ mã hóa và so sánh ba cách phân loại: **zero-shot**, **Logistic Regression** và **bộ phân loại tuyến tính tối ưu bằng AdamW**. Nhóm **không huấn luyện lại CLIP từ đầu**.

Nhóm sử dụng kết quả thực nghiệm lưu trong notebook để tổng hợp các bảng đánh giá và đối chiếu với số liệu trình bày trong slide. Các nhận xét được giới hạn trong phạm vi cấu hình và kết quả của lần thực nghiệm này.

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

Nhóm sử dụng tập validation có nhãn để lựa chọn cấu hình prompt. Vì vậy, cần phân biệt việc dự đoán zero-shot không huấn luyện trên nhãn tập đích với bước lựa chọn cấu hình có sử dụng validation.

## 4. Kết quả trên tập Test

Bảng dưới đây tổng hợp kết quả đánh giá trên 10.000 ảnh thuộc tập Test, theo đầu ra đã lưu trong notebook và bảng kết quả trình bày ở slide 18:

| Phương pháp | Top-1 (%) | Top-5 (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| Zero-shot (ensemble 18 prompt) | 65,12 | 89,24 | 64,31 |
| Logistic Regression | **79,39** | **96,16** | **79,43** |
| AdamW linear probe | 78,84 | 96,04 | 78,86 |

Logistic Regression cao hơn zero-shot **14,27 điểm phần trăm Top-1** (tương ứng 1.427 ảnh dự đoán đúng nhiều hơn trên 10.000 ảnh Test). Chênh lệch Top-1 giữa Logistic Regression và AdamW là **0,55 điểm phần trăm**. Kết quả cho thấy Logistic Regression đạt Top-1 cao nhất trong ba phương pháp được khảo sát. Tuy nhiên, nhóm chưa đánh giá nhiều seed và nhiều cấu hình để kết luận về độ ổn định của chênh lệch này.

Các bảng số liệu: [Kết quả Test](ket_qua/ket_qua_test_phien_cuoi.csv) và [Validation](ket_qua/ket_qua_validation_prompt.csv).

## 5. Cấu trúc thư mục

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── ket_qua/
│   ├── ket_qua_test_phien_cuoi.csv
│   ├── ket_qua_validation_prompt.csv
│   ├── ket_qua_nhom_do_kho_test.csv
│   └── ket_qua_superclass_logistic_regression.csv
├── ma_nguon/
│   ├── CLIP_CIFAR100_BAN_NOP_DA_CHINH_LOGIC (1).ipynb
│   └── HUONG_DAN_NOTEBOOK.md
└── bai_bao_va_slide/
    └── TAI_LIEU_THAM_KHAO.md
```

## 6. Chạy lại và kiểm chứng

1. Chuẩn bị Python, Jupyter/Google Colab và cài các gói trong `requirements.txt`.
2. Cài CLIP gốc: `pip install git+https://github.com/openai/CLIP.git`.
3. Mở notebook `ma_nguon/CLIP_CIFAR100_BAN_NOP_DA_CHINH_LOGIC (1).ipynb` và thực hiện các ô theo thứ tự.
4. Kiểm tra cách chia dữ liệu, seed, prompt, tham số, checkpoint và đầu ra trước khi đánh giá Test.
5. Đối chiếu bảng sinh ra từ notebook với các CSV ở `ket_qua/`.


## 7. Hạn chế

- Mới đánh giá một backbone CLIP trên một bộ dữ liệu.
- AdamW mới có một seed và một cấu hình learning rate; checkpoint tốt nhất ở epoch 100, chưa đủ bằng chứng kết luận hội tụ.
- Lựa chọn prompt có sử dụng nhãn validation.
- Nhóm chưa lưu đầy đủ lịch sử các phiên thử nghiệm trước đó, nên việc kiểm tra lại toàn bộ quá trình lựa chọn cấu hình còn hạn chế.
- Các kết quả được tổng hợp từ đầu ra notebook đã lưu; nhóm chưa thực hiện thêm một lần chạy độc lập để đánh giá khả năng tái lập.

## 8. Tài liệu tham khảo

1. Radford, A. và cộng sự (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. https://arxiv.org/abs/2103.00020
2. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. https://www.cs.toronto.edu/~kriz/cifar.html
3. Mã nguồn CLIP của OpenAI: https://github.com/openai/CLIP
