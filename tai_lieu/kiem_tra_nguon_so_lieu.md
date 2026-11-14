# Đối chiếu nguồn và phạm vi xác thực số liệu

## Nguyên tắc trình bày

Tài liệu này phân biệt rõ: **(1) số liệu do nhóm báo cáo trên slide; (2) số liệu xuất hiện trong đầu ra đã lưu của notebook; (3) số liệu chưa được kiểm chứng bằng chạy lại**. Không suy đoán thêm chữ số thập phân hoặc điền giá trị chưa có nguồn.

## Các nguồn đã kiểm tra

- Slide: `CLIP_Presentation_Group12(1).pptx`, 23 trang, phần thực nghiệm trang 15–21.
- Notebook: `CLIP_CIFAR100_BAN_NOP_DA_CHINH_LOGIC (1).ipynb`, 34 ô; ô cuối phần Test chứa bảng Top-1, Top-5, Macro-F1 và phân tích nhóm.
- Bài báo: Radford và cộng sự (2021), arXiv:2103.00020v1.
- Tài liệu và CSV cũ của repository: chuyển vào thư mục lưu trữ riêng để tránh nhầm với phiên cuối.

## Đối chiếu Test

| Phương pháp | Top-1 | Top-5 | Macro-F1 | Slide tr.18 | Notebook ô Test |
|---|---:|---:|---:|---|---|
| Zero-shot ensemble 18 prompt | 65,12 | 89,24 | 64,31 | Khớp | Khớp |
| Logistic Regression | 79,39 | 96,16 | 79,43 | Khớp | Khớp |
| AdamW linear probe | 78,84 | 96,04 | 78,86 | Khớp | Khớp |

Tất cả giá trị là **phần trăm**, được chép từ nguồn hiển thị. Chưa chạy lại notebook hoặc đối chiếu trực tiếp từng dự đoán trên ảnh.

## Vấn đề phiên chạy cũ

CSV cũ `ket_qua_nhom.csv` ghi **65,13%** Top-1 cho zero-shot ensemble, **80,03%** cho linear probe trên 50.000 ảnh train; trong khi slide và đầu ra notebook của phiên cuối ghi **65,12%** zero-shot và **79,39%** Logistic Regression với chia 40.000 train / 10.000 validation / 10.000 test. Đây **không phải cùng bảng kết quả**; không được ghép hai nguồn làm một thí nghiệm.

File CSV cũ được giữ ở `ket_qua/luu_tru_phien_cu/` chỉ để truy vết lịch sử.

## Đối chiếu số liệu khác

- Slide trang 16: validation zero-shot (tên lớp 60,24/85,37/59,09; một prompt 64,38/88,05/63,54; ensemble 65,33/89,26/64,66); chưa chạy lại để kiểm chứng từng hàng.
- Slide trang 17 và đầu ra notebook: AdamW epoch cuối 100; train loss đầu 3,903921 và cuối 0,541425; validation Top-1 cuối 79,32%. Không kết luận đã hội tụ.
- Slide trang 18 và đầu ra notebook: thời gian huấn luyện phân loại 0 / 81,07 / 56,55 giây; dự đoán 1,318 / 0,050 / 0,008 giây; **không gồm thời gian trích xuất embedding CLIP**.
- Slide trang 19 và đầu ra notebook: các nhóm độ khó A/B/C có 3.500/3.200/3.300 ảnh Test; nhóm được xác định từ validation.
- Slide trang 23 nêu hạn chế lịch sử các phiên chạy trước chưa lưu đủ; vì vậy **không thể tuyên bố kết quả đã được xác minh tuyệt đối hoặc Test hoàn toàn độc lập**.

## Quy ước

- `ket_qua_test_phien_cuoi.csv`: giá trị **trùng giữa slide và đầu ra notebook lưu sẵn**.
- `ket_qua_validation_prompt.csv`: **giá trị theo slide trang 16**, không đồng nghĩa đã tái lập.
- `ket_qua_nhom_do_kho_test.csv` và `ket_qua_superclass_logistic_regression.csv`: **giá trị trích từ đầu ra notebook đã lưu**, không tái chạy.
- Các tài liệu cũ: lưu trữ lịch sử, **không dùng để kết luận phiên cuối**.
