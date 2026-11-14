# Hỏi đáp, kịch bản demo và checklist của Huệ

Mảng Huệ trả lời khi hỏi đáp: **dữ liệu, số liệu, cách chạy demo**.

---

## A. Ba câu hỏi chính kèm đáp án (STT 32, dán vào Docs)

### Câu 1. Zero-shot và linear probe khác nhau thế nào? Vì sao linear probe luôn cao hơn?
- **Zero-shot:** không dùng ảnh huấn luyện nào của tập đích. Tên lớp được đưa qua bộ mã hóa văn bản để tạo trọng số phân loại.
- **Linear probe:** giữ nguyên bộ mã hóa ảnh, nhưng huấn luyện thêm một lớp hồi quy logistic trên toàn bộ tập train có nhãn. Vì được nhìn dữ liệu có nhãn nên cao hơn là hợp lý. Paper coi linear probe gần như là **giới hạn trên** của zero-shot.
- **Số liệu:** với ViT-B/32 trên CIFAR-100, zero-shot đạt 65,1% còn linear probe đạt 80,5% (Bảng 10, 11). Nhìn chung zero-shot thấp hơn 10–25 điểm (Hình 8).
- **Điểm hay:** zero-shot CLIP ngang hồi quy logistic được học từ **4 ảnh/lớp** (Hình 6). Trên ImageNet, nó ngang mức **16 ảnh/lớp** (Hình 7).

### Câu 2. Dữ liệu 400 triệu cặp lấy từ internet, có thể chứa sẵn ảnh test. Vậy kết quả có bị "thổi phồng" không?
- Paper đã kiểm tra việc này ở §5. Họ xây một bộ phát hiện ảnh trùng và quét **35 bộ** dữ liệu đánh giá.
- Tỉ lệ trùng lặp có **trung vị 2,2%** và **trung bình 3,2%**. Có 9 bộ không trùng ảnh nào.
- Độ chính xác hiếm khi thay đổi quá 0,1%. Mức tăng lớn nhất chỉ **0,6%**, ở Birdsnap.
- Country211 trùng nhiều nhất (21,5%) nhưng chỉ tăng 0,2%, vì văn bản đi kèm ảnh thường không nhắc tới địa điểm.
- **Kết luận:** trùng lặp có tồn tại nhưng ảnh hưởng tới kết quả là không đáng kể.

### Câu 3. Vì sao thêm "a photo of a" lại tăng độ chính xác? Số của nhóm có khớp paper không?
- **Lý do 1, đa nghĩa:** nếu chỉ có một từ thì mô hình không biết nghĩa nào. Ví dụ "crane" vừa là cần cẩu vừa là con sếu, "boxer" vừa là giống chó vừa là võ sĩ.
- **Lý do 2, khác biệt phân phối:** khi huấn luyện, văn bản thường là cả câu mô tả ảnh, hiếm khi chỉ có một từ.
- **Số của paper trên ImageNet:** một prompt tăng **+1,3%**, ensemble 80 prompt tăng **thêm 3,5%**, tổng gần **5%**.
- **Nhóm (CIFAR-100):** không prompt đạt 60,44%, một prompt 64,40% (**+3,96 điểm**), ensemble 65,13% (thêm **+0,73 điểm**). Xu hướng giống paper. Ensemble của nhóm đạt 65,13%, **khớp với 65,1% của paper**. Linear probe đạt 80,03% so với 80,5% của paper.
- **Nếu bị hỏi vì sao số lệch:** bộ 18 prompt của nhóm giống hệt bộ tác giả công bố (`data/prompts.md` của repo), nên zero-shot ensemble chỉ lệch nhỏ do phần cứng và phiên bản thư viện. Linear probe lệch nhiều hơn vì nhóm dùng C cố định (paper quét C) và dùng đặc trưng sau phép chiếu (paper dùng trước phép chiếu). README đã ghi rõ các điểm này.

## B. Câu hỏi dự phòng

| Câu hỏi | Trả lời ngắn |
|---|---|
| Vì sao paper ghi 76,2% mà nhóm chỉ được 65,1%? | 76,2% là của **ViT-L/14@336px** (mô hình lớn nhất) trên **ImageNet**. Nhóm dùng ViT-B/32 (nhỏ nhất) trên CIFAR-100. Ảnh CIFAR chỉ 32×32 nên phải phóng to lên 224×224. Paper với đúng cấu hình của nhóm cũng chỉ đạt 65,1% |
| Huấn luyện CLIP tốn bao nhiêu? Nhóm có huấn luyện lại không? | RN50x64 mất 18 ngày trên 592 GPU V100. ViT-L/14 mất 12 ngày trên 256 GPU V100. Nhóm **không huấn luyện lại** mà chỉ dùng trọng số OpenAI công bố để suy luận trên Colab |
| CLIP yếu ở đâu? | Đếm vật, ảnh vệ tinh (EuroSAT, thua 37,1 điểm), ảnh y tế (PatchCamelyon), khoảng cách xe (KITTI). Chỉ đạt 88% trên MNIST, thua cả hồi quy logistic trên pixel thô, vì dữ liệu huấn luyện gần như không có chữ viết tay |
| Vì sao CLIP bền vững hơn? | Zero-shot không được huấn luyện trên phân phối của ImageNet, nên ít học các tương quan giả chỉ đúng trên ImageNet. Bằng chứng: khi cho CLIP học thêm trên ImageNet, độ chính xác ImageNet tăng 9,2% (lên 85,4%) nhưng độ bền vững lại giảm nhẹ (Hình 14) |
| Demo có dùng tiếng Việt được không? | Không. CLIP được huấn luyện trên văn bản tiếng Anh nên nhãn phải viết bằng tiếng Anh. Phần tiếng Việt trên biểu đồ chỉ để hiển thị |
| Dữ liệu WIT được thu thập thế nào? | Dùng 500.000 truy vấn (các từ xuất hiện ít nhất 100 lần trên Wikipedia tiếng Anh, bigram, tên bài Wikipedia, WordNet synset). Mỗi truy vấn lấy tối đa 20.000 cặp để cân bằng. Tổng cộng 400 triệu cặp |

---

## C. Kịch bản video demo dự phòng 1–2 phút (STT 30)

Quay màn hình Colab bằng OBS, Xbox Game Bar (`Win + G`) hoặc Loom. Nên chạy notebook xong trước rồi mới quay, để không phải chờ trong video.

| Thời điểm | Hình ảnh | Lời nói |
|---|---|---|
| 0:00–0:10 | Đầu notebook | "Đây là notebook thực nghiệm CLIP ViT-B/32 của nhóm, chạy trên Google Colab với GPU T4." |
| 0:10–0:30 | Mục 4.1: ảnh CIFAR-100 và top-5 | "Đầu tiên, CLIP nhận một ảnh CIFAR-100 và chọn trong 100 tên lớp mà không cần huấn luyện. Ở đây là ảnh con rắn, CLIP đoán đúng 'snake' với xác suất 67,9%." |
| 0:30–0:55 | Mục 8: bảng và biểu đồ | "Trên 10.000 ảnh test: không dùng prompt đạt 60,4%, thêm 'a photo of a' đạt 64,4%, ensemble 18 prompt đạt 65,1%, khớp đúng con số trong paper. Linear probe đạt 80,0%, sát với 80,5% của paper." |
| 0:55–1:30 | Mục 9: tải 2–3 ảnh tự chụp lên | "Giờ thử với ảnh nhóm tự chụp. Danh sách nhãn này do mình tự đặt, CLIP chưa từng được huấn luyện riêng cho nó. Kết quả: …" |
| 1:30–1:45 | Biểu đồ demo | "Đây chính là zero-shot: muốn nhận dạng lớp mới thì chỉ cần viết thêm một dòng mô tả." |

**Ảnh nên chuẩn bị (3–4 ảnh, chụp rõ, một vật chính):** cốc cà phê, xe máy, bát phở, con mèo hoặc con chó, laptop.
Nên có **một ảnh khó**, ví dụ một bức vẽ tay hình con mèo, để minh họa độ bền vững như ImageNet-Sketch.
Nhớ sửa `LABELS` trong notebook cho khớp với các ảnh.

## D. Kịch bản demo trực tiếp ở slide 6 (STT 38, ~30 giây)

1. Đạt nói: *"Bây giờ Huệ sẽ demo"*. Huệ chuyển sang tab Colab, mở sẵn ở **mục 9** (đã chạy hết các ô trước).
2. Chạy ô demo, chọn ảnh có sẵn trên máy, đọc to kết quả: *"CLIP nhận ra đây là … với …%"*.
3. Nếu còn thời gian: thêm một nhãn mới vào `LABELS` (ví dụ `"a durian": "quả sầu riêng"`) rồi chạy lại. Điều này cho thấy có thể thêm lớp mới mà không cần huấn luyện.
4. **Nếu lỗi hoặc mất mạng quá 10 giây:** nói *"Em xin mở video đã quay sẵn"* và phát video dự phòng.

---

## E. Checklist máy demo – sáng Ngày 5, 21/10 (STT 37)

**Tối Ngày 4:**
- [ ] Notebook chạy hết, không có ô lỗi. Đã lưu bản có output (File → Save).
- [ ] `bieu_do_ket_qua.png` đã gửi Thư; số liệu đã điền vào slide 6, README và Docs.
- [ ] Video demo dự phòng đã lưu **trên máy**, không chỉ trên Drive, và đã mở thử.
- [ ] 3–4 ảnh demo đã nằm trong một thư mục trên máy.
- [ ] Link Colab và video đã dán vào cột Ghi chú / Link (sheet Lịch 5 ngày) và vào README.

**Trước giờ thuyết trình 30 phút:**
- [ ] Máy đã sạc đầy, mang theo sạc và đầu chuyển HDMI.
- [ ] Đã kết nối Wi-Fi phòng thi; mang điện thoại để phát 4G dự phòng.
- [ ] Mở Colab, kết nối GPU, chạy lại các ô 1–3 và mục 9 để mô hình có sẵn trong bộ nhớ.
- [ ] Có 3 tab mở sẵn: slide, Colab (đang ở mục 9), video dự phòng.
- [ ] Tắt thông báo (Focus assist / Không làm phiền), tắt các ứng dụng chat.
- [ ] Phóng to trình duyệt lên 125–150% để người ngồi cuối phòng nhìn rõ.

**Lưu ý:** Colab ngắt kết nối nếu để yên khoảng 90 phút. Vì vậy hãy chạy lại các ô 1–3 **ngay trước** buổi thuyết trình, đừng chạy từ tối hôm trước.
