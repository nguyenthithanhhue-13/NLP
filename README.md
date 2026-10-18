# Thực nghiệm CLIP của nhóm – Hướng dẫn chạy lại (Phụ lục)

Notebook `CLIP_thuc_nghiem.ipynb` kiểm chứng lại một phần kết quả của paper
*Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., 2021, arXiv:2103.00020).
Nhóm dùng mô hình đã huấn luyện sẵn do tác giả công bố và chỉ chạy suy luận, không huấn luyện lại CLIP.

- **Link Colab:** https://colab.research.google.com/drive/1_2B09eHFooiblv3mjbTbMBSJlCqfElFh


## 1. Mục tiêu thực nghiệm

Dùng mô hình CLIP ViT-B/32 của tác giả để trả lời bốn câu hỏi trên bộ CIFAR-100:

1. Mô hình có chạy được trên Colab không (kiểm tra môi trường)?
2. Phân loại zero-shot (không huấn luyện thêm) có đạt mức paper báo cáo không?
3. Prompt engineering và ensemble prompt có làm tăng độ chính xác không?
4. Đặc trưng ảnh của CLIP có tốt không, đo bằng linear probe?

## 2. Dữ liệu thực nghiệm

| Mục | Mô tả |
|---|---|
| **Bộ dữ liệu** | CIFAR-100 (Krizhevsky, Nair, Hinton), là tập con có gán nhãn của bộ *80 Million Tiny Images*. Đây là một trong 27 bộ đánh giá của paper (Bảng 9, tr. 39) |
| **Nguồn tải** | Trang chính thức của Đại học Toronto (https://www.cs.toronto.edu/~kriz/cifar.html), tải tự động qua `torchvision.datasets.CIFAR100` |
| **Kích thước** | 60.000 ảnh màu 32×32 pixel, 100 lớp, mỗi lớp 600 ảnh: 500 ảnh train và 100 ảnh test |
| **Chia tập** | Theo chia chuẩn của bộ dữ liệu: 50.000 ảnh train, 10.000 ảnh test. Nhóm không tự chia lại |
| **Nhãn** | Nhãn có sẵn trong bộ dữ liệu, nhóm không tự gán nhãn. Mỗi ảnh có nhãn *fine* (100 lớp, ví dụ `apple`, `snake`) và nhãn *coarse* (20 nhóm lớp, ví dụ "fruit and vegetables"). Nhóm dùng **nhãn fine** |
| **Từ nhãn thành văn bản** | Tên lớp đổi dấu `_` thành dấu cách (`aquarium_fish` → `aquarium fish`), rồi ghép vào câu mẫu (prompt), ví dụ `a photo of a aquarium fish.`. Danh sách tên lớp khớp với file `data/prompts.md` của repo |
| **Dùng cho** | Zero-shot: **chỉ dùng 10.000 ảnh test** và 100 tên lớp, không dùng ảnh train. Linear probe: dùng 50.000 ảnh train để huấn luyện, đo trên 10.000 ảnh test |
| **Dữ liệu demo** | Ảnh nhóm tự chọn hoặc tự chụp, với danh sách nhãn tự đặt bằng tiếng Anh |

## 3. Tiền xử lý

Theo đúng code của repo openai/CLIP (`clip/clip.py`):

**Ảnh** (hàm `_transform`):
1. **Resize** cạnh ngắn lên 224 pixel, nội suy bicubic (ảnh CIFAR 32×32 được phóng lên 224×224).
2. **CenterCrop**: cắt vùng giữa 224×224.
3. Chuyển sang RGB, **ToTensor**: đưa giá trị pixel về khoảng [0, 1], kích thước `[3, 224, 224]`.
4. **Normalize** theo trung bình (0,4815; 0,4578; 0,4082) và độ lệch chuẩn (0,2686; 0,2613; 0,2758) của CLIP.

**Văn bản** (hàm `clip.tokenize`):
1. Chuẩn hóa và đưa về chữ thường, tách thành token bằng BPE (paper §2.4: từ vựng 49.152).
2. Thêm token bắt đầu `[SOS]` và kết thúc `[EOS]`.
3. Đệm số 0 cho đủ độ dài cố định **77** vị trí → kích thước `[số câu, 77]`.

## 4. Luồng dữ liệu và số chiều tensor

Notebook có một ô in ra số chiều thật ở từng bước (mô hình ViT-B/32, 1 ảnh và K câu văn bản), đối chiếu với pseudocode Hình 3 của paper:

| Bước | Số chiều | Ý nghĩa |
|---|---|---|
| I: ảnh sau tiền xử lý | `[1, 3, 224, 224]` | Resize, cắt giữa, chuẩn hóa |
| Patch embedding | `[1, 768, 7, 7]` | Chia ảnh thành 7×7 = 49 mảnh 32×32, mỗi mảnh thành vector 768 |
| Chuỗi vào Transformer | `[1, 50, 768]` | 49 mảnh + 1 token `[CLS]` |
| I_f: đặc trưng ảnh | `[1, 768]` | Token `[CLS]` ở lớp cuối |
| W_i: ma trận chiếu ảnh | `[768, 512]` | Chiếu 768 → 512 |
| I_e: embedding ảnh | `[1, 512]` | I_f · W_i, chuẩn hóa L2 |
| T: token văn bản | `[K, 77]` | K câu, mỗi câu 77 token |
| T_f: đặc trưng văn bản | `[K, 512]` | Lấy tại token `[EOS]` ở lớp cuối |
| W_t: ma trận chiếu văn bản | `[512, 512]` | Chiếu 512 → 512 |
| T_e: embedding văn bản | `[K, 512]` | T_f · W_t, chuẩn hóa L2 |
| logits | `[1, K]` | Cosine similarity × 100 (logit scale) |
| Xác suất | `[1, K]` | Softmax |

Với CIFAR-100 thì K = 100. Khi chạy trên toàn bộ tập test, ảnh được xử lý theo lô và `logits` có kích thước `[10.000, 100]`.

## 5. Các setting thực nghiệm

Tất cả dùng cùng mô hình CLIP ViT-B/32, cùng 10.000 ảnh test.

| # | Setting | Văn bản cho mỗi lớp | Ảnh train dùng | Có huấn luyện không | Tham số |
|---|---|---|---|---|---|
| 1 | Zero-shot, không prompt | Tên lớp trơn, vd. `apple` | 0 | Không | – |
| 2 | Zero-shot, một prompt | `a photo of a {lớp}.` | 0 | Không | – |
| 3 | Zero-shot, ensemble | Trung bình embedding của 18 prompt chính thức cho CIFAR-100 (`data/prompts.md`) | 0 | Không | 18 prompt |
| 4 | Linear probe | Không dùng văn bản | 50.000 | Có: hồi quy logistic trên embedding ảnh | scikit-learn L-BFGS, C = 0,316, tối đa 1.000 vòng lặp |

**Vì sao không có đường cong huấn luyện và tập validation:**
- Setting 1–3 là zero-shot nên **không có quá trình huấn luyện**, vì vậy không có đường cong loss hay tập validation.
- Setting 4 huấn luyện hồi quy logistic bằng L-BFGS của scikit-learn, với tham số C cố định theo ví dụ của tác giả trong README của repo. Paper (Phụ lục A.3) thì tách tập validation để quét C. Phần này nhóm dự kiến bổ sung ở cuối kì, kèm đường cong huấn luyện.

## 6. Độ đo đánh giá

- **Top-1 accuracy** = (số ảnh có lớp dự đoán cao nhất trùng nhãn đúng) / (tổng số ảnh test).
- **Top-5 accuracy** = (số ảnh có nhãn đúng nằm trong 5 lớp được dự đoán cao nhất) / (tổng số ảnh test).

Paper báo cáo **top-1** cho CIFAR-100 (Bảng 9: độ đo "accuracy"), nên chỉ so sánh top-1 với paper. Top-5 ghi thêm để tham khảo.

## 7. Cách chạy

1. Mở link Colab ở trên, chọn **File → Save a copy in Drive**.
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Chọn **Runtime → Run all**. Ở ô Demo (mục 9), bấm *Choose files* để tải ảnh lên.
4. Kết quả: `ket_qua_nhom.csv` (bảng số liệu) và `bieu_do_ket_qua.png` (biểu đồ) được tải về máy.

**Chỉ chạy demo** (ví dụ khi thuyết trình): chạy ô 1, 2, 3, rồi chạy thẳng ô Demo. Không cần chạy ô 4–8 (tải CIFAR-100 và tính kết quả). Mất khoảng 2–3 phút.

**Thời gian chạy thực tế trên GPU T4 (Colab, 18/10/2026):**

| Bước | Lần 1 | Lần 2 |
|---|---|---|
| Cài đặt và tải mô hình | ~1–2 phút | ~1–2 phút |
| Tải CIFAR-100 (169 MB) | Phụ thuộc máy chủ Đại học Toronto, từ vài phút đến hơn 30 phút | |
| Trích đặc trưng 10.000 ảnh test | 24 giây | 23 giây |
| Trích đặc trưng 50.000 ảnh train | 108 giây | 106 giây |
| Huấn luyện hồi quy logistic | 154 giây | 151 giây |

## 8. Cấu hình máy

| Mục | Paper (ViT-L/14) | Nhóm |
|---|---|---|
| Phần cứng | 256 GPU V100, 12 ngày (huấn luyện) | 1 GPU Tesla T4 (14,6 GB VRAM), 12,7 GB RAM trên Google Colab (chỉ suy luận) |
| Thư viện | – | PyTorch 2.11.0 (CUDA 13.0), Python 3.13, repo openai/CLIP |

## 9. Kết quả (CIFAR-100, 10.000 ảnh test, CLIP ViT-B/32)

| Setting | Top-1 nhóm (%) | Top-5 nhóm (%) | Paper, ViT-B/32, top-1 (%) |
|---|---|---|---|
| 1. Zero-shot, không prompt | 60,44 | 85,62 | – |
| 2. Zero-shot, `a photo of a {}.` | 64,40 | 88,37 | – |
| 3. Zero-shot, ensemble 18 prompt | **65,13** | 89,24 | **65,1** (Bảng 11) |
| 4. Linear probe (50.000 ảnh) | **80,03** | 96,24 | **80,5** (Bảng 10) |

Notebook được chạy **2 lần**, cho kết quả trùng khớp đến hai chữ số thập phân.

**Nhận xét và so sánh:**
- **Prompt có tác dụng:** chỉ thêm `a photo of a` đã tăng **+3,96 điểm** (60,44 → 64,40). Ensemble 18 prompt tăng thêm **+0,73 điểm** (lên 65,13). Tổng cộng **+4,69 điểm**, cùng xu hướng paper mô tả ở §3.1.4 (trên ImageNet: +1,3% rồi +3,5%, tổng gần 5%). Lý do theo paper: khi huấn luyện, văn bản đi kèm ảnh thường là cả câu, hiếm khi chỉ một từ. Prompt giúp đầu vào giống dữ liệu huấn luyện hơn và giảm hiện tượng đa nghĩa.
- **Zero-shot ensemble khớp paper:** 65,13% so với 65,1% trong Bảng 11, chênh 0,03 điểm. Nhóm dùng đúng mô hình, đúng bộ dữ liệu và đúng bộ prompt của tác giả.
- **Linear probe sát paper:** 80,03% so với 80,5% trong Bảng 10, thấp hơn 0,47 điểm. Nguyên nhân hợp lý là hai điểm khác biệt về cách làm nêu ở dưới.
- **Linear probe cao hơn zero-shot 14,9 điểm:** linear probe được học từ 50.000 ảnh có nhãn, còn zero-shot không dùng ảnh nào. Paper cũng ghi nhận zero-shot thường thấp hơn linear probe 10–25 điểm (Hình 8).
- **Kiểm tra môi trường (STT 5):** ảnh CIFAR-100 số 3637 (nhãn đúng *snake*, ảnh ví dụ trong README của repo) được dự đoán đúng ở top-1 với 67,94%. Các dự đoán tiếp theo là turtle 16,28%, lizard 2,05%, crocodile 1,66%, snail 1,34%, đều là động vật gần giống rắn. Cùng ảnh này xuất hiện trong Hình 21 của paper (tr. 42), nơi CLIP cũng dự đoán đúng *snake* ở top-1 (38,02%).

**Khác biệt so với paper:**
- **Chọn C:** paper quét hệ số điều chuẩn λ trên tập validation trong khoảng 10⁻⁶ đến 10⁶. Nhóm dùng C = 0,316 cố định theo README của repo.
- **Đặc trưng dùng cho linear probe:** paper dùng đặc trưng trước phép chiếu (I_f, 768 chiều). Nhóm dùng embedding sau phép chiếu (I_e, 512 chiều), giống README của repo.
- **Bộ prompt:** *giống* paper. 18 prompt và tên lớp lấy đúng từ file `data/prompts.md` của repo openai/CLIP, là bộ tác giả công bố cho CIFAR-100.
- **Phần cứng và phiên bản thư viện:** khác paper, có thể làm số lệch nhỏ (thường dưới 1 điểm).

**Giới hạn:** nhóm chỉ kiểm chứng trên một bộ dữ liệu (CIFAR-100) với một mô hình (ViT-B/32) và không huấn luyện lại CLIP. Lý do: huấn luyện cần hàng trăm GPU trong nhiều ngày (§2.5), và bộ dữ liệu huấn luyện WIT 400 triệu cặp không được công bố. Các kết quả khác của paper (27 bộ dữ liệu, độ bền vững…) được trích từ paper.

## 10. Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| Ô cài đặt `openai/CLIP` báo lỗi | Notebook tự chuyển sang Hugging Face `transformers` (cùng trọng số OpenAI). Có thể đặt thẳng `BACKEND = "hf"` ở ô 2 |
| `Không có GPU` | Runtime → Change runtime type → T4 GPU |
| Hết GPU miễn phí (quota) | Đổi sang tài khoản Google khác, hoặc chạy CPU với `N_TRAIN_PROBE = 5000` (chậm hơn nhiều) |
| Linear probe chạy quá lâu hoặc hết RAM | Đặt `N_TRAIN_PROBE = 5000` (paper dùng toàn bộ 50.000 ảnh, cần ghi rõ khi báo cáo) |
| Tải CIFAR-100 rất chậm | Do máy chủ Đại học Toronto. Giữ tab mở và chờ, không dừng giữa chừng |
| Demo không tải được ảnh | Kéo thả ảnh vào thanh Files bên trái, rồi đặt `IMAGE_PATH = "/content/<tên ảnh>"` |

## 11. Nguồn

- **Paper:** A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," *ICML*, 2021, arXiv:2103.00020.
- **Mã nguồn và trọng số:** https://github.com/openai/CLIP
- **CIFAR-100:** A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," 2009. https://www.cs.toronto.edu/~kriz/cifar.html
