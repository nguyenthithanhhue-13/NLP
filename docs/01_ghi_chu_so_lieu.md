# Ghi chú số liệu – Phần Thực nghiệm & Nghiên cứu liên quan (STT 4)

Paper: *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., 2021), arXiv:2103.00020v1.
Số trang ghi theo file PDF arXiv v1 (48 trang). Mọi số dưới đây đã đối chiếu trực tiếp với PDF.

## 0. Hai khái niệm cần phân biệt

| Khái niệm | Cách làm | Có dùng nhãn của tập đích không? |
|---|---|---|
| **Zero-shot** | Đưa tên lớp vào bộ mã hóa văn bản (vd. "a photo of a dog.") → được vector cho mỗi lớp → ảnh thuộc lớp có cosine similarity cao nhất. Bản chất là bộ phân loại tuyến tính có trọng số do văn bản sinh ra (§3.1.2, tr. 6) | **Không**, 0 ảnh huấn luyện |
| **Linear probe** | Giữ nguyên bộ mã hóa ảnh, trích đặc trưng, rồi huấn luyện hồi quy logistic (scikit-learn L-BFGS, tối đa 1.000 vòng lặp, quét λ từ 10⁻⁶ đến 10⁶) trên tập train của bộ dữ liệu đích (Phụ lục A.3, tr. 38) | **Có**, dùng toàn bộ tập train |

Linear probe đo **chất lượng biểu diễn**. Zero-shot đo **khả năng học tác vụ** (task learning).

## 1. Zero-shot

| Số liệu | Ý nghĩa | Nguồn |
|---|---|---|
| **11,5% → 76,2%** | Zero-shot ImageNet: Visual N-Grams (2017) so với CLIP tốt nhất | Bảng 1, tr. 7 |
| 72,4 → 98,4 (aYahoo); 23,0 → 58,5 (SUN) | Hai bộ còn lại trong so sánh với Visual N-Grams | Bảng 1, tr. 7 |
| 95% top-5 | CLIP top-5 trên ImageNet, ngang Inception-V4 | §3.1.3, tr. 6 |
| 1,28 triệu | Số ảnh gán nhãn của ImageNet mà CLIP **không dùng** nhưng vẫn ngang ResNet-50 gốc | Abstract; §3.1.3 |
| 10× dữ liệu, ~100× compute/dự đoán, >1000× compute huấn luyện | Lý do không nên coi so sánh với Visual N-Grams là so sánh phương pháp trực tiếp | tr. 7 |
| **16/27** | Zero-shot CLIP thắng linear probe trên đặc trưng ResNet-50 ở 16/27 bộ, có ImageNet (+1,9) | Hình 5, tr. 8 |
| CIFAR-100: **+3,0**; CIFAR-10: +3,9 | Chênh lệch zero-shot CLIP so với linear probe ResNet-50 | Hình 5, tr. 8 |
| StanfordCars +28,9; Food101 +22,5; Kinetics700 +14,5; UCF101 +7,7 | Các bộ CLIP thắng đậm | Hình 5, tr. 8 |
| EuroSAT −37,1; KITTI −34,0; PCam −19,5; GTSRB −18,4; CLEVRCounts −18,2 | Các bộ CLIP thua đậm: ảnh vệ tinh, khoảng cách xe, ảnh y tế, biển báo, đếm vật | Hình 5, tr. 8 |
| 99,3% | STL10 zero-shot, SOTA mới mà không dùng ảnh huấn luyện | tr. 8 |
| **Zero-shot ≈ 4-shot** | Zero-shot CLIP ngang hồi quy logistic 4 ảnh/lớp trên cùng không gian đặc trưng; gần bằng 16-shot của BiT-M (mô hình tốt nhất) | Hình 6, tr. 9 |
| Trung vị 5,4 / trung bình 20,8 ảnh/lớp | Số ảnh gán nhãn/lớp mà linear probe cần để bằng zero-shot. ImageNet: zero-shot ≈ 16-shot | Hình 7, tr. 10 |
| r = 0,82 | Tương quan giữa zero-shot và linear probe. Zero-shot thường thấp hơn 10–25 điểm; chỉ 5 bộ chênh ≤ 3 điểm | Hình 8, tr. 10 |
| 44× compute | Lỗi zero-shot giảm theo hàm log-log mượt khi tăng compute (5 mô hình ResNet) | Hình 9, tr. 11 |

## 2. Prompt engineering và ensembling (§3.1.4, tr. 7–8)

| Số liệu | Ý nghĩa |
|---|---|
| **+1,3%** | Chỉ dùng prompt "A photo of a {label}." đã tăng 1,3% trên ImageNet so với dùng tên nhãn trơn |
| **+3,5%** | Ensemble 80 prompt khác nhau: tăng **thêm** 3,5% so với 1 prompt mặc định |
| **gần 5%** | Tổng cộng prompt + ensemble trên ImageNet |
| gần 5 điểm, ≈ 4× compute | Trung bình trên 36 bộ: cải thiện tương đương dùng mô hình lớn gấp 4 lần, nhưng "miễn phí" (Hình 4) |

> **Lưu ý khi trích dẫn:** ensemble 80 prompt chỉ tăng **3,5%**. Con số **gần 5%** là tổng của cả prompt (+1,3%) và ensemble.

- **Lý do prompt giúp ích:** (1) Đa nghĩa: ImageNet có "crane" là cần cẩu và "crane" là con sếu; Oxford Pets có "boxer" là giống chó nhưng cũng là võ sĩ. (2) Văn bản lúc huấn luyện hiếm khi chỉ có một từ.
- **Prompt theo tác vụ:** "a type of pet", "a satellite photo of a {}", đặt chữ cần nhận dạng trong ngoặc kép cho OCR.
- **Ensemble trong không gian embedding:** lấy trung bình vector văn bản rồi cache lại, nên chi phí bằng chi phí dùng 1 prompt.

## 3. Linear probe (§3.2, tr. 11–13; Bảng 10, tr. 40)

| Số liệu | Ý nghĩa | Nguồn |
|---|---|---|
| **21/27** | Linear probe trên CLIP tốt nhất thắng Noisy Student EfficientNet-L2 (mô hình ImageNet tốt nhất) ở 21/27 bộ | Hình 11, tr. 13 |
| SST2 +23,6; Country211 +22,7; HatefulMemes +18,8; StanfordCars +15,9; GTSRB +14,7 | Các bộ CLIP thắng nhiều nhất (OCR, định vị địa lý, xe, biển báo) | Hình 11, tr. 13 |
| ImageNet −3,0; CLEVRCounts −2,4; **CIFAR100 −1,7**; PCam −1,2; CIFAR10 −0,8; Pets −0,5 | 6 bộ CLIP thua. Với CIFAR (ảnh độ phân giải thấp), paper *cho rằng* một phần do CLIP không dùng augmentation theo tỉ lệ | Hình 11; tr. 12 |
| +2,6% → +5% | Mức hơn mô hình tốt nhất trước đó: trên 12 bộ Kornblith là +2,6%, trên 27 bộ là +5% | tr. 11–12 |
| ~3× | CLIP-ViT hiệu quả compute gấp khoảng 3 lần CLIP-ResNet | tr. 11 |
| 66 mô hình × 27 bộ = 1.782 đánh giá | Quy mô đánh giá linear probe | tr. 11 |

## 4. Số tham chiếu để so với nhóm (ViT-B/32, cùng mô hình nhóm chạy)

| Bộ dữ liệu | Zero-shot (Bảng 11, tr. 43) | Linear probe (Bảng 10, tr. 40) |
|---|---|---|
| **CIFAR-100** | **65,1** | **80,5** |
| CIFAR-10 | 91,3 | 95,1 |
| ImageNet | 63,2 | 76,1 |

Tham khảo thêm: linear probe của ResNet-50 (ImageNet) trên CIFAR-100 là 74,5. Linear probe của ViT-L/14-336px trên CIFAR-100 là 87,4, zero-shot là 77,5.

## 5. Độ bền vững khi phân phối thay đổi (§3.3, tr. 13–16)

Hình 13 (tr. 15) so sánh ResNet-101 với zero-shot CLIP ViT-L/14@336px. **Hai mô hình có cùng 76,2% trên ImageNet.**

| Bộ dữ liệu | ResNet-101 | Zero-shot CLIP | Δ |
|---|---|---|---|
| ImageNet | 76,2 | 76,2 | 0% |
| ImageNetV2 | 64,3 | 70,1 | +5,8% |
| ImageNet-R | 37,7 | 88,9 | +51,2% |
| **ObjectNet** | **32,6** | **72,3** | +39,7% |
| ImageNet Sketch | 25,2 | 60,2 | +35,0% |
| ImageNet-A | 2,7 | 77,1 | +74,4% |

- Zero-shot CLIP thu hẹp "khoảng cách bền vững" tới **75%**.
- Trên 7 bộ phân phối tự nhiên mới, ResNet-101 mắc lỗi nhiều gấp **5 lần** so với trên tập validation ImageNet.
- Nếu "thích nghi" CLIP với ImageNet (huấn luyện logistic trên tập train ImageNet), độ chính xác tăng **9,2%**, lên **85,4%**. Nhưng độ bền vững trung bình lại **giảm nhẹ**: ImageNet-R −4,7, ObjectNet −3,8, Sketch −2,8, ImageNet-A −1,9 (Hình 14, tr. 16).

## 6. Trùng lặp dữ liệu (§5, tr. 17–18)

- Trong 35 bộ được kiểm tra, có 9 bộ không phát hiện trùng lặp nào (MNIST, CLEVR, GTSRB, ObjectNet, Hateful Memes…).
- Tỉ lệ trùng lặp: **trung vị 2,2%**, **trung bình 3,2%**.
- Độ chính xác hiếm khi thay đổi quá 0,1% (chỉ 7 bộ vượt mức này, 2 bộ có ý nghĩa thống kê). Mức tăng lớn nhất chỉ **0,6%**, ở Birdsnap (trùng lặp 12,1%).
- Country211 trùng lặp nhiều nhất (21,5%) nhưng độ chính xác chỉ tăng 0,2%.
- Riêng CIFAR-100: ảnh độ phân giải rất thấp nên nhiều "bản trùng" là nhận nhầm (vật nhỏ như chim, máy bay).

## 7. Hạn chế có số liệu (§6, tr. 18–19)

- Zero-shot chỉ đạt **88% trên MNIST**, thua cả hồi quy logistic trên pixel thô. Lý do: dữ liệu huấn luyện gần như không có ảnh giống chữ số viết tay.
- Ước tính cần **~1000× compute** để zero-shot CLIP đạt SOTA tổng thể.
- Kém ở: phân loại chi tiết (mẫu xe, loài hoa, máy bay), đếm vật, các tác vụ mới như khoảng cách tới xe gần nhất (gần mức ngẫu nhiên).

## 8. Dữ liệu và huấn luyện (§2.2, §2.5; Phụ lục F)

| Mục | Giá trị | Nguồn |
|---|---|---|
| Dữ liệu WIT | 400 triệu cặp (ảnh, văn bản); 500.000 truy vấn; tối đa 20.000 cặp/truy vấn | §2.2, tr. 3–4 |
| YFCC100M sau lọc | 100 triệu ảnh → còn 15 triệu (≈ cỡ ImageNet) | §2.2 |
| Hàm mục tiêu | Contrastive hiệu quả hơn **4×** so với dự đoán bag-of-words, và BoW nhanh hơn Transformer LM **3×** | Hình 2, tr. 3 |
| Huấn luyện | 32 epoch, batch 32.768, Adam + weight decay 0,2, cosine LR, warm-up 2.000 vòng, mixed precision | §2.5; Bảng 18 |
| Nhiệt độ τ | Khởi tạo tương đương 0,07, logit nhân tối đa 100 | §2.5 |
| Chi phí | RN50x64: 18 ngày / 592 GPU V100. ViT-L/14: 12 ngày / 256 GPU V100 | §2.5, tr. 5 |
| ViT-B/32 | Embedding 512, ảnh 224, ViT 12 lớp × 768 × 12 head; Text 12 lớp × 512 × 8 head | Bảng 20, tr. 48 |
| CIFAR-100 | 100 lớp, 50.000 train, 10.000 test, độ đo accuracy | Bảng 9, tr. 39 |

## 9. Nghiên cứu liên quan (§1, tr. 1–2; §8, tr. 25–27)

| Hướng | Đại diện | Hạn chế mà CLIP giải quyết |
|---|---|---|
| Học từ văn bản thời kỳ đầu | Mori 1999; Quattoni 2007; Srivastava 2012 | Quy mô nhỏ, hiệu năng thấp |
| Dự đoán từ trong chú thích | Joulin 2016 (bag-of-words trên YFCC100M); **Visual N-Grams** (Li 2017, từ điển 142.806 n-gram) | Zero-shot ImageNet chỉ **11,5%**, thấp hơn cả 50% của thị giác cổ điển và rất xa SOTA 88,4% |
| Học từ văn bản bằng Transformer | **VirTex** (LM tự hồi quy), **ICMLM** (masked LM), **ConVIRT** (contrastive, ảnh y tế) | Chỉ huấn luyện vài ngày GPU trên 100–200 nghìn ảnh. CLIP là "ConVIRT đơn giản hóa", huấn luyện từ đầu |
| Giám sát yếu | Hashtag Instagram (Mahajan 2018, 3,5 tỷ ảnh, tăng hơn 5% trên ImageNet khi fine-tune); JFT-300M (BiT, ViT) | Giới hạn trong 1.000 và 18.291 lớp, softmax tĩnh nên không zero-shot linh hoạt được |
| Webly supervised | Fergus 2005; Divvala 2014 | Chỉ dùng truy vấn (1 từ) làm nhãn; CLIP dùng cả câu văn bản |
| Mô hình thị giác–ngôn ngữ kết hợp | Lu et al. 2019; Tan & Bansal 2019; Chen et al. 2019; Li et al. 2020; Yu et al. 2020 (VQA, suy luận) | Ghép nhiều hệ con đã huấn luyện sẵn. CLIP chỉ tương tác qua **một tích vô hướng** |
| Bộ dữ liệu ảnh–văn bản | MS-COCO, Visual Genome (~100.000 ảnh); Conceptual Captions… (1–10 triệu) | Quá nhỏ so với WIT 400 triệu cặp |
