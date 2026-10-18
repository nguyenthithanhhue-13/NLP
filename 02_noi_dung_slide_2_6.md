# Nội dung slide 2 và 6 gửi Thư (STT 24)

Đạt là người trình bày. Huệ chạy demo ở slide 6. Mỗi slide tối đa 6 gạch đầu dòng.
Phần **Lời nói** là gợi ý cho Đạt, Thư không cần đưa lên slide.

---

## Slide 2 – Nghiên cứu liên quan (1,0 phút)

**Tiêu đề:** Các cách tiếp cận trước CLIP

**Nội dung trên slide:**
- **Học có giám sát (ImageNet):** 1,28 triệu ảnh gán nhãn, 1.000 lớp cố định. Muốn thêm lớp thì phải gán nhãn lại.
- **Giám sát yếu:** hashtag Instagram (3,5 tỷ ảnh), JFT-300M. Vẫn giới hạn 1.000 và 18.291 lớp, không có zero-shot.
- **Học từ văn bản, quy mô nhỏ:** VirTex, ICMLM, ConVIRT. Chỉ 100–200 nghìn ảnh, huấn luyện vài ngày GPU.
- **Zero-shot đầu tiên:** Visual N-Grams (2017). Chỉ đạt **11,5%** trên ImageNet.
- **→ Khoảng trống:** chưa có ai học từ ngôn ngữ tự nhiên ở quy mô lớn. CLIP dùng **400 triệu** cặp (ảnh, văn bản).

**Gợi ý thiết kế:** bảng hoặc dòng thời gian 4 cột "Hướng | Dữ liệu | Hạn chế", cột cuối tô đậm CLIP.
Có thể thêm một thanh nhỏ so sánh mức zero-shot ImageNet: 11,5% (Visual N-Grams) → 76,2% (CLIP).

| Hướng | Quy mô dữ liệu | Hạn chế |
|---|---|---|
| Có giám sát (ImageNet) | 1,28 triệu ảnh gán nhãn | 1.000 lớp cố định |
| Giám sát yếu (Instagram, JFT) | 3,5 tỷ / 300 triệu ảnh | Số lớp cố định, không zero-shot |
| VirTex / ICMLM / ConVIRT | 100–200 nghìn ảnh | Quy mô nhỏ |
| Visual N-Grams | YFCC100M | Zero-shot 11,5% |
| **CLIP** | **400 triệu cặp** | – |

**Lời nói (~1 phút):**
> "Trước CLIP, có ba hướng chính. Một là học có giám sát trên ImageNet, cần 1,28 triệu ảnh gán nhãn tay và chỉ nhận ra 1.000 lớp cố định. Hai là giám sát yếu từ hashtag, dữ liệu lớn hơn nhiều nhưng vẫn bị khóa trong một tập lớp cố định. Ba là học trực tiếp từ văn bản như VirTex hay ConVIRT, đúng hướng nhưng quy mô chỉ vài trăm nghìn ảnh. Visual N-Grams là công trình zero-shot đầu tiên, nhưng chỉ đạt 11,5% trên ImageNet. Khoảng trống ở đây là chưa ai học từ văn bản tự nhiên ở quy mô đủ lớn, và đó là điều CLIP làm."

**Nguồn:** §1 (tr. 1–2), §2.2, §8; Bảng 1.

---

## Slide 6 – Thực nghiệm (2,0 phút)

**Tiêu đề:** Kết quả thực nghiệm

**Nội dung trên slide (chia 2 cột):**

*Cột trái – Paper:*
- Zero-shot ImageNet **76,2%**, ngang ResNet-50 có giám sát mà không dùng ảnh huấn luyện nào.
- Thắng linear probe ResNet-50 ở **16/27** bộ dữ liệu.
- Bền vững: trên ObjectNet đạt **72,3%** so với 32,6% (cùng 76,2% trên ImageNet).
- Prompt và ensemble: **+~5%** "miễn phí".

*Cột phải – Nhóm tự kiểm chứng (CIFAR-100, ViT-B/32):*
- Biểu đồ `bieu_do_ket_qua.png` (Huệ gửi sau khi chạy xong).
- Demo trực tiếp: ảnh tự chụp với nhãn tùy chọn.

### Số liệu để vẽ biểu đồ

**Biểu đồ 1 – Độ bền vững (Hình 13, tr. 15).** Biểu đồ cột nhóm: ResNet-101 so với Zero-shot CLIP.
Đây là biểu đồ thuyết phục nhất. Nếu không đủ chỗ, chỉ giữ ImageNet, ObjectNet, ImageNet-R và Sketch.

| Bộ dữ liệu | ResNet-101 (%) | Zero-shot CLIP (%) |
|---|---|---|
| ImageNet | 76,2 | 76,2 |
| ImageNetV2 | 64,3 | 70,1 |
| ImageNet-R | 37,7 | 88,9 |
| ObjectNet | 32,6 | 72,3 |
| ImageNet Sketch | 25,2 | 60,2 |
| ImageNet-A | 2,7 | 77,1 |

Chú thích dưới biểu đồ: *Hai mô hình có cùng độ chính xác 76,2% trên ImageNet.*

**Biểu đồ 2 – Kết quả nhóm.** Notebook đã vẽ sẵn thành file `bieu_do_ket_qua.png`. Nếu Thư muốn vẽ lại cho cùng phong cách slide, dùng bảng này:

| Phương pháp | Nhóm (%) | Paper ViT-B/32 (%) |
|---|---|---|
| Zero-shot, không prompt | 60,4 | – |
| Zero-shot, "a photo of a {}." | 64,4 | – |
| Zero-shot, ensemble 18 prompt | 65,1 | 65,1 |
| Linear probe | 80,0 | 80,5 |

*(Số nhóm chạy trên Colab GPU T4 ngày 18/10/2026. Số chi tiết hai chữ số thập phân có trong `ket_qua_nhom.csv`.)*

Gợi ý chú thích dưới biểu đồ: *"Prompt +4,0 điểm, ensemble +0,7 điểm. Zero-shot ensemble khớp paper (65,1%), linear probe sát paper (80,0% so với 80,5%)."*

**Biểu đồ dự phòng – Prompt trên ImageNet (§3.1.4).** Dùng nếu Colab lỗi và nhóm không có số tự chạy:
tên lớp trơn → +1,3% (1 prompt) → thêm +3,5% (ensemble 80 prompt) = gần +5%.

### Lời nói (~2 phút, gồm demo)
> "Kết quả nổi bật nhất: CLIP đạt 76,2% zero-shot trên ImageNet, ngang ResNet-50 được huấn luyện với 1,28 triệu ảnh, trong khi CLIP không dùng ảnh nào của ImageNet. Trên 27 bộ dữ liệu, zero-shot CLIP thắng ResNet-50 có giám sát ở 16 bộ.
> Điểm đáng chú ý hơn là độ bền vững. Biểu đồ này so sánh hai mô hình cùng đạt 76,2% trên ImageNet. Khi đổi sang ảnh vật thể thật như ObjectNet, ResNet-101 rơi xuống 32,6%, còn CLIP giữ được 72,3%. Với ảnh vẽ, ảnh hoạt hình ở ImageNet-R, CLIP đạt 88,9% so với 37,7%.
> Nhóm cũng tự chạy lại trên CIFAR-100 với ViT-B/32: chỉ thêm prompt 'a photo of a' đã tăng từ 60,4% lên 64,4%, ensemble 18 prompt đạt 65,1%, khớp đúng con số của paper. Linear probe đạt 80,0%, sát với 80,5% của paper.
> Bây giờ Huệ sẽ demo với ảnh nhóm tự chụp."

**Demo (~30 giây, Huệ thao tác):** xem file `03_hoi_dap_va_demo.md`.

**Nguồn:** Bảng 1, Hình 5, Hình 13, §3.1.4, Bảng 10–11.
