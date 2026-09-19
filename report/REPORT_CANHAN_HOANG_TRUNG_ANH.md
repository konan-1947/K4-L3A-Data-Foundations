# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** [Hoàng Trung Anh]
**Nhóm:** [Tên nhóm]
**Ngày:** [19/9/2026]

> **Trạng thái:** Phần code đạt 42/42 bài kiểm thử. Đã chạy `python bench.py` với 5 câu hỏi chung, corpus 9 tài liệu và `SentenceChunker`; kết quả dùng `MockEmbedder` chỉ kiểm tra luồng truy xuất, chưa phản ánh chất lượng embedding ngữ nghĩa.

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector biểu diễn văn bản hướng gần giống nhau, thường cho thấy hai đoạn nói về nội dung gần nhau. Điểm gần 1 là rất gần về hướng; điểm gần 0 là ít liên quan, còn điểm âm là hướng đối nhau trong không gian vector.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên cần đăng ký học phần trước hạn chót."
- Câu B: "Học viên phải chọn môn học trước ngày kết thúc đăng ký."
- Tại sao tương đồng: Hai câu dùng từ khác nhau nhưng cùng nói về việc đăng ký môn trước thời hạn.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Thư viện cho sinh viên mượn sách trong hai tuần."
- Câu B: "Máy chủ cần được khởi động lại sau khi cập nhật phần mềm."
- Tại sao khác: Một câu nói về dịch vụ thư viện, câu kia nói về vận hành máy chủ.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine đo góc giữa hai vector nên tập trung vào hướng biểu diễn, ít bị ảnh hưởng bởi độ lớn vector. Khoảng cách Euclid còn phụ thuộc độ lớn; hai vector cùng hướng nhưng khác độ dài vẫn có thể bị xem là cách xa nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Bước trượt là `500 - 50 = 450`. Số chunk là `ceil((10.000 - 50) / 450) = ceil(22,11...) = 23`. Kiểm tra bằng `FixedSizeChunker(chunk_size=500, overlap=50)` trên chuỗi 10.000 ký tự cũng cho **23 chunk**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Bước trượt còn `500 - 100 = 400`, nên số chunk là `ceil((10.000 - 100) / 400) = 25`; chạy `FixedSizeChunker` cũng cho **25 chunk**. Overlap lớn hơn giúp giữ ngữ cảnh ở ranh giới hai chunk, nhưng làm tăng số chunk cần lưu và tìm kiếm.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

> Các mô tả dưới đây phản ánh mã nguồn hiện tại trong `src/`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi tách tại khoảng trắng sau `.`, `!`, `?` bằng biểu thức có lookbehind để giữ lại dấu câu, rồi gom tối đa `max_sentences_per_chunk` câu vào một chunk. Văn bản rỗng trả `[]`; các chữ viết tắt và số thập phân vẫn có thể bị tách sai.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Tôi thử các dấu phân cách theo thứ tự `\n\n`, `\n`, `. `, dấu cách, rồi từng ký tự. Đoạn vượt `chunk_size` được tách tiếp bằng mức kế tiếp; các mảnh ngắn liền kề được gom lại đến gần giới hạn. Trường hợp dừng là văn bản rỗng, đoạn đã đủ ngắn hoặc đã hết dấu phân cách.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Tôi dùng kho trong bộ nhớ, lưu mỗi `Document` thành một record gồm ID, nội dung, bản sao metadata và embedding. Khi tìm kiếm, tôi nhúng câu hỏi bằng cùng hàm embedding, tính tích vô hướng với các vector tài liệu đã chuẩn hóa, sắp xếp điểm giảm dần và lấy `top_k`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Tôi lọc ứng viên theo metadata **trước khi** xếp hạng tương đồng, để các kết quả không khớp không chiếm chỗ trong top-k. Khi xóa, tôi bỏ toàn bộ chunk có `metadata['doc_id']` khớp ID tài liệu gốc; hàm trả `True` nếu có bản ghi bị xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Tôi lấy `top_k` chunk liên quan từ store, đánh số và ghi nguồn từng chunk trong ngữ cảnh của prompt. Prompt yêu cầu chỉ trả lời theo ngữ cảnh và trích số đoạn đã dùng; sau đó tôi gọi `llm_fn`. Nếu store không trả về chunk nào, hàm báo không tìm thấy tài liệu mà không gọi LLM.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
Lệnh đã chạy: python -m pytest tests/ -v --tb=short -p no:cacheprovider
Python 3.13.9; pytest 9.1.1; collected 42 items
Tất cả 42 test trong tests/test_solution.py đều PASSED.
============================= 42 passed in 0.06s ==============================
```

**Số lượng bài test vượt qua (pass):** **42 / 42**.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên đăng ký học phần trực tuyến. | Học viên chọn môn học trên cổng đăng ký. | Cao | -0,1665 | Trái dự đoán |
| 2 | Thư viện mở cửa vào cuối tuần. | Giờ hoạt động của thư viện ngày thứ bảy là khi nào? | Cao | -0,1115 | Trái dự đoán |
| 3 | Sinh viên xin học bổng theo thành tích. | Quy trình cập nhật máy chủ gồm ba bước. | Thấp | 0,2139 | Trái dự đoán |
| 4 | Hạn nộp học phí là cuối tháng. | Ngày cuối cùng để đóng học phí là ngày nào? | Cao | 0,1424 | Gần dự đoán |
| 5 | Quy định mượn sách áp dụng cho giảng viên. | Ký túc xá có bao nhiêu phòng trống? | Thấp | -0,0176 | Gần dự đoán |

> Điểm thực tế được tính bằng `compute_similarity(_mock_embed(câu A), _mock_embed(câu B))`. `MockEmbedder` tạo vector giả lập, nên các nhãn "đúng/trái dự đoán" chỉ mô tả kết quả của mock, không đo chất lượng hiểu nghĩa.

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp 3 ít liên quan về nghĩa nhưng lại có điểm cao nhất (0,2139), còn cặp 1 gần nghĩa lại có điểm thấp nhất (-0,1665). Điều này cho thấy `MockEmbedder` phù hợp để kiểm thử luồng code, nhưng không thể dùng kết quả của nó để kết luận embedding hiểu ngữ nghĩa tốt hay kém.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

> Kết quả lấy từ `ket_qua_benchmark.txt` (`SentenceChunker`, tối đa 3 câu/chunk, `MockEmbedder`, 75 chunk). Bảng ghi lượt không lọc để so sánh chung; câu 5 còn có lượt `metadata_filter={"audience": "student"}` trong file kết quả. `bench.py` chỉ đo truy xuất, chưa gọi agent để sinh câu trả lời.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Quy trình đăng ký hai lớp trùng giờ là gì? | `registration-start-times#2` — ví dụ về khung giờ đăng ký | 0,309 | Top-1 không; top-3 có `course-registration#5` nêu bước gửi yêu cầu nhưng thiếu bước phê duyệt | Chưa chạy agent |
| 2 | Sinh viên đại học năm nhất đăng ký vào ngày nào trong kỳ thu/xuân? | `voucher-process-faq#0` — hướng dẫn voucher | 0,333 | Không chứa ngày đăng ký trong top-3 | Chưa chạy agent |
| 3 | Trước khi dùng voucher sau hạn drop/P/NP, sinh viên phải làm gì? | `course-changes#10` — late withdrawal và advisor | 0,290 | Không chứa quy trình voucher trong top-3 | Chưa chạy agent |
| 4 | Sinh viên đại học có bao nhiêu voucher trong toàn khóa và trong một kỳ? | `course-changes#6` — hạn drop | 0,233 | Không chứa số lượng voucher trong top-3 | Chưa chạy agent |
| 5 | Tôi có giờ bắt đầu đăng ký cụ thể không? | `course-registration#8` — cross-registration | 0,413 | Không chứa thông tin giờ được gán trong top-3 | Chưa chạy agent |

**Bao nhiêu câu hỏi trả về chunk chứa đáp án chuẩn trong top-3?** 0/5 chứa đủ đáp án; câu 1 có một chunk liên quan nhưng mới nêu bước gửi yêu cầu. Ở câu 5, lọc `audience=student` không làm thay đổi top-3 vì cả ba chunk vốn đã có `audience=student`. Đây là kết quả của `MockEmbedder`; cần chạy cùng bộ câu hỏi bằng embedding ngữ nghĩa để đánh giá chất lượng ở checkpoint tiếp theo.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Chưa có dữ liệu từ buổi demo; cần ghi lại một nhận xét thực tế sau khi nghe thành viên hoặc nhóm khác trình bày.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | Chờ tự đánh giá / 5 |
| Hướng tiếp cận của tôi (My Approach) | Chờ tự đánh giá / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 (42/42 test) |
| Dự đoán độ tương tự (Similarity Predictions) | Chờ tự đánh giá / 5 |
| Kết quả truy xuất của tôi (Competition Results) | Đã có kết quả mock; chờ đánh giá bằng embedding ngữ nghĩa / 10 |
| **Tổng phần cá nhân** | **Chưa tự đánh giá / 60** |
