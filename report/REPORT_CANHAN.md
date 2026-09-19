# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Vũ Đình Đăng — 2A202602946
**Nhóm:** Akatsuki
**Ngày:** 2026-09-19

## 1. Khởi động

### Cosine similarity

Cosine similarity đo mức cùng hướng của hai embedding: điểm cao thường nghĩa là hai câu có ý nghĩa gần nhau, còn điểm thấp hoặc âm nghĩa là ít liên quan. Ví dụ cao: “Students use SIO to register for classes” và “A student registers for courses in SIO”. Ví dụ thấp: “The library lends books to students” và “Registration start times use student ID digits”.

Cosine phù hợp hơn Euclidean distance vì nó tập trung vào hướng/ý nghĩa tương đối của vector và ít bị ảnh hưởng bởi độ lớn embedding hoặc độ dài câu.

### Bài toán chunking

Với `chunk_size=500`, `overlap=50`: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23` chunks. Với `overlap=100`: `ceil(9900 / 400) = 25` chunks. Overlap lớn hơn tạo thêm chunks nhưng giữ ngữ cảnh xuyên ranh giới tốt hơn.

## 2. Hướng tiếp cận của tôi

- `SentenceChunker` dùng regex `(?<=[.!?])\s+`, giữ dấu kết câu và gom tối đa số câu được cấu hình; văn bản rỗng trả về danh sách rỗng.
- `RecursiveChunker` ưu tiên `\n\n`, `\n`, `. `, khoảng trắng và cuối cùng cắt theo ký tự. Khi một đoạn còn quá dài, hàm gọi đệ quy với separator tiếp theo.
- `EmbeddingStore` lưu record trong bộ nhớ gồm content, embedding, metadata và id. Search nhúng query, chấm dot product, rồi sắp xếp giảm dần.
- `search_with_filter` lọc metadata trước khi xếp hạng; `delete_document` xóa mọi record có cùng `metadata.doc_id`.
- `KnowledgeBaseAgent` lấy top-k chunks, gắn chúng vào prompt có Context và Question, sau đó gọi hàm LLM được truyền vào.

## 3. Hoàn thiện code

```text
python3 -m unittest tests/test_solution.py -q
Ran 42 tests in 0.004s
OK
```

**Số lượng bài test vượt qua:** **42 / 42**.

## 4. Dự đoán similarity

Các giá trị dưới đây được tính bằng `text-embedding-3-small` của OpenAI.

| Cặp | Dự đoán | Điểm thực tế | Nhận xét |
|---|---|---:|---|
| SIO registration / SIO registration | cao | 0.8900 | Đúng kỳ vọng |
| Late-drop voucher / voucher late drop | cao | 0.6956 | Đúng kỳ vọng |
| Library loans / start times | thấp | 0.2396 | Đúng kỳ vọng |
| Withdrawal W / W grade | cao | 0.6188 | Đúng kỳ vọng |
| Staff petition / staff vouchers | trung bình | 0.3059 | Cùng domain nhưng khác thông tin |

Model phân biệt rõ các cặp gần nghĩa và cặp không liên quan. Cặp về staff có điểm trung bình vì cùng bối cảnh non-degree staff nhưng hỏi hai chính sách khác nhau.

## 5. Kết quả truy xuất của tôi

Chiến lược thử nghiệm: chia theo heading Markdown, sau đó `RecursiveChunker` với `chunk_size=900`. Script tái lập: `python3 scripts/evaluate_benchmarks.py`.

| # | Query rút gọn | Top-1 doc (OpenAI) | Gold doc trong top-3? | Ghi chú |
|---|---|---|---|---|
| 1 | full-time units | course-registration | Có, top-1 | Đúng nguồn |
| 2 | course-time conflict | course-registration | Có, top-1 | Đúng nguồn |
| 3 | withdrawal transcript | course-changes | Có, top-1 | Đúng nguồn |
| 4 | undergraduate start times | registration-start-times | Có, top-1 | Dùng filter `audience=student` |
| 5 | staff petition and vouchers | staff-non-degree-registration | Có, top-1 | Đúng nguồn |

**Có chunk liên quan trong top-3:** **5 / 5** với `text-embedding-3-small`. Chưa đánh giá điểm agent-answer vì repo không có LLM thật; không nên ghi câu trả lời sinh bởi mock demo là kết quả factual.

## 6. Failure analysis và bước cải thiện

Mock embedder từng làm Q1 và Q3 thất bại; khi chuyển sang OpenAI, cả hai đều đúng ở top-1. Điều này cho thấy chất lượng embedding ảnh hưởng trực tiếp đến retrieval. Bước tiếp theo, nếu cần điểm agent-answer, là dùng một LLM với prompt grounding và đánh giá câu trả lời dựa trên đúng context.
