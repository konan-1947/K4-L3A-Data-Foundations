# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Chí Công — 2A202602634
**Nhóm:** 2A
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

Các giá trị dưới đây được tính bằng `_mock_embed`; chúng minh họa giới hạn của mock backend, không dùng để kết luận chất lượng ngữ nghĩa.

| Cặp | Dự đoán | Điểm mock | Nhận xét |
|---|---|---:|---|
| SIO registration / SIO registration | cao | -0.1925 | Không đúng kỳ vọng |
| Late-drop voucher / voucher late drop | cao | 0.0344 | Gần 0, không phân biệt được ý nghĩa |
| Library loans / start times | thấp | -0.0450 | Đúng hướng nhưng yếu |
| Withdrawal W / W grade | cao | 0.1288 | Cao hơn một ít nhưng vẫn yếu |
| Staff petition / staff vouchers | trung bình | -0.0428 | Cùng domain nhưng mock không nhận ra |

Điều bất ngờ là các câu gần nghĩa cũng không có điểm cao. Nguyên nhân là mock embedding sinh vector xác định từ toàn bộ chuỗi, gần như ngẫu nhiên theo nghĩa; benchmark chính thức cần local multilingual embedder hoặc API embedding.

## 5. Kết quả truy xuất của tôi

Chiến lược thử nghiệm: chia theo heading Markdown, sau đó `RecursiveChunker` với `chunk_size=900`. Script tái lập: `python3 scripts/evaluate_benchmarks.py`.

| # | Query rút gọn | Top-1 doc (mock) | Gold doc trong top-3? | Ghi chú |
|---|---|---|---|---|
| 1 | full-time units | staff-non-degree-registration | Không | Failure case |
| 2 | course-time conflict | course-registration | Có, top-1 | Đúng nguồn |
| 3 | withdrawal transcript | registration-four-steps | Không | Failure case |
| 4 | undergraduate start times | course-changes | Có, top-2 | Dùng filter `audience=student` |
| 5 | staff petition and vouchers | staff-non-degree-registration | Có, top-1 | Đúng nguồn |

**Có chunk liên quan trong top-3:** **3 / 5** với mock backend. Chưa đánh giá điểm agent-answer vì repo không có LLM thật; không nên ghi câu trả lời sinh bởi mock demo là kết quả factual.

## 6. Failure analysis và bước cải thiện

Q1 và Q3 thất bại vì mock embedder không mã hóa ngữ nghĩa. Các chunk đúng tồn tại nhưng không được xếp hạng cao. Bước tiếp theo là chạy lại cùng 5 query bằng `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, so sánh top-3 với baseline và chỉ chấm answer khi LLM được grounding bằng đúng context.
