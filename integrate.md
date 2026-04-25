# Tích hợp đặc trưng: DINO + EVA → PhoBERT

## Tổng quan luồng dữ liệu

```
Ảnh (B, 3, 224, 224)
       ├─── DinoBackbone ──→ (B, 768)
       └─── EvaBackbone  ──→ (B, 768)
                  │
            [GatedFusion]
                  │
       Fused image features (B, 1, 768)
                  │
Câu hỏi (list[str])
       └─── QuesEmbedding (PhoBERT + LSTM) ──→ (B, 768) → (B, 1, 768)
                  │
         [4× Attention layers]
                  │
       Image-Question context (B, 768)
                  │
Câu trả lời (list[str])
       └─── AnsEmbedding (PhoBERT embeddings) ──→ (B, max_len, 768)
                  │
         [Transformer Decoder (5 layers)]
                  │
         [MLP Head: Linear → vocab_size]
                  │
       Output logits (B, max_len, vocab_size)
```

---

## Phần 1: Kết hợp đặc trưng DINO và EVA

### 1.1 Trích xuất đặc trưng DINO

**File:** [src/models/image_backbones.py](src/models/image_backbones.py) — `DinoBackbone` (dòng 12–23)

DINO sử dụng kiến trúc Vision Transformer (ViT-Base/16) được huấn luyện theo phương pháp self-supervised DINO (Self-DIstillation with NO labels).

```python
self.model = timm.create_model("vit_base_patch16_224.dino", pretrained=False)
self.model.head = nn.Linear(self.model.embed_dim, embedding_dim)
```

**Quá trình forward:**
1. Ảnh đầu vào `(B, 3, 224, 224)` được chia thành các patch 16×16 → 196 patch tokens + 1 CLS token.
2. ViT xử lý chuỗi 197 token qua các lớp Transformer encoder.
3. Token `[CLS]` được lấy làm đại diện toàn cục, đi qua lớp `head` (Linear) để chiếu về chiều `embedding_dim=768`.
4. **Output:** vector `(B, 768)`.

---

### 1.2 Trích xuất đặc trưng EVA

**File:** [src/models/image_backbones.py](src/models/image_backbones.py) — `EvaBackbone` (dòng 89–107)

EVA-02 sử dụng ViT-Base/14 được tiền huấn luyện theo phương pháp Masked Image Modeling (MIM) trên ImageNet-22K.

```python
self.model = timm.create_model("eva02_base_patch14_224.mim_in22k", pretrained=False, num_classes=0)
self.out_dim = self.model.num_features
self.proj = nn.Linear(self.out_dim, embedding_dim)
```

**Quá trình forward:**
1. Ảnh đầu vào `(B, 3, 224, 224)` được chia thành các patch 14×14 → 256 patch tokens.
2. `forward_features()` chạy toàn bộ Transformer encoder, trả về feature map.
3. `forward_head(..., pre_logits=True)` thực hiện pooling (lấy CLS token hoặc global average) → `(B, out_dim)`.
4. Lớp `proj` (Linear) chiếu từ `out_dim` của EVA về `embedding_dim=768`.
5. **Output:** vector `(B, 768)`.

> **Lưu ý:** Hai backbone có patch size khác nhau (16 vs 14) nhưng cùng độ phân giải ảnh 224×224. Projection layer đảm bảo hai đặc trưng có cùng chiều trước khi fusion.

---

### 1.3 Fusion DINO và EVA

**File:** [src/models/image_fusion.py](src/models/image_fusion.py) — `GatedFusion` + `ImageEmbedding`

Mặc định cấu hình sử dụng **Gated Fusion** (`fusion="gated"`). Mô hình hỗ trợ 4 phương pháp fusion, được lựa chọn qua tham số `--fusion` khi huấn luyện.

#### Phương pháp mặc định: Gated Fusion

**File:** [src/models/image_fusion.py](src/models/image_fusion.py) — `GatedFusion` (dòng 6–20)

```python
class GatedFusion(nn.Module):
    def __init__(self, embedding_dim=768):
        self.gate = nn.Linear(embedding_dim * 2, 2)   # học trọng số động
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, emb1, emb2):           # emb1=DINO, emb2=EVA, cả 2 là (B, 768)
        combined = torch.cat([emb1, emb2], dim=-1)          # (B, 1536)
        gate = torch.softmax(self.gate(combined), dim=-1)   # (B, 2)
        fused = gate[..., 0:1] * emb1 + gate[..., 1:2] * emb2   # (B, 768)
        return self.proj(fused)                              # (B, 768)
```

**Cơ chế hoạt động:**
1. Concatenate hai vector DINO và EVA thành `(B, 1536)`.
2. Linear layer `(1536 → 2)` + Softmax học ra **hai trọng số gate** `[g_dino, g_eva]` phụ thuộc vào nội dung ảnh — trọng số thay đổi theo từng mẫu.
3. Tính tổng có trọng số: `fused = g_dino × v_dino + g_eva × v_eva`.
4. Linear projection `(768 → 768)` để trộn thêm thông tin.
5. **Output:** `(B, 768)`.

#### Phương pháp bổ sung: Cross-Attention Fusion

**File:** [src/models/image_fusion.py](src/models/image_fusion.py) — `CrossAttentionFusion` (dòng 22–35)

**Kích hoạt:** truyền `--fusion cross-attention` khi huấn luyện.

```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, emb1, emb2):
        e1 = emb1.unsqueeze(1) if emb1.dim() == 2 else emb1   # (B, 1, 768) — DINO
        e2 = emb2.unsqueeze(1) if emb2.dim() == 2 else emb2   # (B, 1, 768) — EVA

        out1, _ = self.attn(e1, e2, e2)   # Query=DINO, Key=Value=EVA
        out2, _ = self.attn(e2, e1, e1)   # Query=EVA,  Key=Value=DINO
        return (out1 + out2) / 2           # (B, 1, 768)
```

**Cơ chế hoạt động chi tiết:**

Mô-đun dùng một `nn.MultiheadAttention` duy nhất (8 đầu, `batch_first=True`) cho cả hai chiều attention.

**Bước 1 — Reshape:**
Cả hai vector `(B, 768)` được unsqueeze về `(B, 1, 768)` để trở thành sequence có độ dài 1, phù hợp với API của `MultiheadAttention`.

**Bước 2 — Attention DINO → EVA (`out1`):**
```
Q = e1 (DINO),  K = e2 (EVA),  V = e2 (EVA)
Attention score = softmax(Q·Kᵀ / √(768/8))   # shape (B, 1, 1)
out1 = score × V                               # (B, 1, 768)
```
DINO đóng vai trò Query: "tôi cần thông tin gì từ EVA?". Output `out1` là vector DINO đã được làm giàu bởi đặc trưng của EVA.

**Bước 3 — Attention EVA → DINO (`out2`):**
```
Q = e2 (EVA),  K = e1 (DINO),  V = e1 (DINO)
out2 = softmax(Q·Kᵀ / √d_k) × V              # (B, 1, 768)
```
Chiều ngược lại: EVA truy vấn vào DINO, thu được vector EVA đã bổ sung thông tin từ DINO.

**Bước 4 — Kết hợp:**
```python
fused = (out1 + out2) / 2   # (B, 1, 768)
```
Trung bình đối xứng của hai luồng — không ưu tiên backbone nào, đảm bảo đóng góp cân bằng.

**So sánh với Gated Fusion:**

| Tiêu chí | Gated Fusion | Cross-Attention Fusion |
|---|---|---|
| Cơ chế | Trọng số vô hướng học từ `concat([e1,e2])` | Attention score học cách "hỏi đáp" giữa 2 đặc trưng |
| Tham số | `Linear(1536→2)` + `Linear(768→768)` | `MultiheadAttention(768, 8heads)` |
| Tương tác | Cộng có trọng số — mỗi chiều `d` xử lý độc lập | Attention tính trên toàn bộ vector → mỗi chiều ảnh hưởng đến mọi chiều khác |
| Inductive bias | "Backbone nào quan trọng hơn?" | "Thông tin nào trong backbone này có liên quan đến backbone kia?" |
| Output shape | `(B, 768)` → unsqueeze → `(B, 1, 768)` | `(B, 1, 768)` trực tiếp |

**Lý do Cross-Attention phù hợp cho bài toán này:**
- DINO và EVA học đặc trưng từ góc độ khác nhau (ngữ nghĩa vs. chi tiết cấu trúc). Cross-Attention cho phép mỗi backbone "hỏi" backbone kia những thông tin bổ sung mà mình thiếu.
- Với sequence length = 1, attention score luôn bằng 1.0 (softmax của một phần tử duy nhất), nên `out1 = V = e2` và `out2 = V = e1` trong trường hợp lý tưởng — thực tế 8-head projection tạo ra mixing trong không gian con trước khi tính score, nên vẫn có transformation có ý nghĩa.
- Không có projection `proj` hậu xử lý như GatedFusion; nếu muốn thêm non-linearity, có thể bổ sung `nn.Linear(768, 768)` sau bước trung bình.

#### Tổng hợp ImageEmbedding

**File:** [src/models/image_fusion.py](src/models/image_fusion.py) — `ImageEmbedding.forward()` (dòng 66–88)

```python
def forward(self, image):
    embeddings = [encoder(image) for encoder in self.encoders]
    # embeddings = [(B, 768), (B, 768)] cho [DINO, EVA]

    if self.fusion_type == "gated":
        fused = self.fusion(embeddings[0], embeddings[1]).unsqueeze(1)
    # ...

    return fused, image_ids   # fused: (B, 1, 768)
```

**Output cuối cùng của stage 1:** tensor `(B, 1, 768)` — một vector duy nhất đại diện cho ảnh sau khi đã tích hợp cả DINO và EVA.

---

## Phần 2: Kết hợp Fusion ảnh với đặc trưng PhoBERT (câu hỏi)

### 2.1 Trích xuất đặc trưng câu hỏi từ PhoBERT

**File:** [src/models/text_encoders.py](src/models/text_encoders.py) — `QuesEmbedding` (dòng 8–40)

```python
self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
self.lstm = nn.LSTM(input_size=768, output_size=768, batch_first=True)
```

**Quá trình forward:**
1. Tokenize câu hỏi với PhoBERT tokenizer (padding đến `max_len=27`, không có special tokens).
2. PhoBERT encoder trả về `last_hidden_state` có shape `(B, max_len, 768)` — chuỗi ngữ cảnh của từng token.
3. LSTM nhận chuỗi trên, lấy hidden state của bước cuối `h` để nén thành một vector.
4. **Output:** `(B, 768)` — đặc trưng ngữ nghĩa toàn câu hỏi.

> LSTM đóng vai trò pooling có trọng số, nắm bắt thứ tự và phụ thuộc tuần tự tốt hơn mean/CLS pooling đơn thuần.

---

### 2.2 Cơ chế Attention tích hợp ảnh và câu hỏi

**File:** [src/models/attention.py](src/models/attention.py) — `Attention` (dòng 7–26)

**File:** [src/models/vqa.py](src/models/vqa.py) — `VQAModel.forward()` (dòng 143–150)

Đây là giai đoạn cốt lõi: **4 lớp Attention nối tiếp** cho phép đặc trưng ảnh và câu hỏi ảnh hưởng lẫn nhau.

```python
# vqa.py — forward pass
ques_embedds = ques_embeddings.unsqueeze(1)   # (B, 768) → (B, 1, 768)

att_embedds = None
for att_layer in self.an_model:               # 4 lần lặp
    att_embedds = att_layer(image_embeddings, ques_embedds)
```

**Bên trong mỗi lớp Attention:**

```python
# attention.py
def forward(self, vi, vq):                        # vi=(B,1,768), vq=(B,1,768)
    combined_input = torch.cat([vq, vi], dim=1)   # (B, 2, 768) — nối theo chiều sequence

    attn_output, _ = self.attention(              # Self-attention trên sequence gồm 2 token
        combined_input, combined_input, combined_input
    )
    attn_output = self.dropout(attn_output)
    attn_output = self.layer_norm(attn_output)    # (B, 2, 768)

    vi_attended = attn_output[:, 1:, :]           # (B, 1, 768) — phần ứng với ảnh đã attended
    u = vi_attended.sum(dim=1) + vq.squeeze(1)   # (B, 768) — cộng dồn với câu hỏi
    return u
```

**Cơ chế chi tiết của 1 lớp Attention:**
1. Concatenate `[v_question, v_image]` thành sequence 2 token `(B, 2, 768)`.
2. **Self-attention 8 đầu**: mỗi token attend vào cả 2 token → ảnh học được đặc trưng câu hỏi, câu hỏi học được đặc trưng ảnh.
3. Dropout + LayerNorm ổn định huấn luyện.
4. Lấy phần output ứng với token ảnh (`[:, 1:, :]`) → vector ảnh đã được "nhào nặn" bởi câu hỏi.
5. Cộng trực tiếp với vector câu hỏi gốc (residual connection) → `(B, 768)`.

**Sau 4 lớp:**
- Mỗi lớp nhận output của lớp trước làm input.
- Vector `att_embedds` sau 4 lần là biểu diễn **đa phương thức** (ảnh+câu hỏi), trong đó hai nguồn thông tin đã được tích hợp sâu qua nhiều bước attention.

---

### 2.3 Chuẩn bị context và kết hợp với đặc trưng câu trả lời

**File:** [src/models/vqa.py](src/models/vqa.py) (dòng 150–175)

```python
att_embedds = self.tanh(att_embedds)        # (B, 768) — chuẩn hóa phi tuyến
att_embedds = self.dropout(att_embedds)     # regularization

# Mở rộng context thành sequence để Decoder xử lý song song
x = att_embedds.unsqueeze(1).expand(-1, max_len, -1)   # (B, max_len, 768)
```

**PhoBERT embedding câu trả lời:**

**File:** [src/models/text_encoders.py](src/models/text_encoders.py) — `AnsEmbedding` (dòng 43–72)

```python
# Chỉ dùng tầng embedding (không encoder) — cần chuỗi token level
self.phobert_embed = AutoModel.from_pretrained("vinai/phobert-base").embeddings
```

- Tokenize câu trả lời, tra bảng embedding của PhoBERT (không qua Transformer encoder).
- **Output:** `(input_ids, embeddings)` trong đó `embeddings` có shape `(B, max_len, 768)`.

---

### 2.4 Transformer Decoder sinh câu trả lời

**File:** [src/models/decoder.py](src/models/decoder.py) — `DecoderLayer` (dòng 98–126)

Decoder gồm 5 lớp `DecoderLayer`, mỗi lớp có 3 sublayer:

```
Input: x = context ảnh+câu hỏi (B, max_len, 768)
       y = PhoBERT embeddings câu trả lời (B, max_len, 768)
       decoder_mask = causal mask (max_len, max_len)
```

**Sublayer 1 — Masked Self-Attention trên câu trả lời:**
```python
y = self.self_attention(y, mask=decoder_mask)   # (B, max_len, 768)
y = self.norm1(y + residual)
```
- Mỗi token câu trả lời chỉ attend vào các token đứng trước nó (causal mask → tam giác trên = -inf).
- Cho phép model học phụ thuộc nội bộ trong chuỗi câu trả lời.

**Sublayer 2 — Cross-Attention từ câu trả lời → context ảnh+câu hỏi:**
```python
# x = context (encoder/image+ques), y = answer sequence (decoder)
kv = self.kv_layer(x)   # Key, Value từ context ảnh+câu hỏi
q  = self.q_layer(y)    # Query từ chuỗi câu trả lời
y = self.encoder_decoder_attention(x, y)
y = self.norm2(y + residual)
```
- Câu trả lời (Query) truy vấn thông tin từ context ảnh+câu hỏi (Key, Value).
- Đây là bước **tích hợp cuối cùng** giữa thông tin thị giác/ngôn ngữ và quá trình sinh văn bản.

**Sublayer 3 — Feed-Forward Network:**
```python
y = self.ffn(y)         # Linear(768→2048) → ReLU → Linear(2048→768)
y = self.norm3(y + residual)
```

**Output Decoder:** `(B, max_len, 768)`

---

### 2.5 Phân loại token và sinh câu trả lời

**File:** [src/models/vqa.py](src/models/vqa.py) (dòng 175–177)

```python
out = self.decoder(x, ans_embedds, decoder_mask)   # (B, max_len, 768)
output_logits = self.mlp(out)                       # (B, max_len, vocab_size)
```

MLP head gồm `Dropout(0.3) → Linear(768, vocab_size)` chiếu mỗi position về không gian từ vựng.

---

## Tóm tắt kiến trúc và kích thước tensor

| Bước | Mô-đun | Input shape | Output shape |
|------|--------|-------------|--------------|
| 1a | `DinoBackbone` | `(B, 3, 224, 224)` | `(B, 768)` |
| 1b | `EvaBackbone` | `(B, 3, 224, 224)` | `(B, 768)` |
| 2 | `GatedFusion` | `(B, 768) × 2` | `(B, 768)` |
| 3 | `ImageEmbedding` (unsqueeze) | `(B, 768)` | `(B, 1, 768)` |
| 4 | `QuesEmbedding` (PhoBERT+LSTM) | `list[str]` → `(B, max_len, 768)` | `(B, 768)` |
| 5 | `Attention ×4` | `(B,1,768) + (B,1,768)` | `(B, 768)` |
| 6 | Tanh + Dropout + expand | `(B, 768)` | `(B, max_len, 768)` |
| 7 | `AnsEmbedding` (PhoBERT embed) | `list[str]` | `(B, max_len, 768)` |
| 8 | `Decoder ×5` (cross-attention) | `(B, max_len, 768) × 2` | `(B, max_len, 768)` |
| 9 | `MLP` (Linear) | `(B, max_len, 768)` | `(B, max_len, vocab_size)` |

---

## Thiết kế đáng chú ý

**Tại sao DINO + EVA?**
- DINO học đặc trưng ngữ nghĩa cấp cao (phân cụm đối tượng tốt không cần nhãn).
- EVA học đặc trưng chi tiết từ Masked Image Modeling (tái tạo pixel → hiểu cấu trúc tốt hơn).
- Kết hợp bổ sung nhau: DINO mạnh về "cái gì", EVA mạnh về "như thế nào".

**Tại sao Gated Fusion?**
- Trọng số gate học được từ dữ liệu, linh hoạt hơn trung bình cố định.
- Với câu hỏi về màu sắc/chi tiết, EVA có thể được ưu tiên; với câu hỏi về danh mục, DINO có thể được ưu tiên — gate tự điều chỉnh theo nội dung.

**Tại sao PhoBERT + LSTM cho câu hỏi (không phải CLS token)?**
- LSTM nắm bắt phụ thuộc tuần tự theo chiều thời gian, phù hợp với câu hỏi tiếng Việt có cấu trúc ngữ pháp phức tạp.
- Hidden state cuối LSTM mang thông tin từ toàn bộ chuỗi, được nhấn mạnh ở cuối.

**Tại sao AnsEmbedding chỉ dùng embedding layer (không encoder)?**
- Trong quá trình sinh câu trả lời, model cần embed các token đã sinh trước để đưa vào Decoder.
- Dùng full encoder sẽ cực kỳ tốn kém (mỗi bước sinh phải chạy lại toàn bộ PhoBERT).
- Embedding layer đủ để cung cấp biểu diễn token-level làm input Decoder.
