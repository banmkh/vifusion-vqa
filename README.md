# VQA with DINO & EVA

Dự án triển khai pipeline VQA (Visual Question Answering) dựa trên notebook `notebooks/vqa-with-dino-and-eva.ipynb`, đã được tách các module xử lý dữ liệu để tái sử dụng.

## Cấu trúc thư mục
```
.
├─ notebooks/
│  └─ vqa-with-dino-and-eva.ipynb
├─ src/
│  ├─ data/
│  ├─ models/
│  ├─ training/
│  └─ utils/
├─ scripts/
│  └─ preprocess.py
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ outputs/
│  ├─ checkpoints/
│  ├─ logs/
│  └─ figures/
├─ configs/
├─ tests/
├─ requirements.txt
└─ README.md
```

## Chuẩn bị dữ liệu
Copy dữ liệu vào `data/raw/` theo cấu trúc:
- `data/raw/training-images/`
- `data/raw/test-images/`
- `data/raw/dev-images/`
- `data/raw/training-annotations.json`
- `data/raw/test-annotations.json`
- `data/raw/dev-annotations.json`

## Cài đặt
```
pip install -r requirements.txt
```

## Preprocess
```
python -m scripts.preprocess --root .
```

Tuỳ chọn:
- `--limit N` để giới hạn số mẫu
- `--skip-normalize` để bỏ qua normalize text

## Test
```
python -m pytest -q
```

## Train
```
python -m scripts.train --root .
```

Tuỳ chọn:
- `--epochs N`
- `--lr LR`
- `--batch-size B`
- `--device cpu|cuda`
- `--image-encoders dino,eva,beit`
- `--fusion gated|attention|linear`

Ví dụ:
```
python -m scripts.train --root . --image-encoders dino,beit --fusion attention
```

## Ghi chú
- Các module xử lý dữ liệu được đặt trong `src/data/`.
- Notebook đã được refactor để gọi module trong `src/data/` thay vì code inline.
- Các hàm metrics và visualization được tách ra `src/utils/metrics.py` và `src/utils/visualize.py`.
- Training loop tối thiểu được đặt trong `src/training/trainer.py` và được dùng lại trong notebook + `scripts/train.py`.
- Cấu hình thống nhất nằm ở `src/config.py` (gom `DataConfig`, `ModelConfig`, `TrainConfig`).
- Helper nhỏ `count_parameters`, `build_vocab_swap` và plot loss/time ở `src/utils/helpers.py`, `src/utils/plot.py`.

## Model Architecture
Kiến trúc mô hình được tách thành các module trong `src/models/` và bám sát notebook.

Sơ đồ luồng chính:
```
Image -> Image Encoders (DINO/EVA/BEiT/...) -> Fusion -> Image Embedding
Question -> PhoBERT -> LSTM -> Question Embedding
Image Embedding + Question Embedding -> Multi-Head Attention (xN) -> Joint Embedding
Answer (teacher forcing) -> PhoBERT Embeddings -> Decoder (Masked Self-Attn + Cross-Attn)
Decoder Output -> MLP -> Vocab Logits
```

Thành phần chính:
- `ImageEmbedding` nhận nhiều encoder và hợp nhất bằng `gated`, `attention`, hoặc `linear` fusion.
- `QuesEmbedding` dùng `AutoTokenizer` + `AutoModel` (PhoBERT) và LSTM để ra vector câu hỏi.
- `Attention` là multi-head self-attention trên chuỗi ghép `[vq, vi]` để tạo vector hợp nhất.
- `Decoder` là Transformer decoder với masked self-attention và cross-attention vào vector hợp nhất.
- `VQAModel` ghép tất cả module lại và xuất logits theo vocab.

Khởi tạo nhanh (ví dụ):
```python
from src.models import VQAModel, ModelConfig

cfg = ModelConfig()
model = VQAModel(
    vocab_size=None,
    text_model=cfg.text_model,
    image_encoders=list(cfg.image_encoders),
    fusion=cfg.fusion,
    d_model=cfg.d_model,
    ffn_hidden=cfg.ffn_hidden,
    num_heads=cfg.num_heads,
    num_layers=cfg.num_layers,
    num_att_layers=cfg.num_att_layers,
    dropout=cfg.dropout,
    device="cuda",
)
```
