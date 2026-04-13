from __future__ import annotations

import torch
import torch.nn as nn

from .attention import Attention
from .decoder import Decoder
from .image_fusion import ImageEmbedding
from .text_encoders import QuesEmbedding, AnsEmbedding


def build_causal_mask(max_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full([max_len, max_len], float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


class VQAModel(nn.Module):
    def __init__(
        self,
        vocab_size: int | None,
        text_model: str,
        image_encoders: list[str] | tuple[str, ...],
        fusion: str = "gated",
        image_weights: dict[str, str] | None = None,
        use_safetensors: bool = True,
        local_files_only: bool = False,
        d_model: int = 768,
        ffn_hidden: int = 2048,
        num_heads: int = 8,
        num_layers: int = 5,
        num_att_layers: int = 4,
        dropout: float = 0.3,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.image_model = ImageEmbedding(
            encoders=image_encoders,
            fusion=fusion,
            embedding_dim=d_model,
            device=device,
            image_weights=image_weights,
        )
        self.ques_model = QuesEmbedding(
            text_model=text_model,
            device=device,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
        )
        self.ans_model = AnsEmbedding(
            text_model=text_model,
            device=device,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
        )

        if vocab_size is None:
            vocab_size = len(self.ans_model.tokenizer.get_vocab())

        self.an_model = nn.ModuleList(
            [Attention(d=d_model, num_heads=num_heads, dropout=0.5) for _ in range(num_att_layers)]
        )

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

        self.decoder = Decoder(d_model, ffn_hidden, num_heads, dropout, num_layers)

        self.mlp = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(d_model, vocab_size),
        )

    @torch.no_grad()
    def generate(self, images: torch.Tensor, questions: list, max_len: int = 27,
                 min_len: int = 2) -> torch.Tensor:
        """
        Sinh câu trả lời autoregressively — KHÔNG dùng ground truth answers.

        Args:
            min_len: Số token tối thiểu trước khi cho phép EOS.
                     Ngăn model predict EOS ngay lập tức.

        Returns:
            generated: Tensor (B, max_len) chứa token IDs
        """
        tokenizer = self.ans_model.tokenizer
        pad_id = tokenizer.pad_token_id
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        B = images.size(0)

        # Tính image + question context một lần duy nhất
        image_embeddings, _ = self.image_model(images.to(self.device))
        ques_embeddings = self.ques_model(questions, max_len=max_len)
        ques_embedds = ques_embeddings.unsqueeze(1)

        att_embedds = None
        for att_layer in self.an_model:
            att_embedds = att_layer(image_embeddings.to(self.device), ques_embedds.to(self.device))

        if att_embedds is None:
            raise RuntimeError("Attention stack produced no output")

        att_embedds = self.tanh(att_embedds)

        x = att_embedds.unsqueeze(1).expand(-1, max_len, -1)  # (B, max_len, D)
        decoder_mask = build_causal_mask(max_len, device=x.device)

        # Khởi tạo sequence với BOS token, phần còn lại là PAD
        generated = torch.full((B, max_len), pad_id, dtype=torch.long, device=self.device)
        generated[:, 0] = bos_id
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        for step in range(max_len - 1):
            y = self.ans_model.phobert_embed(input_ids=generated)  # (B, max_len, D)

            out = self.decoder(x, y, decoder_mask)      # (B, max_len, D)
            logits = self.mlp(out)                       # (B, max_len, vocab_size)

            # Block EOS/PAD trước min_len tokens — buộc model phải generate content
            if step < min_len:
                logits[:, step, eos_id] = float("-inf")
                logits[:, step, pad_id] = float("-inf")

            next_token = logits[:, step, :].argmax(dim=-1)  # (B,)

            next_token = next_token.masked_fill(finished, pad_id)
            generated[:, step + 1] = next_token

            finished = finished | (next_token == eos_id)
            if finished.all():
                break

        return generated  # (B, max_len) token IDs

    def forward(self, images, questions, answers, anno_ids=None, mask: bool = True,
                max_len: int = 27, teacher_forcing_ratio: float = 1.0):
        image_embeddings, _ = self.image_model(images.to(self.device), anno_ids)
        ques_embeddings = self.ques_model(questions, max_len=max_len)
        ques_embedds = ques_embeddings.unsqueeze(1)

        att_embedds = None
        for att_layer in self.an_model:
            att_embedds = att_layer(image_embeddings.to(self.device), ques_embedds.to(self.device))

        if att_embedds is None:
            raise RuntimeError("Attention stack produced no output")

        att_embedds = self.tanh(att_embedds)
        att_embedds = self.dropout(att_embedds)

        ans_vocab, ans_embedds = self.ans_model(answers, max_len=max_len)

        x = att_embedds.to(self.device).unsqueeze(1).expand(-1, max_len, -1)
        decoder_mask = build_causal_mask(max_len, device=x.device) if mask else None

        # Scheduled Sampling: trộn ground truth embeddings với predicted embeddings
        # teacher_forcing_ratio=1.0 → 100% ground truth (giống cũ)
        # teacher_forcing_ratio=0.5 → 50% dùng predicted token
        if teacher_forcing_ratio >= 1.0 or not self.training:
            # Pure teacher forcing (backward-compatible)
            out = self.decoder(x, ans_embedds, decoder_mask)
        else:
            B = x.size(0)
            # Bắt đầu từ ground truth embeddings
            y = ans_embedds.clone()
            all_logits = []

            for t in range(max_len):
                out_t = self.decoder(x, y, decoder_mask)   # (B, max_len, D)
                logit_t = self.mlp(out_t[:, t, :])         # (B, vocab_size)
                all_logits.append(logit_t)

                # Với xác suất (1 - teacher_forcing_ratio), thay thế
                # embedding tại position t+1 bằng predicted token
                if t + 1 < max_len and torch.rand(1).item() > teacher_forcing_ratio:
                    pred_token = logit_t.argmax(dim=-1)  # (B,)
                    y[:, t + 1, :] = self.ans_model.phobert_embed(
                        input_ids=pred_token.unsqueeze(1)
                    ).squeeze(1)

            output_logits = torch.stack(all_logits, dim=1)  # (B, max_len, vocab_size)
            return output_logits, ans_vocab

        output_logits = self.mlp(out)
        return output_logits, ans_vocab
