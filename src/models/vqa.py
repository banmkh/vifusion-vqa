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
        )
        self.ques_model = QuesEmbedding(text_model=text_model, device=device)
        self.ans_model = AnsEmbedding(text_model=text_model, device=device)

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

    def forward(self, images, questions, answers, anno_ids=None, mask: bool = True, max_len: int = 27):
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
        y = ans_embedds

        decoder_mask = build_causal_mask(max_len, device=x.device) if mask else None
        out = self.decoder(x, y, decoder_mask)

        output_logits = self.mlp(out)
        return output_logits, ans_vocab
