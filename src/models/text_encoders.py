from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class QuesEmbedding(nn.Module):
    def __init__(
        self,
        text_model: str,
        input_size: int = 768,
        output_size: int = 768,
        device: str = "cuda",
        use_safetensors: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.phobert = AutoModel.from_pretrained(
            text_model,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
        )
        self.lstm = nn.LSTM(input_size, output_size, batch_first=True)
        self.to(self.device)

    def forward(self, ques, max_len: int):
        tokenized = self.tokenizer(
            ques,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
            add_special_tokens=False,
        )
        ques_hidden = self.phobert(**tokenized.to(self.device)).last_hidden_state
        _, (h, _) = self.lstm(ques_hidden)
        return h.squeeze(0)


class AnsEmbedding(nn.Module):
    def __init__(
        self,
        text_model: str,
        input_size: int = 768,
        device: str = "cuda",
        use_safetensors: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.phobert_embed = AutoModel.from_pretrained(
            text_model,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
        ).embeddings
        self.to(self.device)

    def forward(self, ans, max_len: int):
        tokenized = self.tokenizer(
            ans,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=False,
        )
        emb = self.phobert_embed(**tokenized.to(self.device))
        return tokenized["input_ids"], emb
