from __future__ import annotations

import torch
import torch.nn as nn
import timm
from transformers import AutoImageProcessor, BeitModel, AutoModel


class DinoBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 768, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = timm.create_model("vit_base_patch16_224.dino", pretrained=True)
        self.model.head = nn.Linear(self.model.embed_dim, embedding_dim)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.blocks[-2:].parameters():
            p.requires_grad = True
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))


class SwinBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 768, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, embedding_dim)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.layers[-1:].parameters():
            p.requires_grad = True
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))


class ConvNeXtBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 768, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = timm.create_model("convnext_base", pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, embedding_dim)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.stages[-1:].parameters():
            p.requires_grad = True
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))


class BeitBackbone(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/beit-base-patch16-224-pt22k-ft22k"
        )
        self.model = BeitModel.from_pretrained(
            "microsoft/beit-base-patch16-224-pt22k-ft22k"
        )
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.encoder.layer[-2:].parameters():
            p.requires_grad = True

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.to(self.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        unnormalized = images * self.std + self.mean
        unnormalized = unnormalized.clamp(0, 1)

        batch_images = [
            unnormalized[i].permute(1, 2, 0).cpu().numpy()
            for i in range(unnormalized.size(0))
        ]
        inputs = self.processor(batch_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(**inputs).last_hidden_state
            emb = out[:, 0, :]
        return emb


class EvaBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 768, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = timm.create_model(
            "eva02_base_patch14_224.mim_in22k",
            pretrained=True,
            num_classes=0,
        )
        self.out_dim = self.model.num_features
        self.proj = nn.Linear(self.out_dim, embedding_dim) if self.out_dim != embedding_dim else nn.Identity()
        self.to(self.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(images.to(self.device))
        pooled = self.model.forward_head(feats, pre_logits=True)
        return self.proj(pooled)


class SigLIPBackbone(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.vision_model.encoder.layers[-2:].parameters():
            p.requires_grad = True

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.to(self.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        unnormalized = images * self.std + self.mean
        unnormalized = unnormalized.clamp(0, 1)

        batch_images = [
            unnormalized[i].permute(1, 2, 0).cpu().numpy()
            for i in range(unnormalized.size(0))
        ]
        inputs = self.processor(batch_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(**inputs).last_hidden_state
            emb = out[:, 0, :]
        return emb
