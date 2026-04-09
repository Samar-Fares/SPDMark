import os
from typing import Any, Tuple, Optional
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd
import torch.nn as nn
from decord import VideoReader, cpu
import hmac, hashlib, struct
from huggingface_hub import HfFolder, whoami
import itertools


class FrameWiseExtractor(nn.Module):
    def __init__(self, total_slots):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2 * total_slots) 

    def forward(self, video_frames):
        B, T, C, H, W = video_frames.shape

        x = video_frames.view(B * T, C, H, W)  # [B*T, 3, H, W]
        x = (x / 2) + 0.5
        x = x.clamp(0, 1)
        mean = x.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        x = (x - mean) / std
        logits = self.backbone(x)  
        logits = logits.view(B, T, -1)              # [B, T, 2*total_slots]
        return logits

normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                           std=[0.5, 0.5, 0.5]) 

def vqgan_transform(img_size: int):
    return  transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize_vqgan
                ])

def get_video_dataloader(
    metadata_path: str,
    data_dir: str,
    num_frames: int,
    frame_interval: int,
    transform: transforms,
    num_videos: int = None,
    batch_size: int = 1,
    num_workers: int = 8,
    shuffle: bool = False,
    collate_fn: Any = None
):
    """
    Returns a DataLoader for locally downloaded OpenVid-1M videos.
    """
    df = pd.read_csv(metadata_path)
    df["video_path"] = df["video"].apply(lambda v: os.path.join(data_dir, v))
    df = df[df["video_path"].apply(os.path.exists)]
    print(f"Found {len(df)} existing videos in {data_dir}")

    if num_videos is not None and len(df) > num_videos:
        df = df.sample(n=num_videos, random_state=42).reset_index(drop=True)
        print(f"Using random subset of {len(df)} videos")
    dataset = SubOpenVid(df, num_frames, frame_interval, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )



def selective_temporal_loss(gen, ref, threshold=0.05):
    y_gen = 0.299*gen[:, :, 0:1] + 0.587*gen[:, :, 1:2] + 0.114*gen[:, :, 2:3]
    y_ref = 0.299*ref[:, :, 0:1] + 0.587*ref[:, :, 1:2] + 0.114*ref[:, :, 2:3]
    gd = y_gen[:, 1:] - y_gen[:, :-1]
    rd = y_ref[:, 1:] - y_ref[:, :-1]
    return F.l1_loss(gd, rd)



def derive_base_key(seed: int, img_id: int, salt=b"SPDMark-v1"):
    return hashlib.sha256(salt + f"{seed}_{img_id}".encode()).digest()[:16]


def build_message_sequence(T: int, total_bits: int = 28, base_key: bytes = None):
    """
    Each frame t gets total_bits  = HMAC(base_key, t)
    """
    if base_key is None:
        base_key = os.urandom(16)  # secret per video
    msgs = []
    for t in range(T):
        digest = hmac.new(base_key, struct.pack(">I", t), hashlib.sha256).digest()
        bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))[:total_bits]
        msgs.append(torch.tensor(bits, dtype=torch.float32))
    msg_bits = torch.stack(msgs, dim=0).unsqueeze(0)  # [1, T, total_bits]
    return msg_bits

class SubOpenVid(torch.utils.data.Dataset):
    def __init__(self, df, num_frames, frame_interval, transform=None):
        self.df = df
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        try:
            vr = VideoReader(video_path, ctx=cpu())
            total_frames = len(vr)
            if total_frames < self.num_frames:
                raise ValueError("Too few frames")

            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = [torch.from_numpy(vr[i].asnumpy()).permute(2, 0, 1).float() / 255.0
                      for i in frame_indices]
            video = torch.stack(frames, dim=0)  

            if self.transform is not None:
                if isinstance(self.transform, transforms.Compose):
                    processed = []
                    for f in video:
                        try:
                            processed.append(self.transform(transforms.functional.to_pil_image(f)))
                        except Exception:
                            processed.append(self.transform(f))
                    video = torch.stack(processed, dim=0)
                else:
                    video = self.transform(video)
            return video
        except Exception as e:
            print(f"Skipping {video_path}: {e}")
            return torch.empty(0)

def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def get_params_optimize(vaed, extractor):
    params_to_optimize = itertools.chain(vaed.parameters(), extractor.parameters())
    return params_to_optimize

def get_dataloader(args) -> Tuple[DataLoader, DataLoader]:
    transform = vqgan_transform(256)
    train_loader = get_video_dataloader(
        metadata_path=args.metadata_path,
        data_dir=args.data_dir,
        num_frames=8,
        frame_interval=8,
        transform=transform,
        num_videos=10000,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    val_loader = get_video_dataloader(
        metadata_path=args.metadata_path,
        data_dir=args.data_dir,
        num_frames=8,
        frame_interval=8,
        transform=transform,
        num_videos=100,
        batch_size=args.train_batch_size,
        shuffle=False,
    )
    return train_loader, val_loader

def get_all_lora_params(model):
    lora_params = []
    for name, module in model.named_modules():
        for attr in ["lora1", "lora2", "q_adapters", "k_adapters", "v_adapters"]:
            if hasattr(module, attr):
                adapters = getattr(module, attr)
                if isinstance(adapters, (list, nn.ModuleList)):
                    for adapter in adapters:
                        for p in adapter.parameters():
                            if p.requires_grad:
                                lora_params.append(p)
        if hasattr(module, "adapters"):
            for adapter in getattr(module, "adapters"):
                for p in adapter.parameters():
                    if p.requires_grad:
                        lora_params.append(p)
    lora_params = list(set(lora_params))
    return lora_params