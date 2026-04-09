import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import ResnetBlock2D
from typing import Optional
from diffusers.models.unets.unet_3d_blocks import SpatioTemporalResBlock


class LoRAAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, rank=32):
        super().__init__()
        self.lora_down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        nn.init.normal_(self.lora_down.weight, std=1 / rank)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.lora_up(self.lora_down(x))


class RoutingLoRAResnetBlock(nn.Module):
    def __init__(self, base_block, num_paths: int = 4, lora_rank: int = 32):
        super().__init__()
        self.base = base_block
        for p in self.base.parameters():
            p.requires_grad = False  # freeze base weights

        Cin1, Cout1 = self.base.conv1.in_channels, self.base.conv1.out_channels
        Cin2, Cout2 = self.base.conv2.in_channels, self.base.conv2.out_channels

        self.lora1 = nn.ModuleList([LoRAAdapter(Cin1, Cout1, rank=lora_rank) for _ in range(num_paths)])
        self.lora2 = nn.ModuleList([LoRAAdapter(Cin2, Cout2, rank=lora_rank) for _ in range(num_paths)])

        self.alpha1 = lora_rank / 2
        self.alpha2 = lora_rank / 2
        self.num_paths = num_paths

    def forward(
        self,
        input_tensor: torch.Tensor,
        routing_slice: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ):
        x = input_tensor
        BxT = x.shape[0]

        if routing_slice is None:
            routing_slice = getattr(self, "_current_routing_slice", None)

        if routing_slice is None:
            routing_slice = torch.zeros(BxT, self.num_paths, device=x.device)
            routing_slice[:, 0] = 1.0

        if routing_slice.ndim == 3:
            routing_slice = routing_slice.reshape(BxT, -1)

        idx = routing_slice.argmax(dim=1)  # [B*T]

        # --- norm1 + act ---
        hidden_states = self.base.norm1(x)
        hidden_states = self.base.nonlinearity(hidden_states)

        # --- up/downsample if needed ---
        if self.base.upsample is not None:
            x = self.base.upsample(x)
            hidden_states = self.base.upsample(hidden_states)
        elif self.base.downsample is not None:
            x = self.base.downsample(x)
            hidden_states = self.base.downsample(hidden_states)

        # --- conv1 + routed LoRA (vectorized) ---
        conv1_out = self.base.conv1(hidden_states)
        lora1_out = torch.zeros_like(conv1_out)
        for p in range(self.num_paths):
            mask = (idx == p)
            if mask.any():
                lora1_out[mask] = self.lora1[p](hidden_states[mask])
        hidden_states = conv1_out + self.alpha1 * lora1_out

        # --- time embedding projection ---
        temb_proj = None
        if (self.base.time_emb_proj is not None) and (temb is not None):
            temb_use = temb
            if not getattr(self.base, "skip_time_act", False):
                temb_use = self.base.nonlinearity(temb_use)
            temb_proj = self.base.time_emb_proj(temb_use)[:, :, None, None]

        # --- norm2 + act + dropout ---
        if self.base.time_embedding_norm == "default":
            if temb_proj is not None:
                hidden_states = hidden_states + temb_proj
            hidden_states = self.base.norm2(hidden_states)
        elif self.base.time_embedding_norm == "scale_shift":
            if temb_proj is None:
                raise ValueError("temb required for scale_shift norm")
            time_scale, time_shift = torch.chunk(temb_proj, 2, dim=1)
            hidden_states = self.base.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.base.norm2(hidden_states)

        hidden_states = self.base.nonlinearity(hidden_states)
        hidden_states = self.base.dropout(hidden_states)

        # --- conv2 + routed LoRA ---
        conv2_out = self.base.conv2(hidden_states)
        lora2_out = torch.zeros_like(conv2_out)
        for p in range(self.num_paths):
            mask = (idx == p)
            if mask.any():
                lora2_out[mask] = self.lora2[p](hidden_states[mask])
        hidden_states = conv2_out + self.alpha2 * lora2_out

        # --- shortcut connection ---
        if self.base.conv_shortcut is not None:
            x = self.base.conv_shortcut(x.contiguous())

        output_tensor = (x + hidden_states) / self.base.output_scale_factor
        return output_tensor



def message_to_route_mask_video(bits, num_resnets, num_paths=4):
    bits_per_decision = int(torch.log2(torch.tensor(num_paths)).item())

    if bits.dim() == 2:
        B, bit_len = bits.shape
        T = 1
        bits_in = bits.unsqueeze(1)  # [B, 1, bit_len]
    elif bits.dim() == 3:
        B, T, bit_len = bits.shape
        bits_in = bits
    else:
        raise ValueError(f"bits must be [B, bit_len] or [B, T, bit_len], got {bits.shape}")

    needed = (num_resnets) * bits_per_decision
    assert bit_len >= needed, f"Need {needed} bits, got {bit_len}"
    masks = []
    powers = (2 ** torch.arange(bits_per_decision, device=bits.device)).float()

    for t in range(T):
        bits_t = bits_in[:, t, :]  # [B, bit_len] for this frame
        ptr = 0
        frame_masks = []

        for _ in range(num_resnets):
            reps = 1 
            for _ in range(reps):
                chunk = bits_t[:, ptr:ptr + bits_per_decision]; ptr += bits_per_decision
                idx = (chunk * powers).sum(dim=1).long() % num_paths
                idx = torch.as_tensor(idx, device=bits.device, dtype=torch.long)
                frame_masks.append(F.one_hot(idx, num_classes=num_paths).float())
        frame_mask_tensor = torch.stack(frame_masks, dim=1)  # [B, D, P]
        masks.append(frame_mask_tensor)

    mask_tensor = torch.stack(masks, dim=1)  # [B, T, D, P]
    return mask_tensor



def inject_routing_blocks_into_video_vae_decoder(vae, num_paths=4):
    num_resnet_blocks = 0
    inserted_modules = []

    def wrap_with_routing(module, module_path):
        nonlocal num_resnet_blocks

        if isinstance(module, ResnetBlock2D):
            num_resnet_blocks += 1
            new_block = RoutingLoRAResnetBlock(module, num_paths=num_paths)
            inserted_modules.append((module_path, "RoutingLoRAResnetBlock"))
            return new_block, True
        elif isinstance(module, SpatioTemporalResBlock):
            spatial_block = getattr(module, "spatial_res_block", None)
            if isinstance(spatial_block, ResnetBlock2D):
                wrapped, _ = wrap_with_routing(spatial_block, f"{module_path}.spatial_res_block")
                module.spatial_res_block = wrapped
            return module, True

        return module, False

    def replace_blocks(parent, parent_path="decoder"):
        for name, child in parent.named_children():
            module_path = f"{parent_path}.{name}"
            new_child, replaced = wrap_with_routing(child, module_path)
            if replaced:
                setattr(parent, name, new_child)
            else:
                replace_blocks(child, module_path)
    replace_blocks(vae.decoder)

    print("\n Routing blocks injected at:")
    for path, cls in inserted_modules:
        print(f"  {path:<70s} --> {cls}")
    total_slots = num_resnet_blocks 


    def decode_with_routing(self, z, routing_mask, temb=None, num_frames=None):
        from types import SimpleNamespace
        if z.dim() != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(z.shape)}")
        B, C, T, H, W = z.shape
        z_flat = z.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        if routing_mask.dim() == 4:
            routing_flat = routing_mask.reshape(B*T, routing_mask.shape[2], routing_mask.shape[3])
        else:
            raise ValueError(f"routing_mask must be [B,T,L,P], got {tuple(routing_mask.shape)}")

        if getattr(self, "post_quant_conv", None) is not None:
            z_flat = self.post_quant_conv(z_flat)
        sample = self.decoder.conv_in(z_flat)
        route_ptr = 0
        for res in getattr(self.decoder.mid_block, "resnets", []):
            res._current_routing_slice = routing_flat[:, route_ptr, :]
            route_ptr += 1
        sample = self.decoder.mid_block(sample)

        for up in self.decoder.up_blocks:
            if hasattr(up, "resnets"):
                for res in up.resnets:
                    res._current_routing_slice = routing_flat[:, route_ptr, :]
                    route_ptr += 1
            sample = up(sample)

        sample = self.decoder.conv_norm_out(sample)
        sample = self.decoder.conv_act(sample)
        sample = self.decoder.conv_out(sample)

        return SimpleNamespace(sample=sample)

    vae.decode_with_routing = decode_with_routing.__get__(vae)

    return vae, num_resnet_blocks, total_slots






