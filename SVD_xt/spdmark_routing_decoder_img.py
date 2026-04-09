import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.attention import Attention as AttentionBlock
from typing import  Optional
import math
from diffusers.models.unets.unet_3d_blocks import SpatioTemporalResBlock
from types import SimpleNamespace
import itertools


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
            routing_slice = routing_slice.reshape(BxT, self.num_paths)
        idx = routing_slice.argmax(dim=1)  # [B*T]

        hidden_states = self.base.norm1(x)
        hidden_states = self.base.nonlinearity(hidden_states)

        if self.base.upsample is not None:
            x = self.base.upsample(x)
            hidden_states = self.base.upsample(hidden_states)
        elif self.base.downsample is not None:
            x = self.base.downsample(x)
            hidden_states = self.base.downsample(hidden_states)

        conv1_out = self.base.conv1(hidden_states)
        lora1_out = torch.zeros_like(conv1_out)
        for p in range(self.num_paths):
            mask = (idx == p)
            if mask.any():
                lora1_out[mask] = self.lora1[p](hidden_states[mask])
        hidden_states = conv1_out + self.alpha1 * lora1_out

        temb_proj = None
        if (self.base.time_emb_proj is not None) and (temb is not None):
            temb_use = temb
            if not getattr(self.base, "skip_time_act", False):
                temb_use = self.base.nonlinearity(temb_use)
            temb_proj = self.base.time_emb_proj(temb_use)[:, :, None, None]

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

        conv2_out = self.base.conv2(hidden_states)
        lora2_out = torch.zeros_like(conv2_out)
        for p in range(self.num_paths):
            mask = (idx == p)
            if mask.any():
                lora2_out[mask] = self.lora2[p](hidden_states[mask])
        hidden_states = conv2_out + self.alpha2 * lora2_out

        if self.base.conv_shortcut is not None:
            x = self.base.conv_shortcut(x.contiguous())

        output_tensor = (x + hidden_states) / self.base.output_scale_factor
        return output_tensor

def message_to_route_mask_video(bits, num_resnets, num_attns, num_paths=4, per_block=True):
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

    if per_block:
        needed = (num_resnets + 3 * num_attns) * bits_per_decision
    assert bit_len >= needed, f"Need {needed} bits, got {bit_len}"
    masks = []
    powers = (2 ** torch.arange(bits_per_decision, device=bits.device)).float()

    for t in range(T):
        bits_t = bits_in[:, t, :]  # [B, bit_len] for this frame
        ptr = 0
        frame_masks = []

        for _ in range(num_resnets):
            reps = 1 if per_block else 2
            for _ in range(reps):
                chunk = bits_t[:, ptr:ptr + bits_per_decision]; ptr += bits_per_decision
                idx = (chunk * powers).sum(dim=1).long() % num_paths
                idx = torch.as_tensor(idx, device=bits.device, dtype=torch.long)
                frame_masks.append(F.one_hot(idx, num_classes=num_paths).float())

        for _ in range(num_attns):
            for _ in range(3):
                chunk = bits_t[:, ptr:ptr + bits_per_decision]; ptr += bits_per_decision
                idx = (chunk * powers).sum(dim=1).long() % num_paths
                idx = torch.as_tensor(idx, device=bits.device, dtype=torch.long)
                frame_masks.append(F.one_hot(idx, num_classes=num_paths).float())

        frame_mask_tensor = torch.stack(frame_masks, dim=1)  # [B, D, P]
        masks.append(frame_mask_tensor)

    mask_tensor = torch.stack(masks, dim=1)  # [B, T, D, P]
    return mask_tensor

def inject_routing_blocks_into_video_vae_decoder(vae, num_paths=4, enable_attention=False):
    num_resnet_blocks = 0
    num_attention_blocks = 0
    inserted_modules = []

    def wrap_with_routing(module, module_path):
        nonlocal num_resnet_blocks, num_attention_blocks

        if isinstance(module, ResnetBlock2D):
            num_resnet_blocks += 1
            new_block = RoutingLoRAResnetBlock(module, num_paths=num_paths)
            inserted_modules.append((module_path, "RoutingLoRAResnetBlock"))
            return new_block, True

        elif enable_attention and isinstance(module, AttentionBlock):
            num_attention_blocks += 1
            new_block = RoutingLoRAAttentionBlock(module, num_paths=num_paths)
            inserted_modules.append((module_path, "RoutingLoRAAttentionBlock"))
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
    total_slots = num_resnet_blocks + 3 * num_attention_blocks


    def decode_with_routing(self, z, routing_mask, temb=None, num_frames=None, enable_attention: bool = False):
        def _list_attentions(block):
            attns = []
            if hasattr(block, "attentions"):
                attns.extend(list(block.attentions))
            if hasattr(block, "temporal_attentions"):
                attns.extend(list(block.temporal_attentions))
            return attns

        if z.dim() == 5:
            B, C, T, H, W = z.shape
            assert C == 4
            z_bt = z.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        elif z.dim() == 4:
            BT, C, H, W = z.shape
            assert C == 4
            assert num_frames is not None
            T = int(num_frames)
            assert BT % T == 0
            B = BT // T
            z_bt = z
        else:
            raise ValueError(f"z must be 4D/5D, got {z.shape}")

        indicator = torch.zeros(B, T, dtype=z_bt.dtype, device=z_bt.device)

        def _count_resnets(decoder) -> int:
            cnt = 0
            for res in getattr(decoder.mid_block, "resnets", []):
                if isinstance(res, SpatioTemporalResBlock) and hasattr(res, "spatial_res_block"):
                    cnt += 1
            for up in decoder.up_blocks:
                for res in getattr(up, "resnets", []):
                    if isinstance(res, SpatioTemporalResBlock) and hasattr(res, "spatial_res_block"):
                        cnt += 1
            return cnt
        def _count_attn_sites(decoder) -> int:
            n = 0
            for attn in _list_attentions(decoder.mid_block):
                if isinstance(attn, RoutingLoRAAttentionBlock):
                    n += 3
            for up in decoder.up_blocks:
                for attn in _list_attentions(up):
                    if isinstance(attn, RoutingLoRAAttentionBlock):
                        n += 3
            return n

        L_res = _count_resnets(self.decoder)
        L_attn = _count_attn_sites(self.decoder) if enable_attention else 0
        L_total = L_res + L_attn

        if routing_mask is None:
            P = 4
            routing_bt = z_bt.new_zeros((B * T, L_total, P))
            routing_bt[..., 0] = 1.0
        else:
            if routing_mask.dim() == 4:
                if routing_mask.shape[1] == 1:
                    routing_mask = routing_mask.expand(-1, T, -1, -1)
                assert routing_mask.shape[0] == B and routing_mask.shape[1] == T
                routing_bt = routing_mask.reshape(B * T, routing_mask.shape[2], routing_mask.shape[3]).contiguous()
            elif routing_mask.dim() == 3:
                assert routing_mask.shape[0] == B * T
                routing_bt = routing_mask
            else:
                raise ValueError(f"routing_mask must be [BT,L,P] or [B,T,L,P], got {tuple(routing_mask.shape)}")

            assert routing_bt.shape[1] == L_total, f"routing L={routing_bt.shape[1]} must match expected L_total={L_total}"

        sample = self.decoder.conv_in(z_bt)

        route_ptr = 0
        L_mask = routing_bt.shape[1]

        for res in getattr(self.decoder.mid_block, "resnets", []):
            if isinstance(res, SpatioTemporalResBlock) and hasattr(res, "spatial_res_block"):
                res.spatial_res_block._current_routing_slice = routing_bt[:, route_ptr, :] if route_ptr < L_mask else None
                route_ptr += 1

        if enable_attention:
            for attn in _list_attentions(self.decoder.mid_block):
                if isinstance(attn, RoutingLoRAAttentionBlock):
                    attn._current_routing_slice = routing_bt[:, route_ptr:route_ptr + 3, :] if (route_ptr + 2) < L_mask else None
                    route_ptr += 3

        upscale_dtype = next(itertools.chain(self.decoder.up_blocks.parameters(),
                                            self.decoder.up_blocks.buffers())).dtype

        sample = self.decoder.mid_block(sample, image_only_indicator=indicator)
        sample = sample.to(upscale_dtype)

        for up in self.decoder.up_blocks:
            if hasattr(up, "resnets"):
                for res in up.resnets:
                    if isinstance(res, SpatioTemporalResBlock) and hasattr(res, "spatial_res_block"):
                        res.spatial_res_block._current_routing_slice = routing_bt[:, route_ptr, :] if route_ptr < L_mask else None
                        route_ptr += 1

            if enable_attention:
                for attn in _list_attentions(up):
                    if isinstance(attn, RoutingLoRAAttentionBlock):
                        attn._current_routing_slice = routing_bt[:, route_ptr:route_ptr + 3, :] if (route_ptr + 2) < L_mask else None
                        route_ptr += 3

            sample = up(sample, image_only_indicator=indicator)

        sample = self.decoder.conv_norm_out(sample)
        sample = self.decoder.conv_act(sample)
        sample = self.decoder.conv_out(sample)

        bt, c, h, w = sample.shape
        sample_5d = sample[None, :].reshape(B, T, c, h, w).permute(0, 2, 1, 3, 4)
        sample_5d = self.decoder.time_conv_out(sample_5d)
        sample = sample_5d.permute(0, 2, 1, 3, 4).reshape(bt, c, h, w)

        return SimpleNamespace(sample=sample)
    vae.decode_with_routing = decode_with_routing.__get__(vae)

    return vae, num_resnet_blocks, num_attention_blocks, total_slots

class AttentionLoRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.lora_up(self.lora_down(x))

class RoutedAttentionLinear(nn.Module):
    def __init__(self, base_linear: nn.Module, num_paths: int = 4, lora_rank: int = 8, alpha: Optional[float] = None):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.num_paths = num_paths
        self.rank = lora_rank
        self.alpha = float(alpha if alpha is not None else lora_rank)

        in_features = self.base.in_features
        out_features = self.base.out_features
        self.adapters = nn.ModuleList([
            AttentionLoRAAdapter(in_features, out_features, rank=lora_rank) for _ in range(num_paths)
        ])
        self._current_routing_slice = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)

        routing_slice = getattr(self, '_current_routing_slice', None)
        self._current_routing_slice = None

        if routing_slice is None:
            return out

        if routing_slice.ndim == 1:
            routing_slice = routing_slice.unsqueeze(0)
        idx = routing_slice.argmax(dim=-1)

        lora_out = torch.zeros_like(out)
        for p in range(self.num_paths):
            mask = (idx == p)
            if mask.any():
                lora_out[mask] = self.adapters[p](x[mask])

        return out + (self.alpha / self.rank) * lora_out


class RoutingLoRAAttentionBlock(nn.Module):
    def __init__(self, base_block: AttentionBlock, num_paths: int = 4, lora_rank: int = 8, patch_out: bool = False):
        super().__init__()
        self.base = base_block
        for p in self.base.parameters():
            p.requires_grad = False

        self.num_paths = num_paths
        self.base.to_q = RoutedAttentionLinear(self.base.to_q, num_paths=num_paths, lora_rank=lora_rank)
        self.base.to_k = RoutedAttentionLinear(self.base.to_k, num_paths=num_paths, lora_rank=lora_rank)
        self.base.to_v = RoutedAttentionLinear(self.base.to_v, num_paths=num_paths, lora_rank=lora_rank)

        self.has_out_adapter = False
        if patch_out and hasattr(self.base, 'to_out'):
            if isinstance(self.base.to_out, nn.ModuleList) and len(self.base.to_out) > 0 and isinstance(self.base.to_out[0], nn.Linear):
                self.base.to_out[0] = RoutedAttentionLinear(self.base.to_out[0], num_paths=num_paths, lora_rank=lora_rank)
                self.has_out_adapter = True
            elif isinstance(self.base.to_out, nn.Sequential) and len(self.base.to_out) > 0 and isinstance(self.base.to_out[0], nn.Linear):
                self.base.to_out[0] = RoutedAttentionLinear(self.base.to_out[0], num_paths=num_paths, lora_rank=lora_rank)
                self.has_out_adapter = True

    def forward(self, x: torch.Tensor, *args, routing_slice: Optional[torch.Tensor] = None, **kwargs):
        if routing_slice is None:
            routing_slice = getattr(self, '_current_routing_slice', None)
        self._current_routing_slice = None

        if routing_slice is None:
            return self.base(x, *args, **kwargs)

        if routing_slice.ndim != 3 or routing_slice.shape[1] < 3:
            raise ValueError(f'routing_slice for attention must be [B,3,P], got {tuple(routing_slice.shape)}')

        self.base.to_q._current_routing_slice = routing_slice[:, 0, :]
        self.base.to_k._current_routing_slice = routing_slice[:, 1, :]
        self.base.to_v._current_routing_slice = routing_slice[:, 2, :]

        return self.base(x, *args, **kwargs)
