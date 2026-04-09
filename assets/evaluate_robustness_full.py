import os, json, math, random
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import shutil, subprocess, tempfile, cv2
from PIL import ImageDraw, ImageFont, Image
from diffusers.utils import  export_to_video
import torchvision.transforms.functional as TF
from torchvision import transforms as _T
import torch.nn as nn
from contextlib import contextmanager
from scipy.stats import binom

def _ffprobe_video_meta(path: str) -> dict:
    if shutil.which("ffprobe") is None:
        return {}
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,codec_long_name,bit_rate,avg_frame_rate,pix_fmt,width,height",
             "-of", "json", path],
            stderr=subprocess.STDOUT
        ).decode("utf-8", "ignore")
        j = json.loads(out)
        return j.get("streams", [{}])[0] if "streams" in j else {}
    except Exception:
        return {}

def atk_recompress(frames, crf=28, bitrate=None, codec_preference=None, fps=7):
    """
    Recompress frames [T,3,H,W] in [-1,1]. Returns (frames_out, meta).
    meta has keys: applied, tried_codecs, chosen_codec, ffmpeg_cmd, ffmpeg_err,
                   ffprobe, frames_in, frames_out, mean_abs_diff, crf, bitrate, fps
    """
    device = frames.device
    T, C, H, W = frames.shape
    meta = {
        "applied": False,
        "tried_codecs": [],
        "chosen_codec": None,
        "ffmpeg_cmd": None,
        "ffmpeg_err": None,
        "ffprobe": {},
        "frames_in": int(T),
        "frames_out": 0,
        "mean_abs_diff": 0.0,
        "crf": int(crf) if crf is not None else None,
        "bitrate": bitrate,
        "fps": int(fps),
    }

    # Decide codec order (preference → availability → safe fallback)
    candidates = []
    if codec_preference:
        candidates.append(codec_preference)
    has_x264 = _ffmpeg_has_encoder("libx264")
    has_x265 = _ffmpeg_has_encoder("libx265")
    has_openh264 = _ffmpeg_has_encoder("libopenh264")
    if has_x264: candidates.append("libx264")
    if has_x265: candidates.append("libx265")
    if has_openh264: candidates.append("libopenh264")
    candidates.append("mpeg4")  # usually available even if others fail
    # dedupe while preserving order
    seen = set(); candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    if shutil.which("ffmpeg") is None:
        meta["ffmpeg_err"] = "ffmpeg not found; returning JPEG fallback"
        try:
            return _jpeg_frame_recompress(frames, quality=45), meta
        except Exception as e:
            meta["ffmpeg_err"] += f" | JPEG fallback failed: {e}"
            return frames, meta

    # Prepare temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in  = os.path.join(tmpdir, "in.mp4")
        tmp_out = os.path.join(tmpdir, "out.mp4")

        # Ensure even dimensions for yuv420p
        even_w, even_h = (W // 2) * 2, (H // 2) * 2
        frames_pil = [TF.to_pil_image((f.clamp(-1,1)+1)/2) for f in frames]
        export_to_video(frames_pil, tmp_in, fps=fps)

        vf = f"scale={even_w}:{even_h}:flags=lanczos,fps={int(fps)}"

        last_err = None
        for codec in candidates:
            meta["tried_codecs"].append(codec)
            if codec in ("libx264", "libx265"):
                cmd = [
                    "ffmpeg","-hide_banner","-loglevel","error","-y",
                    "-i", tmp_in,
                    "-c:v", codec, "-preset", "veryfast", "-crf", str(int(crf)),
                    "-pix_fmt", "yuv420p", "-vf", vf, "-an",
                    "-movflags","+faststart", tmp_out
                ]
            elif codec == "libopenh264":
                br = bitrate or "1200k"
                buf = f"{max(1, int(br.rstrip('k'))*2)}k"
                cmd = [
                    "ffmpeg","-hide_banner","-loglevel","error","-y",
                    "-i", tmp_in,
                    "-c:v","libopenh264",
                    "-b:v", br, "-maxrate", br, "-bufsize", buf,
                    "-pix_fmt","yuv420p", "-vf", vf, "-an",
                    "-movflags","+faststart", tmp_out
                ]
            else:  # mpeg4: approximate CRF with qscale
                qscale = max(2, min(31, int((int(crf) - 18) * 1.2 + 2)))
                cmd = [
                    "ffmpeg","-hide_banner","-loglevel","error","-y",
                    "-i", tmp_in,
                    "-c:v","mpeg4","-qscale:v", str(qscale),
                    "-pix_fmt","yuv420p", "-vf", vf, "-an",
                    tmp_out
                ]

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                meta["chosen_codec"] = codec
                meta["ffmpeg_cmd"] = " ".join(cmd)
            except subprocess.CalledProcessError as e:
                last_err = e.stderr.decode("utf-8", "ignore") if e.stderr else str(e)
                continue  # try next codec

            # decode result
            cap = cv2.VideoCapture(tmp_out)
            out_frames = []
            while True:
                ret, fr = cap.read()
                if not ret: break
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(fr).permute(2,0,1).float()/127.5 - 1.0
                out_frames.append(t)
            cap.release()

            if out_frames:
                out_t = torch.stack(out_frames, 0).to(device)
                Tm = min(out_t.shape[0], T)
                mad = (out_t[:Tm] - frames[:Tm]).abs().mean().item()
                meta["applied"] = True
                meta["frames_out"] = int(out_t.shape[0])
                meta["mean_abs_diff"] = float(mad)
                meta["ffprobe"] = _ffprobe_video_meta(tmp_out)
                return out_t, meta

        # If all codecs failed → JPEG fallback or return original
        meta["ffmpeg_err"] = last_err or "all codecs failed"
        try:
            out = _jpeg_frame_recompress(frames, quality=45)
            meta["chosen_codec"] = "jpeg_per_frame"
            meta["applied"] = True
            Tm = min(out.shape[0], T)
            meta["frames_out"] = int(out.shape[0])
            meta["mean_abs_diff"] = float((out[:Tm]-frames[:Tm]).abs().mean().item())
            return out, meta
        except Exception as e:
            meta["ffmpeg_err"] += f" | JPEG fallback failed: {e}"
            return frames, meta

def tau_from_fpr_frame(M, fpr_frame=0.01):
    for tau in range(M+1):
        if binom.sf(tau-1, M, 0.5) <= fpr_frame:
            print("Tau: ", tau)
            return tau
    return M+1 

def K_from_fpr_video(n_trials, p_frame, fpr_video=0.01):
    for K in range(n_trials+1):
        if binom.sf(K-1, n_trials, p_frame) <= fpr_video:
            return K
    return n_trials+1

def build_similarity_matrix(gt_bits: torch.Tensor, pred_bits: torch.Tensor) -> np.ndarray:
    gt = gt_bits[0].int()     # [Tg, M_g]
    pr = pred_bits[0].int()   # [Tp, M_p]
    Tg, Mg = gt.shape
    Tp, Mp = pr.shape
    M_eff = min(Mg, Mp)
    gt = gt[:, :M_eff]
    pr = pr[:, :M_eff]
    sim = torch.zeros(Tg, Tp, dtype=torch.float32)
    for i in range(Tg):
        sim[i] = (pr == gt[i]).float().mean(dim=1)
    return sim.cpu().numpy()


def eval_alignment_and_detection(
    gt_bits: torch.Tensor, pred_bits: torch.Tensor,
    M_target: int,                      # payload 
    fpr_frame: float = 0.01,            # per-frame false alarm target
    fpr_video: float = 0.01,             # video-level false alarm target,
):
    Mg = int(gt_bits.shape[2]); Mp = int(pred_bits.shape[2])
    M_used = min(Mg, Mp, M_target)
    gt_bits = gt_bits[:, :, :M_used]
    pred_bits = pred_bits[:, :, :M_used]

    sim = build_similarity_matrix(gt_bits, pred_bits)
    r,c = linear_sum_assignment(-sim)
    pairs_all = list(zip(r.tolist(), c.tolist()))
    scores = sim[r,c]

    n_trials = len(pairs_all)           

    # Per-frame threshold from FPR
    tau_f = tau_from_fpr_frame(M_used, fpr_frame=fpr_frame)
    p_f  = float(binom.sf(tau_f - 1, M_used, 0.5)) 

    # Count valid pairs 
    valid = []
    valid_scores = []
    hits = 0
    for (i, j), s in zip(pairs_all, scores):
        Dt = int((gt_bits[0, i] == pred_bits[0, j]).sum().item())
        if Dt >= tau_f:
            hits += 1
            valid.append((i, j))
            valid_scores.append(s)
    num_valid = len(valid)
    bitacc_defined = num_valid > 0
    avg_valid_sim = float(np.mean(valid_scores)) if bitacc_defined else 0.0
    avg_match_sim = float(np.mean(scores)) if n_trials > 0 else 0.0

    # 4) Video-level decision using
    if n_trials == 0:
        K_star = 1
        is_watermarked = False
    else:
        K_star = K_from_fpr_video(n_trials, p_frame=p_f, fpr_video=fpr_video)
        is_watermarked = (hits > K_star)

    # Order preservation over valid pairs
    in_order = all(valid[k][1] < valid[k+1][1] for k in range(max(0, num_valid-1)))

    return {
        "T_gt": int(gt_bits.shape[1]),
        "T_pred": int(pred_bits.shape[1]),
        "M_used": int(M_used),
        "tau_frame": int(tau_f),
        "p_frame": float(p_f),
        "K_star": int(K_star),
        "n_trials": int(n_trials),
        "hits": int(hits),
        "num_valid": int(num_valid),
        "avg_valid_sim": float(avg_valid_sim),   # mean over valid pairs
        "avg_match_sim": float(avg_match_sim),   # mean over all matches
        "bitacc_defined": bool(bitacc_defined),
        "is_watermarked": bool(is_watermarked),
        "is_in_order": bool(in_order),
        "pairs_all": pairs_all,
        "pairs_valid": valid,
    }

@contextmanager
def test_time_bn(model: nn.Module):
    """
    Temporarily make ONLY BatchNorm layers use batch statistics,
    while keeping the rest of the model in eval mode.
    Also restores running stats afterward so you don't corrupt them.
    """
    was_training = model.training
    model.eval()
    bn_layers = []
    saved = {}

    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append((name, m))
            saved[name] = (
                m.training,
                m.momentum,
                m.track_running_stats,
                m.running_mean.detach().clone() if m.running_mean is not None else None,
                m.running_var.detach().clone() if m.running_var is not None else None,
                int(m.num_batches_tracked.detach().item()) if hasattr(m, "num_batches_tracked") else None,
            )
            m.train()
            m.momentum = 1.0
            m.track_running_stats = True 
    try:
        yield
    finally:
        for name, m in bn_layers:
            tr, mom, trs, rm, rv, nbt = saved[name]
            m.training = tr
            m.momentum = mom
            m.track_running_stats = trs
            if rm is not None: m.running_mean.copy_(rm)
            if rv is not None: m.running_var.copy_(rv)
            if nbt is not None and hasattr(m, "num_batches_tracked"):
                m.num_batches_tracked.fill_(nbt)
        model.train(was_training)

@torch.no_grad()
def decode_bits_per_frame(video_bTchw: torch.Tensor, extractor) -> torch.Tensor:
    # video_bTchw: [B, T, 3, H, W]
    with test_time_bn(extractor):
        logits = extractor(video_bTchw)
    return (torch.sigmoid(logits) > 0.5).int()

def bit_acc_per_frame(pred_bits: torch.Tensor, gt_bits: torch.Tensor):
    assert pred_bits.shape == gt_bits.shape
    per_frame = (pred_bits == gt_bits).float().mean(dim=2).squeeze(0)  # [T]
    return per_frame, float(per_frame.mean().item())


def atk_gaussian(frames, sigma=0.05):
    return (frames + torch.randn_like(frames) * sigma).clamp(-1, 1)

def atk_blur_cv2(frames, ksize=7, sigma=1.5):
    out = []
    frames_np = ((frames + 1) * 127.5).cpu().numpy().astype(np.uint8)  # [T,3,H,W] 0..255
    for f in frames_np:
        bgr = cv2.cvtColor(f.transpose(1,2,0), cv2.COLOR_RGB2BGR)
        blur = cv2.GaussianBlur(bgr, (ksize, ksize), sigma)
        rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        out.append(torch.from_numpy(rgb).permute(2,0,1).float() / 127.5 - 1)
    return torch.stack(out, 0)

def atk_center_crop(frames, ratio=0.9):
    T, C, H, W = frames.shape
    new_h, new_w = int(H*ratio), int(W*ratio)
    out = []
    for f in frames:
        pil = TF.to_pil_image((f+1)/2)
        crop = TF.center_crop(pil, (new_h, new_w))
        resized = TF.resize(crop, (H, W), interpolation=_T.InterpolationMode.BICUBIC)
        out.append(TF.to_tensor(resized)*2 - 1)
    return torch.stack(out)

def atk_rotate(frames, angle=15):
    out = []
    for f in frames:
        pil = TF.to_pil_image((f+1)/2)
        rot = TF.rotate(pil, angle)
        out.append(TF.to_tensor(rot)*2 - 1)
    return torch.stack(out)

def atk_rescale(frames, down=0.5):
    out = []
    T, C, H, W = frames.shape
    for f in frames:
        pil = Image.fromarray(((f.clamp(-1,1)+1)/2*255).permute(1,2,0).byte().cpu().numpy())
        small = pil.resize((int(W*down), int(H*down)), Image.BICUBIC)
        up = small.resize((W, H), Image.BICUBIC)
        arr = torch.from_numpy(np.array(up)).permute(2,0,1).float()/127.5 - 1
        out.append(arr)
    return torch.stack(out, 0)


def atk_color_jitter(frames, strength=0.1):
    out = []
    for f in frames:
        alpha = random.uniform(1-strength, 1+strength)  # contrast
        beta  = random.uniform(-strength, strength)     # brightness
        f_new = (f * alpha + beta).clamp(-1, 1)
        out.append(f_new)
    return torch.stack(out)

def atk_crop_drop(frames, ratio=0.5, mode="drop"):
    out = []
    T, C, H, W = frames.shape
    for f in frames:
        pil = TF.to_pil_image((f+1)/2)
        if mode == "crop":
            left = random.randint(0, int(W*ratio))
            top = random.randint(0, int(H*ratio))
            right = left + int(W*(1-ratio))
            bottom = top + int(H*(1-ratio))
            crop = pil.crop((left, top, right, bottom))
            crop = TF.resize(crop, (H,W))
            out.append(TF.to_tensor(crop)*2-1)
        else:  # "drop" mode
            mask = torch.ones_like(f)
            w0 = random.randint(0, int(W*(1-ratio)))
            h0 = random.randint(0, int(H*(1-ratio)))
            mask[:, h0:h0+int(H*ratio), w0:w0+int(W*ratio)] = 0
            flipped = f * mask - f * (1-mask)  # flip content sign for visible change
            out.append(flipped)
    return torch.stack(out)


def atk_drop(frames: torch.Tensor, drop_ratio=0.5, seed=0):
    rng = random.Random(seed)
    T = frames.shape[0]
    keep = sorted(rng.sample(range(T), max(1, int(math.ceil(T*(1-drop_ratio))))))
    out = frames[keep]
    return out, {"type":"drop", "kept_idx": keep, "T_orig": T}

def atk_swap(frames: torch.Tensor, swap_fraction=0.5, seed=0):
    rng = random.Random(seed)
    T = frames.shape[0]
    idx = list(range(T))
    num_swaps = int(T*swap_fraction)
    for _ in range(num_swaps):
        i, j = rng.randrange(T), rng.randrange(T)
        idx[i], idx[j] = idx[j], idx[i]
    out = frames[idx]
    return out, {"type":"perm", "perm": idx}

def atk_swap_adjacent(frames: torch.Tensor):
    T = frames.shape[0]
    idx = list(range(T))
    for p in range(2, T-1, 4):
        idx[p], idx[p+1] = idx[p+1], idx[p]
    out = frames[idx]
    return out, {"type":"perm", "perm": idx}

def atk_insert(frames: torch.Tensor, seed=0, mode="duplicate"):
    rng = random.Random(seed)
    T = frames.shape[0]
    p = rng.randint(1, T-1)
    if mode == "duplicate":
        insert_frame = frames[p-1].clone()
    elif mode == "noise":
        insert_frame = torch.randn_like(frames[p-1]).clamp(-1,1)
    else:
        raise ValueError("mode must be 'duplicate' or 'noise'")
    out = torch.cat([frames[:p], insert_frame.unsqueeze(0), frames[p:]], dim=0)
    return out, {"type": "insert", "insert_idx": p, "T_orig": T}

def evaluate_temporal_detection(result, meta):
    if meta["type"] == "drop":
        Tg = meta["T_orig"]
        kept = set(meta["kept_idx"])
        dropped_gt = {i for i in range(Tg) if i not in kept}
        matched_gt = {gt for gt, pred in result["pairs_valid"]}
        dropped_pred = set(range(Tg)) - matched_gt

        # precision/recall on dropped detection
        tp = len(dropped_gt & dropped_pred)
        fp = len(dropped_pred - dropped_gt)
        fn = len(dropped_gt - dropped_pred)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2*prec*rec / (prec+rec+1e-8)
        return {
            "eval_type": "drop",
            "gt_dropped": sorted(list(dropped_gt)),
            "pred_dropped": sorted(list(dropped_pred)),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
        }
    if meta["type"] == "perm":
        # recovered order
        order = sorted(result["pairs_valid"], key=lambda p: p[1])
        recovered_gt_sequence = [gt for gt, _ in order]
        gt_perm = meta["perm"]    
        T = len(gt_perm)
        rank = {v: i for i, v in enumerate(recovered_gt_sequence)}

        total_pairs = 0
        correct_pairs = 0
        for i in range(T):
            for j in range(i+1, T):
                a, b = gt_perm[i], gt_perm[j]
                if a not in rank or b not in rank:
                    continue
                total_pairs += 1
                if rank[a] < rank[b]:
                    correct_pairs += 1
        order_acc = correct_pairs / max(1, total_pairs)
        return {
            "eval_type": "perm",
            "gt_perm": gt_perm,
            "recovered_gt_sequence": recovered_gt_sequence,
            "pairwise_order_accuracy": float(order_acc),
        }
    if meta["type"] == "insert":
        return {
            "eval_type": "insert",
            "insert_idx": meta["insert_idx"],
            "T_orig": meta["T_orig"],
            "detected_insert": True
        }
    return {"eval_type": "none"}


def save_video_tensor(frames, name, SAVE_DIR, fps=7):
    frames_pil = [TF.to_pil_image((f.clamp(-1,1) + 1) / 2) for f in frames]
    export_to_video(frames_pil, os.path.join(SAVE_DIR, f"{name}.mp4"), fps=fps)
    print(f"Saved {name}.mp4")

def _ffmpeg_has_encoder(name: str) -> bool:
    """Return True if ffmpeg -encoders lists `name`."""
    try:
        out = subprocess.check_output(["ffmpeg", "-encoders"], stderr=subprocess.STDOUT, text=True)
        return name in out
    except Exception:
        return False

def _jpeg_frame_recompress(frames, quality=30):
    """
    Fallback recompression using per-frame JPEG compression (pure-Python).
    frames: [T,3,H,W] in [-1,1] torch tensor
    Returns tensor [T,3,H,W] in [-1,1]
    """
    from io import BytesIO
    out_frames = []
    for f in frames:
        pil = TF.to_pil_image((f.clamp(-1,1)+1)/2)
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        pil2 = Image.open(buf).convert("RGB")
        out_frames.append(TF.to_tensor(pil2)*2 - 1)
    return torch.stack(out_frames, 0)


def atk_overlay_text(frames, text="Auto-caption", font_size=32, box_alpha=180):
    out = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    for f in frames:
        base = TF.to_pil_image((f+1)/2).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        w,h = base.size
        try:
            bbox = draw.textbbox((0,0), text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            try: tw, th = draw.textsize(text, font=font)
            except Exception: tw, th = font.getsize(text)
        x = (w - tw)//2; y = h - th - 40
        draw.rectangle((x-12,y-8,x+tw+12,y+th+8), fill=(0,0,0,box_alpha))
        draw.text((x,y), text, font=font, fill=(255,255,255,255))
        merged = Image.alpha_composite(base, overlay).convert("RGB")
        out.append(TF.to_tensor(merged)*2 - 1)
    return torch.stack(out, 0)

def atk_multistage_recompress(frames, crf1=28, bitrate2="600k", fps=7,
                             codec1="libx264", codec2="libx265"):
    """
    Multi-Stage Recompression:
      (1) H.264 CRF=28
      (2) decode + H.265 (libx265) at 600 kbps
    frames: [T,3,H,W] in [-1,1]
    returns (frames_out, meta)
    """
    device = frames.device
    T, C, H, W = frames.shape
    meta = {
        "applied": False,
        "stage1": {},
        "stage2": {},
        "frames_in": int(T),
        "frames_out": 0,
        "mean_abs_diff": 0.0,
        "fps": int(fps),
    }
    if shutil.which("ffmpeg") is None:
        meta["error"] = "ffmpeg not found"
        return frames, meta

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in   = os.path.join(tmpdir, "in.mp4")
        tmp_s1   = os.path.join(tmpdir, "s1.mp4")
        tmp_out  = os.path.join(tmpdir, "out.mp4")
        # write input video
        frames_pil = [TF.to_pil_image((f.clamp(-1,1)+1)/2) for f in frames]
        export_to_video(frames_pil, tmp_in, fps=fps)
        # ensure even dims
        even_w, even_h = (W // 2) * 2, (H // 2) * 2
        vf = f"scale={even_w}:{even_h}:flags=lanczos,fps={int(fps)}"
        # ---- Stage 1: H.264 CRF=28 ----
        cmd1 = [
            "ffmpeg","-hide_banner","-loglevel","error","-y",
            "-i", tmp_in,
            "-c:v", codec1, "-preset","veryfast", "-crf", str(int(crf1)),
            "-pix_fmt","yuv420p", "-vf", vf, "-an",
            "-movflags","+faststart", tmp_s1
        ]
        try:
            subprocess.run(cmd1, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            meta["stage1"]["cmd"] = " ".join(cmd1)
            meta["stage1"]["codec"] = codec1
            meta["stage1"]["crf"] = int(crf1)
        except subprocess.CalledProcessError as e:
            meta["stage1"]["err"] = e.stderr.decode("utf-8","ignore") if e.stderr else str(e)
            return frames, meta

        # ---- Stage 2: H.265 libx265 @ 600 kbps ----
        cmd2 = [
            "ffmpeg","-hide_banner","-loglevel","error","-y",
            "-i", tmp_s1,
            "-c:v", codec2,
            "-b:v", bitrate2, "-maxrate", bitrate2, "-bufsize", "1200k",
            "-pix_fmt","yuv420p", "-vf", vf, "-an",
            "-movflags","+faststart", tmp_out
        ]
        try:
            subprocess.run(cmd2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            meta["stage2"]["cmd"] = " ".join(cmd2)
            meta["stage2"]["codec"] = codec2
            meta["stage2"]["bitrate"] = bitrate2
        except subprocess.CalledProcessError as e:
            meta["stage2"]["err"] = e.stderr.decode("utf-8","ignore") if e.stderr else str(e)
            return frames, meta
        # decode final
        cap = cv2.VideoCapture(tmp_out)
        out_frames = []
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(fr).permute(2,0,1).float()/127.5 - 1.0
            out_frames.append(t)
        cap.release()
        if not out_frames:
            meta["error"] = "no frames decoded after multi-stage recompress"
            return frames, meta
        out_t = torch.stack(out_frames, 0).to(device)
        Tm = min(out_t.shape[0], T)
        meta["applied"] = True
        meta["frames_out"] = int(out_t.shape[0])
        meta["mean_abs_diff"] = float((out_t[:Tm] - frames[:Tm]).abs().mean().item())
        return out_t, meta

def atk_screen_record_approx(frames):
    """
    Approximate phone screen capture: downscale, add noise, vignette, recompress.
    frames: [T,3,H,W] in [-1,1]
    """
    T, C, H, W = frames.shape
    out = []
    for f in frames:
        down = F.interpolate(f.unsqueeze(0), scale_factor=0.7, mode="bilinear", align_corners=False)
        f = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
        f = (f + torch.randn_like(f) * 0.03).clamp(-1, 1)
        _, h, w = f.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=f.device),
            torch.linspace(-1, 1, w, device=f.device),
            indexing="ij",
        )
        vign = torch.exp(-2.0 * (xx**2 + yy**2))
        f = f * vign.unsqueeze(0)
        out.append(f)
    out = torch.stack(out, 0)
    return atk_recompress(out, bitrate="600k")

def atk_denoise(frames, ksize=3):
    T,C,H,W = frames.shape
    out=[]
    for f in frames:
        arr = ((f+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        arr = cv2.GaussianBlur(arr,(ksize,ksize),sigmaX=1)
        M = np.float32([[1,0,random.uniform(-2,2)],[0,1,random.uniform(-2,2)]])
        arr = cv2.warpAffine(arr,M,(W,H),borderMode=cv2.BORDER_REFLECT)
        out.append(torch.from_numpy(arr).permute(2,0,1).float()/127.5-1)
    return torch.stack(out)

def atk_trim(frames, trim_start=2, trim_end=2):
    T = frames.shape[0]
    out = frames[trim_start: T-trim_end]
    return out, {"type":"trim", "idx_start":trim_start, "idx_end":T-trim_end}


def atk_inpaint_sttn(frames: torch.Tensor, mask: torch.Tensor = None, model_path: str = "weights/sttn.pth", device="cuda"):
    model = STTN().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    T, C, H, W = frames.shape
    frames_in = frames.unsqueeze(0).to(device)  # [1,T,3,H,W]

    # build random rectangular mask if none provided
    if mask is None:
        mask = torch.zeros((1, T, 1, H, W), device=device)
        h0, w0 = random.randint(0, H//4), random.randint(0, W//4)
        h1, w1 = h0 + H//3, w0 + W//3
        mask[:, :, :, h0:h1, w0:w1] = 1.0

    with torch.no_grad():
        output = model(frames_in, mask)  # [1,T,3,H,W]
        out = output.clamp(-1, 1).squeeze(0)

    meta = {"type": "inpaint", "method": "STTN", "mask_sum": mask.sum().item()}
    return out, meta