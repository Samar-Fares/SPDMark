import argparse
import os, json, sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from assets import utils as utils
from assets import evaluate_robustness_full as eval_utils
import torch
from spdmark_i2v_pipeline import StableVideoDiffusionPipeline
from torchvision import transforms
import numpy as np
from diffusers.models import AutoencoderKLTemporalDecoder
from spdmark_routing_decoder_img import  inject_routing_blocks_into_video_vae_decoder, message_to_route_mask_video
import glob
import math
from PIL import Image



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--decoder_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--extractor_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default='videos/SVD',
        required=False,
    )
    parser.add_argument(
        "--images_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args

def main():
    device = "cuda"
    args = parse_args()
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    vae, num_resnet_blocks, num_attention_blocks, total_slots = inject_routing_blocks_into_video_vae_decoder(vae, num_paths=4, enable_attention=False)
    state_dict = torch.load(args.decoder_path, map_location="cpu")
    vae.decoder.load_state_dict(state_dict)
    vae.requires_grad_(False)

    weight_dtype = torch.float32
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        revision=args.revision,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)

    NUM_PATHS = 4
    TOTAL_SLOTS = num_resnet_blocks + 3 * num_attention_blocks  
    num_frames = 25
    BITS_PER_DECISION = int(math.log2(NUM_PATHS))  # = 2 for 4 paths
    TOTAL_DECISIONS =  num_resnet_blocks + 3 * num_attention_blocks  # total_slots
    BIT_LEN_VAE = BITS_PER_DECISION * TOTAL_DECISIONS


    extractor = utils.FrameWiseExtractor(TOTAL_SLOTS)
    state_dict = torch.load(args.extractor_path, map_location="cpu")
    extractor.load_state_dict(state_dict)
    extractor.to(device)



    BASE_SAVE_DIR = args.save_dir
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    water_save_dir = os.path.join(BASE_SAVE_DIR+'/water')
    os.makedirs(water_save_dir, exist_ok=True)


    all_results = []  


    for seed in [42, 0, 1789, 2025]:
        for i in range(2):
            img_path = f'{args.images_file}/{i}.png'
            img_name = i
            SAVE_DIR = os.path.join(BASE_SAVE_DIR+f'/{i}')
            os.makedirs(SAVE_DIR, exist_ok=True)
            attacked_img_save_dir = os.path.join(SAVE_DIR, "attacked_image_attacks")
            attacked_vid_save_dir = os.path.join(SAVE_DIR, "attacked_video_attacks")
            os.makedirs(attacked_img_save_dir, exist_ok=True)
            os.makedirs(attacked_vid_save_dir, exist_ok=True)
            print(f"\n=== Processing prompt image: {img_name}.png ===")
            image = Image.open(img_path).convert("RGB").resize((1024, 576))
            base_key = utils.derive_base_key(seed, img_name)   
            msg_bits = utils.build_message_sequence(
                num_frames, total_bits=BIT_LEN_VAE,
                base_key=base_key,
            ).to("cuda")
            routing_mask_vae = message_to_route_mask_video(
                msg_bits, num_resnets=num_resnet_blocks, num_attns=num_attention_blocks, num_paths=4,
                per_block=True
            )
            stem = f"{seed}_{img_name}"
            # ---- Save GT bits + key ----
            torch.save(msg_bits.detach().cpu(), os.path.join(SAVE_DIR, f"{stem}_msg_bits.pt"))
            with open(os.path.join(SAVE_DIR, f"{stem}_base_key.bin"), "wb") as f:
                f.write(base_key)

            # ---- Save metadata needed for later evaluation ----
            meta = {
                "seed": int(seed),
                "img_id": int(img_name),
                "stem": stem,
                "num_frames_requested": int(num_frames),
                "NUM_PATHS": int(NUM_PATHS),
                "TOTAL_SLOTS": int(TOTAL_SLOTS),
                "num_resnet_blocks": int(num_resnet_blocks),
                "num_attention_blocks": int(num_attention_blocks),
                "BITS_PER_DECISION": int(BITS_PER_DECISION),
                "BIT_LEN_VAE": int(BIT_LEN_VAE),
            }
            json.dump(meta, open(os.path.join(SAVE_DIR, f"{stem}_meta.json"), "w"), indent=2)
            generator = torch.manual_seed(seed)
            water_frames = pipe(
                image=image, name='water',
                routing_mask_vae=routing_mask_vae, num_frames=num_frames,
                decode_chunk_size=8, generator=generator,
            ).frames[0]
            def preprocess_frames(frames):
                frames_tensor = torch.stack([
                    transforms.ToTensor()(f) * 2 - 1 for f in frames
                ], dim=0)
                return frames_tensor.unsqueeze(0)
            water_video_tensor = preprocess_frames(water_frames).to("cuda")
            torch.save(water_video_tensor.squeeze(0).detach().cpu(), os.path.join(SAVE_DIR, f"{stem}_water_tensor.pt"))
            gt_bits = msg_bits.int()
            pred_bits = eval_utils.decode_bits_per_frame(water_video_tensor, extractor)
            per_frame_acc, avg_acc = eval_utils.bit_acc_per_frame(pred_bits, gt_bits.int())
            print("BitAcc per frame:", per_frame_acc.cpu().numpy().round(3).tolist())
            print(f"BitAcc avg: {avg_acc*100:.2f}%")
            attacks_img = {
                "none":        lambda x: x,
                "gaussian":    lambda x: eval_utils.atk_gaussian(x, sigma=0.05),
                "blur":        lambda x: eval_utils.atk_blur_cv2(x, ksize=11, sigma=2.0),
                "crop":        lambda x: eval_utils.atk_center_crop(x, ratio=0.9),
                "rotate":      lambda x: eval_utils.atk_rotate(x, angle=15),
                "rescale":     lambda x: eval_utils.atk_rescale(x, down=0.5),
                "colorjitter": lambda x: eval_utils.atk_color_jitter(x, strength=0.1),
                "cropdrop":    lambda x: eval_utils.atk_crop_drop(x, ratio=0.5, mode="crop"),
                "denoise":      lambda x: eval_utils.atk_denoise(x),
            }
            attacks_vid = {
                "drop_50":        lambda x: eval_utils.atk_drop(x, drop_ratio=0.5, seed=123),
                "insert_dup":     lambda x: eval_utils.atk_insert(x, seed=123, mode="duplicate"),
                "insert_noise":   lambda x: eval_utils.atk_insert(x, seed=123, mode="noise"),
                "swap_random":    lambda x: eval_utils.atk_swap(x, swap_fraction=0.5, seed=123),
                "swap_adjacent":  lambda x: eval_utils.atk_swap_adjacent(x),
            }
            attacks_img.update({
                "recomp": lambda x: eval_utils.atk_multistage_recompress(x, crf1=28, bitrate2="600k"),
                "subtitle": lambda x: eval_utils.atk_overlay_text(x, text="Auto-generated captions"),
                "screenrec": lambda x: eval_utils.atk_screen_record_approx(x),
            })
            attacks_vid.update({
                "trim": lambda x: eval_utils.atk_trim(x, trim_start=2, trim_end=2),
            })
            FPR_FRAME = 0.001        # per-frame false alarm target
            FPR_VIDEO = 0.001        # video-level false alarm target
            results = {}
            # --------------------- IMAGE ATTACKS ---------------------
            for name, fn in attacks_img.items():
                result = fn(water_video_tensor.squeeze(0))
                if isinstance(result, tuple):
                    attacked, meta = result
                else:
                    attacked, meta = result, {}
                with torch.no_grad():
                    logits = extractor(attacked.unsqueeze(0).to("cuda"))
                    pred_bits_att = (torch.sigmoid(logits) > 0.5).int()
                det = eval_utils.eval_alignment_and_detection(
                    gt_bits.int(), pred_bits_att.int(),
                    M_target=BIT_LEN_VAE, fpr_frame=FPR_FRAME, fpr_video=FPR_VIDEO
                )
                results[f"img_{name}"] = det
                if meta:
                    results[f"img_{name}"]["meta"] = meta
                eval_utils.save_video_tensor(attacked, f"{img_name}_img_{name}", attacked_img_save_dir)
                print(
                    f"\n[IMG] {name}: watermarked={det['is_watermarked']}, "
                    f"tamper={(det['num_valid'] < det['T_gt']) or (not det['is_in_order'])}, "
                    f"valid={det['num_valid']}/{det['T_gt']}, "
                    f"avg_valid_sim={det['avg_valid_sim']:.3f} (defined={det['bitacc_defined']})"
                )

            # --------------------- VIDEO ATTACKS ---------------------
            for name, fn in attacks_vid.items():
                attacked, meta = fn(water_video_tensor.squeeze(0).to("cuda"))
                with torch.no_grad():
                    logits = extractor(attacked.unsqueeze(0).to("cuda"))
                    pred_bits_att = (torch.sigmoid(logits) > 0.5).int()
                det = eval_utils.eval_alignment_and_detection(
                    gt_bits.int(), pred_bits_att.int(),
                    M_target=BIT_LEN_VAE, fpr_frame=FPR_FRAME, fpr_video=FPR_VIDEO
                )
                eval_meta = eval_utils.evaluate_temporal_detection(det, meta) 
                results[f"vid_{name}"] = {"detect": det, "eval": eval_meta}
                print(f"\n[VID] {name}: watermarked={det['is_watermarked']}, {eval_meta}")
                eval_utils.save_video_tensor(attacked, f"{img_name}_vid_{name}", attacked_vid_save_dir)
            # Save per-video results
            json.dump(results, open(os.path.join(SAVE_DIR, f"tamper_eval.json"), "w"), indent=2)
            eval_utils.save_video_tensor(water_video_tensor.squeeze(0), f"{seed}_{img_name}_watermarked", water_save_dir)


        all_results = {}
        json_paths = sorted(glob.glob(os.path.join(BASE_SAVE_DIR, "*/tamper_eval.json")))

        for path in json_paths:
            with open(path, "r") as f:
                res = json.load(f)

            for attack_name, info in res.items():
                if attack_name not in all_results:
                    all_results[attack_name] = {
                        "avg_valid_sim": [],
                        "avg_valid_sim_defined": [],  
                        "num_valid": [],
                        "T_gt": [],
                        "is_watermarked": [],
                        "pairwise_order_accuracy": [],
                        "precision": [],
                        "recall": [],
                        "f1": []
                    }

                entry = info
                if attack_name.startswith("vid_") and isinstance(info, dict) and "detect" in info:
                    entry = info["detect"]

                if isinstance(entry, dict) and "is_watermarked" in entry:
                    all_results[attack_name]["is_watermarked"].append(int(entry["is_watermarked"]))

                if "avg_valid_sim" in entry:
                    all_results[attack_name]["avg_valid_sim"].append(entry["avg_valid_sim"])
                    all_results[attack_name]["avg_valid_sim_defined"].append(
                        int(entry.get("bitacc_defined", entry.get("num_valid", 0) > 0))
                    )

                if "num_valid" in entry:
                    all_results[attack_name]["num_valid"].append(entry["num_valid"])
                if "T_gt" in entry:
                    all_results[attack_name]["T_gt"].append(entry["T_gt"])

                if attack_name.startswith("vid_") and isinstance(info, dict) and "eval" in info:
                    eval_ = info["eval"]
                    if eval_.get("eval_type") == "drop":
                        for k in ["precision", "recall", "f1"]:
                            if k in eval_:
                                all_results[attack_name][k].append(eval_[k])
                    if eval_.get("eval_type") == "perm":
                        if "pairwise_order_accuracy" in eval_:
                            all_results[attack_name]["pairwise_order_accuracy"].append(
                                eval_["pairwise_order_accuracy"]
                            )
        summary = {}
        for attack_name, vals in all_results.items():
            summary[attack_name] = {
                k: float(np.mean(v))
                for k, v in vals.items()
                if len(v) > 0 and k not in ["avg_valid_sim", "avg_valid_sim_defined", "is_watermarked"]
            }

            if len(vals["avg_valid_sim"]) > 0:
                av = np.array(vals["avg_valid_sim"], float)
                mask = np.array(vals["avg_valid_sim_defined"], int) == 1
                if mask.sum() > 0:
                    summary[attack_name]["avg_valid_sim"] = float(av[mask].mean())
                summary[attack_name]["no_valid_rate"] = float(1.0 - mask.mean())

            # attacked-watermarked detection rate
            if len(vals["is_watermarked"]) > 0:
                preds = np.array(vals["is_watermarked"], int)
                summary[attack_name]["DetectionRate"] = float(preds.mean())

        print("\n=== AVERAGE RESULTS ACROSS VIDEOS ===")
        for k, v in summary.items():
            print(f"{k:18s} -> " + ", ".join([f"{kk}: {vv:.3f}" for kk, vv in v.items()]))

        json.dump(summary, open(os.path.join(BASE_SAVE_DIR, f"avg_summary_{seed}.json"), "w"), indent=2)
        print(f"Saved overall summary to {os.path.join(BASE_SAVE_DIR, f'avg_summary_{seed}.json')}")
if __name__ == "__main__":
    # exit()
    main()

