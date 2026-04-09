import argparse
import logging
import math
import os, sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from assets import utils as utils
from pathlib import Path
import torch
import torch.nn.functional as F
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from huggingface_hub import  Repository, create_repo
from tqdm.auto import tqdm
os.environ["WANDB_START_METHOD"] = "thread"
import wandb
import lpips
from spdmark_routing_decoder_txt import inject_routing_blocks_into_video_vae_decoder, message_to_route_mask_video
logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="SPDMark_VAE_Res_Rank32_text",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="ali-vilab/text-to-video-ms-1.7b",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to OpenVid/data/train/OpenVid-1M.csv",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to OpenVid/video",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="MS-model-VAE",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--train_steps_per_epoch",
        type=int,
        default=1000,
        help="Number of training steps per epoch. If provided, limits the number of iterations for each epoch",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=6000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_restarts",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--cosine_cycle",
        type=int,
        default=1000,
        help=(
            "cosine_with_restarts option for cycle"
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
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
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def main():
    args = parse_args()
    wandb.init(name=args.exp_name, project="WatermarkLora")
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    metrics = []
    os.environ['WANDB_DISABLE_SERVICE'] = 'true'

    global accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = utils.get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )

    import copy
    vae_frozen = copy.deepcopy(vae)
    accelerator.print(" *** vae_frozen.")
    vae.requires_grad_(False)
    vae_frozen.requires_grad_(False)
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(accelerator.device)
    vae, num_resnet_blocks, total_slots = inject_routing_blocks_into_video_vae_decoder(vae, num_paths=4)
    extractor = utils.FrameWiseExtractor(total_slots)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    extractor.to(accelerator.device, dtype=weight_dtype)
    vae_frozen.to(accelerator.device, dtype=weight_dtype)

    lora_params_vae = utils.get_all_lora_params(vae.decoder)
    params = lora_params_vae + list(extractor.parameters())
    # Optimizer setup
    optimizer = optimizer_cls(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_loader, val_loader = utils.get_dataloader(args)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.train_steps_per_epoch is None:
        train_steps_per_epoch = num_update_steps_per_epoch
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * train_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles = args.cosine_cycle * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    vae, extractor, optimizer, train_loader, lr_scheduler = accelerator.prepare(
         vae, extractor, optimizer, train_loader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.train_steps_per_epoch is None:
        args.train_steps_per_epoch = num_update_steps_per_epoch
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * args.train_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = 24
    # math.ceil(args.max_train_steps / args.train_steps_per_epoch)
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2vid-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    # Setup all metrics
    for metric in metrics:
        metric.setup(accelerator, args)
    NUM_PATHS = 4
    BITS_PER_DECISION = int(math.log2(NUM_PATHS)) 
    TOTAL_DECISIONS =  num_resnet_blocks 
    BIT_LEN_VAE = BITS_PER_DECISION * TOTAL_DECISIONS
    for epoch in range(first_epoch, args.num_train_epochs):
        vae.train()
        extractor.train()
        local_step = 0
        train_loss = 0.0
        list_train_bit_acc = []


        for step, frames in enumerate(train_loader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(accelerator.unwrap_model(vae).decoder), accelerator.accumulate(extractor):
                bs, num_frames, c, h, w = frames.shape
                imgs_bt = frames.view(bs * num_frames, c, h, w).to(accelerator.device, dtype=weight_dtype)

                latents_bt = accelerator.unwrap_model(vae).encode(imgs_bt).latent_dist.mode()
                latents_bt = latents_bt * vae.config.scaling_factor


                msg_bits = utils.build_message_sequence(num_frames, total_bits=BIT_LEN_VAE).to(latents_bt.device)
                route_mask = message_to_route_mask_video(
                    msg_bits, num_resnets=num_resnet_blocks,
                    num_paths=NUM_PATHS
                )
                target_bits = msg_bits 
                latents_bt_for_decode = latents_bt / vae.config.scaling_factor

                # 3) [B,4,T,H',W']
                C_lat, H_lat, W_lat = latents_bt_for_decode.shape[1:]
                latents_bcthw = (
                    latents_bt_for_decode.view(bs, num_frames, C_lat, H_lat, W_lat)
                    .permute(0, 2, 1, 3, 4)
                    .contiguous()
                )
                out_routed = vae.decode_with_routing(latents_bcthw, routing_mask=route_mask, num_frames=num_frames)
                frames_bt = out_routed.sample                 # [B*T, 3, H, W]
                generated_bcthw = frames_bt.view(bs, num_frames, 3, h, w).permute(0, 2, 1, 3, 4).contiguous()

                out_frozen = vae_frozen.decode(latents_bt_for_decode)
                frozen_bt = out_frozen.sample                 # [B*T, 3, H, W]
                frozen_bcthw = frozen_bt.view(bs, num_frames, 3, h, w).permute(0, 2, 1, 3, 4).contiguous()

                gen_bTchw = generated_bcthw.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,3,H,W]
                fro_bTchw = frozen_bcthw.permute(0, 2, 1, 3, 4).contiguous()
                pred_logits = extractor(gen_bTchw)
                tgt = target_bits  # [B, T, BIT_LEN_VAE]

                loss_key = F.binary_cross_entropy_with_logits(pred_logits, tgt)

                pred_bits = (torch.sigmoid(pred_logits) > 0.5).int()
                gt_bits = tgt.int()
                bit_acc = (pred_bits.eq(gt_bits).float().mean(dim=(1, 2))).mean()
                list_train_bit_acc.append(bit_acc.item())

                gen_bt = gen_bTchw.view(bs * num_frames, 3, h, w)
                fro_bt = fro_bTchw.view(bs * num_frames, 3, h, w)
                loss_lpips_reg = loss_fn_vgg(gen_bt, fro_bt).mean()

                loss_temporal_smooth = utils.selective_temporal_loss(generated_bcthw, frozen_bcthw)
                if global_step < 2000:
                    loss = loss_key
                else: 
                    loss = loss_key + loss_lpips_reg  + loss_temporal_smooth
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_optimize = utils.get_params_optimize(
                        accelerator.unwrap_model(vae).decoder, extractor
                    )
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # sync/log
            if accelerator.sync_gradients:
                progress_bar.update(1)
                local_step += 1
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"key_loss": loss_key.item()}, step=global_step)
                accelerator.log({"loss_temporal": loss_temporal_smooth.item()}, step=global_step)
                accelerator.log({"lpips_reg": loss_lpips_reg.item()}, step=global_step)
                train_loss = 0.0

                if (global_step % args.checkpointing_steps) == 0:
                    torch.save(accelerator.unwrap_model(vae).decoder.state_dict(), args.output_dir + "/vae_decoder.pth")
                    torch.save(extractor.state_dict(), args.output_dir + "/extractor.pth")

            logs = {
                "loss_key": loss_key.detach().item(),
                "loss_lpips": loss_lpips_reg.item(),
                "loss_temporal": loss_temporal_smooth.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if local_step >= args.train_steps_per_epoch or global_step >= args.max_train_steps:
                break

        # epoch summary
        train_acc = torch.tensor(list_train_bit_acc).mean()
        print(f"Training Acc: Bit-wise Acc in Epoch {epoch}: {train_acc:.4f}")
        wandb.log({"Train Acc": train_acc.item()})
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        extractor = accelerator.unwrap_model(extractor)
        torch.save(accelerator.unwrap_model(vae).decoder.state_dict(), args.output_dir + "/vae_decoder.pth")
        torch.save(extractor.state_dict(), args.output_dir + "/extractor.pth")
        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()
    wandb.finish()

if __name__ == "__main__":
    main()










 




