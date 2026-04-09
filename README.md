
# SPDMark: Selective Parameter Displacement for Robust Video Watermarking [CVPR 2026]

Official code release for **SPDMark**, a robust video watermarking framework based on selective parameter displacement in the decoder of video generation models.
---

### Folder description

- `assets/`  
  Shared utilities for data loading, key/message construction, extraction, and robustness evaluation.

- `ModelScope/`  
  Text-to-video SPDMark generation and training code.

- `SVD_xt/`  
  Image-to-video SPDMark generation and training code.

---

## Installation

Create and activate your environment, then install the required packages:

```bash
pip install -r requirements.txt
```

---
## Text-to-Video (ModelScope)

### Training

Train the SPDMark decoder and extractor for text-to-video:

```bash
python ModelScope/train_spdmark_decoder_txt.py \
  --metadata_path /path/to/OpenVid-1M.csv \
  --data_dir /path/to/OpenVid/videos \
```

### Watermarked Generation

Generate watermarked videos from text prompts:

```bash
python ModelScope/generate_watermarked_txt2videos.py \
  --decoder_path /path/to/decoder.pt \
  --extractor_path /path/to/extractor.pt \
  --prompts_txt_file /path/to/prompts.txt \
  --save_dir ./videos/ModelScope
```



---

## Image-to-Video (SVD-XT)

### Training

Train the SPDMark decoder and extractor for image-to-video:

```bash
python SVD_xt/train_spdmark_decoder_img.py \
  --metadata_path /path/to/OpenVid-1M.csv \
  --data_dir /path/to/OpenVid/videos \
```

To enable attention routing:

```bash
python SVD_xt/train_spdmark_decoder_img.py \
  --metadata_path /path/to/OpenVid-1M.csv \
  --data_dir /path/to/OpenVid/videos \
  --output_dir ./checkpoints \
  --exp_name SPDMark_VAE_Res_Rank32_image \
  --enable_attention_routing
```

### Watermarked Generation

Generate watermarked videos from input images:

```bash
python SVD_xt/generate_watermarked_img2videos.py \
  --decoder_path /path/to/decoder.pt \
  --extractor_path /path/to/extractor.pt \
  --images_file /path/to/images.txt \
  --save_dir ./videos/SVD_xt
```
---

## Robustness Evaluation

Robustness utilities are implemented in:

```text
assets/evaluate_robustness_full.py
```

The generation scripts automatically evaluate watermark robustness under multiple attacks.

Saved outputs include attacked videos and JSON summaries of detection results.

---

## Checkpoints

Before running generation, you need:

- pretrained backbone checkpoint from Hugging Face
- trained SPDMark decoder weights
- trained extractor weights

Please update command paths according to your local setup.

---

## Notes

- Run scripts from the repository root.
- The repository assumes CUDA-based execution for generation and training.
- Some scripts use `wandb` for experiment logging.
- The provided `requirements.txt` reflects the working environment used for development and may include extra packages beyond the minimum runtime set.

---

## Citation

If you find this repository useful, please cite the paper:
```bibtex
@inproceedings{fares2026spdmark,
  title={{SPDM}ark: Selective Parameter Displacement for Robust Video Watermarking},
  author={Samar Fares and Nurbek Tastan and Karthik Nandakumar},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026},
}
```
