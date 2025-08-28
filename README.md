## 原創性聲明(Statement of Originality)
本人聲明，Karcher-merge包括idea及code均為本人（在o3-mini的輔助下）的個人智力成果。

Karcher-merge的目的是更好的模型合併，其核心思想是將神經網絡權重視為黎曼流形上的點，利用Karcher平均（也稱為Fréchet平均或幾何平均）來計算它們的內在平均值，同時考慮權重張量的方向和幅度。該方法將每個權重張量分解為方向分量（位於單位球面上）和幅度分量，通過叠代算法計算方向的Karcher平均，並使用加權算術平均計算合並幅度。這種幾何方法能有效處理正交或近正交權重，保持權重張量的方向信息和適當縮放，最後可以達到超越model stock等傳統模型合併方法的效果。本人之所以再次公開此聲明是因為sthenno（qq號為：1829321520）企圖借合作之名竊取本人成果來達成與阿里巴巴（qwen團隊？）的論文合作，在騙取我的代碼和論文後，sthenno(qq號為：1829321520)將我在各種社交媒體上拉黑（詳情見附件1，2），導致本人無法正常聲索本人智力成果的所有權。

本人從某些途徑(Sthenno本人與dc和qq群中)獲知，該論文已通過了阿里的內部流程。因此，本人迫不得已把已有代碼和論文主要思路公開，望周知。 

希望阿里巴巴相關工作人員看見本文後可盡快聯系本人（jmes100010@gmail.com，qq:3695314497）。

I hereby declare that Karcher-merge, including both the idea and code, is my personal intellectual achievement (with the assistance of o3-mini). The purpose of Karcher-merge is to achieve better model merging, with its core concept being the treatment of neural network weights as points on a Riemannian manifold, utilizing the Karcher mean (also known as Fréchet mean or geometric mean) to compute their intrinsic average while considering both the direction and magnitude of weight tensors. This method decomposes each weight tensor into a directional component (lying on the unit sphere) and a magnitude component, computing the Karcher mean of directions through an iterative algorithm and using weighted arithmetic mean for merged magnitude. This geometric approach effectively handles orthogonal or nearly orthogonal weights, preserves directional information in weight tensors with appropriate scaling, and ultimately achieves results that surpass traditional model averaging methods such as model stock. I am making this public declaration again because sthenno (QQ number: 1829321520) attempted to steal my work under the pretense of collaboration to establish a paper partnership with Alibaba (Qwen team?). After fraudulently obtaining my code and paper, sthenno (QQ number: 1829321520) blocked me on various social media platforms (see attachments 1 and 2 for details), preventing me from properly claiming ownership of my intellectual achievement.
I have learned through certain channels that this paper has passed through Alibaba's internal process. 
Therefore, I am compelled to make the existing code and the main ideas of the paper public. 
Please take note. I hope that relevant staff members at Alibaba who see this article can contact me as soon as possible (via email, phone).

See the evidence for details(# 證據(Evidence))

# Karcher merge
## Overview
`Karcher_merge.py` is a Python script for merging model weights using the Karcher mean, a concept from Riemannian geometry. It supports both `.safetensors` and `.bin` formats and allows merging up to 100 model weights.

## parper
See the paper:

[Karcher-paper](https://github.com/win10ogod/Karcher-merge/blob/main/Karcher-paper.pdf)

## Features
- Supports `.safetensors` and `.bin` model formats
- Uses Karcher mean for weighted averaging of model parameters
- Customizable weight distribution via `--alphas`
- Runs on `cpu` or `cuda`
- Optionally copies extra non-weight files from the first model’s directory

## Installation
Ensure you have Python 3 and install dependencies:
```bash
pip install torch safetensors
```

## How It Works
The script implements the Karcher mean method to merge model weights iteratively:

1. **Normalize and align tensors**: Ensures tensors have compatible shapes.
2. **Compute Karcher mean**: Uses Riemannian gradient descent to find the mean of tensors in the tangent space.

   Given tensors \( T_1, T_2, \dots, T_n \) and weights \( \alpha_1, \alpha_2, \dots, \alpha_n \), the Karcher mean \( T^* \) minimizes the sum of squared Riemannian distances:
   
   ```math
   T^* = \arg\min_T \sum_{i=1}^{n} \alpha_i d^2(T, T_i)
   ```
   
   where \( d(T, T_i) \) is the Riemannian distance between tensors, given by:
   
   ```math
   d(T, T_i) = \| \log(T^{-1} T_i) \|
   ```
   
   The algorithm iteratively updates \( T^* \) using gradient descent along the manifold:
   
   ```math
   T_{k+1} = \exp_{T_k} \left( \sum_{i=1}^{n} \alpha_i \log_{T_k} (T_i) \right)
   ```
   
3. **Rescale merged tensors**: Applies global scaling based on original tensor norms:
   
   ```math
   s = \sum_{i=1}^{n} \alpha_i \|T_i\|
   ```
   
   The final merged tensor is computed as:
   
   ```math
   T^* = s \cdot U
   ```
   
   where \( U \) is the unit-weighted mean computed in the tangent space.

4. **Save output**: Writes merged weights to a `.safetensors` file.

## Notes
- Ensure models have compatible architectures before merging.
- Large models may require substantial memory.
- Different `--alphas` values will influence how model weights are blended.

## License

This project is licensed under a custom license.  
**Commercial and academic use is strictly prohibited without explicit written permission.**  
See [LICENSE](./LICENSE) for full details.


## 算法由葉佐俊（dc ID:win100，我本人）版權所有，如若盜用必定追究責任，且將被學術界認定為垃圾！
I have to protect myself because of the actions of some people.

---

---

# Karcher Merge Fork Information
## Overview
#### ~~vibe coding warning~~
#### This modification of the Karcher mean script was used to create **[Karmix-XL v0](https://huggingface.co/chemwolf/Karmix-XL-v0)** [(Article)](https://rentry.co/-introducing-karcher-mean-experimental-model-v0) 

**Differences from original:**

- Layer-specific weights: adds `--alphas-te`, `--alphas-unet-in`, `--alphas-unet-mid`, `--alphas-unet-out` with validation/normalization.
- VAE handling: keys `first_stage_model.*` are copied from the first model instead of being merged.
- Numerical stability: `float32` accumulation on target device, clamped dot product, skip when `sin(theta) ≈ 0`, unit re-normalization per iteration.
- Non-float tensors: copied from the first model (e.g., int/bool buffers) instead of merging.
- Shape alignment: zero-padding of the last two dimensions with extra checks/warnings; skip layer on mismatch.
- Logging: `--log-details` writes a per-layer merge log (alpha source, copied/skipped layers).
- Progress: `tqdm` progress bar during merging.
- `.bin` loading: attempts `torch.load(..., weights_only=True)` with safe fallback and clearer errors.
- Memory/saving: merged tensors moved to CPU before saving to reduce VRAM usage spikes.
- Extra files: more robust copying `(shutil.copy2)`, excludes `.bin`/`.safetensors`/`.pt`/`.ckpt`, creates output dir, reports counts.
- CLI changes: new flags, explicit exit codes (0/1).
- Dependency: requires `tqdm` for progress display.

## Usage
Usage example (Layered):
```bash
python Karcher_merge.py --models modelA.safetensors modelB.safetensors \
    --alphas 0.5 0.5 \
    --alphas-te 0.1 0.9 \
    --alphas-unet-in 0.6 0.4 \
    --alphas-unet-mid 0.6 0.4 \
    --alphas-unet-out 0.35 0.65 \
    --output merged_model.safetensors --device cuda --karcher-iter 10 --karcher-tol 1e-5
```
Usage example (Uniform alphas):
```bash
python Karcher_merge.py --models modelA.safetensors modelB.safetensors \
    --alphas 0.3 0.7 \
    --output merged_model.safetensors --device cuda --karcher-iter 10 --karcher-tol 1e-5
```

### Arguments
| Argument | Description |
|----------|-------------|
| `--models` | List of model weight files to merge (2-100 files) |
| `--alphas` | Weight coefficients for merging (default: equal weights) |
| `--device` | Compute device: `cpu` or `cuda` (default: `cpu`) |
| `--output` | Output filename (default: `merged_model.safetensors`) |
| `--copy-extra-files` | Copy additional non-weight files from first model |
| `--karcher-iter` | Maximum iterations for Karcher mean computation (default: 10) |
| `--karcher-tol` | Convergence tolerance for Karcher mean algorithm (default: `1e-5`) |
| `--alphas-te` | Specific weights for Text Encoder layers (e.g., `conditioner.*`) |
| `--alphas-unet-in` | Specific weights for UNet Input Blocks (e.g., `model.diffusion_model.input_blocks.*`) |
| `--alphas-unet-mid` | Specific weights for UNet Middle Block (e.g., `model.diffusion_model.middle_block.*`) |
| `--alphas-unet-out` | Specific weights for UNet Output Blocks (e.g., `model.diffusion_model.output_blocks.*`) |
| `--log-details` | Log the specific alphas used for each merged layer to a text file. |

# Special Thanks:
- **[win10ogod](https://github.com/win10ogod)** for his work and for the [Karcher mean](https://github.com/win10ogod/Karcher-merge) merge method script. 
- **su momo** from [SDCN](https://t.me/StableDiffusion_CN) for creating this modification, implementing detailed logging, VAE fix and Alphas ratio management for **TE/UNet-in/UNet-mid/UNet-out.**
