#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Karcher_merge.py

This script merges 2-100 model weights (supports .safetensors and .bin formats)
using Riemannian (Karcher) mean fusion, with optional layer-specific alpha weights.

Usage example (Layered):
  python Karcher_merge.py --models modelA.safetensors modelB.safetensors \
    --alphas 0.5 0.5 \
    --alphas-te 0.1 0.9 \
    --alphas-unet-in 0.6 0.4 \
    --alphas-unet-mid 0.6 0.4 \
    --alphas-unet-out 0.2 0.8 \
    --output merged_model.safetensors --device cuda

Usage example (Uniform alphas):
  python Karcher_merge.py --models modelA.safetensors modelB.safetensors \
    --alphas 0.3 0.7 \
    --output merged_model.safetensors --device cuda
"""

import argparse
import os
import shutil
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file

###############################################################################
# General utilities: Reading, saving, and tensor alignment (resizing)
###############################################################################

class BinDataHandler:
    """Simple wrapper for .bin files providing get_tensor() and keys()."""
    def __init__(self, data):
        self.data = data

    def keys(self):
        return list(self.data.keys())

    def get_tensor(self, key):
        return self.data[key]

def read_tensors(file_path, device="cpu"):
    """
    Reads .safetensors or .bin files based on extension,
    returns (handler, key_set).
    """
    if file_path.endswith(".safetensors"):
        f = safe_open(file_path, framework="pt", device=device)
        return f, set(f.keys())
    elif file_path.endswith(".bin"):
        try:
            data = torch.load(file_path, map_location=torch.device(device), weights_only=True)
        except RuntimeError as e:
            print(f"[Warning] Failed to load with weights_only=True ({e}), trying without it. Ensure the file is from a trusted source.")
            data = torch.load(file_path, map_location=torch.device(device))
        except Exception as e:
            print(f"[Error] Failed to load model file {file_path}: {e}")
            raise
        f = BinDataHandler(data)
        return f, set(data.keys())
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def save_safetensors(tensor_dict, output_path):
    """Saves merged weights as safetensors file."""
    metadata = {"format": "pt"}
    try:
        save_file(tensor_dict, output_path, metadata=metadata)
    except Exception as e:
        print(f"[Error] Failed to save file to {output_path}: {e}")
        raise

def resize_tensors(t1, t2):
    """
    Aligns tensors using zero-pad if last two dimensions don't match.
    """
    # --- Zero Padding ---
    if len(t1.shape) < 2 or len(t2.shape) < 2 or t1.shape == t2.shape:
        return t1, t2

    shape1 = t1.shape
    shape2 = t2.shape

    h1, w1 = shape1[-2], shape1[-1]
    h2, w2 = shape2[-2], shape2[-1]

    pad_w = abs(w1 - w2)
    pad_h = abs(h1 - h2)

    padding1 = (0, pad_w if w1 < w2 else 0, 0, pad_h if h1 < h2 else 0)
    padding2 = (0, pad_w if w2 < w1 else 0, 0, pad_h if h2 < h1 else 0)

    if sum(padding1) > 0:
        t1 = F.pad(t1, padding1)
    if sum(padding2) > 0:
        t2 = F.pad(t2, padding2)

    if t1.shape != t2.shape:
        print(f"[Warning] Padding failed to align shapes for tensors with initial shapes {shape1} and {shape2}. Final shapes: {t1.shape} vs {t2.shape}. Skipping might occur.")
        return t1, t2

    # --- End Zero Padding ---
    return t1, t2

def copy_extra_files_if_needed(args):
    """
    If --copy-extra-files is specified, copies non-weight files
    (not ending with .bin, .safetensors or .pt) from the first model's
    directory to the output directory.
    """
    if not args.copy_extra_files or not args.models:
        return

    first_model_path = args.models[0]
    if os.path.isdir(first_model_path):
        first_dir = first_model_path
    elif os.path.isfile(first_model_path):
        first_dir = os.path.dirname(first_model_path)
    else:
        print(f"[Warning] First model path '{first_model_path}' is not a valid file or directory. Cannot copy extra files.")
        return

    out_dir = os.path.dirname(args.output)
    if not out_dir:
        out_dir = "."

    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
            print(f"[Info] Created output directory: {out_dir}")
        except OSError as e:
            print(f"[Error] Could not create output directory {out_dir}: {e}")
            return

    if os.path.isdir(first_dir) and os.path.isdir(out_dir):
        print(f"[Info] Attempting to copy extra files from {first_dir} to {out_dir}")
        copied_count = 0
        skipped_count = 0
        for fname in os.listdir(first_dir):
            fpath = os.path.join(first_dir, fname)
            if (os.path.isfile(fpath) and not fname.startswith(".") and
                not fname.lower().endswith((".bin", ".safetensors", ".pt", ".ckpt"))):
                tgt = os.path.join(out_dir, fname)
                try:
                    print(f"[Info] Copying extra file: {fname} -> {out_dir}")
                    shutil.copy2(fpath, tgt)
                    copied_count += 1
                except Exception as e:
                    print(f"[Error] Failed to copy {fname}: {e}")
                    skipped_count += 1
        print(f"[Info] Extra file copy complete. Copied: {copied_count}, Skipped/Failed: {skipped_count}")
    elif not os.path.isdir(first_dir):
        print(f"[Warning] Source directory {first_dir} does not exist. Cannot copy extra files.")

###############################################################################
# Algorithm: Karcher Mean-based Weight Fusion
###############################################################################

def karcher_merge_tensors(tensors, alphas, device, max_iter=10, tol=1e-5):
    """
    Fixed Karcher mean merging function using specified device.
    """
    if not tensors:
        raise ValueError("Tensor list cannot be empty.")
    if len(tensors) == 1:
        return tensors[0]
    if len(tensors) != len(alphas):
        raise ValueError(f"Number of tensors ({len(tensors)}) must match number of alphas ({len(alphas)}).")

    norms = []
    units = []
    ref_dtype = tensors[0].dtype
    ref_shape = tensors[0].shape

    for i, t in enumerate(tensors):
        if t.shape != ref_shape:
            print(f"[Warning] Mismatched shape detected in karcher_merge_tensors: tensor {i} shape {t.shape} vs ref shape {ref_shape}. Skipping layer is likely.")
            return torch.zeros(ref_shape, dtype=ref_dtype, device=device)

        # Compute on target device in float32
        t_float = t.to(device=device, dtype=torch.float32)

        n = torch.linalg.norm(t_float)
        n_val = n.item()

        if n_val < tol:
            norms.append(0.0)
            # Zero tensor with correct type and device
            units.append(torch.zeros_like(t_float, dtype=ref_dtype, device=device))
        else:
            norms.append(n_val)
            # Normalize on device, then cast back to original dtype
            unit_vec = (t_float / n).to(ref_dtype)
            units.append(unit_vec)

    # Keep only tensors with non-negligible norm
    valid_indices = [i for i, n_val in enumerate(norms) if n_val >= tol]

    if not valid_indices:
        return torch.zeros(ref_shape, dtype=ref_dtype, device=device)

    if len(valid_indices) == 1:
        idx = valid_indices[0]
        s = sum(a * n for a, n in zip(alphas, norms))
        return (s * units[idx]).to(ref_dtype)

    valid_alphas = [alphas[i] for i in valid_indices]
    alpha_sum = sum(valid_alphas)
    if alpha_sum < tol:
        print("[Warning] Sum of alphas for non-zero tensors is near zero. Returning zero tensor.")
        return torch.zeros(ref_shape, dtype=ref_dtype, device=device)
    normalized_alphas = [a / alpha_sum for a in valid_alphas]
    valid_units = [units[i] for i in valid_indices]

    # Initial guess: weighted mean of unit vectors (accumulate in float32)
    u = torch.zeros_like(valid_units[0], dtype=torch.float32, device=device)
    for a, ui in zip(normalized_alphas, valid_units):
        u += a * ui.to(torch.float32)

    norm_u = torch.linalg.norm(u)  # Compute norm on float32 tensor
    if norm_u.item() < tol:
        print("[Debug] Initial Karcher guess is near zero, starting from first valid unit vector.")
        u = valid_units[0].clone().to(torch.float32)
        norm_u = torch.linalg.norm(u)
        if norm_u.item() < tol:
            print("[Warning] First valid unit vector is near zero after clone. Returning zero tensor.")
            return torch.zeros(ref_shape, dtype=ref_dtype, device=device)
        u = u / norm_u
    else:
        u = u / norm_u

    u = u.to(ref_dtype)

    # Iterative Karcher mean computation
    for iter_count in range(max_iter):
        # Ensure T accumulates in float32 on the correct device
        T = torch.zeros_like(u, dtype=torch.float32, device=device)

        u_float = u.to(torch.float32)  # Use float32 for calculations within loop
        u_flat_float = u_float.flatten()  # Flatten for dot product

        for a, ui in zip(normalized_alphas, valid_units):
            ui_float = ui.to(torch.float32)
            ui_flat_float = ui_float.flatten()

            # Dot product on float32 vectors (clamped for stability)
            dot = torch.dot(u_flat_float, ui_flat_float)
            dot = torch.clamp(dot, -1.0, 1.0)

            theta = torch.arccos(dot)
            theta_val = theta.item()

            if theta_val < tol:
                continue

            sin_theta = torch.sin(theta)
            if sin_theta.item() < tol:
                print(f"[Debug] Skipping contribution for nearly opposite vectors (theta ~ pi, sin(theta) ~ 0), iter {iter_count}")
                continue

            # Tangent contribution in float32
            factor = a * (theta / sin_theta)
            tangent_component = ui_float - dot * u_float
            T += factor * tangent_component

        norm_T = torch.linalg.norm(T)
        norm_T_val = norm_T.item()

        if norm_T_val < tol:
            break

        cos_norm_T = torch.cos(norm_T)
        sin_norm_T = torch.sin(norm_T)

        if norm_T_val < tol:
            print("[Warning] norm_T is near zero before exp map update, skipping update.")
            u_new_float = u_float
        else:
            normalized_T = T / norm_T
            u_new_float = cos_norm_T * u_float + sin_norm_T * normalized_T

        norm_u_new = torch.linalg.norm(u_new_float)
        if norm_u_new.item() < tol:
            print(f"[Warning] Updated Karcher mean vector norm collapsed at iter {iter_count+1}. Keeping previous estimate.")
            u = u_float.to(ref_dtype)
            break
        else:
            u = (u_new_float / norm_u_new).to(ref_dtype)

    s = sum(a * n for a, n in zip(alphas, norms))
    final_tensor = (s * u).to(device=device, dtype=ref_dtype)
    return final_tensor

###############################################################################
# Main Program
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Layered Model merging script (Karcher mean fusion algorithm)"
    )
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Model files (.safetensors or .bin), supports 2-100 models")
    parser.add_argument("--output", type=str, default="merged_model.safetensors",
                        help="Output filename (safetensors format)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu",
                        help="Computing device (cpu or cuda)")

    # --- Alphas Arguments ---
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="Default weights for each model (equal weights if not specified). Used for layers not matching specific rules.")
    parser.add_argument("--alphas-te", type=float, nargs="+", default=None,
                        help="Specific weights for Text Encoder layers (e.g., conditioner.*)")
    parser.add_argument("--alphas-unet-in", type=float, nargs="+", default=None,
                        help="Specific weights for UNet Input Blocks (e.g., model.diffusion_model.input_blocks.*)")
    parser.add_argument("--alphas-unet-mid", type=float, nargs="+", default=None,
                        help="Specific weights for UNet Middle Block (e.g., model.diffusion_model.middle_block.*)")
    parser.add_argument("--alphas-unet-out", type=float, nargs="+", default=None,
                        help="Specific weights for UNet Output Blocks (e.g., model.diffusion_model.output_blocks.*)")

    # --- Other Arguments ---
    parser.add_argument("--copy-extra-files", action="store_true",
                        help="Copy non-weight files from first model's directory to output dir")
    parser.add_argument("--karcher-iter", type=int, default=10,
                        help="Maximum iterations for Karcher mean algorithm")
    parser.add_argument("--karcher-tol", type=float, default=1e-5,
                        help="Convergence tolerance for Karcher mean algorithm")
    parser.add_argument("--log-details", action="store_true",
                        help="Log the specific alphas used for each merged layer to a text file.")

    args = parser.parse_args()

    global_device = torch.device(args.device)
    print(f"[Info] Using device: {global_device}")

    N = len(args.models)
    if N < 2 or N > 100:
        raise ValueError("This script currently supports merging 2-100 models.")
    print(f"[Info] Merging {N} models.")

    # --- Alpha Validation and Normalization ---
    def validate_and_normalize_alphas(alpha_list, num_models, name="alphas"):
        if alpha_list is None:
            return None
        if len(alpha_list) != num_models:
            raise ValueError(f"Number of --{name} ({len(alpha_list)}) must match number of models ({num_models})")
        s = sum(alpha_list)
        if abs(s) < 1e-9:
            # Allow zero-sum for layer-specific alphas (warn)
            print(f"[Warning] --{name} sum to zero. This might lead to zero tensors for layers using these weights.")
            return alpha_list
        normalized = [x / s for x in alpha_list]
        print(f"[Info] Normalized --{name}: {list(map(lambda x: round(x, 3), normalized))}")
        return normalized

    if args.alphas is None:
        default_alphas = [1.0 / N] * N
        print(f"[Info] Using default equal alphas: {list(map(lambda x: round(x, 3), default_alphas))}")
    else:
        default_alphas = validate_and_normalize_alphas(args.alphas, N, "alphas")
        if default_alphas is None:
            raise ValueError("Default --alphas were provided but failed validation.")

    layer_alphas = {
        "te": validate_and_normalize_alphas(args.alphas_te, N, "alphas-te"),
        "unet_in": validate_and_normalize_alphas(args.alphas_unet_in, N, "alphas-unet-in"),
        "unet_mid": validate_and_normalize_alphas(args.alphas_unet_mid, N, "alphas-unet-mid"),
        "unet_out": validate_and_normalize_alphas(args.alphas_unet_out, N, "alphas-unet-out"),
    }

    # --- End Alpha Setup ---

    log_file_path = None
    if args.log_details:
        output_dir = os.path.dirname(args.output)
        output_basename = os.path.basename(args.output)
        log_filename = os.path.splitext(output_basename)[0] + "_merge_log.txt"
        if not output_dir:
            output_dir = "."
        log_file_path = os.path.join(output_dir, log_filename)
        print(f"[Info] Layer details will be logged to: {log_file_path}")
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                print(f"[Warning] Could not create output directory for log file {output_dir}: {e}. Log might fail.")

    handlers = []
    all_keys = []
    print("[Info] Loading models...")
    for i, mpath in enumerate(args.models):
        print(f"  Loading model {i+1}/{N}: {os.path.basename(mpath)}")
        try:
            h, keys = read_tensors(mpath, device=args.device)
            handlers.append((h, keys))
            all_keys.append(keys)
        except Exception as e:
            print(f"[Error] Failed to load model: {mpath}. Error: {e}")
            return 1
    print("[Info] Models loaded.")

    if not all_keys:
        print("[Error] No keys found in loaded models.")
        return 1

    common_keys = set.intersection(*all_keys)
    if not common_keys:
        print("[Error] No common keys found among the models. Cannot merge.")
        return 1

    print(f"[Info] Found {len(common_keys)} common keys for merging.")

    # --- Merging Loop ---
    merged_tensors = {}
    layer_log_data = {}
    skipped_count = 0
    print("[Info] Starting tensor merging process...")

    key_list = sorted(list(common_keys))

    from tqdm import tqdm

    for key in tqdm(key_list, desc="Merging Layers"):
        current_alphas = default_alphas
        alpha_source_name = "Default"
        key_lower = key.lower()

        # --- VAE Skip Logic ---
        if key_lower.startswith("first_stage_model."):
            try:
                vae_tensor = handlers[0][0].get_tensor(key).clone().cpu()
                merged_tensors[key] = vae_tensor
                if args.log_details:
                    layer_log_data[key] = ("Copied VAE from Model 1", None)
            except Exception as e:
                print(f"\n[Warning] Error loading VAE tensor for key '{key}' from first model. Skipping this key. Error: {e}")
                skipped_count += 1
                if args.log_details:
                    layer_log_data[key] = ("Skipped (VAE Load Error)", None)
            continue
        # --- End VAE Skip Logic ---

        if key_lower.startswith("conditioner.") and layer_alphas["te"] is not None:
            current_alphas = layer_alphas["te"]
            alpha_source_name = "TE"
        elif key_lower.startswith("model.diffusion_model.input_blocks.") and layer_alphas["unet_in"] is not None:
            current_alphas = layer_alphas["unet_in"]
            alpha_source_name = "UNet IN"
        elif key_lower.startswith("model.diffusion_model.middle_block.") and layer_alphas["unet_mid"] is not None:
            current_alphas = layer_alphas["unet_mid"]
            alpha_source_name = "UNet MID"
        elif key_lower.startswith("model.diffusion_model.output_blocks.") and layer_alphas["unet_out"] is not None:
            current_alphas = layer_alphas["unet_out"]
            alpha_source_name = "UNet OUT"

        weight_list = []
        try:
            for i, (handler, _) in enumerate(handlers):
                w = handler.get_tensor(key)
                weight_list.append(w.to(global_device))
        except Exception as e:
            print(f"\n[Warning] Error loading tensor for key '{key}'. Skipping this key. Error: {e}")
            skipped_count += 1
            if args.log_details:
                layer_log_data[key] = ("Skipped (Load Error)", None)
            continue

        if not all(t.is_floating_point() for t in weight_list):
            print(f"\n[Info] Key '{key}' has non-float tensors. Copying from first model.")
            merged_tensors[key] = weight_list[0].clone().cpu()
            if args.log_details:
                layer_log_data[key] = ("Copied from Model 1", None)
            continue

        if not all(t.is_floating_point() for t in weight_list):
            print(f"\n[Info] Key '{key}' has non-float tensors. Copying from first model.")
            merged_tensors[key] = weight_list[0].clone()
            continue

        try:
            ref_shape = weight_list[0].shape
            max_shape_dims = list(ref_shape)
            for i in range(1, N):
                current_shape = weight_list[i].shape
                if len(current_shape) >= 2 and len(max_shape_dims) >= 2:
                    max_shape_dims[-2] = max(max_shape_dims[-2], current_shape[-2])
                    max_shape_dims[-1] = max(max_shape_dims[-1], current_shape[-1])
            target_shape = tuple(max_shape_dims)

            needs_resize = False
            for i in range(N):
                if weight_list[i].shape != target_shape:
                    needs_resize = True
                    pass

            if needs_resize:
                temp_list = [w.clone() for w in weight_list]
                for i in range(1, N):
                    w0r, wir = resize_tensors(temp_list[0], temp_list[i])
                    temp_list[0] = w0r
                    temp_list[i] = wir
                weight_list = temp_list

        except Exception as e:
            print(f"\n[Warning] Error during resizing for key '{key}'. Skipping this key. Error: {e}")
            skipped_count += 1
            if args.log_details:
                layer_log_data[key] = ("Skipped (Resize Error)", None)
            continue

        final_shape = weight_list[0].shape
        if not all(w.shape == final_shape for w in weight_list):
            print(f"\n[Warning] Key '{key}' weight shapes inconsistent after resize ({[w.shape for w in weight_list]}). Skipping layer.")
            skipped_count += 1
            if args.log_details:
                layer_log_data[key] = ("Skipped (Shape Mismatch)", None)
            continue

        ref_dtype = weight_list[0].dtype
        try:
            for i in range(N):
                if weight_list[i].dtype != ref_dtype:
                    print(f"\n[Warning] Key '{key}': Converting tensor {i} from {weight_list[i].dtype} to {ref_dtype}.")
                    weight_list[i] = weight_list[i].to(ref_dtype)
        except Exception as e:
            print(f"\n[Warning] Error during dtype conversion for key '{key}'. Skipping this key. Error: {e}")
            skipped_count += 1
            if args.log_details:
                layer_log_data[key] = ("Skipped (Dtype Error)", None)
            continue

        try:
            merged = karcher_merge_tensors(weight_list, current_alphas,
                                           device=global_device,
                                           max_iter=args.karcher_iter,
                                           tol=args.karcher_tol)
            merged_tensors[key] = merged.cpu()
            if args.log_details:
                formatted_alphas = [round(a, 4) for a in current_alphas]
                layer_log_data[key] = (alpha_source_name, formatted_alphas)
        except Exception as e:
            print(f"\n[Error] Failed to merge key '{key}'. Skipping. Error: {e}")
            skipped_count += 1
            if args.log_details:
                layer_log_data[key] = ("Skipped (Merge Error)", None)
            continue

    print(f"\n[Info] Merging complete. Merged {len(merged_tensors)} layers. Skipped {skipped_count} layers due to errors or inconsistencies.")

    if args.log_details and log_file_path:
        print(f"[Info] Writing layer details to log file: {log_file_path}")
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write("--- Model Merge Log ---\n\n")
                f.write("Merged Models:\n")
                for i, model_path in enumerate(args.models):
                    f.write(f"  {i}: {os.path.basename(model_path)}\n")
                f.write("\n--- Layer Details ---\n")
                for key in sorted(layer_log_data.keys()):
                    source_or_action, alphas = layer_log_data[key]
                    if alphas is not None:
                        f.write(f"{key}: Alphas={alphas}, Source={source_or_action}\n")
                    elif source_or_action.startswith("Skipped"):
                        f.write(f"{key}: Action={source_or_action}\n")
                    elif source_or_action == "Copied from Model 1":
                        f.write(f"{key}: Action=Copied from Model 1 (non-float or merge error)\n")
                    else:
                        f.write(f"{key}: Status={source_or_action}\n")
                f.write(f"\n--- Summary ---\n")
                f.write(f"Total layers merged: {len(merged_tensors)}\n")
                f.write(f"Total layers skipped: {skipped_count}\n")
            print("[Info] Log file written successfully.")
        except Exception as e:
            print(f"[Error] Failed to write log file {log_file_path}: {e}")

    if not merged_tensors:
        print("[Error] No tensors were successfully merged. Cannot save output file.")
        return 1

    # --- Saving ---
    print(f"[Info] Saving merged model to {args.output}...")
    save_safetensors(merged_tensors, args.output)
    print(f"[Info] Output file saved successfully: {args.output}")

    # --- Copy Extra Files ---
    try:
        copy_extra_files_if_needed(args)
    except Exception as e:
        print(f"[Error] Failed during optional file copying: {e}")

    print("[Info] Script finished.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
