"""Utilities for mixed training strategies: LoRA on old components, full training on new heads."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Set, Any
import torch
import torch.nn as nn
from safetensors.torch import load_file


VALUE_FUNCTION_COMPONENT_KEYS = {
    "value_function_encoder",
    "value_function_decoder",
}


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    """Normalize checkpoint payload into a plain state_dict."""
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")

    # Already a raw state_dict.
    if payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload

    # Common wrappers used by training frameworks.
    for key in ("state_dict", "model_state_dict", "model", "module"):
        wrapped = payload.get(key)
        if isinstance(wrapped, dict) and wrapped and all(isinstance(v, torch.Tensor) for v in wrapped.values()):
            return wrapped

    raise ValueError(
        "Could not find a tensor state_dict in checkpoint payload. "
        f"Top-level keys: {list(payload.keys())[:20]}"
    )


def _load_checkpoint_payload(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load checkpoint from file or directory (supports sharded safetensors)."""
    if os.path.isdir(checkpoint_path):
        index_candidates = [
            "model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
        ]
        for index_name in index_candidates:
            index_path = os.path.join(checkpoint_path, index_name)
            if os.path.exists(index_path):
                print(f"Loading sharded safetensors from index: {index_path}")
                with open(index_path, "r") as f:
                    index = json.load(f)
                state_dict: Dict[str, torch.Tensor] = {}
                for shard_file in sorted(set(index["weight_map"].values())):
                    shard_path = os.path.join(checkpoint_path, shard_file)
                    print(f"Loading shard: {shard_path}")
                    state_dict.update(load_file(shard_path))
                return state_dict

        safetensor_candidates = [
            "model.safetensors",
            "diffusion_pytorch_model.safetensors",
            "pytorch_model.safetensors",
        ]
        for filename in safetensor_candidates:
            file_path = os.path.join(checkpoint_path, filename)
            if os.path.exists(file_path):
                print(f"Loading safetensors checkpoint: {file_path}")
                return load_file(file_path)

        torch_candidates = [
            "pytorch_model.bin",
            "model.pt",
            "model.pth",
            "checkpoint.pt",
            "checkpoint.pth",
        ]
        for filename in torch_candidates:
            file_path = os.path.join(checkpoint_path, filename)
            if os.path.exists(file_path):
                print(f"Loading torch checkpoint: {file_path}")
                payload = torch.load(file_path, map_location="cpu")
                return _extract_state_dict(payload)

        preview = sorted(os.listdir(checkpoint_path))[:20]
        raise FileNotFoundError(
            f"No supported checkpoint file found in directory: {checkpoint_path}. "
            f"Directory preview: {preview}"
        )

    if checkpoint_path.endswith(".safetensors"):
        return load_file(checkpoint_path)

    payload = torch.load(checkpoint_path, map_location="cpu")
    return _extract_state_dict(payload)


def _extract_model_state_dict(model: nn.Module, full_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract/normalize keys so they match `model.state_dict()` namespace."""
    model_keys = set(model.state_dict().keys())

    # Already a direct match.
    overlap = sum(1 for k in full_state_dict.keys() if k in model_keys)
    if overlap > 0:
        print(f"Using direct checkpoint keys ({overlap} matching parameters)")
        return full_state_dict

    # Try common wrapping prefixes in full checkpoints.
    prefix_candidates = [
        "action_head.model.",
        "module.action_head.model.",
        "model.",
        "module.model.",
    ]

    best_state_dict: Dict[str, torch.Tensor] = {}
    best_overlap = 0
    best_prefix = None

    for prefix in prefix_candidates:
        candidate = {
            key[len(prefix):]: value
            for key, value in full_state_dict.items()
            if key.startswith(prefix)
        }
        if not candidate:
            continue
        candidate_overlap = sum(1 for k in candidate.keys() if k in model_keys)
        if candidate_overlap > best_overlap:
            best_overlap = candidate_overlap
            best_state_dict = candidate
            best_prefix = prefix

    if best_overlap > 0:
        print(
            f"Using checkpoint prefix '{best_prefix}' "
            f"({best_overlap} matching parameters after stripping prefix)"
        )
        return best_state_dict

    sample_keys = list(full_state_dict.keys())[:20]
    sample_model_keys = list(model_keys)[:20]
    raise ValueError(
        "Could not map checkpoint keys to model keys. "
        f"Checkpoint sample keys: {sample_keys}. "
        f"Model sample keys: {sample_model_keys}."
    )


def get_component_keys_from_checkpoint(checkpoint_path: str) -> Set[str]:
    """Extract available component keys from a checkpoint."""
    state_dict = _load_checkpoint_payload(checkpoint_path)
    
    # Extract top-level component names (e.g., "value_function_encoder", "model.blocks.0", etc.)
    component_keys = set()
    for key in state_dict.keys():
        parts = key.split(".")
        if len(parts) > 0:
            component_keys.add(parts[0])
    
    return component_keys


def load_pretrained_checkpoint_selective(
    model: nn.Module,
    checkpoint_path: str,
    exclude_keys: Optional[Set[str]] = None,
    skip_missing: bool = True,
) -> Dict[str, List[str]]:
    """
    Load pretrained checkpoint while excluding specific components.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to pretrained checkpoint
        exclude_keys: Set of component keys to skip loading (e.g., {"value_function_encoder"})
        skip_missing: If True, don't fail if some keys are missing in checkpoint
        
    Returns:
        Dict with 'missing_keys' and 'unexpected_keys' lists
    """
    if exclude_keys is None:
        exclude_keys = set()
    
    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Excluding components: {exclude_keys}")
    
    # Load checkpoint (supports file path or directory path)
    raw_state_dict = _load_checkpoint_payload(checkpoint_path)
    full_state_dict = _extract_model_state_dict(model, raw_state_dict)
    
    # Filter out excluded keys
    filtered_state_dict = {}
    excluded_count = 0
    for key, value in full_state_dict.items():
        # Check if this key belongs to an excluded component
        component = key.split(".")[0]
        if component in exclude_keys:
            excluded_count += 1
            continue
        filtered_state_dict[key] = value
    
    print(f"Excluded {excluded_count} keys belonging to excluded components")
    
    # Load filtered state dict
    if skip_missing:
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=True)
    
    result = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}
    
    if missing_keys:
        print(f"Missing keys: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")
    
    return result


def apply_mixed_training_strategy(
    model: nn.Module,
    backbone_strategy: str = "lora",
    train_value_function: bool = True,
    train_legacy_action_heads: bool = False,
    freeze_backbone: bool = False,
    add_lora_fn=None,
    lora_rank: int = 4,
    lora_alpha: int = 4,
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2",
    init_lora_weights: str = "kaiming",
) -> nn.Module:
    """
    Apply mixed training strategy: selectively train different components.
    
    Args:
        model: The model to configure
        backbone_strategy: One of 'lora' (LoRA), 'full' (full training), 'frozen' (no gradients)
        train_value_function: Whether value function head should be trainable
        freeze_backbone: If True, freeze all backbone components (ignores backbone_strategy)
        add_lora_fn: Function to apply LoRA (must accept model and config)
        lora_rank: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_target_modules: Comma-separated list of module patterns to apply LoRA to
        init_lora_weights: LoRA weight initialization strategy
        
    Returns:
        Modified model with appropriate requires_grad settings
    """
    print(f"\nApplying mixed training strategy:")
    print(f"  backbone_strategy: {backbone_strategy}")
    print(f"  train_value_function: {train_value_function}")
    print(f"  train_legacy_action_heads: {train_legacy_action_heads}")
    print(f"  freeze_backbone: {freeze_backbone}")
    
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Handle backbone training strategy
    if freeze_backbone:
        print("  → Freezing entire backbone (ignore backbone_strategy)")
        # All params already frozen, nothing to do
    elif backbone_strategy == "lora":
        print("  → Applying LoRA to backbone")
        if add_lora_fn is None:
            raise ValueError("add_lora_fn must be provided for LoRA strategy")
        
        # Apply LoRA (this will set requires_grad=True for LoRA params)
        model = add_lora_fn(
            model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            init_lora_weights=init_lora_weights,
        )
        
        # Optionally enable pretrained action/state heads.
        if train_legacy_action_heads:
            if hasattr(model, "state_encoder"):
                model.state_encoder.requires_grad_(True)
                print("  → Enabled state_encoder")
            if hasattr(model, "action_encoder"):
                model.action_encoder.requires_grad_(True)
                print("  → Enabled action_encoder")
            if hasattr(model, "action_decoder"):
                model.action_decoder.requires_grad_(True)
                print("  → Enabled action_decoder")
        else:
            print("  → Keeping state/action encoder-decoder heads frozen")
        if hasattr(model, "value_function_encoder") and not train_value_function:
            model.value_function_encoder.requires_grad_(False)
            print("  → Disabled value_function_encoder (train_value_function=False)")
        if hasattr(model, "value_function_decoder") and not train_value_function:
            model.value_function_decoder.requires_grad_(False)
            print("  → Disabled value_function_decoder (train_value_function=False)")
    
    elif backbone_strategy == "full":
        print("  → Full fine-tuning of backbone")
        for param in model.parameters():
            param.requires_grad = True
    
    elif backbone_strategy == "frozen":
        print("  → Keeping backbone frozen")
        for param in model.parameters():
            param.requires_grad = False
    
    else:
        raise ValueError(f"Unknown backbone_strategy: {backbone_strategy}")
    
    # Handle value function head training
    if train_value_function:
        print("  → Enabling value_function encoder/decoder for training")
        if hasattr(model, "value_function_encoder"):
            model.value_function_encoder.requires_grad_(True)
        if hasattr(model, "value_function_decoder"):
            model.value_function_decoder.requires_grad_(True)
    else:
        print("  → Freezing value_function encoder/decoder")
        if hasattr(model, "value_function_encoder"):
            model.value_function_encoder.requires_grad_(False)
        if hasattr(model, "value_function_decoder"):
            model.value_function_decoder.requires_grad_(False)
    
    # Print summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nMixed training strategy applied:")
    print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    return model


def print_trainable_parameters_by_component(model: nn.Module):
    """Print trainable parameters grouped by component."""
    components = {}
    
    for name, param in model.named_parameters():
        component = name.split(".")[0]
        if component not in components:
            components[component] = {"trainable": 0, "total": 0}
        
        param_count = param.numel()
        components[component]["total"] += param_count
        if param.requires_grad:
            components[component]["trainable"] += param_count
    
    print("\nTrainable parameters by component:")
    total_trainable = 0
    total_params = 0
    for component, counts in sorted(components.items()):
        trainable = counts["trainable"]
        total = counts["total"]
        total_trainable += trainable
        total_params += total
        pct = 100 * trainable / total if total > 0 else 0
        print(f"  {component:25s}: {trainable:10,} / {total:10,} ({pct:5.1f}%)")
    
    pct = 100 * total_trainable / total_params if total_params > 0 else 0
    print(f"  {'TOTAL':25s}: {total_trainable:10,} / {total_params:10,} ({pct:5.1f}%)")
