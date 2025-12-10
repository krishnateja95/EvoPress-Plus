#!/usr/bin/env python3
"""
Quantized Layer Generator for HIGGS FP Optimizer (llmcompressor-based)

This script generates quantized layer weights for LLMs (Llama, Qwen, etc.) 
at different precision formats using llmcompressor's actual quantization.

Supported precision formats:
- FP8_Dynamic: 8-bit floating point with per-channel dynamic scaling
- FP8_Block: 8-bit floating point with block-wise scaling  
- NVFP4: 4-bit NVIDIA floating point with group-wise scaling

The output directory structure is:
    layer-dir/
    ├── metadata.json
    ├── model.layers.0.self_attn.q_proj/
    │   ├── FP8_Dynamic.pth
    │   ├── FP8_Block.pth
    │   └── NVFP4.pth
    ├── model.layers.0.self_attn.k_proj/
    │   ├── FP8_Dynamic.pth
    │   ├── FP8_Block.pth
    │   └── NVFP4.pth
    └── ...

Usage:
    python create_quantized_layers.py \
        --model meta-llama/Llama-2-7b-hf \
        --output-dir ./quantized_layers \
        --precisions FP8_Dynamic FP8_Block NVFP4 \
        --num-calibration-samples 32 \
        --device cuda

Requirements:
    pip install llmcompressor transformers datasets torch --break-system-packages
"""

import argparse
import logging
import os
import json
import gc
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Precision Format Definitions and Recipes
# ============================================================================

PRECISION_CONFIGS = {
    "FP8_Dynamic": {
        "effective_bits": 8.0,
        "num_bits": 8,
        "strategy": "channel",
        "recipe_template": """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: channel
                        dynamic: false
                        symmetric: true
                        observer: minmax
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: [{targets}]
"""
    },
    "FP8_Block": {
        "effective_bits": 8.0,
        "num_bits": 8,
        "strategy": "block",
        "recipe_template": """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: block
                        block_structure: [128, 128]
                        dynamic: false
                        symmetric: true
                        observer: minmax
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: group
                        group_size: 128
                        dynamic: true
                        symmetric: true
                    targets: [{targets}]
"""
    },
    "NVFP4": {
        "effective_bits": 4.0,
        "num_bits": 4,
        "strategy": "tensor_group",
        "recipe_template": """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: float
                        strategy: tensor_group
                        dynamic: false
                        symmetric: true
                        group_size: 16
                    input_activations:
                        num_bits: 4
                        type: float
                        strategy: tensor_group
                        dynamic: local
                        symmetric: true
                        group_size: 16
                    targets: [{targets}]
"""
    }
}


# ============================================================================
# Model and Data Loading
# ============================================================================

def load_model(model_path: str, device: str = "auto") -> Tuple[nn.Module, Any]:
    """Load a Hugging Face model and tokenizer."""
    logger.info(f"Loading model from {model_path}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map=device if device != "cpu" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded: {model.config.model_type}")
    return model, tokenizer


def load_calibration_dataset(
    tokenizer,
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    num_samples: int = 32,
    max_length: int = 2048
):
    """Load and prepare calibration dataset for quantization."""
    logger.info(f"Loading calibration data from {dataset_name}")
    
    from datasets import load_dataset
    
    ds = load_dataset(dataset_name, split=f"train_sft[:{num_samples}]")
    ds = ds.shuffle(seed=42)
    
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }
    
    ds = ds.map(preprocess)
    
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )
    
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    
    logger.info(f"Loaded {len(ds)} calibration samples")
    return ds


# ============================================================================
# Layer Extraction
# ============================================================================

def get_linear_layer_names(model: nn.Module) -> List[str]:
    """
    Extract names of all Linear layers from the model.
    Excludes lm_head and embedding layers.
    """
    linear_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip lm_head and embedding layers
            if "lm_head" in name or "embed" in name:
                continue
            linear_layers.append(name)
    
    logger.info(f"Found {len(linear_layers)} linear layers")
    return linear_layers


def get_layer_weight(model: nn.Module, layer_name: str) -> Optional[torch.Tensor]:
    """Extract weights for a specific layer from the model."""
    try:
        current = model
        for attr in layer_name.split('.'):
            current = getattr(current, attr)
        
        if hasattr(current, 'weight'):
            return current.weight.data.clone()
        return None
    except AttributeError:
        return None


def extract_all_weights(model: nn.Module, layer_names: List[str]) -> Dict[str, torch.Tensor]:
    """Extract weights from all specified layers."""
    weights = {}
    for name in layer_names:
        weight = get_layer_weight(model, name)
        if weight is not None:
            weights[name] = weight
    return weights


# ============================================================================
# Quantization with llmcompressor
# ============================================================================

def build_recipe(precision: str, target_layers: List[str]) -> str:
    """Build a quantization recipe for the specified precision and layers."""
    config = PRECISION_CONFIGS[precision]
    
    # Format target layer names as quoted strings
    targets = ", ".join(f'"{layer}"' for layer in target_layers)
    
    recipe = config["recipe_template"].format(targets=targets)
    return recipe


def quantize_model_with_precision(
    model_path: str,
    precision: str,
    target_layers: List[str],
    calibration_dataset,
    num_calibration_samples: int,
    max_sequence_length: int
) -> nn.Module:
    """
    Quantize a model at a specific precision using llmcompressor.
    
    Returns the quantized model.
    """
    from transformers import AutoModelForCausalLM
    from llmcompressor import oneshot
    
    logger.info(f"Quantizing model to {precision}...")
    
    # Load a fresh copy of the model for this precision
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    # Build recipe
    recipe = build_recipe(precision, target_layers)
    
    logger.debug(f"Recipe for {precision}:\n{recipe}")
    
    # Run quantization
    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=max_sequence_length,
        num_calibration_samples=num_calibration_samples,
    )
    
    logger.info(f"Quantization to {precision} complete")
    return model


# ============================================================================
# Error Computation
# ============================================================================

def compute_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor
) -> float:
    """
    Compute relative Frobenius norm error.
    
    t²_l = ||W^quantized - W^original||²_F / ||W^original||²_F
    """
    with torch.no_grad():
        original_float = original.float().cpu()
        quantized_float = quantized.float().cpu()
        
        diff = quantized_float - original_float
        error_norm_sq = torch.norm(diff, p='fro') ** 2
        original_norm_sq = torch.norm(original_float, p='fro') ** 2
        
        if original_norm_sq < 1e-10:
            return 0.0
        
        return (error_norm_sq / original_norm_sq).item()


# ============================================================================
# Main Pipeline
# ============================================================================

def create_quantized_layers(
    model_path: str,
    output_dir: str,
    precisions: List[str],
    num_calibration_samples: int = 32,
    max_sequence_length: int = 2048,
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    verbose: bool = False
) -> Dict[str, Dict[str, str]]:
    """
    Create quantized versions of all linear layers at specified precisions
    using llmcompressor.
    
    Args:
        model_path: HuggingFace model path or name
        output_dir: Output directory for quantized layers
        precisions: List of precision formats to generate
        num_calibration_samples: Number of calibration samples
        max_sequence_length: Maximum sequence length for calibration
        dataset_name: Calibration dataset name
        verbose: Enable verbose logging
    
    Returns:
        Dictionary mapping layer names to precision -> file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer to get layer names and original weights
    logger.info("Loading original model to extract layer information...")
    original_model, tokenizer = load_model(model_path, device="cpu")
    
    # Get linear layer names
    layer_names = get_linear_layer_names(original_model)
    
    # Extract original weights for error computation
    logger.info("Extracting original weights...")
    original_weights = extract_all_weights(original_model, layer_names)
    
    # Free up memory
    del original_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load calibration dataset
    calibration_dataset = load_calibration_dataset(
        tokenizer,
        dataset_name=dataset_name,
        num_samples=num_calibration_samples,
        max_length=max_sequence_length
    )
    
    # Dictionary to store all layer paths
    layer_paths: Dict[str, Dict[str, str]] = {name: {} for name in layer_names}
    
    # Process each precision format
    for precision in precisions:
        logger.info("=" * 60)
        logger.info(f"Processing precision: {precision}")
        logger.info("=" * 60)
        
        if precision not in PRECISION_CONFIGS:
            logger.error(f"Unknown precision format: {precision}")
            continue
        
        try:
            # Quantize model at this precision
            quantized_model = quantize_model_with_precision(
                model_path=model_path,
                precision=precision,
                target_layers=layer_names,
                calibration_dataset=calibration_dataset,
                num_calibration_samples=num_calibration_samples,
                max_sequence_length=max_sequence_length
            )
            
            # Extract quantized weights and save
            logger.info(f"Extracting and saving {precision} weights...")
            
            for layer_name in layer_names:
                # Create directory for this layer
                layer_dir = output_path / layer_name
                layer_dir.mkdir(parents=True, exist_ok=True)
                
                # Get quantized weight
                quantized_weight = get_layer_weight(quantized_model, layer_name)
                
                if quantized_weight is None:
                    logger.warning(f"Could not extract weight for {layer_name}")
                    continue
                
                # Compute quantization error
                original_weight = original_weights.get(layer_name)
                if original_weight is not None:
                    error = compute_quantization_error(original_weight, quantized_weight)
                else:
                    error = -1.0  # Unknown
                
                # Save quantized weight
                save_path = layer_dir / f"{precision}.pth"
                
                save_data = {
                    "weight": quantized_weight.cpu(),
                    "quant_info": {
                        "precision": precision,
                        "effective_bits": PRECISION_CONFIGS[precision]["effective_bits"],
                        "strategy": PRECISION_CONFIGS[precision]["strategy"],
                        "quantization_error": error,
                    },
                    "original_shape": list(quantized_weight.shape),
                    "original_dtype": str(original_weight.dtype) if original_weight is not None else "unknown"
                }
                
                torch.save(save_data, save_path)
                layer_paths[layer_name][precision] = str(save_path)
                
                if verbose:
                    logger.info(f"  {layer_name}: error={error:.6f}")
            
            # Free memory
            del quantized_model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info(f"Completed {precision}: saved {len(layer_names)} layers")
            
        except Exception as e:
            logger.error(f"Failed to process {precision}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save metadata
    metadata = {
        "model_path": model_path,
        "num_layers": len(layer_names),
        "layer_names": layer_names,
        "precisions": precisions,
        "precision_configs": {
            name: {
                "effective_bits": config["effective_bits"],
                "num_bits": config["num_bits"],
                "strategy": config["strategy"]
            }
            for name, config in PRECISION_CONFIGS.items()
            if name in precisions
        },
        "calibration": {
            "dataset": dataset_name,
            "num_samples": num_calibration_samples,
            "max_sequence_length": max_sequence_length
        },
        "layer_paths": layer_paths
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("QUANTIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Layers processed: {len(layer_names)}")
    logger.info(f"  Precisions generated: {precisions}")
    logger.info(f"  Metadata saved to: {metadata_path}")
    
    return layer_paths


# ============================================================================
# Alternative: Per-Layer Quantization (More Memory Efficient)
# ============================================================================

def create_quantized_layers_per_layer(
    model_path: str,
    output_dir: str,
    precisions: List[str],
    num_calibration_samples: int = 32,
    max_sequence_length: int = 2048,
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    verbose: bool = False
) -> Dict[str, Dict[str, str]]:
    """
    Alternative approach: Quantize one layer at a time.
    
    This is more memory efficient but slower, as it requires running
    oneshot() for each layer-precision combination.
    
    Use this if you run into memory issues with the full-model approach.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    from transformers import AutoModelForCausalLM
    from llmcompressor import oneshot
    
    # Load model and tokenizer
    logger.info("Loading model...")
    model, tokenizer = load_model(model_path, device="cpu")
    
    # Get linear layer names
    layer_names = get_linear_layer_names(model)
    
    # Extract original weights
    logger.info("Extracting original weights...")
    original_weights = extract_all_weights(model, layer_names)
    
    # Free model memory
    del model
    gc.collect()
    
    # Load calibration dataset
    calibration_dataset = load_calibration_dataset(
        tokenizer,
        dataset_name=dataset_name,
        num_samples=num_calibration_samples,
        max_length=max_sequence_length
    )
    
    layer_paths: Dict[str, Dict[str, str]] = {name: {} for name in layer_names}
    
    total_combinations = len(layer_names) * len(precisions)
    current = 0
    
    for layer_name in layer_names:
        # Create directory for this layer
        layer_dir = output_path / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        for precision in precisions:
            current += 1
            logger.info(f"[{current}/{total_combinations}] {layer_name} -> {precision}")
            
            if precision not in PRECISION_CONFIGS:
                logger.warning(f"Unknown precision: {precision}")
                continue
            
            try:
                # Load fresh model
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    trust_remote_code=True
                )
                
                # Build recipe for single layer
                recipe = build_recipe(precision, [layer_name])
                
                # Quantize
                oneshot(
                    model=model,
                    dataset=calibration_dataset,
                    recipe=recipe,
                    max_seq_length=max_sequence_length,
                    num_calibration_samples=num_calibration_samples,
                )
                
                # Extract quantized weight
                quantized_weight = get_layer_weight(model, layer_name)
                
                if quantized_weight is None:
                    logger.warning(f"Could not extract weight for {layer_name}")
                    del model
                    gc.collect()
                    continue
                
                # Compute error
                original_weight = original_weights.get(layer_name)
                if original_weight is not None:
                    error = compute_quantization_error(original_weight, quantized_weight)
                else:
                    error = -1.0
                
                # Save
                save_path = layer_dir / f"{precision}.pth"
                
                save_data = {
                    "weight": quantized_weight.cpu(),
                    "quant_info": {
                        "precision": precision,
                        "effective_bits": PRECISION_CONFIGS[precision]["effective_bits"],
                        "strategy": PRECISION_CONFIGS[precision]["strategy"],
                        "quantization_error": error,
                    },
                    "original_shape": list(quantized_weight.shape),
                    "original_dtype": str(original_weight.dtype) if original_weight is not None else "unknown"
                }
                
                torch.save(save_data, save_path)
                layer_paths[layer_name][precision] = str(save_path)
                
                if verbose:
                    logger.info(f"  Error: {error:.6f}")
                
                # Cleanup
                del model
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                logger.error(f"Failed: {e}")
                continue
    
    # Save metadata
    metadata = {
        "model_path": model_path,
        "num_layers": len(layer_names),
        "layer_names": layer_names,
        "precisions": precisions,
        "precision_configs": {
            name: {
                "effective_bits": config["effective_bits"],
                "num_bits": config["num_bits"],
                "strategy": config["strategy"]
            }
            for name, config in PRECISION_CONFIGS.items()
            if name in precisions
        },
        "calibration": {
            "dataset": dataset_name,
            "num_samples": num_calibration_samples,
            "max_sequence_length": max_sequence_length
        },
        "layer_paths": layer_paths
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Quantization complete!")
    return layer_paths


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate quantized layer weights using llmcompressor"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HuggingFace model name (e.g., meta-llama/Llama-2-7b-hf)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for quantized layers"
    )
    
    parser.add_argument(
        "--precisions",
        type=str,
        nargs="+",
        default=["FP8_Dynamic", "FP8_Block", "NVFP4"],
        choices=["FP8_Dynamic", "FP8_Block", "NVFP4"],
        help="Precision formats to generate (default: all three)"
    )
    
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=32,
        help="Number of calibration samples (default: 32)"
    )
    
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration (default: 2048)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset name (default: HuggingFaceH4/ultrachat_200k)"
    )
    
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="Use per-layer quantization (slower but more memory efficient)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n" + "=" * 70)
    print("QUANTIZED LAYER GENERATOR (llmcompressor-based)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Precisions: {args.precisions}")
    print(f"Calibration samples: {args.num_calibration_samples}")
    print(f"Max sequence length: {args.max_sequence_length}")
    print(f"Calibration dataset: {args.dataset}")
    print(f"Mode: {'Per-layer' if args.per_layer else 'Full-model'}")
    print("=" * 70 + "\n")
    
    # Select quantization approach
    if args.per_layer:
        layer_paths = create_quantized_layers_per_layer(
            model_path=args.model,
            output_dir=args.output_dir,
            precisions=args.precisions,
            num_calibration_samples=args.num_calibration_samples,
            max_sequence_length=args.max_sequence_length,
            dataset_name=args.dataset,
            verbose=args.verbose
        )
    else:
        layer_paths = create_quantized_layers(
            model_path=args.model,
            output_dir=args.output_dir,
            precisions=args.precisions,
            num_calibration_samples=args.num_calibration_samples,
            max_sequence_length=args.max_sequence_length,
            dataset_name=args.dataset,
            verbose=args.verbose
        )
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── metadata.json")
    
    # Show sample directory structure
    sample_layers = list(layer_paths.keys())[:3]
    for layer_name in sample_layers:
        print(f"  ├── {layer_name}/")
        for precision in args.precisions:
            print(f"  │   ├── {precision}.pth")
    
    if len(layer_paths) > 3:
        print(f"  └── ... ({len(layer_paths) - 3} more layers)")
    
    print("\n" + "=" * 70)
    print("Next step: Run the HIGGS FP optimizer with:")
    print(f"  python fp_higgs_optimizer.py \\")
    print(f"      --model {args.model} \\")
    print(f"      --layer-dir {args.output_dir} \\")
    print(f"      --output-config optimal_config.txt \\")
    print(f"      --output-model ./optimized_model \\")
    print(f"      --target-precision NVFP4")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())