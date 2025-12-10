#!/usr/bin/env python3
"""
HIGGS FP Dynamic Quantization Optimizer

A Python implementation of the dynamic quantization optimization algorithm modified
for FP8 and NVFP4 precision formats. Based on the paper:
"Pushing the Limits of Large Language Model Quantization via the Linearity Theorem"

This tool optimizes precision allocation across neural network layers to minimize
quantization error while meeting a target average precision constraint.

Supported precision formats:
- FP8_Dynamic: 8-bit floating point with per-channel dynamic scaling
- FP8_Block: 8-bit floating point with block-wise scaling
- NVFP4: 4-bit NVIDIA floating point with group-wise scaling

Usage:
    python fp_higgs_optimizer.py \
        --model meta-llama/Llama-2-7b-hf \
        --layer-dir ./quantized_layers \
        --output-config optimal_config.txt \
        --output-model ./optimized_model \
        --target-precision NVFP4
"""

import argparse
import json
import logging
import os
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from ortools.linear_solver import pywraplp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Precision Format Definitions
# ============================================================================

@dataclass
class PrecisionFormat:
    """Represents a precision format with its properties."""
    name: str
    effective_bits: float
    num_bits: int
    strategy: str
    block_structure: Optional[Tuple[int, int]] = None
    group_size: Optional[int] = None
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, PrecisionFormat):
            return self.name == other.name
        return False


# Define available precision formats
PRECISION_FORMATS = {
    "FP8_Dynamic": PrecisionFormat(
        name="FP8_Dynamic",
        effective_bits=8.0,
        num_bits=8,
        strategy="channel"
    ),
    "FP8_Block": PrecisionFormat(
        name="FP8_Block",
        effective_bits=8.0,
        num_bits=8,
        strategy="block",
        block_structure=(128, 128)
    ),
    "NVFP4": PrecisionFormat(
        name="NVFP4",
        effective_bits=4.0,
        num_bits=4,
        strategy="tensor_group",
        group_size=16
    )
}

# Target precision to effective bits mapping
TARGET_PRECISION_BITS = {
    "FP8_Dynamic": 8.0,
    "FP8_Block": 8.0,
    "NVFP4": 4.0,
    "FP8": 8.0,  # Alias
    "FP4": 4.0   # Alias
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class QuantizationOption:
    """Represents a quantization option with its precision and error."""
    precision: str
    effective_bits: float
    error: float
    layer_path: str
    quant_info: Dict[str, Any] = field(default_factory=dict)


class ErrorDatabase:
    """Manages the error database for all layers and quantization options."""

    def __init__(self):
        self.layers: Dict[str, Dict[str, QuantizationOption]] = {}
        self.layer_names: List[str] = []
        self.available_precisions: List[str] = []

    def add_layer_option(
        self,
        layer_name: str,
        precision: str,
        effective_bits: float,
        error: float,
        layer_path: str,
        quant_info: Optional[Dict[str, Any]] = None
    ):
        """Add a quantization option for a specific layer."""
        if layer_name not in self.layers:
            self.layers[layer_name] = {}
            self.layer_names.append(layer_name)

        self.layers[layer_name][precision] = QuantizationOption(
            precision=precision,
            effective_bits=effective_bits,
            error=error,
            layer_path=layer_path,
            quant_info=quant_info or {}
        )

        if precision not in self.available_precisions:
            self.available_precisions.append(precision)

    def get_layer_count(self) -> int:
        return len(self.layer_names)

    def get_layer_options(self, layer_name: str) -> Dict[str, QuantizationOption]:
        return self.layers.get(layer_name, {})
    
    def get_effective_bits(self) -> Dict[str, float]:
        """Get effective bits for each available precision."""
        bits = {}
        for layer_name in self.layer_names:
            for precision, option in self.layers[layer_name].items():
                if precision not in bits:
                    bits[precision] = option.effective_bits
        return bits


# ============================================================================
# Model Loading Utilities
# ============================================================================

def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """Load a Hugging Face model."""
    logger.info(f"Loading model from {model_path}")

    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device if device != "cpu" else None
        )
        
        if device == "cpu":
            model = model.to(device)

        model.eval()
        return model
    
    except Exception as exc:
        logger.error("Failed to load Hugging Face model: %s", exc)
        raise


def load_tokenizer(model_path: str):
    """Load a tokenizer if available."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as exc:
        logger.warning("Could not load tokenizer: %s", exc)
        return None


def get_layer_weights(model: nn.Module, layer_name: str) -> Optional[torch.Tensor]:
    """Extract weights for a specific layer from the model."""
    try:
        current = model
        for attr in layer_name.split('.'):
            current = getattr(current, attr)

        if hasattr(current, 'weight'):
            return current.weight
        else:
            logger.error(f"Layer {layer_name} does not have a 'weight' attribute")
            return None

    except AttributeError as e:
        logger.error(f"Could not access layer {layer_name}: {e}")
        return None


def set_layer_weights(model: nn.Module, layer_name: str, weights: torch.Tensor):
    """Set weights for a specific layer in the model."""
    try:
        current = model
        parts = layer_name.split('.')
        
        # Navigate to parent
        for attr in parts[:-1]:
            current = getattr(current, attr)
        
        # Get the layer and set weights
        layer = getattr(current, parts[-1])
        if hasattr(layer, 'weight'):
            layer.weight.data = weights.to(layer.weight.dtype).to(layer.weight.device)
            return True
        else:
            logger.error(f"Layer {layer_name} does not have a 'weight' attribute")
            return False

    except AttributeError as e:
        logger.error(f"Could not set weights for layer {layer_name}: {e}")
        return False


# ============================================================================
# Error Computation
# ============================================================================

def compute_layer_error(original_weights: torch.Tensor, quantized_weights: torch.Tensor) -> float:
    """
    Compute the relative quantization error t²_l.
    t²_l = ||W^quantized - W^original||²_F / ||W^original||²_F
    """
    with torch.no_grad():
        original = original_weights.float()
        quantized = quantized_weights.float().to(original.device)

        diff = quantized - original
        error_norm_sq = torch.norm(diff, p='fro') ** 2
        original_norm_sq = torch.norm(original, p='fro') ** 2

        if original_norm_sq < 1e-10:
            return 0.0

        return (error_norm_sq / original_norm_sq).item()


# ============================================================================
# Error Database Building
# ============================================================================

def build_error_database(model: nn.Module, layer_dir: str) -> ErrorDatabase:
    """
    Build the error database by computing quantization errors for all layers
    and precision formats.
    """
    logger.info("Building error database...")

    error_db = ErrorDatabase()
    layer_dir_path = Path(layer_dir)
    
    # Load metadata if available
    metadata_path = layer_dir_path / "metadata.json"
    precision_bits = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if "precision_configs" in metadata:
                for name, config in metadata["precision_configs"].items():
                    precision_bits[name] = config.get("effective_bits", 8.0)
    
    # Default precision bits if not in metadata
    for name, fmt in PRECISION_FORMATS.items():
        if name not in precision_bits:
            precision_bits[name] = fmt.effective_bits

    # Iterate through all layer directories
    for layer_path in layer_dir_path.iterdir():
        if not layer_path.is_dir():
            continue

        layer_name = layer_path.name
        
        # Skip metadata directory or files
        if layer_name.startswith('.') or layer_name == "metadata.json":
            continue
            
        logger.info(f"Processing layer: {layer_name}")

        # Get original weights from the model
        original_weights = get_layer_weights(model, layer_name)
        if original_weights is None:
            logger.warning(f"Could not find original weights for layer: {layer_name}")
            continue

        # Process each quantized version
        for quant_file in layer_path.iterdir():
            if not quant_file.suffix == '.pth':
                continue

            # Get precision name from filename (e.g., "FP8_Dynamic.pth" -> "FP8_Dynamic")
            precision = quant_file.stem
            
            if precision not in PRECISION_FORMATS and precision not in precision_bits:
                logger.warning(f"Unknown precision format: {precision}")
                continue

            try:
                # Load quantized weights
                data = torch.load(quant_file, map_location='cpu')
                
                if isinstance(data, dict):
                    quantized_weights = data.get('weight', data)
                    quant_info = data.get('quant_info', {})
                else:
                    quantized_weights = data
                    quant_info = {}
                
                # Get effective bits
                effective_bits = precision_bits.get(
                    precision,
                    PRECISION_FORMATS.get(precision, PRECISION_FORMATS["FP8_Dynamic"]).effective_bits
                )

                # Compute error
                error = compute_layer_error(original_weights, quantized_weights)

                error_db.add_layer_option(
                    layer_name=layer_name,
                    precision=precision,
                    effective_bits=effective_bits,
                    error=error,
                    layer_path=str(quant_file),
                    quant_info=quant_info
                )

                logger.debug(f"Layer {layer_name}, {precision}: error = {error:.6f}")

            except Exception as e:
                logger.error(f"Error processing {quant_file}: {e}")
                continue

    logger.info(f"Error database built with {error_db.get_layer_count()} layers")
    logger.info(f"Available precisions: {error_db.available_precisions}")
    return error_db


# ============================================================================
# Alpha Coefficient Estimation
# ============================================================================

def iter_lines(filepath: str, limit: Optional[int] = None) -> Iterable[str]:
    """Yield non-empty lines from a file up to an optional limit."""
    with open(filepath, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            text = line.strip()
            if text:
                yield text


def compute_prompt_loss(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    device: str,
    max_length: int = 256
) -> float:
    """Compute average negative log-likelihood for a batch of prompts."""
    if tokenizer is None:
        raise ValueError("Tokenizer is required for loss-based alpha estimation")

    model_device = next(model.parameters()).device
    
    total_loss = 0.0
    total_tokens = 0

    for prompt in prompts:
        with torch.no_grad():
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            )
            input_ids = encoded["input_ids"].to(model_device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)

            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.detach().cpu().float().item()

            total_loss += loss * input_ids.shape[1]
            total_tokens += input_ids.shape[1]

    return total_loss / max(total_tokens, 1)


def apply_noise_and_measure(
    model: nn.Module,
    layer_name: str,
    epsilon: float,
    loss_fn: Callable[[], float]
) -> float:
    """Apply Gaussian noise to a layer, measure loss difference, and restore weights."""
    weight = get_layer_weights(model, layer_name)
    if weight is None:
        raise ValueError(f"Layer {layer_name} has no weights to perturb")

    noise = torch.randn_like(weight) * epsilon

    with torch.no_grad():
        weight.add_(noise)
        perturbed_loss = loss_fn()
        weight.sub_(noise)

    return perturbed_loss


def estimate_alpha_coefficients(
    model: nn.Module,
    error_db: ErrorDatabase,
    tokenizer,
    method: str = "hybrid",
    calibration_file: Optional[str] = None,
    calibration_samples: int = 16,
    alpha_epsilon: float = 1e-3,
    device: str = "cpu",
    max_length: int = 256,
    num_samples: int = 8,
) -> Dict[str, float]:
    """
    Estimate layer sensitivities using heuristics and/or noise probing.
    
    Methods:
    - heuristic: uses structural information (parameter count, depth, type)
    - noise: perturbs each layer and measures loss deltas
    - hybrid: blends both approaches
    """
    method = method.lower()
    assert method in {"heuristic", "noise", "hybrid"}, "Unknown alpha estimation method"

    logger.info("Estimating alpha coefficients (%s mode)...", method)
    logger.info("=" * 50)

    prompts: List[str] = []
    base_loss: Optional[float] = None

    if calibration_file:
        prompts = list(iter_lines(calibration_file, limit=calibration_samples))
        logger.info("Loaded %d calibration prompts from %s", len(prompts), calibration_file)

    if method in {"noise", "hybrid"}:
        if not prompts or tokenizer is None:
            logger.warning("Noise-based alpha estimation requested but calibration data missing")
            method = "heuristic"
        else:
            base_loss = compute_prompt_loss(model, tokenizer, prompts, device=device, max_length=max_length)
            logger.info("Baseline calibration loss: %.6f", base_loss)

    alphas: Dict[str, float] = {}
    layer_stats: List[Dict[str, object]] = []
    total_params = 0

    for i, layer_name in enumerate(error_db.layer_names):
        original_weights = get_layer_weights(model, layer_name)
        if original_weights is None:
            logger.warning(f"Could not get weights for {layer_name}, using default alpha=1.0")
            alphas[layer_name] = 1.0
            continue

        layer_size = original_weights.numel()
        total_params += layer_size

        # Extract layer depth and type information
        layer_parts = layer_name.split('.')
        layer_depth = 0
        layer_type = "unknown"

        if 'layers' in layer_parts:
            try:
                layer_depth = int([p for p in layer_parts if p.isdigit()][0])
            except (IndexError, ValueError):
                layer_depth = 0

        if 'attn' in layer_name or 'attention' in layer_name:
            layer_type = "attention"
        elif 'mlp' in layer_name or 'ffn' in layer_name:
            layer_type = "mlp"
        elif 'embed' in layer_name:
            layer_type = "embedding"

        # Heuristic alpha computation
        base_alpha = np.log(layer_size + 1)
        depth_factor = 1 + layer_depth * 0.05

        type_multiplier = 1.0
        if layer_type == "attention":
            type_multiplier = 1.2
        elif layer_type == "embedding":
            type_multiplier = 1.5
        elif layer_type == "mlp":
            type_multiplier = 0.9

        heuristic_alpha = base_alpha * depth_factor * type_multiplier
        alpha_value = heuristic_alpha

        # Optional noise probing
        if method in {"noise", "hybrid"} and base_loss is not None:
            loss_fn = lambda: compute_prompt_loss(model, tokenizer, prompts, device=device, max_length=max_length)
            deltas: List[float] = []

            for _ in range(num_samples):
                perturbed_loss = apply_noise_and_measure(model, layer_name, alpha_epsilon, loss_fn)
                delta = max(perturbed_loss - base_loss, 0.0)
                deltas.append(delta)

            mean_delta = float(np.mean(deltas)) if deltas else 0.0
            noise_alpha = mean_delta / max(alpha_epsilon ** 2, 1e-8)

            if method == "noise":
                alpha_value = noise_alpha
            else:  # hybrid
                alpha_value = 0.5 * heuristic_alpha + 0.5 * noise_alpha

        alphas[layer_name] = float(alpha_value)

        layer_stats.append({
            'name': layer_name,
            'alpha': alpha_value,
            'size': layer_size,
            'depth': layer_depth,
            'type': layer_type
        })

        if i < 5 or i % 20 == 0:
            logger.info(f"Layer {i + 1:3d}: {layer_name}")
            logger.info(f"  Size: {layer_size:,} params")
            logger.info(f"  Depth: {layer_depth}, Type: {layer_type}")
            logger.info(f"  Alpha ({method}): {alpha_value:.4f}")

    # Log summary statistics
    alpha_values = list(alphas.values())
    logger.info("=" * 50)
    logger.info("Alpha coefficient statistics:")
    logger.info(f"  Total layers: {len(alphas)}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Alpha range: [{min(alpha_values):.4f}, {max(alpha_values):.4f}]")
    logger.info(f"  Alpha mean: {np.mean(alpha_values):.4f}")

    return alphas


# ============================================================================
# Optimization
# ============================================================================

def solve_optimal_quantization(
    error_db: ErrorDatabase,
    alphas: Dict[str, float],
    target_bits: float,
    force_mixed: bool = False
) -> Dict[str, str]:
    """
    Solve the optimal quantization problem using linear programming.
    
    Minimizes: Σ αₗ × t²ₗ,pₗ
    Subject to: Σ bₚₗ ≤ b_target × L
    
    Where pₗ ∈ {available precisions}
    
    Returns a dictionary mapping layer names to precision format names.
    """
    logger.info(f"Solving optimal quantization for target bits: {target_bits}")

    total_layers = error_db.get_layer_count()
    available_precisions = error_db.available_precisions
    effective_bits = error_db.get_effective_bits()
    
    logger.info(f"Database statistics:")
    logger.info(f"  Total layers: {total_layers}")
    logger.info(f"  Available precisions: {available_precisions}")
    logger.info(f"  Effective bits per precision: {effective_bits}")
    logger.info(f"  Total budget: {target_bits * total_layers} bits")

    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("Could not create solver")

    logger.info("Creating decision variables...")

    # Decision variables: x[layer][precision] = 1 if layer uses precision, 0 otherwise
    x = {}
    total_variables = 0

    for layer_name in error_db.layer_names:
        x[layer_name] = {}
        layer_options = error_db.get_layer_options(layer_name)

        if not layer_options:
            logger.warning(f"Layer {layer_name} has no available quantization options!")
            continue

        for precision in layer_options.keys():
            x[layer_name][precision] = solver.IntVar(0, 1, f'x_{layer_name}_{precision}')
            total_variables += 1

    logger.info(f"Created {total_variables} decision variables")

    # Constraint 1: Each layer must use exactly one quantization option
    logger.info("Adding layer selection constraints...")
    constraint_count = 0

    for layer_name in error_db.layer_names:
        if layer_name not in x:
            continue

        constraint = solver.Constraint(1, 1, f'one_choice_{layer_name}')

        for precision in x[layer_name]:
            constraint.SetCoefficient(x[layer_name][precision], 1)

        constraint_count += 1

    logger.info(f"Added {constraint_count} layer selection constraints")

    # Constraint 2: Average bitwidth constraint
    logger.info("Adding bitwidth budget constraint...")
    max_total_bits = target_bits * total_layers
    bitwidth_constraint = solver.Constraint(
        -solver.infinity(),
        max_total_bits,
        'bitwidth_constraint'
    )

    min_possible_bits = 0
    max_possible_bits = 0

    for layer_name in error_db.layer_names:
        if layer_name not in x:
            continue

        layer_options = error_db.get_layer_options(layer_name)
        layer_bits = [option.effective_bits for option in layer_options.values()]

        min_possible_bits += min(layer_bits)
        max_possible_bits += max(layer_bits)

        for precision in x[layer_name]:
            bits = layer_options[precision].effective_bits
            bitwidth_constraint.SetCoefficient(x[layer_name][precision], bits)

    logger.info(f"Bitwidth constraint details:")
    logger.info(f"  Target total bits: {max_total_bits:.1f}")
    logger.info(f"  Minimum possible total: {min_possible_bits}")
    logger.info(f"  Maximum possible total: {max_possible_bits}")
    logger.info(f"  Problem feasible: {min_possible_bits <= max_total_bits <= max_possible_bits}")

    if max_total_bits < min_possible_bits:
        logger.error(f"INFEASIBLE: Target bitwidth {target_bits} is too low!")
        logger.error(f"Minimum average bitwidth possible: {min_possible_bits / total_layers:.2f}")
        raise RuntimeError("Optimization problem is infeasible - target bitwidth too low")

    # Add constraint to force mixed solutions if requested
    if force_mixed:
        logger.info("Adding constraints to prevent uniform solutions...")
        
        for precision in available_precisions:
            max_uniform_layers = int(0.7 * total_layers)
            uniform_constraint = solver.Constraint(
                -solver.infinity(),
                max_uniform_layers,
                f'limit_uniform_{precision}'
            )

            for layer_name in error_db.layer_names:
                if layer_name not in x:
                    continue
                if precision in x[layer_name]:
                    uniform_constraint.SetCoefficient(x[layer_name][precision], 1)

    # Objective: Minimize weighted sum of errors
    logger.info("Setting up objective function...")
    objective = solver.Objective()
    objective.SetMinimization()

    for layer_name in error_db.layer_names:
        if layer_name not in x:
            continue

        alpha = alphas.get(layer_name, 1.0)
        layer_options = error_db.get_layer_options(layer_name)

        for precision, option in layer_options.items():
            coefficient = alpha * option.error
            objective.SetCoefficient(x[layer_name][precision], coefficient)

    # Solve
    solver.SetTimeLimit(300000)  # 5 minutes
    
    logger.info("=" * 60)
    logger.info("STARTING OPTIMIZATION")
    logger.info("=" * 60)

    status = solver.Solve()

    logger.info("=" * 60)
    logger.info("OPTIMIZATION COMPLETED")
    logger.info("=" * 60)

    status_messages = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
    }

    status_msg = status_messages.get(status, f"UNKNOWN({status})")
    logger.info(f"Solver status: {status_msg}")
    logger.info(f"Solve time: {solver.WallTime()} ms")

    if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        logger.error(f"Solver failed with status: {status_msg}")
        raise RuntimeError(f"Solver failed with status: {status_msg}")

    # Extract solution
    logger.info("Extracting solution...")

    solution = {}
    total_bits = 0
    total_weighted_error = 0
    precision_distribution = {}

    for layer_name in error_db.layer_names:
        if layer_name not in x:
            continue

        for precision in x[layer_name]:
            if x[layer_name][precision].solution_value() > 0.5:
                solution[layer_name] = precision
                
                layer_options = error_db.get_layer_options(layer_name)
                option = layer_options[precision]
                
                total_bits += option.effective_bits
                precision_distribution[precision] = precision_distribution.get(precision, 0) + 1

                alpha = alphas.get(layer_name, 1.0)
                total_weighted_error += alpha * option.error
                break

    avg_bits = total_bits / len(solution) if solution else 0

    # Log solution analysis
    logger.info("=" * 60)
    logger.info("SOLUTION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Solution found for {len(solution)}/{total_layers} layers")
    logger.info(f"Total effective bits: {total_bits:.1f}")
    logger.info(f"Average bits per layer: {avg_bits:.3f}")
    logger.info(f"Target bits: {target_bits:.3f}")
    logger.info(f"Total weighted error: {total_weighted_error:.6f}")

    logger.info("Precision distribution:")
    for precision in sorted(precision_distribution.keys()):
        count = precision_distribution[precision]
        percentage = (count / len(solution)) * 100 if solution else 0
        logger.info(f"  {precision}: {count} layers ({percentage:.1f}%)")

    logger.info("=" * 60)

    return solution


# ============================================================================
# Model Generation
# ============================================================================

def generate_quantized_model(
    model: nn.Module,
    solution: Dict[str, str],
    error_db: ErrorDatabase,
    output_path: str,
    tokenizer=None
):
    """
    Generate and save the quantized model based on the optimal configuration.
    
    Args:
        model: Original model
        solution: Dictionary mapping layer names to precision formats
        error_db: Error database with quantized weight paths
        output_path: Path to save the quantized model
        tokenizer: Optional tokenizer to save with the model
    """
    logger.info("Generating quantized model...")
    logger.info(f"Output path: {output_path}")
    
    # Create a deep copy to avoid modifying the original
    # Note: For large models, this might be memory-intensive
    # In production, you might want to modify in-place or use a different strategy
    
    layers_updated = 0
    layers_failed = 0
    
    for layer_name, precision in solution.items():
        layer_options = error_db.get_layer_options(layer_name)
        
        if precision not in layer_options:
            logger.warning(f"Precision {precision} not found for layer {layer_name}")
            layers_failed += 1
            continue
        
        option = layer_options[precision]
        
        try:
            # Load quantized weights
            data = torch.load(option.layer_path, map_location='cpu')
            
            if isinstance(data, dict):
                quantized_weights = data.get('weight', data)
            else:
                quantized_weights = data
            
            # Set weights in model
            if set_layer_weights(model, layer_name, quantized_weights):
                layers_updated += 1
            else:
                layers_failed += 1
                
        except Exception as e:
            logger.error(f"Failed to load weights for {layer_name}: {e}")
            layers_failed += 1
    
    logger.info(f"Layers updated: {layers_updated}")
    logger.info(f"Layers failed: {layers_failed}")
    
    # Save the model
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        model.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Tokenizer saved to {output_dir}")
        
        # Save quantization configuration
        config = {
            "quantization_config": {
                "quant_method": "higgs_fp_optimized",
                "layer_precisions": solution,
                "precision_distribution": {},
            }
        }
        
        for precision in set(solution.values()):
            count = sum(1 for p in solution.values() if p == precision)
            config["quantization_config"]["precision_distribution"][precision] = count
        
        config_path = output_dir / "quantization_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Quantization config saved to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise
    
    return output_dir


def build_llmcompressor_recipe(solution: Dict[str, str]) -> str:
    """
    Build an llmcompressor recipe YAML string from the optimal configuration.
    
    This can be used with llmcompressor's oneshot() function for proper quantization.
    """
    # Group layers by precision
    precision_groups: Dict[str, List[str]] = {}
    for layer_name, precision in solution.items():
        if precision not in precision_groups:
            precision_groups[precision] = []
        precision_groups[precision].append(layer_name)
    
    recipe = "quant_stage:\n    quant_modifiers:\n        QuantizationModifier:\n            ignore: [\"lm_head\"]\n            config_groups:\n"
    
    group_idx = 0
    for precision, layers in precision_groups.items():
        if precision == "FP8_Block":
            recipe += f"""                group_{group_idx}:
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
                    targets: ["""
        elif precision == "FP8_Dynamic":
            recipe += f"""                group_{group_idx}:
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
                    targets: ["""
        elif precision == "NVFP4":
            recipe += f"""                group_{group_idx}:
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
                    targets: ["""
        
        # Add layer targets
        for i, layer in enumerate(layers):
            recipe += f'"{layer}"'
            if i < len(layers) - 1:
                recipe += ", "
        recipe += "]\n"
        
        group_idx += 1
    
    return recipe


# ============================================================================
# Save/Load Functions
# ============================================================================

def save_error_database(error_db: ErrorDatabase, filepath: str):
    """Save the error database to a JSON file."""
    logger.info(f"Saving error database to {filepath}")

    db_data = {
        'layers': {},
        'available_precisions': error_db.available_precisions,
        'layer_names': error_db.layer_names
    }

    for layer_name in error_db.layer_names:
        layer_options = error_db.get_layer_options(layer_name)
        db_data['layers'][layer_name] = {}

        for precision, option in layer_options.items():
            db_data['layers'][layer_name][precision] = {
                'precision': option.precision,
                'effective_bits': option.effective_bits,
                'error': option.error,
                'layer_path': option.layer_path,
                'quant_info': option.quant_info
            }

    with open(filepath, 'w') as f:
        json.dump(db_data, f, indent=2)

    logger.info("Error database saved successfully")


def load_error_database(filepath: str) -> ErrorDatabase:
    """Load a previously saved error database."""
    logger.info(f"Loading error database from {filepath}")

    error_db = ErrorDatabase()

    with open(filepath, 'r') as f:
        data = json.load(f)

    for layer_name, layer_data in data['layers'].items():
        for precision, option_data in layer_data.items():
            error_db.add_layer_option(
                layer_name=layer_name,
                precision=option_data['precision'],
                effective_bits=option_data['effective_bits'],
                error=option_data['error'],
                layer_path=option_data['layer_path'],
                quant_info=option_data.get('quant_info', {})
            )

    logger.info(f"Loaded error database with {error_db.get_layer_count()} layers")
    return error_db


def save_configuration(solution: Dict[str, str], output_path: str):
    """Save the optimal quantization configuration to a file."""
    logger.info(f"Saving configuration to {output_path}")

    with open(output_path, 'w') as f:
        f.write("# HIGGS FP Optimized Quantization Configuration\n")
        f.write("# Format: layer_name: precision_format\n\n")
        
        for layer_name, precision in sorted(solution.items()):
            f.write(f"{layer_name}: {precision}\n")

    logger.info("Configuration saved successfully")


def load_configuration(config_path: str) -> Dict[str, str]:
    """Load a quantization configuration from a file."""
    logger.info(f"Loading configuration from {config_path}")
    
    solution = {}
    
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(':')
            if len(parts) == 2:
                layer_name = parts[0].strip()
                precision = parts[1].strip()
                solution[layer_name] = precision
    
    logger.info(f"Loaded configuration with {len(solution)} layers")
    return solution


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="HIGGS FP Dynamic Quantization Optimizer"
    )
    
    parser.add_argument(
        '--model',
        required=True,
        help="Path or name of the Hugging Face model"
    )
    
    parser.add_argument(
        '--layer-dir',
        required=True,
        help="Directory containing quantized layers"
    )
    
    parser.add_argument(
        '--output-config',
        required=True,
        help="Output configuration file path"
    )
    
    parser.add_argument(
        '--output-model',
        type=str,
        default=None,
        help="Output directory for the quantized model (optional)"
    )
    
    parser.add_argument(
        '--output-recipe',
        type=str,
        default=None,
        help="Output path for llmcompressor recipe YAML (optional)"
    )
    
    parser.add_argument(
        '--target-precision',
        type=str,
        default="NVFP4",
        choices=["FP8_Dynamic", "FP8_Block", "NVFP4", "FP8", "FP4"],
        help="Target average precision (default: NVFP4)"
    )
    
    parser.add_argument(
        '--target-bits',
        type=float,
        default=None,
        help="Target average bits (overrides --target-precision)"
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
        help="Device for alpha estimation (cpu/cuda)"
    )
    
    parser.add_argument(
        '--save-error-db',
        type=str,
        help="Save error database to file"
    )
    
    parser.add_argument(
        '--load-error-db',
        type=str,
        help="Load error database from file"
    )
    
    parser.add_argument(
        '--load-config',
        type=str,
        help="Load existing configuration (skip optimization)"
    )
    
    parser.add_argument(
        '--alpha-method',
        type=str,
        default="heuristic",
        choices=['heuristic', 'noise', 'hybrid'],
        help="Strategy for alpha estimation"
    )
    
    parser.add_argument(
        '--calibration-file',
        type=str,
        help="Text file with one prompt per line"
    )
    
    parser.add_argument(
        '--calibration-samples',
        type=int,
        default=16,
        help="Number of calibration prompts to read"
    )
    
    parser.add_argument(
        '--alpha-epsilon',
        type=float,
        default=1e-3,
        help="Noise scale for loss-based sensitivity"
    )
    
    parser.add_argument(
        '--alpha-samples',
        type=int,
        default=8,
        help="Number of noise probes per layer"
    )
    
    parser.add_argument(
        '--calibration-max-length',
        type=int,
        default=256,
        help="Maximum sequence length for calibration prompts"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine target bits
    if args.target_bits is not None:
        target_bits = args.target_bits
    else:
        target_bits = TARGET_PRECISION_BITS.get(args.target_precision, 4.0)

    print("\n" + "=" * 70)
    print("HIGGS FP DYNAMIC QUANTIZATION OPTIMIZER")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Layer directory: {args.layer_dir}")
    print(f"Target precision: {args.target_precision}")
    print(f"Target bits: {target_bits}")
    print(f"Alpha method: {args.alpha_method}")
    print("=" * 70 + "\n")

    try:
        # Load the model
        model = load_model(args.model, device=args.device)
        tokenizer = load_tokenizer(args.model)

        # Load or build error database
        if args.load_error_db:
            error_db = load_error_database(args.load_error_db)
        else:
            error_db = build_error_database(model, args.layer_dir)

            if error_db.get_layer_count() == 0:
                logger.error("No layers found in error database")
                return 1

            if args.save_error_db:
                save_error_database(error_db, args.save_error_db)

        # Load existing configuration or run optimization
        if args.load_config:
            solution = load_configuration(args.load_config)
        else:
            # Estimate alpha coefficients
            alphas = estimate_alpha_coefficients(
                model,
                error_db,
                tokenizer,
                method=args.alpha_method,
                calibration_file=args.calibration_file,
                calibration_samples=args.calibration_samples,
                alpha_epsilon=args.alpha_epsilon,
                device=args.device,
                max_length=args.calibration_max_length,
                num_samples=args.alpha_samples,
            )

            # Solve optimal quantization
            solution = solve_optimal_quantization(error_db, alphas, target_bits)

            # Check if solution is uniform
            if len(set(solution.values())) == 1:
                uniform_precision = list(solution.values())[0]
                logger.info(f"Uniform solution found ({uniform_precision} for all layers)")
                
                # Try forcing mixed solution
                logger.info("Trying to find better mixed solution...")
                try:
                    mixed_solution = solve_optimal_quantization(
                        error_db, alphas, target_bits, force_mixed=True
                    )

                    # Compare solutions
                    uniform_error = sum(
                        alphas.get(layer, 1.0) * error_db.get_layer_options(layer)[prec].error
                        for layer, prec in solution.items()
                    )
                    mixed_error = sum(
                        alphas.get(layer, 1.0) * error_db.get_layer_options(layer)[prec].error
                        for layer, prec in mixed_solution.items()
                    )

                    if mixed_error < uniform_error - 1e-6:
                        logger.info(f"Mixed solution is better! Error reduction: {uniform_error - mixed_error:.6f}")
                        solution = mixed_solution
                    else:
                        logger.info("Uniform solution is indeed optimal")

                except Exception as e:
                    logger.warning(f"Failed to find mixed solution: {e}")

        # Save configuration
        save_configuration(solution, args.output_config)

        # Generate llmcompressor recipe if requested
        if args.output_recipe:
            recipe = build_llmcompressor_recipe(solution)
            with open(args.output_recipe, 'w') as f:
                f.write(recipe)
            logger.info(f"Recipe saved to {args.output_recipe}")

        # Generate quantized model if requested
        if args.output_model:
            generate_quantized_model(
                model, solution, error_db, args.output_model, tokenizer
            )

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"Configuration saved to: {args.output_config}")
        if args.output_model:
            print(f"Quantized model saved to: {args.output_model}")
        if args.output_recipe:
            print(f"LLMCompressor recipe saved to: {args.output_recipe}")
        
        # Print precision distribution
        distribution = {}
        for precision in solution.values():
            distribution[precision] = distribution.get(precision, 0) + 1
        
        print("\nPrecision distribution:")
        for precision, count in sorted(distribution.items()):
            percentage = (count / len(solution)) * 100
            print(f"  {precision}: {count} layers ({percentage:.1f}%)")
        
        print("=" * 70 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())