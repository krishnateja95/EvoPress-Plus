import json
import os
from pathlib import Path
from safetensors import safe_open
from collections import defaultdict
from huggingface_hub import hf_hub_download, list_repo_files
import torch

def get_tensor_size_gb(dtype_str, shape):
    """Calculate tensor size in GB based on dtype and shape"""
    # Determine bytes per element
    dtype_to_bytes = {
        'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2,
        'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1,
        'U8': 1, 'BOOL': 1, 'F8_E4M3': 1, 'F8_E5M2': 1
    }
    
    bytes_per_element = dtype_to_bytes.get(dtype_str, 4)
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    
    size_bytes = num_elements * bytes_per_element
    size_gb = size_bytes / (1024**3)
    return round(size_gb, 6)

def download_model_files(repo_id):
    """Download necessary files from HuggingFace"""
    print(f"Downloading files from {repo_id}...")
    
    # List all files in the repository
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        print(f"Error listing files for {repo_id}: {e}")
        return None, []
    
    # Download index file
    index_file = None
    if "model.safetensors.index.json" in files:
        try:
            index_file = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
            print(f"  Downloaded index file")
        except Exception as e:
            print(f"  Error downloading index: {e}")
            return None, []
    else:
        print(f"  No index file found for {repo_id}")
        return None, []
    
    # Load index to get shard files
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    # Get unique shard files
    shard_files = list(set(index_data['weight_map'].values()))
    
    # Download all shard files
    downloaded_shards = []
    for shard_file in shard_files:
        try:
            shard_path = hf_hub_download(repo_id=repo_id, filename=shard_file)
            downloaded_shards.append(shard_path)
            print(f"  Downloaded {shard_file}")
        except Exception as e:
            print(f"  Error downloading {shard_file}: {e}")
    
    return index_file, downloaded_shards

def extract_base_layer_names(weight_map):
    """Extract base layer names from weight_map by removing tensor suffixes"""
    base_names = set()
    
    for tensor_name in weight_map.keys():
        # Remove known suffixes to get base name
        base_name = tensor_name
        for suffix in ['.weight', '.weight_scale', '.input_global_scale', 
                       '.weight_global_scale', '.weight_packed']:
            if base_name.endswith(suffix):
                base_name = base_name.replace(suffix, '')
                break
        base_names.add(base_name)
    
    return sorted(base_names)

def analyze_model(repo_id, base_layer_names=None):
    """Analyze a single model from HuggingFace and return layer sizes
    
    Args:
        repo_id: HuggingFace model repository ID
        base_layer_names: Optional reference list of base layer names (e.g., from FP16 model)
    """
    
    # Download model files
    index_file, shard_files = download_model_files(repo_id)
    
    if index_file is None:
        return {}
    
    # Load index file
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    weight_map = index_data['weight_map']
    
    # If no base layer names provided, extract them from this model
    if base_layer_names is None:
        base_layer_names = extract_base_layer_names(weight_map)
    
    # Get the directory where files are cached
    cache_dir = os.path.dirname(index_file)
    
    # Dictionary to store layer sizes
    layer_sizes = {}
    
    # First pass: calculate all tensor sizes
    tensor_sizes = {}
    
    # Group tensors by shard file
    shard_to_tensors = defaultdict(list)
    for tensor_name, shard_file in weight_map.items():
        shard_to_tensors[shard_file].append(tensor_name)
    
    # Process each shard to get tensor sizes
    for shard_file, tensor_names in shard_to_tensors.items():
        shard_path = os.path.join(cache_dir, shard_file)
        
        if not os.path.exists(shard_path):
            print(f"  Warning: Shard file not found: {shard_path}")
            continue
        
        # Open safetensors file
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for tensor_name in tensor_names:
                    try:
                        tensor = f.get_slice(tensor_name)
                        dtype = str(tensor.get_dtype())
                        shape = tensor.get_shape()
                        
                        # Calculate size
                        size_gb = get_tensor_size_gb(dtype, shape)
                        tensor_sizes[tensor_name] = size_gb
                    except Exception as e:
                        print(f"  Error processing tensor {tensor_name}: {e}")
        except Exception as e:
            print(f"  Error opening shard {shard_path}: {e}")
    
    # Second pass: group tensors by base layer name
    # For each base layer name, sum all tensors that start with that name
    for base_name in base_layer_names:
        total_size = 0.0
        
        # Find all tensors that match this base name (prefix matching)
        for tensor_name, size in tensor_sizes.items():
            if tensor_name.startswith(base_name):
                total_size += size
        
        if total_size > 0:
            layer_sizes[base_name] = round(total_size, 6)
    
    return layer_sizes

def main():
    # Define model repositories with their quantization scheme names
    models = {
        "FP16": "meta-llama/Llama-3.1-8B-Instruct",
        "FP8-DYNAMIC": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic",
        "FP8-BLOCK": "RedHatAI/Llama-3.1-8B-Instruct-FP8-block",
        "NVFP4": "RedHatAI/Llama-3.1-8B-Instruct-NVFP4",
        "INT8": "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
    }
    
    # Store results for each model
    model_results = {}
    base_layer_names = None
    
    # Analyze each model
    for quant_scheme, repo_id in models.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {quant_scheme} ({repo_id})...")
        print(f"{'='*60}")
        
        # Use FP16 base layer names for all other models
        if quant_scheme == "FP16":
            layer_sizes = analyze_model(repo_id)
            # Extract base layer names from FP16 to use as reference
            if layer_sizes:
                base_layer_names = sorted(layer_sizes.keys())
        else:
            layer_sizes = analyze_model(repo_id, base_layer_names)
        
        if layer_sizes:
            model_results[quant_scheme] = layer_sizes
            print(f"\n✓ Successfully analyzed {quant_scheme}")
            print(f"  Total size: {round(sum(layer_sizes.values()), 6)} GB")
            print(f"  Number of layers: {len(layer_sizes)}")
        else:
            print(f"\n✗ Failed to analyze {quant_scheme}")
    
    # Reorganize data by layer
    layer_comparison = {}
    
    # Get all unique layer names across all models
    all_layers = set()
    for model_data in model_results.values():
        all_layers.update(model_data.keys())
    
    # For each layer, create a subdictionary with quantization schemes
    for layer_name in sorted(all_layers):
        layer_comparison[layer_name] = {}
        for quant_scheme, layer_sizes in model_results.items():
            if layer_name in layer_sizes:
                layer_comparison[layer_name][quant_scheme] = layer_sizes[layer_name]
            else:
                layer_comparison[layer_name][quant_scheme] = None
    
    # Calculate total sizes for summary
    totals = {}
    for quant_scheme, layer_sizes in model_results.items():
        totals[quant_scheme] = round(sum(layer_sizes.values()), 6)
    
    # Create final output structure
    output = {
        "layer_sizes": layer_comparison,
        "total_sizes_gb": totals
    }
    
    # Save results to JSON
    output_file = "model_layer_sizes.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
    
    # Print summary table
    print("\n=== SUMMARY - TOTAL MODEL SIZES ===")
    print(f"{'Quantization Scheme':<20} {'Total Size (GB)':<20}")
    print("-" * 40)
    for quant_scheme, total_size in totals.items():
        print(f"{quant_scheme:<20} {total_size:<20}")
    
    # Print sample layer comparison
    print("\n=== SAMPLE LAYER COMPARISON ===")
    sample_layers = list(layer_comparison.keys())[:5]  # Show first 5 layers
    for layer_name in sample_layers:
        print(f"\n{layer_name}:")
        for quant_scheme, size in layer_comparison[layer_name].items():
            if size is not None:
                print(f"  {quant_scheme:<15} {size} GB")

if __name__ == "__main__":
    main()