
```
python fp_higgs_optimizer.py \
    --model meta-llama/Llama-3.1-8B-Instruct\
    --layer-dir ./quantized_layers \
    --output-config optimal_config.txt \
    --output-model ./optimized_model \
    --output-recipe recipe.yaml \
    --target-precision NVFP4 \
    --alpha-method heuristic \
    --target-bits 6
```