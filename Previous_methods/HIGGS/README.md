
```
python create_quantized_layers.py \
     --model meta-llama/Llama-3.1-8B-Instruct  \
     --output-dir ./quantized_layers \
     --precisions FP8_Dynamic FP8_Block NVFP4 \
     --num-calibration-samples 32
```


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