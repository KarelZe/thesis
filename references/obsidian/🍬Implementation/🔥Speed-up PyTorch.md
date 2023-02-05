- Train at a lower precision e. g. with 16 bit floats. Numeric issues are already adressed. https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/ or https://www.youtube.com/watch?v=OqCrNkjN_PM
- Avoid non-deterministic operations. Likely not applicable here. https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
- Speed up model with `TensorRT`. Will probably not affect training, but only on finished model. https://medium.com/analytics-vidhya/1-line-of-python-code-that-will-speed-up-your-ai-by-up-to-6x-667a9bf53d7d
- Increase `num_workers` as explained here: https://wandb.ai/srishti-gureja-wandb/posts/How-to-Eliminate-the-Data-Processing-Bottleneck-with-PyTorch--VmlldzoyNDMxNzM1
- 
## Decoupeling of serving batches and gpu operations
**Status quo:**
Similar to this one:
```python
all_inputs, all_true_labels = /

pd.read_csv(input_data).iloc[:,0:100], pd.read_csv(input_data).iloc[:, 100]

for epoch in range(n_epochs):

for i in range(num_batches):

batch_inputs, batch_true_labels = /

all_inputs.iloc[i*64:(1+i)*64, :], all_true_labels.iloc[i*64:(1+i)*64, :]

# applying any transformations to batched data etc.

output = model(batch_inputs)

# model training steps
```

![[pre-data-loader.png]]
**To be:**
![[after-data-loader.png]]
(adapted from https://wandb.ai/srishti-gureja-wandb/posts/How-to-Eliminate-the-Data-Processing-Bottleneck-with-PyTorch--VmlldzoyNDMxNzM1)
## Single GPU:

![[techniques-to-speed-up-training.png]]
(https://huggingface.co/docs/transformers/perf_train_gpu_one)

Monitor memory usage using.
```
pip install transformers datasets accelerate nvidia-ml-py3
```

## Batch size
- See recommendations on batch size: https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#batch-size
- See recommendations for tensors: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

## Dimension Quantization Effects
- https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization

## Optimizer
- Optimized implementation of AdamW `adamw_apex_fused` 
- See: https://github.com/NVIDIA/apex (Integrated into PyTorch already?)
- Patch of fusedAdamW not merged? https://github.com/pytorch/pytorch/pull/88015/files
## Anatomy of Model's Operations

Transformers architecture includes 3 main groups of operations grouped below by compute-intensity.

1.  **Tensor Contractions**
    
    Linear layers and components of Multi-Head Attention all do batched **matrix-matrix multiplications**. These operations are the most compute-intensive part of training a transformer.
    
2.  **Statistical Normalizations**
    
    Softmax and layer normalization are less compute-intensive than tensor contractions, and involve one or more **reduction operations**, the result of which is then applied via a map.
    
3.  **Element-wise Operators**
    
    These are the remaining operators: **biases, dropout, activations, and residual connections**. These are the least compute-intensive operations.
    

This knowledge can be helpful to know when analyzing performance bottlenecks.

This summary is derived from [Data Movement Is All You Need: A Case Study on Optimizing Transformers 2020](https://arxiv.org/abs/2007.00072)


## Multi-GPU training:
- https://huggingface.co/docs/transformers/perf_train_gpu_many

## Timing PyTorch
- https://pytorch.org/tutorials/recipes/recipes/benchmark.html


## Model
- See various implementation tricks used here: https://github.com/microsoft/DeepSpeed-MII/tree/main/examples/benchmark/txt2img


Data parallelism or model parallelism:


## Data Loader
- Use pinned memory in data loader: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/20 and https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
- use gpu-accelerated data loaders: https://github.com/NVIDIA-Merlin/dataloader

-   `DataLoader(pin_memory=True, ...)` which ensures that the data gets preloaded into the pinned memory on CPU and typically leads to much faster transfers from CPU to GPU memory.
-   `DataLoader(num_workers=4, ...)` - spawn several workers to pre-load data faster - during training watch the GPU utilization stats and if it’s far from 100% experiment with raising the number of workers. Of course, the problem could be elsewhere so a very big number of workers won’t necessarily lead to a better performance. (copied from https://huggingface.co/docs/transformers/perf_train_gpu_one#dataloader)

## Profiler
https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html


https://huggingface.co/docs/transformers/performance

https://spell.ml/blog/gradient-checkpointing-pytorch-YGypLBAAACEAefHs

## Scientific papers
- https://arxiv.org/pdf/2007.00072.pdf


## Gradient checkpointing
- Decreases speed, but lets one handle models that would not fit into gpu ram otherwise. See https://spell.ml/blog/gradient-checkpointing-pytorch-YGypLBAAACEAefHs
- In PyTorch `torch.utils.checkpoint.checkpoint`
- See paper https://arxiv.org/abs/1604.06174

