# kaggle-jane-street-forecasting

## Model Implementation
### Time-Series Transformer
The overall [model architecture](https://github.com/bcollico/kaggle-jane-street-forecasting/blob/main/attention_model/attention_model.py) is a decoder-only, autoregressive model based on [Llama 3](https://ai.meta.com/blog/meta-llama-3/). Each transformer block includes [Group Query Attention](https://arxiv.org/abs/2305.13245) (GQA), [Rotary Positional Encoding](https://arxiv.org/abs/2104.09864) (RoPE), SwiGLU feed-forward, and scaled dot product attention.

The first transformer layer is cross-attention between the feature embedding and the responder embedding. The remaining layers are self-attention.

The model is trained using [Backprop Through Time](https://en.wikipedia.org/wiki/Backpropagation_through_time) (BPTT), leveraging gradient checkpointing to reduce peak GPU memory in exchange for lower training throughput. This is adopted to allow the model to take advantage of the memory mechanism, learning to leverage historical/temporal information effectively and unlock long-context reasoning. 

#### Model Inputs
The input to the model is
- `date_ids`: The integer valued absolute date ID in [0, 365).
- `time_ids`: The integer time ID within a date in [0, 1000).
- `symbol_ids`: The integer market symbol IDs in [0, 255).
- `features`: Input features for N timesteps [t_1, t_N+1]. Multiple feature symbols are present at each timestep, and the number of symbols at each timestep may vary.
- `responders` The lagged responders for each timestep [t_0, t_N]. The number of responders at each timestep matches the number of fetaures.

#### Causal Masking
To ensure that future information is not leaked to the model, a causal mask is crafted which prevents features from attending to future responders. 
```
E.g. id_mat_1 = [1, 1, 2, 2, 2, 3], id_mat_2 = [0, 1, 2, 2, 3]

causal attention mask -- attend to rows with the same ID or less.

output = [False, False,  True,  True,  True],
         [False, False,  True,  True,  True],
         [False, False, False, False,  True],
         [False, False, False, False,  True],
         [False, False, False, False,  True],
         [False, False, False, False, False]]
```

#### Tokenization
The inputs are tokenized to a standard feature dimension before passing them into the first transformer block. Each of the integer IDs (date, time, symbol) are mapped to a learnable `nn.Embedding` vector. The features and responders are projected to the token dimension via an `nn.Linear` layer.

The features and responders have separate embeddings to allow separate information to be learned about the input and output data. In both cases, the input tokens are formed by summing all of the embeddings for the features/responders + date + time + symbol.

#### Model Outputs
The model outputs predictions clamped to [-5.0, 5.0] predicting the responders from timestep [t_1, t_n+1]. The regression loss is calcuated against the difference between the predicted responders and the actual responders.

### Memory Mechanism
Each transformer block includes the [infini-attention](https://arxiv.org/abs/2404.07143) memory mechanism. See the paper for details and the implementation [here](https://github.com/bcollico/kaggle-jane-street-forecasting/blob/4d86267a2396b2743e53bbc0d3cc8ab67ad5810a/attention_layers/attention_layers.py#L355).

### Efficient Grouped Query Attention
The built-in Pytorch[ GQA implementation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) uses an inefficient `repeat_interleave` to multiply each key and value against its respective query group.
>
> ```
> if enable_gqa:
>   key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
>   value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
> ```

This can be more efficiently posed by viewing the queries as a rectangular matrix for multiplication against the key matrix. The matrix shapes are
- `query`: (batch_size, num_heads, num_queries * sequence_length, head_feature_dim)
- `key`: (batch_size, num_heads, sequence_length, head_feature_dim)
- `value`: (batch_size, num_heads, sequence_length, head_feature_dim)

Attention is calculated as usual `softmax(query @ key^T / scale) @ value` and reshaped to (batch_size, num_heads * num_queries, sequence_length, feature_dim).

This was [benchmarked](https://github.com/bcollico/kaggle-jane-street-forecasting/blob/4d86267a2396b2743e53bbc0d3cc8ab67ad5810a/attention_layers/attention_layers_bm.py#L127) using various input tensor shapes and posted >15% improvement over the built-in Pytorch GQA.

## Data Loading
Various data loading and sampling strategies were implemented for Parquet tabular data under the assumption that the dataset is too large to load into memory all at once. Naively loading 1 row at a time randomly from a set of parquet files is prohibitvely slow and requires optimization in the form of caching. Strategies include:
- Randomly load a full row group into memory at a time. Exhaustively and randomly sample from within this row-group before loading the next row group.
- Row-group caching can be extended to loading a batch of row-groups spanning temporal windows of interest. This is implemented for time-series training where temporal coherence is necessary.

Huge gains were made by profiling with [Scalene](https://github.com/plasma-umass/scalene) to remove redundant operations and optimize I/O patterns (interact with the file system as little as possible, loading as many row groups as possible at a time). This makes the current dataloading scheme extensible to larger scale datasets (even though this Kaggle dataset fits in memory on a moderately sized GPU).

## Getting Started
1. Install required packages using `pip install -r requirements.txt`. It is recommended to use Conda or other virtual environment.
1. Setup the Kaggle API following the [official instructions](https://github.com/Kaggle/kaggle-api?tab=readme-ov-file#installation).
2. Create the data directory and download the dataset
```
mkdir -p /kaggle/input \
 && kaggle competitions download -c jane-street-real-time-market-data-forecasting \
 && mkdir jane-street-real-time-market-data-forecasting \
 && unzip jane-street-real-time-market-data-forecasting.zip -d jane-street-real-time-market-data-forecasting \
 && mv jane-street-real-time-market-data-forecasting /kaggle/input/
```
3. To train a model, run `python train/train.py`
