"""Simple benchmark for comparing custom scaled_dot_product_attention for grouped head attention
to the pytorch implementation."""

import torch
import torch.utils.benchmark as bm

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from attention_model.attention_model import GroupedRecurrentMultiHeadAttention

N_RUNS = 100000


def run_attention_model_bm():
    """Runs two implementations of scaled_dot_product_attention for grouped query
    attention and prints their timing output to console.

    Compares timing of torch implementation and custom implementation. Note that
    torch.utils.benchmark must be used to properly capture the timing due to
    complexities around GPU synchronization. E.g. when trying to use time.time() for this
    benchmark, the order of the function calls would vary the total time from < 0.1s to
    > 10s for a small amount of runs.

    See https://pytorch.org/tutorials/recipes/recipes/benchmark.html for details.
    
    Last results ("naive" torch implementation is faster!):
        <torch.utils.benchmark.utils.common.Measurement object at 0x75508013d840>
    torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, enable_gqa=True)
    12.81 ms
    1 measurement, 1000 runs , 1 thread
    <torch.utils.benchmark.utils.common.Measurement object at 0x75508013d7e0>
    layer.scaled_dot_product_attention(query=query, key=key, value=value)
    10.55 ms
    1 measurement, 1000 runs , 1 thread
    """

    n_query = 4
    n_head = 8
    d_head = 128
    seq_len = 100
    batch_size = 100
    d_model = n_query * n_head * d_head
    layer = GroupedRecurrentMultiHeadAttention(
        d_model=d_model,
        n_query=n_query,
        n_head=n_head,
    ).cuda()

    query = torch.randn(batch_size, seq_len, d_model).cuda()
    key = torch.randn(batch_size, seq_len, d_head * n_head).cuda()
    value = torch.randn(batch_size, seq_len, d_head * n_head).cuda()

    k = key.reshape(batch_size, seq_len, n_head, -1).transpose(1, 2)
    q = query.reshape(batch_size, seq_len, -1, d_head).transpose(1, 2)
    v = value.reshape(batch_size, seq_len, n_head, -1).transpose(1, 2)

    torch_time = bm.Timer(
        stmt="torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, enable_gqa=True)",
        globals={"q": q, "k": k, "v": v},
    )

    layer_time = bm.Timer(
        stmt="layer.scaled_dot_product_attention(query=query, key=key, value=value)",
        globals={"layer": layer, "query": query, "key": key, "value": value},
    )

    print(torch_time.timeit(N_RUNS))
    print(layer_time.timeit(N_RUNS))


if __name__ == "__main__":
    run_attention_model_bm()
