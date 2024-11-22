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

from attention_model.attention_layers import (
    InfiniGroupedQueryAttention,
    RotaryPositionalEncoding,
)

N_RUNS = 1000

N_QUERY = 8
N_HEAD = 4
D_HEAD = 64
SEQ_LEN = 200
BATCH_SIZE = 10

D_MODEL = N_QUERY * N_HEAD * D_HEAD


def run_torch_scaled_attention():
    """Run torch attention without positional encoding."""
    query = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).cuda()
    key = torch.randn(BATCH_SIZE, SEQ_LEN, D_HEAD * N_HEAD).cuda()
    value = (
        torch.randn(BATCH_SIZE, SEQ_LEN, D_HEAD * N_HEAD)
        .cuda()
        .reshape(BATCH_SIZE, SEQ_LEN, N_HEAD, -1)
        .transpose(1, 2)
    )

    key = key.reshape(BATCH_SIZE, SEQ_LEN, N_HEAD, -1).transpose(1, 2)
    query = query.reshape(BATCH_SIZE, SEQ_LEN, -1, D_HEAD).transpose(1, 2)
    value = value.reshape(BATCH_SIZE, SEQ_LEN, N_HEAD, -1).transpose(1, 2)

    torch_time = bm.Timer(
        stmt=(
            "torch.nn.functional.scaled_dot_product_attention("
            "query=q, "
            "key=k, "
            "value=v, "
            "enable_gqa=True)"
        ),
        globals={
            "q": query,
            "k": key,
            "v": value,
        },
    )

    print(torch_time.timeit(N_RUNS))


def run_custom_layer():
    """Run custom attention implementation with RoPE positional encoding."""

    query = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).cuda()
    key = torch.randn(BATCH_SIZE, SEQ_LEN, D_HEAD * N_HEAD).cuda()
    value = torch.randn(BATCH_SIZE, SEQ_LEN, D_HEAD * N_HEAD).cuda()

    # Instantiating and pre-computing the rope object does not change the
    # timing compared to letting the layer instantiate it's own rope.
    rope = RotaryPositionalEncoding(d_model=D_MODEL, device=key.device)

    rope.forward(x=query)

    layer = InfiniGroupedQueryAttention(
        d_model=D_MODEL,
        n_query=N_QUERY,
        n_head=N_HEAD,
    ).cuda()

    layer_time = bm.Timer(
        stmt="layer.scaled_dot_product_attention(query=query, key=key, value=value)",
        globals={"layer": layer, "query": query, "key": key, "value": value},
    )

    print(layer_time.timeit(N_RUNS))


def run_rope():
    """Benchmark the positional encoding forward pass by itself."""
    query = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).cuda()
    key = torch.randn(BATCH_SIZE, SEQ_LEN, D_HEAD * N_HEAD).cuda()

    # Instantiating and pre-computing the rope object does not change the
    # timing compared to letting the layer instantiate it's own rope.
    rope = RotaryPositionalEncoding(d_model=D_MODEL, device=key.device)
    rope.forward(query)

    rope_time = bm.Timer(
        stmt="rope.forward(x=query), rope.forward(x=key)",
        globals={"rope": rope, "query": query, "key": key},
    )

    print(rope_time.timeit(N_RUNS))


def run_attention_layers_bm():
    """Runs two implementations of scaled_dot_product_attention for grouped query
    attention and prints their timing output to console.

    Compares timing of torch implementation and custom implementation. Note that
    torch.utils.benchmark must be used to properly capture the timing due to
    complexities around GPU synchronization. E.g. when trying to use time.time() for this
    benchmark, the order of the function calls would vary the total time from < 0.1s to
    > 10s for a small amount of runs.

    See https://pytorch.org/tutorials/recipes/recipes/benchmark.html for details.

    Last results:
    rope.forward(x=query), rope.forward(x=key)
        16.24 ms
        1 measurement, 1000 runs , 1 thread
    torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, enable_gqa=True)
        16.81 ms
        1 measurement, 1000 runs , 1 thread
    layer.scaled_dot_product_attention(query=query, key=key, value=value)
        16.95 ms
        1 measurement, 1000 runs , 1 thread

    NB: you get a speedup of about 25% by using torch.compile to optimize the SDPA functions.
    """

    run_rope()
    run_torch_scaled_attention()
    run_custom_layer()


if __name__ == "__main__":
    run_attention_layers_bm()
