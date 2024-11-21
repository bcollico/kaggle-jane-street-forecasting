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

from attention_model.attention_model import (
    GroupedRecurrentMultiHeadAttention,
    RotaryPositionalEncoding,
)

N_RUNS = 1000

n_query = 4
n_head = 8
d_head = 1024
seq_len = 20
batch_size = 10
d_model = n_query * n_head * d_head


def run_torch_scaled_attention():
    """Run torch attention without positional encoding."""
    query = torch.randn(batch_size, seq_len, d_model).cuda()
    key = torch.randn(batch_size, seq_len, d_head * n_head).cuda()
    value = (
        torch.randn(batch_size, seq_len, d_head * n_head)
        .cuda()
        .reshape(batch_size, seq_len, n_head, -1)
        .transpose(1, 2)
    )

    key = key.reshape(batch_size, seq_len, n_head, -1).transpose(1, 2)
    query = query.reshape(batch_size, seq_len, -1, d_head).transpose(1, 2)
    value = value.reshape(batch_size, seq_len, n_head, -1).transpose(1, 2)

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


def run_torch_scaled_attention_with_rope():
    """Run torch attention function with RoPE encoding."""

    query = torch.randn(batch_size, seq_len, d_model).cuda()
    key = torch.randn(batch_size, seq_len, d_head * n_head).cuda()
    value = (
        torch.randn(batch_size, seq_len, d_head * n_head)
        .cuda()
        .reshape(batch_size, seq_len, n_head, -1)
        .transpose(1, 2)
    )

    rope = RotaryPositionalEncoding(d_model=d_model)

    torch_time = bm.Timer(
        stmt=(
            "torch.nn.functional.scaled_dot_product_attention("
            "query=rope.forward(q).reshape(batch_size, seq_len, -1, d_head).transpose(1, 2), "
            "key=rope.forward(k).reshape(batch_size, seq_len, n_head, -1).transpose(1, 2), "
            "value=v, enable_gqa=True)"
        ),
        globals={
            "q": query,
            "k": key,
            "v": value,
            "rope": rope,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "d_head": d_head,
            "n_head": n_head,
        },
    )

    print(torch_time.timeit(N_RUNS))


def run_custom_layer_with_rope():
    """Run custom attention implementation with RoPE positional encoding."""
    layer = GroupedRecurrentMultiHeadAttention(
        d_model=d_model,
        n_query=n_query,
        n_head=n_head,
    ).cuda()

    query = torch.randn(batch_size, seq_len, d_model).cuda()
    key = torch.randn(batch_size, seq_len, d_head * n_head).cuda()
    value = torch.randn(batch_size, seq_len, d_head * n_head).cuda()

    layer_time = bm.Timer(
        stmt="layer.scaled_dot_product_attention(query=query, key=key, value=value)",
        globals={"layer": layer, "query": query, "key": key, "value": value},
    )

    print(layer_time.timeit(N_RUNS))


def run_rope():
    """Benchmark the positional encoding forward pass by itself."""
    query = torch.randn(batch_size, seq_len, d_model).cuda()
    key = torch.randn(batch_size, seq_len, d_head * n_head).cuda()

    rope = RotaryPositionalEncoding(d_model=d_model, device=key.device)

    rope_time = bm.Timer(
        stmt="rope.forward(x=query), rope.forward(x=key)",
        globals={"rope": rope, "query": query, "key": key},
    )

    print(rope_time.timeit(N_RUNS))


def run_attention_model_bm():
    """Runs two implementations of scaled_dot_product_attention for grouped query
    attention and prints their timing output to console.

    Compares timing of torch implementation and custom implementation. Note that
    torch.utils.benchmark must be used to properly capture the timing due to
    complexities around GPU synchronization. E.g. when trying to use time.time() for this
    benchmark, the order of the function calls would vary the total time from < 0.1s to
    > 10s for a small amount of runs.

    See https://pytorch.org/tutorials/recipes/recipes/benchmark.html for details.

    Last results (they are about the same):
    <torch.utils.benchmark.utils.common.Measurement object at 0x737c62d416f0>
    torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, enable_gqa=True)
    16.47 ms
    1 measurement, 1000 runs , 1 thread
    <torch.utils.benchmark.utils.common.Measurement object at 0x737c62d425c0>
    layer.scaled_dot_product_attention(query=query, key=key, value=value)
    16.69 ms
    1 measurement, 1000 runs , 1 thread
    """

    run_rope()
    run_torch_scaled_attention()
    run_torch_scaled_attention_with_rope()
    run_custom_layer_with_rope()


if __name__ == "__main__":
    run_attention_model_bm()
