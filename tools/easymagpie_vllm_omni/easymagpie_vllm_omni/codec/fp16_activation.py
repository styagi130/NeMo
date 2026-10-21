"""Default-off component: reproduce each retained FP16 pointwise rounding boundary.

This changes no convolution, history, layout or precision policy. Actual GPU
bitwise equivalence is a qualification gate, not an assumed property of fusion.
"""

from collections import Counter
import torch
from triton.language.extra.cuda import libdevice
from vllm.triton_utils import tl, triton

COUNTS = Counter()


@triton.jit
def _packed_half_snake_fp16_kernel(
    input_ptr,
    residual_ptr,
    alpha_ptr,
    output_ptr,
    half_numel,
    CHANNELS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One lane handles a matched snake/leaky pair, so sin runs only for the
    # snake half. Tensor boundaries below deliberately round back to Half.
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pair < half_numel
    half_channels = CHANNELS // 2
    channel = pair % half_channels
    index = (pair // half_channels) * CHANNELS + channel
    snake = tl.load(input_ptr + index, mask=mask, other=0.0).to(tl.float32)
    leaky = tl.load(input_ptr + index + half_channels, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        snake_res = tl.load(residual_ptr + index, mask=mask, other=0.0).to(tl.float32)
        leaky_res = tl.load(residual_ptr + index + half_channels, mask=mask, other=0.0).to(tl.float32)
        snake = (snake_res + snake).to(tl.float16).to(tl.float32)
        leaky = (leaky_res + leaky).to(tl.float16).to(tl.float32)
    alpha = tl.load(alpha_ptr + channel, mask=mask, other=1.0).to(tl.float32)
    multiplied = (alpha * snake).to(tl.float16).to(tl.float32)
    sine = libdevice.sin(multiplied).to(tl.float16).to(tl.float32)
    squared = (sine * sine).to(tl.float16).to(tl.float32)
    denominator = (alpha + 1.0e-9).to(tl.float16).to(tl.float32)
    quotient = libdevice.div_rn(squared, denominator).to(tl.float16).to(tl.float32)
    periodic = (snake + quotient).to(tl.float16)
    negative = (leaky * 0.01).to(tl.float16)
    leaky_out = tl.where(leaky > 0.0, leaky.to(tl.float16), negative)
    tl.store(output_ptr + index, periodic, mask=mask)
    tl.store(output_ptr + index + half_channels, leaky_out, mask=mask)


def packed_half_snake_fp16(inputs, alpha, residual=None):
    if (
        not inputs.is_cuda
        or inputs.dtype != torch.float16
        or alpha.dtype != torch.float16
        or inputs.ndim != 2
        or inputs.shape[1] != 2 * alpha.numel()
        or not inputs.is_contiguous()
        or not alpha.is_contiguous()
        or inputs.device != alpha.device
    ):
        raise ValueError('exact contiguous CUDA FP16 packed input/alpha required')
    if residual is not None and (
        residual.dtype != inputs.dtype
        or residual.shape != inputs.shape
        or residual.device != inputs.device
        or not residual.is_contiguous()
    ):
        raise ValueError('residual must match contiguous CUDA FP16 input')
    output = torch.empty_like(inputs, memory_format=torch.contiguous_format)
    if not inputs.numel():
        return output
    block = 256
    _packed_half_snake_fp16_kernel[(triton.cdiv(inputs.numel() // 2, block),)](
        inputs,
        inputs if residual is None else residual,
        alpha,
        output,
        inputs.numel() // 2,
        CHANNELS=inputs.shape[1],
        HAS_RESIDUAL=residual is not None,
        BLOCK=block,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    COUNTS['launch_requests'] += 1
    COUNTS['output_elements'] += inputs.numel()
    return output
