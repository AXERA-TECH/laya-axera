"""Experimental Metal GELU/gate kernel, using MLX's same erf implementation.

The vendored erf/expm1 helpers are from MLX v0.32.2, copyright Apple and
Norbert Juffa. Their original notices are retained in vendor/ and MLX_LICENSE.
Only this experiment uses them; the production package is unaffected.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

VENDOR = Path(__file__).with_name("vendor")
helpers = "\n".join(
    line
    for filename in ("mlx_expm1f.h", "mlx_erf.h")
    for line in (VENDOR / filename).read_text().splitlines()
    if not line.startswith(("#include", "#pragma"))
)
GELU_GATE = mx.fast.metal_kernel(
    name="laya_experimental_exact_gelu_gate",
    input_names=["inp"],
    output_names=["out"],
    header="namespace laya_erf { using namespace metal;\n" + helpers + "\n}",
    source="""
        uint elem = thread_position_in_grid.x;
        uint row = elem / I;
        uint col = elem % I;
        T value = inp[row * (2 * I) + col];
        T gate = inp[row * (2 * I) + I + col];
        T scaled = value / T(1.4142135623730951);
        T erf_value = T(laya_erf::erf(float(scaled)));
        T gelu = T(T(value * T(T(1) + erf_value)) / T(2));
        out[elem] = gelu * gate;
    """,
    compile_options={"math_mode": "safe"},
)


def metal_gelu_gate(x):
    if x.dtype != mx.float16:
        raise ValueError("This research kernel has only been implemented and tested for FP16")
    if x.shape[-1] % 2:
        raise ValueError("GELU/gate input needs two equal-width branches")
    intermediate = x.shape[-1] // 2
    shape = (*x.shape[:-1], intermediate)
    return GELU_GATE(
        inputs=[x],
        template=[("T", x.dtype), ("I", intermediate)],
        grid=(x.size // 2, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[x.dtype],
    )[0]


class MetalMLP(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.Wi, self.Wo = layer.Wi, layer.Wo

    def __call__(self, x):
        return self.Wo(metal_gelu_gate(self.Wi(x)))
