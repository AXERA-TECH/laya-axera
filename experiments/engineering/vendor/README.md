# MLX math helpers for an experimental FP16 kernel

These two unmodified source files come from Apple MLX **v0.32.2**:

- [mlx_erf.h](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/erf.h)
- [mlx_expm1f.h](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/expm1f.h)

The experimental `kernels.py` combines their source into a namespaced Metal
header, omitting the original preprocessor includes at runtime. Reusing MLX's
own error-function implementation preserves the GELU definition used in the
baseline instead of substituting a tanh or sigmoid approximation. Floating-point
rounding parity is checked separately by `microbench.py` and full-model probes.

Apple's MIT license is retained in [MLX_LICENSE](MLX_LICENSE). The original
Norbert Juffa notice and redistribution terms are also retained in
`mlx_expm1f.h`. These helpers are used only by the research prototype and are not
part of the production `laya_mlx` package. The custom kernel supports FP16 only.
