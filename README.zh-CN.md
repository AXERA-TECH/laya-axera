# laya-mlx

[English](README.md) · [本机 benchmark](BENCHMARKS.md) · [上游 Laya](https://github.com/NandhaKishorM/laya)

在 Apple silicon 上用 **原生 MLX** 运行 Laya 的结构化决策模型。支持 `choice`（分类）、`score`（有序评分）和 `noul`（P(true)），以及原版的问题格式、概率校准、语言路由和应用预设。

编码器、决策 Transformer、评分头和 action 头都在 MLX 中计算。推理运行时依赖 Rust tokenizer，**不需要 PyTorch 或 Transformers**。三个检查点均可直接读取上游 safetensors，也可导出独立的 MLX 权重目录。

| 检查点 | 编码器 | 参数量 | 最大上下文 |
|---|---|---:|---:|
| `convaiinnovations/laya` | ModernBERT-large | 421M | 512 |
| `convaiinnovations/laya-multilingual` | mmBERT-base | 322M | 1,024 |
| `convaiinnovations/laya-typed-decisions` | ModernBERT-large | 421M | 1,024 |

上下文预算包含问题、选项和输入状态。中文等非英语输入应使用 multilingual 检查点。本项目实现推理与权重转换；RLCD 训练和微调继续使用上游项目。

## 安装与运行

需要 Apple silicon Mac、macOS 26+ 和 Python 3.11+。本机实测环境为 M3 Max（40 核 GPU、128 GB 内存）、macOS 27.2、Python 3.12.13、MLX 0.32.2。锁定依赖中的 MLX wheel 要求 macOS 26+。

```bash
gh repo clone mizorewww/laya-mlx
cd laya-mlx
uv sync
uv run python examples/quickstart.py
```

或从 GitHub 直接安装：

```bash
python -m pip install 'git+https://github.com/mizorewww/laya-mlx.git'
```

Python 示例：

```python
import laya_mlx as laya

agent = laya.load("convaiinnovations/laya-multilingual")
result = agent.predict(
    "发票被重复扣款，请今天退款。",
    {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": ["billing", "technical", "sales"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
    },
)
print(result["answers"])
```

默认使用 FP16。需要更接近原版 FP32 的数值时使用 `dtype="float32"`。`batch_size=16` 控制每次计算的问题数，更多问题会分批处理。概率按原版格式保留四位小数；不同精度可能造成小幅差异，实测误差见 benchmark 报告。

命令行支持文本或 JSON 状态：

```bash
uv run laya-mlx predict \
  --model convaiinnovations/laya-multilingual \
  --state '发票被重复扣款，请退款。' \
  --questions examples/questions.json
```

本仓库已下载的权重位于 `models/` 时，将 `--model` 改成相应本地目录即可避免再次下载。

## 转换权重

```bash
uv run laya-mlx convert \
  --model convaiinnovations/laya \
  --dtype float16 \
  --output models/laya-mlx-fp16
```

转换后可以通过 `laya.load("./models/laya-mlx-fp16")` 直接加载。输出目录包含模型、配置、tokenizer 和来源元数据。已有目录不会被覆盖，模型权重不会提交到 GitHub。

原始检查点本身存储的是 FP16 权重；这里的转换调整参数命名与计算精度，不涉及重新训练或低比特量化。

## 路由、测试和 benchmark

`Router`、`triage_questions`、`email_questions`、`guard_questions`、`moderation_questions` 等接口保留上游用法，将导入名改为 `laya_mlx` 即可。typed-decisions 检查点可通过 `task="typed_decisions"` 显式指定；`Router(preload=True)` 可预加载三个模型。

详细的 API、测试和复现命令见 [英文 README](README.md)。[BENCHMARKS.md](BENCHMARKS.md) 包含本机 PyTorch MPS FP32、MLX FP32 与 MLX FP16 的端到端 P50/P95、吞吐量、内存、数值一致性、重复运行和固定抽样分类测试。所有原始测量数据位于 [benchmarks/results](benchmarks/results)，GPU 测试应串行运行。

这是独立的 MLX 移植，模型能力及其限制来自上游；模型输出概率不等于答案必然正确。采用 Apache-2.0，原作者与移植说明见 [NOTICE](NOTICE)。
