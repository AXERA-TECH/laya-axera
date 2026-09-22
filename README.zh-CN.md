# Laya-AXERA

**在爱芯元智（AXERA）边缘 NPU 上运行的开源权重类型化决策模型。**

基于 [PyAXEngine](https://github.com/AXERA-TECH/pyaxengine) 推理
[AXERA-TECH/Laya](https://huggingface.co/AXERA-TECH/Laya) 的 AXModel checkpoint，
并提供网页演示：决策台 + 每一步都由 Laya 实时决策的贪吃蛇。

单张 AX8850（AXCL）上 multilingual checkpoint **约 31 ms/问题**，**0 个输出 token**。
不依赖 PyTorch / Transformers 运行时 / 云端 API —— 分词用 Hugging Face Rust tokenizer，
编码器全部在 NPU 上执行。

[English](README.md) ·
[模型包](https://huggingface.co/AXERA-TECH/Laya) ·
[上游 Laya](https://github.com/NandhaKishorM/laya) ·
[所适配的 MLX 移植版](https://github.com/mizorewww/laya-mlx)

Laya 是双向决策模型：对文本或结构化状态回答受约束的问题，单次前向完成，不生成文本。

- `choice`：在 2–4 个命名选项上给出概率分布。
- `score`：在 2–4 个有序等级上给出概率与期望分。
- `noul`：命题成立的概率 P(true)。

## 支持平台

| 平台 | Provider | 说明 |
|---|---|---|
| AX8850 板端（片上） | `AxEngineExecutionProvider` | aarch64, NPU3 |
| x86/arm64 主机 + AXCL 卡（PCIe / M.2） | `AXCLRTExecutionProvider` | `device_id` 选卡 |

Provider 自动选择，也可通过 `provider=` 强制指定。

## 安装

```bash
git clone https://github.com/AXERA-TECH/laya-axera.git
cd laya-axera
pip install -e '.[web]'
```

PyAXEngine 不在 PyPI 上，请先在设备/主机上安装
[pyaxengine releases](https://github.com/AXERA-TECH/pyaxengine/releases) 的 `axengine` wheel。
解析 ModernBERT / mmBERT 的 tokenizer 需要 `tokenizers>=0.21`
（Transformers 4.41 自带的版本过旧）。

下载模型包（三个 checkpoint，约 1.5 GiB）：

```bash
hf download AXERA-TECH/Laya --local-dir models/Laya
# 国内可用 hf-mirror.com 或 ModelScope
```

| Checkpoint | 骨干 | 单问题 NPU 延迟 | 推荐用途 |
|---|---|---:|---|
| `english/` | ModernBERT-large | 片上约 70 ms，AXCL 约 74 ms | 英文路由、护栏、工单分流 |
| `multilingual/` | mmBERT-base | 片上约 28 ms，AXCL 约 31 ms | 中文及多语言输入 |
| `typed-decisions/` | ModernBERT-large | 片上约 70 ms，AXCL 约 74 ms | 发票、安全、Agent 轨迹 |

延迟用 `python examples/bench.py <checkpoint>` 实测：片上为 AX8850 开发板
（multilingual 28.8 ms、english 71.1 ms，PyAXEngine，模型经 NFS 挂载），AXCL 为空闲
x86 主机卡。贪吃蛇每步问 3 个问题，单步耗时约为 3 倍单问题延迟。

## Python API

```python
import laya_axera as laya

agent = laya.load("models/Laya/multilingual")   # device_id=0，provider 自动选择
result = agent.predict(
    "发票4411重复扣款，请今天退还多扣的金额，否则我们会取消服务。",
    {
        "department": {
            "type": "choice",
            "instructions": "这条请求应由哪个团队处理？",
            "criteria": {"billing": "扣款与退款", "technical": "故障报错", "sales": "价格采购"},
        },
        "urgency": {
            "type": "score",
            "instructions": "这条请求有多紧急？",
            "criteria": ["常规", "尽快", "今天必须解决"],
        },
        "refund": {"type": "noul", "instructions": "用户是否明确要求退款？"},
    },
)
print(result["answers"]["department"]["choice"])        # billing
```

`system_one` 是 `predict` 的别名；`predict_request` 直接接受模型包的
`{"state": ..., "questions": ...}` 请求对象。输出 schema（choice / probabilities /
score / legend / noul / confidence / `action.act_probability`）与上游 Laya 及打包的
`axllm` 运行时一致，另附每个问题实测的 `npu_latency_ms`。

## 命令行

```bash
# 单个请求文件（与模型包 sample_request.json 同格式）
laya-axera run models/Laya/multilingual --input models/Laya/multilingual/sample_request.json

# 驻留 JSON Lines 模式：每行一个请求，/exit 退出
laya-axera run models/Laya/multilingual --device 1

# 网页演示：8010 端口，三个 checkpoint 首次使用时加载
laya-axera serve --root models/Laya --port 8010
```

## 网页演示

- **决策台** —— 编辑 state 和 questions，提交到 NPU，以概率条形式查看答案与逐问题延迟；
  一键载入所选 checkpoint 的板端验证示例。
- **Flappy Bird** —— 每步一个二选一决策（拍翅/滑翔），单问题约 30ms：规划器描述两个动作的后果，模型选择，护栏只纠正致命提议。
- **贪吃蛇** —— laya-mlx 贪吃蛇演示的网页版。每一步向驻留 checkpoint 问三个问题
  （走向 / 风险 / 食物），确定性的循环安全护栏会纠正不安全的提议并统计每次干预。
  multilingual + 单张 AXCL AX8850 约 10 步/秒。

接口：`GET /api/info`、`GET /api/samples/{name}`、`POST /api/predict`、
`POST /api/snake/new`、`POST /api/snake/step`。

## 与板端验证输出的一致性

三个 checkpoint 均复现了模型包内 `axllm` 在 AX8850 板上录制的 `sample_output.json`：
选中标签完全一致，概率对齐到小数点后 4 位（如 multilingual：billing 1.0000、
urgency 1.9359、refund 0.9925、churn 0.9400）。

```bash
pytest tests/test_common.py                                   # 无需硬件
LAYA_AXERA_MODEL_DIR=models/Laya/multilingual pytest tests/   # 在 NPU 上
```

## 图约束

AXModel 为固定形状：batch 1、256 token、至多 4 个选项、每个问题一次 NPU 前向。
量化可能使概率发生偏移；在自动化高影响动作前，请用有代表性的数据校准阈值。

## 致谢与许可

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。提示构造、校准、输出 schema 与
贪吃蛇演示改编自 [laya-mlx](https://github.com/mizorewww/laya-mlx) 及上游
[Laya](https://github.com/NandhaKishorM/laya)（Convai Innovations）。AXModel、tokenizer
与 PyAXEngine 参考脚本来自 [AXERA-TECH/Laya](https://huggingface.co/AXERA-TECH/Laya)
部署包。模型权重需单独下载，不包含在本仓库中。
