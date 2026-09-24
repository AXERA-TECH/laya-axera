# Laya-AXERA

**在爱芯元智（AXERA）边缘 NPU 上运行的开源权重类型化决策模型。**

基于 [PyAXEngine](https://github.com/AXERA-TECH/pyaxengine) 推理
[AXERA-TECH/Laya](https://huggingface.co/AXERA-TECH/Laya) 的 AXModel checkpoint，
并提供网页演示：一个决策台和五个小游戏，其中一个可以和 AI 对战。

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

`laya-axera serve` 提供六个页面和一套 JSON 接口，共用同一份驻留的 checkpoint。每个游戏都可以
交给 AI 玩，也可以切到「手动」自己玩：键盘用方向键或 WASD，空格执行动作，也可以点页面上的按钮。
手动模式下 AI 仍然每步都在做判断，页面会显示你和它是否选得一样。

- **决策台**：编辑 state 和 questions，在 NPU 上运行，看每个问题的概率分布和耗时；
  可一键载入 checkpoint 自带的验证示例。
- **贪吃蛇**：每步问三个问题（走向 / 风险 / 食物）。确定性的循环护栏会纠正不安全的走法，并统计次数。
- **飞鸟**：每步一个二选一（拍翅 / 滑翔），穿过高低不一的石柱空隙。
- **落块**：启发式先挑出 4 个候选落点，模型用 `noul` 对每个单独打分，放在分数最高的位置。
- **打方块**：每步在左移 / 右移 / 不动之间三选一。实测 400 步与规划器 400/400 一致。
- **坦克对决**：你和 AI 实时对战。战场在浏览器里运行，只有 AI 的每一步来自 `/api/tank/decide`。
  难度（简单 / 困难 / 地狱）限制 AI 的决策频率（每 900 / 450 / 220 ms 一次）以及移动和射击速度。

各游戏的提示措辞都经过模型概率实测选定。坦克对决的选项用罗盘方向（north / south / west /
east）：`right` 在英文里同时有「正确」的意思，会把概率吸到自己身上，`fire` 也一样，所以开火
被做成某个方向的一种执行方式。状态句固定、选项描述只有四档，整个输入空间一共 108 种组合，
已经全部在 NPU 上跑过，模型在每一种情况下都会执行规划器给出的最佳动作。

接口：`GET /api/info`、`GET /api/samples/{name}`、`POST /api/predict`，
`POST /api/{snake,bird,blocks,bricks}/new`，`POST /api/{snake,bird,bricks}/step`（带 `action`
即为手动操作），`POST /api/blocks/step`、`POST /api/blocks/place`，以及 `POST /api/tank/decide`。

**决策台**

![决策台](docs/playground.png)

**贪吃蛇**

![贪吃蛇](docs/snake.png)

**飞鸟**

![飞鸟](docs/bird.png)

**落块**

![落块](docs/blocks.png)

**打方块**

![打方块](docs/bricks.png)

**坦克对决**

![坦克对决](docs/tank.png)

## 与板端验证输出的一致性

三个 checkpoint 均复现了模型包内 `axllm` 在 AX8850 板上录制的 `sample_output.json`：
选中标签完全一致，概率对齐到小数点后 4 位（如 multilingual：billing 1.0000、
urgency 1.9359、refund 0.9925、churn 0.9400）。

```bash
pytest tests/test_common.py tests/test_tank_planner.py       # 无需硬件
node tests/js/tank_sim.test.js                                # 坦克对决的对战模拟
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
