<div align="center">

# 🏥 Med-Bench-Arena

### 统一、配置驱动的医学与中医大模型 / 智能体评测竞技场

*一套规范化 schema 解耦 **数据集 · 后端 · 指标** —— 任意模型、任意基准、任意指标自由组合，零胶水代码。*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-13%2F13%20passing-brightgreen.svg)](tests/)
[![Benchmarks](https://img.shields.io/badge/benchmarks-40%2B%20已实测-8A2BE2.svg)](DATASETS.md)
[![TCM](https://img.shields.io/badge/中医-原生支持-c1272d.svg)](#-中医原生支持)
[![Ethics & Safety](https://img.shields.io/badge/伦理·安全-原生支持-2E8B57.svg)](configs/catalog_ethics_safety.yaml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-参与贡献)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/Med-Bench-Arena/blob/main/notebooks/Med_Bench_Arena_Colab.ipynb)

[English](README.md) · **简体中文**

</div>

---

**Med-Bench-Arena** 是 *MedEval 协议* 的参考实现：把 **40+ 个医学 / 中医 / 伦理安全基准** 接入 **任意** 大模型或智能体后端，并用 **12 种指标** 评测 —— 从最朴素的选择题正确率，到 rubric 评分的安全性、结构化的 **方剂 / 辨证** 匹配，再到交互式智能体的 `pass^k`。一套 **规范化 schema** 横亘在数据集、模型后端与指标之间，于是你得到的是免费的 `N 数据集 × M 后端 × K 指标` 组合，而非 `N×M×K` 的硬编码。

> 🔬 `configs/` 中的**每一个**数据集条目都经过调研，并**针对其线上源**（HuggingFace datasets-server / 仓库原始文件）**逐一核验** —— 真实的仓库 id、split、列名与答案编码。凡是只能部分复现的源（受限评分器、隐藏答案），其局限都**如实记录、绝不掩盖**。详见 [`DATASETS.md`](DATASETS.md)。

```bash
# 全程无需 API key、无需 GPU、无需联网 —— 确定性 mock 后端
python -m medeval run configs/example_smoke.yaml
```

---

## ✨ 核心亮点

- 🧩 **解耦式设计** —— 数据集、后端、指标只依赖 schema，彼此互不依赖。接入相似数据集 = 改一段 YAML，零代码。
- 📚 **40+ 个已实测基准** —— 选择题、开放问答（LLM 评审）、安全、多模态（舌象 / 影像）、交互式智能体，覆盖中英文。
- ⚖️ **医学伦理与安全，原生支持** —— 专设 [`catalog_ethics_safety.yaml`](configs/catalog_ethics_safety.yaml)：四原则伦理选择题（MedEthicsQA · PrinciplismQA · MedEthicEval · TCM_Humanities）、开放式伦理两难、AI 安全红队（CARES-18K，拒答 / 越狱 / 过度拒答）—— 正是模型最薄弱的一环。
- 🀄 **中医原生支持** —— 辨证证型链、方剂结构匹配（君臣佐使）、经络腧穴、古籍本体、舌象 / 脉象、真实名老中医医案，外加可下载的经典文献**知识图谱**。
- 🤖 **真实智能体回合** —— AgentClinic（OSCE + NEJM）、对接真实 **FHIR** 电子病历服务的 **MedAgentBench**、主动追问的 **MediQ** —— 均以 `pass^k` 评分。
- 🔌 **任意后端** —— 本地 **HF/vLLM**（批量）、**Poe**、**LiteLLM**（100+ 提供商 + 推荐评审模型），改一行 YAML 即可切换。
- ⚖️ **忠实评分** —— HealthBench 逐条 rubric、MedAgentBench 逐任务 FHIR 载荷校验（+ 官方受限 `refsol.py`）、带符号分值的安全 rubric。
- ⚡ **横向扩展** —— 天然并行的跨步分片，支持 **本地 / Ray / Slurm**；可断点续跑，无中心服务器。
- 📤 **可直接提交** —— 一键导出 **OpenCompass** / **MedBench** 上传格式。

---

## 📑 目录

[设计动机](#-设计动机) · [架构](#-架构) · [安装](#-安装) · [快速开始](#-快速开始) · [可靠性](#-可靠性与可复现) · [基准目录](#-基准目录) · [后端](#-后端) · [模型清单](#-模型清单医学与中医) · [指标](#-指标与评审) · [智能体](#-智能体) · [多模态](#-多模态舌象--影像) · [中医](#-中医原生支持) · [分布式](#-分布式调度) · [榜单提交](#-榜单提交) · [扩展](#-扩展) · [项目结构](#-项目结构) · [引用](#-引用) · [贡献](#-参与贡献) · [许可](#-许可)

---

## 💡 设计动机

三组张力驱动了整体设计（也决定了数据集的取舍）：

1. **考试饱和 vs. 真实临床** → 既要静态选择题，也要 rubric 评分的开放任务，更要交互式智能体环境。
2. **单轮问答 vs. 序贯诊疗** → 用 `pass^k` 评分的智能体回合（AgentClinic / MedAgentBench / MediQ），而非一次性问答。
3. **正确性 vs. 安全与伦理**（两者长期滞后） → 安全与伦理是独立的任务类型，配有独立的 rubric（CSEDB / MedSafetyBench / MTCMB-SE / **CARES-18K** 红队），外加专门的医学伦理竞技场（**MedEthicsQA / PrinciplismQA / MedEthicEval / MedEthicsBench**）。

---

## 🏗 架构

```
Config (YAML)         声明式运行规格：models / datasets / eval
Runner                调度 · 并发 · 缓存 · 断点续跑 · 实时进度 · 榜单
 ├ DatasetAdapter     load() -> Sample ;  parse(text) -> Prediction
 ├ ModelProvider      agenerate() ; HF 重载 agenerate_many = vLLM 批量
 └ Metric             score() ; aggregate()
Canonical Schema      Sample · Generation · Prediction · Score   （基石）
```

中间三层**只**依赖 schema，彼此互不依赖 —— 因此任意一层都能被独立替换或扩展。这正是关键：**`N × M × K` 组合，而非 `N·M·K` 胶水代码。**

---

## 📦 安装

```bash
pip install -e .                 # 核心（pyyaml）+ 命令行
pip install -e ".[all]"          # + datasets、litellm、openai、ray（无需 GPU）
# 本地 HF 后端（GPU，可选）：
pip install vllm transformers torch peft
```

> 需要 **Python 3.10+**。`example_smoke.yaml` 单条运行仅需 `pyyaml`；完整离线测试套件还需 `datasets` 附加依赖（有两个套件会加载它）—— `pip install -e ".[all]"`。

---

## 🚀 快速开始

**1 —— 离线 smoke 测试**（无 key、无 GPU、无网络 —— 确定性 `mock` 后端同时充当模型与评审）：

```bash
python -m medeval run configs/example_smoke.yaml
python tests/test_smoke.py && python tests/test_adapters.py     # 完整离线测试套件
```

**2 —— 真实数据集 + mock 模型**（下载**真实**基准并端到端跑通 —— 适合验证可访问性）：

```bash
python -m medeval run configs/catalog_mcq.yaml        --limit 5   # MedQA · MedMCQA · PubMedQA · MMLU · CMB · CMExam · TCMBench
python -m medeval run configs/catalog_en_med.yaml     --limit 5   # MedXpertQA · MedCalc · MedHallu · MLEC-QA · MediQ …
python -m medeval run configs/catalog_multimodal.yaml --limit 5   # MedFrameQA · SLAKE · TCM-Vision（需视觉模型）
python -m medeval run configs/catalog_ethics_safety.yaml --limit 5 # 医学伦理 MedEthicsQA · PrinciplismQA · MedEthicEval · CARES-18K 安全
python -m medeval run configs/example_tcm.yaml        --limit 3   # CMB + 辨证 SDT + 方剂 + 安全（评审打分）
python -m medeval run configs/example_agentclinic.yaml --limit 5  # pass^k，全程离线
```

**3 —— 真实模型 + 真实评审** —— 编辑任意 config 的 `models:`（见 `configs/example_api_backends.yaml`）：

```yaml
eval: {judge_model: deepseek-r1}
models:
  - {id: my-model,    type: litellm, model: openai/gpt-4o,            api_key_env: OPENAI_API_KEY}
  - {id: deepseek-r1, type: litellm, model: deepseek/deepseek-reasoner, api_key_env: DEEPSEEK_API_KEY, judge_only: true}
```

输出落在 `results/<run>/`：逐样本 `detail__<model>__<ds>.jsonl`，以及 `leaderboard.json` 和 `leaderboard.md`。

```python
# …或在 Python 中调用
import yaml, medeval
medeval.run_config(yaml.safe_load(open("configs/example_tcm.yaml")))
```

---

## 🔬 可靠性与可复现

选择题评测要可信，前提是数据确实是你以为的那份；而一次全量评测要跑得完，前提是它能扛住中断。五道防线：

- **固定 revision** —— 核心选择题目录（[`catalog_mcq.yaml`](configs/catalog_mcq.yaml)，唯一例外是 TCMBench-demo：上游不发布稳定 ref，且它本就是 `demo` 层）、模型目录的 MCQ 切片和 TCM-Ladder 均锁定到不可变提交，这些评测集永不会在你脚下悄悄改变。HF 仓库用 `revision: <sha>`（传给 `load_dataset`）；原始文件源把提交嵌进 URL（`…/resolve/<sha>/…`、`raw.githubusercontent/…/<sha>/…`）。其余目录（`catalog_en_med` / `catalog_multimodal` / `catalog_cn_tcm` / `catalog_ethics_safety` 的一部分）目前**跟踪 `main`**——各文件头有注明；发布这些目录的数字前请先固定 `revision:`。大文件经一个原子、**支持 HTTP Range 断点续传**的下载器获取 —— 能扛住会截断大响应的代理，且一次失败的下载绝不会污染缓存。
- **`preflight`** —— **无需模型**即可剖析每个数据集：样本数、选项数分布、**答案解析成功率**、以及前几条样例。在花掉任何一个 token 之前先跑它：

```bash
python -m medeval preflight configs/catalog_mcq.yaml          # 所有数据集，完整加载
python -m medeval preflight configs/catalog_mcq.yaml --strict # CI：任一解析率 < 100% 则非零退出
```

```
✓ cmb_test   [hf_mcq]
    样本数 samples        : 11200 of 11200 rows
    选项数 option dist     : {3: 1, 4: 1201, 5: 9956, 6: 42}
    解析率 answer parse    : 100.0%  ████████████████████
```

解析率低于 100% 意味着有行被丢弃（`field_map` 映射错、答案编码异常、选项解析不出来）—— `preflight` 会按原因列出它们，让你去修配置，而不是去猜症状。

- **可比性分级（`split_type`）** —— 每条结果都带 `split_type`，使**官方可比**的运行绝不会和内部运行混在一张榜上。**默认就是 `unverified`**——数据集必须主动挣得 `official`；光在 YAML 里写一行绝不能授予它（见 `medeval.eligibility`）。leaderboard 会渲染成 **✅ Official** 与 **⚠️ Internal / non-comparable** 两个独立分区。取值：`official` · `validation` · `demo` · `sample` · `gated` · `approximated` · `reimplementation` · `unverified`。于是 **CMB-val**（validation）、**TCMBench-demo**（demo）、**CSEDB-sample**（sample）、**MedAgentBench 的内置评分器**（approximated，除非你提供官方 `refsol_path`）、以及只接入部分角色的 AgentClinic 或任何 MediQ 运行（`reimplementation`，除非所有 support 角色都接好）都被清楚地隔离在完整官方榜单之外。
- **每次 Web 会话 + CI 自动执行** —— 一个 `SessionStart` 钩子（`.claude/`）会安装依赖并跑 `preflight --strict`，让每个 Claude Code on the web 会话在开工前就剖析好评测集；GitHub Actions（`.github/workflows/ci.yml`）在每次 push/PR 上既跑离线测试套件，又把 `preflight --strict` 作为数据闸门。
- **确定性、可断点续跑** —— 每个 `(模型, 数据集)` 的生成结果按「模型 + 其**实际**采样参数」的**稳定**哈希（用 `hashlib`，而非按进程加盐的内置 `hash()`）缓存到磁盘，于是被中断的全量评测（Colab 超时、抢占式实例回收）会**从上一个检查点续跑**，而不是从头重算。运行时显示**实时**逐数据集进度条（`tqdm`），生成结果按 `checkpoint_every`（默认 64，对 Google Drive / 网络挂载友好）批量落盘，每完成一个数据集即重写榜单；遇到崩溃截断的缓存行会被跳过，绝不污染续跑。
- **内容寻址缓存 —— 绝不用陈旧生成结果给新数据打分。** 缓存文件名折入了*整个*数据集协议（revision、field_map、prompt 模板、instruction、adapter 类 + 手工维护的 `ADAPTER_PROTOCOL_VERSION`；智能体还包括 `k`/`max_turns`/support 模型身份），任何一项变化都会路由到全新的缓存文件。在此之上，每条缓存记录还携带**样本实际内容**的哈希（渲染后的消息 + 选项 + 参考答案）；如果一个未固定版本的数据源在上游悄悄漂移——同一个 `sample_id`，不同的题目/参考答案——该样本的哈希就不再匹配，会被重新生成，而不是静默地用变了的标准答案去评判一个陈旧输出。指标/rubric/评审配置**特意排除**在签名之外（评分从不缓存、每次都重新计算，改 rubric 不会读到陈旧结果——也不会浪费一次真实的生成调用）。
- **`run_manifest.json`** —— 每次运行都会写一份：git commit、adapter 协议版本、完整运行配置（密钥已脱敏）、每个数据集解析后的 `split_type` + 是否有固定版本证据 + 类 `preflight` 的加载统计，以及每个模型的解析身份——这样任何一个榜单数字都能准确追溯到产生它的确切配置。

---

## 📊 基准目录

下面是有代表性的一部分（均已接入并对线上源核验；完整 **30+** 详见 [`DATASETS.md`](DATASETS.md)）：

| 基准 | 适配器 | 任务 / 指标 | 获取方式 |
|---|---|---|---|
| **MedQA**（USMLE） | `hf_mcq` | mcq_accuracy | `GBaker/MedQA-USMLE-4-options` |
| **MedMCQA** | `hf_mcq` | mcq_accuracy | `openlifescienceai/medmcqa`（用 *validation*） |
| **PubMedQA** | `hf_mcq` | mcq_accuracy | `qiaojin/PubMedQA`（注入 yes/no/maybe） |
| **MMLU-medical** | `hf_mcq` | mcq_accuracy | `cais/mmlu`（6 个学科） |
| **CMB** | `hf_mcq` | mcq_accuracy | `FreedomIntelligence/CMB` —— **完整 test 11,200**，按 `id` 拼接 GitHub 标准答案 |
| **CMExam** | `hf_mcq` | mcq_accuracy | `williamliujl/CMExam`（内联选项，多选） |
| **TCMBench** | `tcmbench` | mcq_accuracy | `ywjawmw/TCMBench` —— **全部 14 个 demo 文件**（完整题库受限） |
| **TCM-Ladder** | `hf_mcq` | mcq_accuracy | `timzzyus/TCM-Ladder` —— 12,778 文本 + 8,802 图像选择题（开放） |
| **HealthBench** | `local_json` | llm_judge `per_criterion` | OpenAI simple-evals —— **3 个变体**，忠实逐条 rubric 评分 |
| **LLMEval-Med** | `local_json` | llm_judge | `llmeval/LLMEval-Med`（清单式 rubric） |
| **TCMEval-SDT** 辨证 | `local_json` | llm_judge + syndrome_chain | `zhuyan166/TCMEval` |
| **MTCMB** 方剂/安全 | `local_json` | llm_judge + prescription_match | `Wayyuanyuan/MTCMB` |
| **MedSafetyBench** | `local_json` | llm_judge（安全） | `AI4LIFE-GROUP/med-safety-bench` —— **全部 18 个 CSV**（9 类 AMA × 2） |
| **MedEthicsQA** ⚖️ | `hf_mcq` | mcq_accuracy | `JianhuiWei7/MedEthicsQA` —— **5,623** 道伦理选择题（中英），解析率 100% |
| **PrinciplismQA** ⚖️ | `hf_mcq` + `local_json` | mcq_accuracy + llm_judge | `FreedomIntelligence/PrinciplismQA-Demo` —— 100 选择 + 126 rubric |
| **MedEthicEval** ⚖️ | `hf_mcq` + `local_json` | mcq_accuracy + llm_judge | `X-LANCE/MedEthicEval`（NAACL'25）—— 629 知识 + 伦理两难 + 违规检测 |
| **TCM_Humanities** ⚖️ | `hf_mcq` | mcq_accuracy | `TCMLM/TCM_Humanities` —— **500** 道医学人文/伦理/卫生法选择题（多选） |
| **MedEthicsBench** ⚖️ | `local_json` | llm_judge（rubric） | `pariskang/MedEthicsBench` —— 关键点 rubric（前向兼容，待上游生成） |
| **CARES-18K** 🛡️ | `local_json` | llm_judge（安全） | `HFXM/CARES-18K` —— **9,239** 红队提示（8 原则 × 4 危害 × 4 策略） |
| **名老中医医案** 🀄 | `local_json` | syndrome_chain + llm_judge | `TCMLM/real_clinical_cases…` —— **500** 则真实医案（辨证论治） |
| **AgentClinic** | `agentclinic` | pass_k | `SamuelSchmidgall/AgentClinic` —— MedQA **214** + NEJM **120** |
| **MedAgentBench** | `medagentbench` | pass_k | 真实 FHIR（Docker）；**逐任务载荷评分器**（+ 受限 `refsol.py`） |
| **MediQ** | `mediq` | pass_k | `stellalisy/MediQ` —— 主动追问 |

> **无法仅靠配置接入**（已在 `DATASETS.md` 记录）：**MLEC-QA**（Google-Drive 登录）、**TCM-3CEval**（MedBench 隐藏答案）、**AgentClinic-MIMIC-IV**（PhysioNet 资质审核）。

---

## 🔌 后端

| `type` | 是什么 | 说明 |
|---|---|---|
| `hf` | 本地 checkpoint / 仓库 / LoRA | vLLM 批量（重载 `agenerate_many`），transformers 兜底 |
| `poe` | `https://api.poe.com/v1` | bot 名即模型；内置约 500 rpm 限流 |
| `litellm` | 100+ 提供商 + 任意 OpenAI 兼容端点 | 重试、计费；**评审模型放这里** |
| `mock` | 离线确定性 | smoke 测试；感知评审 / 选择题 / 智能体 |

**模式 A**（默认）：各后端进程内运行 —— HF 离线批量，Poe/LiteLLM 走 API。**模式 B**（生产）：用 `vllm serve` 起 HF，所有请求统一经 LiteLLM 走一层重试 / 缓存 / 计费 / 限流。

---

## 🤖 模型清单（医学与中医）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/Med-Bench-Arena/blob/main/notebooks/Med_Bench_Arena_Colab.ipynb)

[`configs/catalog_med_models.yaml`](configs/catalog_med_models.yaml) 已将 **17 个医学与中医大模型**接入为开箱即用的 HF/vLLM 后端 —— 每个仓库 id、基座架构、`dtype`、上下文长度与 `trust_remote_code` 标志都**对照 HuggingFace 线上页面逐一核验**（完整表与逐模型说明见 [`MODELS.md`](MODELS.md)）。每次用 `--models` 选一个跑（vLLM 会把模型常驻显存）：

```bash
python -m medeval run configs/catalog_med_models.yaml --models zhongjing-2-1_8b --limit 20
python -m medeval run configs/catalog_med_models.yaml --models biancang        --limit 20
```

| 分组 | 模型（`--models <id>`） |
|---|---|
| 🀄 **中医 TCM** | `zhongjing-2-1_8b` · `dao1-30b-a3b` · `biancang` · `taiyi` · `disc-medllm` |
| 🩺 **中文医学** | `huatuogpt2-7b` · `huatuogpt-o1-7b` · `aquilamed-rl` · `baichuan-m1-14b` · `baichuan-m2-32b` · `clinicalgpt-r1` |
| ⚕️ **英文 / 国际** | `meditron-70b` · `biomistral-7b` · `deepseek-r1-32b` · `medgemma-27b-it` · `citrus-70b` |
| 🖼 **多模态** | `lingshu-7b`（及 `medgemma-27b-it`） |

已自动处理的坑：**仲景 ZhongJing-2** 是挂在 `Qwen1.5-1.8B-Chat` 上的 LoRA；**Baichuan / 太一 / DISC / AquilaMed** 需 `trust_remote_code`；**Meditron / MedGemma** 受限（需接受许可 + `huggingface-cli login`）；推理模型（**DeepSeek-R1 / HuatuoGPT-o1 / Baichuan-M2 / ClinicalGPT-R1**）给更大的 `max_tokens`；vLLM 加载失败会自动回退到 transformers 而非崩溃。*（**Qibo 岐黄** 与 **Qilin-Med 麒麟文本版**未在 HF 公开发布，已在 [`MODELS.md`](MODELS.md) 如实标注；中医可用 BianCang / Dao1 替代。）*

▶️ **GPU 一键运行** —— 点徽章直接在 Google Colab 打开运行（无需本地环境；克隆 · 装 vLLM · 选模型 · **跑全部基准、全部题目** · 结果写入 **Google Drive**）。Notebook 默认 `LIMIT = 0`，即对每个数据集跑**完整题库**（设 `LIMIT = 50` 可做快速冒烟测试）。全量评测会显示**实时逐数据集进度**、每个检查点**把结果写入 Drive**，并在 Colab 会话断开后**从断点续跑** —— 重新运行该单元格即可：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/Med-Bench-Arena/blob/main/notebooks/Med_Bench_Arena_Colab.ipynb)

---

## 📐 指标与评审

`mcq_accuracy` · `pass_k` · `llm_judge` · `f1` · `rouge` · `bleu` · `numeric_match`，外加**五种结构化中医指标**。

<details>
<summary><b>全部 12 种指标详解</b></summary>

- **`mcq_accuracy`** —— 默认采用**零样本 CoT** 提示（逐步推理 → `Answer: X`），解析器**优先匹配结构化答案行、取最后一次命中**，从而忽略推理过程中提到的干扰项，并先剥离 `<think>` 推理痕迹。鲁棒的字母/序号/文本抽取；同时支持单选**与**多选。
- **`pass_k`** —— *k* 次独立回合必须**全部**成功（同时报告 pass@1）。智能体动作解析前会先剥离推理模型的 `<think>` 痕迹。
- **`llm_judge`** —— 评审就是一个普通的 provider。rubric 来自数据集（HealthBench 分值、CSEDB 分数、LLMEval 清单）或任务默认（open_qa / **辨证证型链** / **方剂** / **安全**）。采用明确的 0/0.5/1 评分锚点；评审输出的不规范 JSON 经 **`json-repair`** 流水线修复；评分按 id/文本/归一化文本/位置匹配（绝不会静默判 0）。评审重试后仍无法给出可解析结果时，该样本被**排除**（`value=None`，聚合中以 `judge_failures` 显示）而不是记 0 —— 评审自身的故障绝不算作模型的 0 分；key 部分匹配时未被评到的准则同样从分母中排除，而不是强制记 0。带符号分值被尊重 —— 含**负分**的 rubric 一律走逐条路径，确保惩罚保持正确符号；`per_criterion: true` 运行 **忠实的 HealthBench 算法**（逐条一次调用，布尔 `criteria_met`，带符号得分 / 正分值）。
- **`f1` / `rouge` / `bleu`** —— 与参考答案的 token 重合；**中日韩感知**分词（中文按字、拉丁按词，装了 `jieba` 则用之）。
- **`numeric_match`** —— 计算类任务（MedCalc-Bench）：优先抽取带标签的 `Answer: <number>` 行（其次为最后一次出现的标记，支持科学计数法），判断是否落入容差 / `[lower, upper]` 区间。
- **`prescription_match`** —— **方剂结构匹配**：药味集合 P/R/F1（君臣佐使）+ 方名 + 治法重合，读自结构化标准答案。
- **`syndrome_chain`** —— **证型链结构分**：对 症状→病机→证型 链打分，含 **同病异治 部分得分**。
- **`meridian_acupoint`** —— **经络腧穴**：对 12 正经 + 奇经及腧穴做集合 F1，含别名归一。
- **`tongue_pulse`** —— **舌象/脉象**：按子句锚定，对舌（舌色/舌形/苔）与脉（脉）特征做集合 F1。
- **`classics_ontology`** —— **古籍本体**：回答是否落到了正确的经典出处？集合 F1 + 最长匹配去重；别名取自知识图谱。

> 结构化指标（方剂/证型链/经络腧穴/舌脉/古籍）遇到**参考答案中提取不到 gold** 的样本时，一律排除（`value=None`，聚合报告 `n_scored` / `skipped_no_gold`）而不是记 0 —— 无标可评不是模型的错。

</details>

支持一个数据集配多种指标 —— 例如 TCMEval-SDT 跑 `[llm_judge, syndrome_chain, bleu, rouge]`。

### ⚖️ 评审校准 —— 开放式分数必须挣得上榜资格

LLM 评审给出的分数，只有在它与**人类专家**意见一致时才值得信任。`medeval calibrate` 在一份冻结的、**医生标注**的数据集上测量这种一致性，决定开放式分数能否登上榜首，还是必须以**辅助**指标形式呈现。

锚点是 **HealthBench 的官方元评测集** —— 每个 *(对话, 回答, rubric 条目)* 都附有 2 位以上医生的二元判断，因此我们能离线计算*真实*的评审-人类一致性。我们逐字复刻了 OpenAI simple-evals 的一致性指标（**met/unmet 两类的平衡 F1**），报告 **Cohen's κ** + 原始一致率，并附带 bootstrap 95% 置信区间，把每个评审者与**医生互评天花板**做比较。

一个强前沿模型在**盲评 120 条**（医生标签被隐藏）上使用逐字复刻的 HealthBench 评分模板进行了打分。结果：

| vs 医生 | 平衡 F1 | Cohen's κ | 原始一致率 |
|---|---|---|---|
| **强模型评审** | **0.697** [0.61, 0.77] | **0.394** [0.23, 0.54] | 0.736 |
| _医生天花板（人类）_ | _0.719_ [0.63, 0.80] | _0.437_ | _0.745_ |

这个评审**达到了医生水平**——在两项指标上都落在人类天花板 0.05 以内，置信区间重叠。但天花板本身只是**中等**水平（κ≈0.44）：rubric 逐条评分本身就带有主观性。因此按策略，该评审**不具备上榜资格**（绝对 κ < 0.40），开放式分数保持**辅助**状态——这正是保守的默认行为。leaderboard 强制执行这一点：评审打分的数据集会落入独立的 *🧪 Auxiliary* 分区，直到某份校准报告将该评审标记为可上榜（`data/calibration/calibration_report.md`）。

**一份校准报告只绑定它实际测量过的东西——绝不是"这个评审总体没问题"的通行证。** 每份报告都携带一个 `signature`：具体的评审模型 + 版本（仅限 `--config --judge` 现场模式）、以及被测的评分 prompt/协议（目前只有 HealthBench 逐条风格）。leaderboard 只有在**这一行自己的**评审模型+版本、以及评分 prompt 与该签名**完全匹配**时，才会把它从 Auxiliary 分区提升出来——在 HealthBench 英文 criteria 上校准了 `gpt-4.1`，绝不会让 `deepseek-r1` 获得上榜资格，也不会让走默认（非逐条）rubric prompt 的中文中医辨证/伦理/安全数据集获得上榜资格，哪怕磁盘上确实存在一份 `calibrated: true` 的报告。`--labels` 模式（强模型/人工的盲评一次性通过，而非现场评审运行）根本没有具体的评审模型可绑定，因此**永远不会自动生效**——它只是"医生水平的一致性是可以达到的"这一证据，而不是可现场验证的保证；只有 `--config --judge <id>` 才会产出可绑定、被 leaderboard 认可的报告。

```bash
# 重新生成冻结的医生标注集，再给评审者的盲标打分
python -m medeval calibrate --rebuild-from <oss_meta_eval.jsonl URL/path>
python -m medeval calibrate --labels data/calibration/healthbench_meta_strongmodel_labels.jsonl
# ……或对照同一份医生金标校准一个现场 API 评审（与 leaderboard 用的是同一条代码路径，
# 也是唯一会产出可绑定签名的模式）：
python -m medeval calibrate --config configs/example_open_safety.yaml --judge gpt-4.1 --strict
```

> TCMEval-SDT / MTCMB 未提供医生标签，因此它们的开放式分数默认继承 **auxiliary** 状态，直到有中医领域专家提供标签文件——评分逻辑完全相同，但那将是绑定在该数据集**自己的**评审+prompt 签名上的独立校准，而不是继承自 HealthBench 的。

---

## 🤖 智能体

医生智能体在 `AgentEnvironment(reset/step)` 回合中运行同一套 `ModelProvider` 策略。

- **AgentClinic** —— OSCE（MedQA）+ 影像病例（NEJM）。病人 / 检查 / 仲裁可各自作为独立 LLM 经 `support:` 接入，或离线规则驱动。`split_type` 反映的是协议忠实度，而非配置的一厢情愿：**三个角色全部**接入 → `official`；接入部分（非全部）→ `reimplementation`；一个都没接 → `approximated`——只接一个角色不能自称"忠实设定"。
- **MediQ** —— 主动信息搜集：医生向病人追问原子事实（仅在相关提问时披露），用 `ANSWER: <letter>`（可选附 `(confidence: NN%)`）作答，或在证据确实不足时 `ABSTAIN`。`pass_k` 在医生给出置信度时还会报告 `abstain_rate`、`avg_questions`（提问效率）与 `mean_confidence` / `confidence_brier_score`（校准度）。默认绝不为 `official`——规则式病人是 `approximated`，LLM 病人是 `reimplementation`（两者都未逐字复现 MediQ 原始 Patient/Expert System 提示词）；只有显式在配置中覆盖才能声明 `official`，且这份声明的责任在使用者而非框架本身。
- **MedAgentBench** —— 真实 **FHIR 电子病历服务**。智能体发出 `GET <url>` / `POST <url>\n<json>` / `FINISH([...])`；评分用官方受限 `refsol.py`（设置 `refsol_path`）**或**内置的**逐任务载荷评分器**，校验 `resourceType` + `subject → Patient/{MRN}` + 正确的 flowsheet/SNOMED/NDC/LOINC 编码。设计上保守 —— 绝不误判为通过。**隔离风险**：默认情况下每个回合共用同一个未重置的真实 FHIR 服务器（一次写入可能泄漏进后续回合的读取）——并发数会被强制降到 1，`split_type` 也会被限制在 `official` 以下，直到你接入 `episode_reset` 钩子。

```bash
docker run -p 8080:8080 jyxsu6/medagentbench:latest          # 提供 :8080/fhir
python -m medeval run configs/example_medagentbench.yaml --limit 10
```

---

## 🖼 多模态（舌象 / 影像）

`Message` 可携带可选的 `images`（http/data URI 或本地路径 → 自动转 data-URI）；`to_openai()` 产出 OpenAI/LiteLLM **内容块**，因此 LiteLLM 与 Poe 视觉模型无需改动即可使用。本地 **HF 视觉模型**（灵枢 / MedGemma）同样可用：携带图片的 batch 会路由到 **vLLM 的 `chat()` 多模态 API**；若加载的后端无法处理图片（transformers 回退、过旧的 vLLM），provider 会**直接报错而不是静默丢图**、让模型只凭文本被评分。`hf_mcq` 适配器接受 `image` 字段（URL / 本地路径 / HF `Image` 字典 / **parquet 原始字节** / PIL）。对于图片以独立 `images.zip` 发布的数据集，设置 `image_zip:` + `image_base:`，适配器会**自动下载并解压**一次（幂等）；也可用 `python -m medeval fetch <url>` 预取。

---

## 🀄 中医原生支持

中医不是补丁 —— 它有专属的任务类型、指标、数据集与本体：

- **辨证** → `syndrome_chain` 对 症状→病机→证型 推理链打分，含同病异治部分得分。
- **方剂** → `prescription_match` 比对药味集合（君臣佐使）、方名与治法。
- **经络腧穴 · 古籍本体 · 舌象/脉象** → 各有专属结构化指标。
- **知识图谱** —— 经典文献本体是一张真实、可下载的图谱（35 部经典 + 作者 + 朝代 + 25 首经典名方），是 `classics_ontology` 的唯一事实来源：

```bash
python -m medeval kg --out data/kg --stats          # 116 节点，157 边 → JSON / Turtle / GraphML
```
```python
from medeval import get_kg
get_kg().source_of_formula("银翘散")     # -> 温病条辨
```

---

## ⚡ 分布式调度

评测网格天然并行，因此 MedEval 采用**跨步分片** —— 每个 worker 跑 `samples[i::N]`，各写各的分片文件，可独立续跑。无中心服务器。

```bash
# 单机 N worker（可每 GPU 一个）：
python -m medeval pool configs/catalog_mcq.yaml --num-shards 4 --gpus 0,1,2,3

# Ray 集群：
python -m medeval pool cfg.yaml --num-shards 8 --backend ray --ray-num-gpus 1

# Slurm：生成并提交作业数组 + 依赖式 merge：
python -m medeval slurm cfg.yaml --num-shards 64 --partition gpu --gpus-per-task 1

# 多机（共享文件系统）：各跑分片，最后合并一次：
python -m medeval merge /shared/run        # 重新聚合逐样本得分（非"均值的均值"）
```

---

## 📤 榜单提交

```bash
# OpenCompass 预测：  predictions/<model>/<dataset>.json
python -m medeval export results/mcq --format opencompass --out oc_out

# MedBench（答案隐藏 —— 本地生成，平台打分）：
python -m medeval export results/medbench --format medbench --out submission \
       --medbench-test-dir /path/to/MedBench
```

---

## 🧱 扩展

```python
from medeval import register_provider, register_dataset, register_metric

@register_provider("myllm")    # 实现 agenerate()
@register_dataset("mybench")   # 实现 load() / parse()
@register_metric("my_metric")  # 实现 score() / aggregate()
```

导入该模块（加进 `runner.py` 的 import 列表）后，在 YAML 里按名引用即可。

---

## 📁 项目结构

```
medeval/
├── schema.py                  # 规范化类型（基石）
├── providers/                 # hf · poe · litellm · mock
├── datasets/                  # hf_mcq · local_json · agent_env · tcmbench · medbench · medagentbench_grader
├── metrics/                   # mcq · llm_judge · text_match · prescription · syndrome · tcm_struct · numeric
├── kg/tcm_classics.py         # 经典文献知识图谱（JSON / Turtle / GraphML）
├── eligibility.py              # official 层级门禁：内容类适配器的固定版本证据
├── runner.py                  # 编排器；内容寻址缓存 + run_manifest.json
├── distributed.py             # 分片 · 合并 · 本地/Ray/Slurm
├── submit.py                  # OpenCompass / MedBench 导出
├── assets.py                  # 自动下载 + 解压 images.zip
└── cli.py                     # python -m medeval run|preflight|list|export|merge|pool|slurm|kg|fetch
configs/                       # 声明式、已实测的运行规格（含 catalog_med_models.yaml）
notebooks/                     # 医学 / 中医模型清单的 Colab 运行器
tests/                         # 13 个离线测试套件（无 key / GPU / 网络；需安装 datasets 附加依赖）
DATASETS.md                    # 各数据集获取说明、注意事项、字段映射
MODELS.md                      # 17 个模型清单：核验过的仓库 id、架构、授权、注意事项
```

---

## 📖 引用

如果 Med-Bench-Arena 对你的研究有帮助，请引用：

```bibtex
@software{med_bench_arena,
  title  = {Med-Bench-Arena: A Unified Arena for Evaluating Medical and TCM LLMs and Agents},
  author = {Med-Bench-Arena contributors},
  year   = {2026},
  url    = {https://github.com/pariskang/Med-Bench-Arena}
}
```

---

## 🤝 参与贡献

欢迎贡献！接入一个基准通常**只需改配置**（字段映射词汇见 `DATASETS.md`）。若要新增适配器/指标/后端，用上面的装饰器注册，并在 `tests/` 下补一个测试。提 PR 前请先跑离线测试套件：

```bash
pip install -e ".[all]"                             # 测试套件需要 datasets 附加依赖
for t in tests/test_*.py; do python "$t"; done      # 13 个都应打印 OK
```

---

## 📄 许可

[MIT](LICENSE) © Med-Bench-Arena contributors.

---

## 🙏 致谢

本项目站在它所接入的开放基准的肩膀上 —— MedQA、MedMCQA、PubMedQA、MMLU、CMB、CMExam、TCMBench、TCM-Ladder、HealthBench、LLMEval-Med、TCMEval、MTCMB、CSEDB、MedSafetyBench、AgentClinic、MedAgentBench、MediQ，以及伦理与安全数据集 MedEthicsBench、MedEthicsQA、PrinciplismQA、MedEthicEval、TCM_Humanities、CARES-18K 与 TCMLM 真实名老中医医案语料 —— 以及运行它们的后端（HuggingFace、vLLM、Poe、LiteLLM）。感谢每一位数据集作者。

<div align="center"><sub>为严谨、可复现的医学与中医模型评测而生。⭐ 如果有用，欢迎 Star！</sub></div>
