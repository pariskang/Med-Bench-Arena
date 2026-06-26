<div align="center">

# 🏥 Med-Bench-Arena

### 统一、配置驱动的医学与中医大模型 / 智能体评测竞技场

*一套规范化 schema 解耦 **数据集 · 后端 · 指标** —— 任意模型、任意基准、任意指标自由组合，零胶水代码。*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen.svg)](tests/)
[![Benchmarks](https://img.shields.io/badge/benchmarks-40%2B%20已实测-8A2BE2.svg)](DATASETS.md)
[![TCM](https://img.shields.io/badge/中医-原生支持-c1272d.svg)](#-中医原生支持)
[![Ethics & Safety](https://img.shields.io/badge/伦理·安全-原生支持-2E8B57.svg)](configs/catalog_ethics_safety.yaml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-参与贡献)

[English](README.md) · **简体中文**

</div>

---

**Med-Bench-Arena** 是 *MedEval 协议* 的参考实现：把 **30+ 个医学 / 中医基准** 接入 **任意** 大模型或智能体后端，并用 **12 种指标** 评测 —— 从最朴素的选择题正确率，到 rubric 评分的安全性、结构化的 **方剂 / 辨证** 匹配，再到交互式智能体的 `pass^k`。一套 **规范化 schema** 横亘在数据集、模型后端与指标之间，于是你得到的是免费的 `N 数据集 × M 后端 × K 指标` 组合，而非 `N×M×K` 的硬编码。

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

[设计动机](#-设计动机) · [架构](#-架构) · [安装](#-安装) · [快速开始](#-快速开始) · [可靠性](#-可靠性与可复现) · [基准目录](#-基准目录) · [后端](#-后端) · [指标](#-指标与评审) · [智能体](#-智能体) · [多模态](#-多模态舌象--影像) · [中医](#-中医原生支持) · [分布式](#-分布式调度) · [榜单提交](#-榜单提交) · [扩展](#-扩展) · [项目结构](#-项目结构) · [引用](#-引用) · [贡献](#-参与贡献) · [许可](#-许可)

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
Runner                调度 · 并发 · 缓存 · 续跑 · 榜单
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

> 需要 **Python 3.10+**。离线 smoke 测试仅需 `pyyaml`。

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

选择题评测要可信，前提是数据确实是你以为的那份。四道防线：

- **固定 revision** —— 每个核心选择题基准都锁定到一个不可变的提交，评测集永不会在你脚下悄悄改变。HF 仓库用 `revision: <sha>`（传给 `load_dataset`）；原始文件源把提交嵌进 URL（`…/resolve/<sha>/…`、`raw.githubusercontent/…/<sha>/…`）。大文件经一个原子、**支持 HTTP Range 断点续传**的下载器获取 —— 能扛住会截断大响应的代理，且一次失败的下载绝不会污染缓存。
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

- **可比性分级（`split_type`）** —— 每条结果都带 `split_type`，使**官方可比**的运行绝不会和内部运行混在一张榜上。leaderboard 会渲染成 **✅ Official** 与 **⚠️ Internal / non-comparable** 两个独立分区。取值：`official` · `validation` · `demo` · `sample` · `gated` · `approximated`。于是 **CMB-val**（validation）、**TCMBench-demo**（demo）、**CSEDB-sample**（sample）、以及 **MedAgentBench 的内置评分器**（approximated，除非你提供官方 `refsol_path`）都被清楚地隔离在完整官方榜单之外。
- **每次 Web 会话 + CI 自动执行** —— 一个 `SessionStart` 钩子（`.claude/`）会安装依赖并跑 `preflight --strict`，让每个 Claude Code on the web 会话在开工前就剖析好评测集；GitHub Actions（`.github/workflows/ci.yml`）在每次 push/PR 上既跑离线测试套件，又把 `preflight --strict` 作为数据闸门。

---

## 📊 基准目录

下面是有代表性的一部分（均已接入并对线上源核验；完整 **40+** 见 [`DATASETS.md`](DATASETS.md)）：

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

### 🤖 内置医学 / 中医模型清单

[`configs/catalog_med_models.yaml`](configs/catalog_med_models.yaml) 已将 **18 个医学与中医大模型**接入为开箱即用的 HF/vLLM 后端 —— 每个仓库 id、基座架构、`dtype`、上下文长度与 `trust_remote_code` 标志都**对照 HuggingFace 线上页面逐一核验**（完整表见 [`MODELS.md`](MODELS.md)）。每次跑选一个模型（vLLM 会把模型常驻显存）：

```bash
python -m medeval run configs/catalog_med_models.yaml --models biancang --limit 20
python -m medeval run configs/catalog_med_models.yaml --models huatuogpt-o1-7b
```

已覆盖：**Dao1-30B-A3B · 扁仓 BianCang · 太一 Taiyi · 仲景 ZhongJing-2 · DISC-MedLLM**（中医/中文）· **HuatuoGPT-II · HuatuoGPT-o1 · AquilaMed · Baichuan-M1/M2 · ClinicalGPT-R1**（中文医学）· **Meditron-70B · BioMistral · DeepSeek-R1-32B · MedGemma-27B · Citrus**（国际）· **灵枢 Lingshu**（多模态）。已处理的坑：Baichuan/太一/DISC/AquilaMed 需 `trust_remote_code`；仲景是 LoRA（挂载在 `Qwen1.5-1.8B-Chat` 上）；Meditron/MedGemma 受限需授权；推理模型给更大的 `max_tokens`。

▶️ **GPU 一键运行**：[`notebooks/Med_Bench_Arena_Colab.ipynb`](notebooks/Med_Bench_Arena_Colab.ipynb) —— 克隆 · 装 vLLM · 选模型 · 在 MedQA + CMB 上打分。

> ⚠️ **Qibo（岐黄）** 与 **Qilin-Med（麒麟）文本版**未在 HuggingFace 公开发布（仅论文/或仅多模态版），已在 `MODELS.md` 与配置中如实标注；中医可用 BianCang / Dao1 替代。

---

## 📐 指标与评审

`mcq_accuracy` · `pass_k` · `llm_judge` · `f1` · `rouge` · `bleu` · `numeric_match`，外加**五种结构化中医指标**。

<details>
<summary><b>全部 12 种指标详解</b></summary>

- **`mcq_accuracy`** —— 鲁棒的字母/序号/文本抽取；同时支持单选**与**多选。
- **`pass_k`** —— *k* 次独立回合必须**全部**成功（同时报告 pass@1）。
- **`llm_judge`** —— 评审就是一个普通的 provider。rubric 来自数据集（HealthBench 分值、CSEDB 分数、LLMEval 清单）或任务默认（open_qa / **辨证证型链** / **方剂** / **安全**）。带符号分值被尊重；`per_criterion: true` 运行 **忠实的 HealthBench 算法**（逐条一次调用，布尔 `criteria_met`，带符号得分 / 正分值）。
- **`f1` / `rouge` / `bleu`** —— 与参考答案的 token 重合；**中日韩感知**分词（中文按字、拉丁按词，装了 `jieba` 则用之）。
- **`numeric_match`** —— 计算类任务（MedCalc-Bench）：抽取最终数值，判断是否落入容差 / `[lower, upper]` 区间。
- **`prescription_match`** —— **方剂结构匹配**：药味集合 P/R/F1（君臣佐使）+ 方名 + 治法重合，读自结构化标准答案。
- **`syndrome_chain`** —— **证型链结构分**：对 症状→病机→证型 链打分，含 **同病异治 部分得分**。
- **`meridian_acupoint`** —— **经络腧穴**：对 12 正经 + 奇经及腧穴做集合 F1，含别名归一。
- **`tongue_pulse`** —— **舌象/脉象**：按子句锚定，对舌（舌色/舌形/苔）与脉（脉）特征做集合 F1。
- **`classics_ontology`** —— **古籍本体**：回答是否落到了正确的经典出处？集合 F1 + 最长匹配去重；别名取自知识图谱。

</details>

支持一个数据集配多种指标 —— 例如 TCMEval-SDT 跑 `[llm_judge, syndrome_chain, bleu, rouge]`。

---

## 🤖 智能体

医生智能体在 `AgentEnvironment(reset/step)` 回合中运行同一套 `ModelProvider` 策略。

- **AgentClinic** —— OSCE（MedQA）+ 影像病例（NEJM）。病人 / 检查 / 仲裁可各自作为独立 LLM 经 `support:` 接入（忠实设定），或离线规则驱动。
- **MediQ** —— 主动信息搜集：医生向病人追问原子事实（仅在相关提问时披露），或用 `ANSWER: <letter>` 作答。
- **MedAgentBench** —— 真实 **FHIR 电子病历服务**。智能体发出 `GET <url>` / `POST <url>\n<json>` / `FINISH([...])`；评分用官方受限 `refsol.py`（设置 `refsol_path`）**或**内置的**逐任务载荷评分器**，校验 `resourceType` + `subject → Patient/{MRN}` + 正确的 flowsheet/SNOMED/NDC/LOINC 编码。设计上保守 —— 绝不误判为通过。

```bash
docker run -p 8080:8080 jyxsu6/medagentbench:latest          # 提供 :8080/fhir
python -m medeval run configs/example_medagentbench.yaml --limit 10
```

---

## 🖼 多模态（舌象 / 影像）

`Message` 可携带可选的 `images`（http/data URI 或本地路径 → 自动转 data-URI）；`to_openai()` 产出 OpenAI/LiteLLM **内容块**，因此 LiteLLM 与 Poe 视觉模型无需改动即可使用。`hf_mcq` 适配器接受 `image` 字段（URL / 本地路径 / HF `Image` 字典 / **parquet 原始字节** / PIL）。对于图片以独立 `images.zip` 发布的数据集，设置 `image_zip:` + `image_base:`，适配器会**自动下载并解压**一次（幂等）；也可用 `python -m medeval fetch <url>` 预取。

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
├── runner.py                  # 编排器
├── distributed.py             # 分片 · 合并 · 本地/Ray/Slurm
├── submit.py                  # OpenCompass / MedBench 导出
├── assets.py                  # 自动下载 + 解压 images.zip
└── cli.py                     # python -m medeval run|preflight|list|export|merge|pool|slurm|kg|fetch
configs/                       # 声明式、已实测的运行规格（含 catalog_med_models.yaml）
notebooks/                     # 医学 / 中医模型清单的 Colab 运行器
tests/                         # 12 个离线测试套件（无 key / GPU / 网络）
DATASETS.md                    # 各数据集获取说明、注意事项、字段映射
MODELS.md                      # 18 个模型清单：核验过的仓库 id、架构、授权、注意事项
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
for t in tests/test_*.py; do python "$t"; done      # 12 个都应打印 OK
```

---

## 📄 许可

[MIT](LICENSE) © Med-Bench-Arena contributors.

---

## 🙏 致谢

本项目站在它所接入的开放基准的肩膀上 —— MedQA、MedMCQA、PubMedQA、MMLU、CMB、CMExam、TCMBench、TCM-Ladder、HealthBench、LLMEval-Med、TCMEval、MTCMB、CSEDB、MedSafetyBench、AgentClinic、MedAgentBench、MediQ，以及伦理与安全数据集 MedEthicsBench、MedEthicsQA、PrinciplismQA、MedEthicEval、TCM_Humanities、CARES-18K 与 TCMLM 真实名老中医医案语料 —— 以及运行它们的后端（HuggingFace、vLLM、Poe、LiteLLM）。感谢每一位数据集作者。

<div align="center"><sub>为严谨、可复现的医学与中医模型评测而生。⭐ 如果有用，欢迎 Star！</sub></div>
