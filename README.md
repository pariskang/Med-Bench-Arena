<div align="center">

# 🏥 Med-Bench-Arena

### A unified, config-driven arena for evaluating medical & Traditional-Chinese-Medicine LLMs and agents

*One canonical schema decouples **datasets · backends · metrics** — so any model meets any benchmark under any metric, with zero glue code.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-10%2F10%20passing-brightgreen.svg)](tests/)
[![Benchmarks](https://img.shields.io/badge/benchmarks-30%2B%20live--verified-8A2BE2.svg)](DATASETS.md)
[![TCM](https://img.shields.io/badge/中医-first--class-c1272d.svg)](#-traditional-chinese-medicine-中医)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

**English** · [简体中文](README.zh-CN.md)

</div>

---

**Med-Bench-Arena** is a reference implementation of the *MedEval protocol*: plug **30+ medical / TCM benchmarks** into **any** LLM or agent backend and score them with **12 metrics** — from plain MCQ accuracy to rubric-graded safety, structured 方剂/辨证 matching, and interactive agent `pass^k`. A single **canonical schema** sits between datasets, model backends, and metrics, so you get free `N datasets × M backends × K metrics` composition instead of `N×M×K` hard-coding.

> 🔬 **Every** dataset entry in `configs/` was researched and **verified against its live source** (HuggingFace datasets-server / raw repo files) — real repo ids, splits, column names, and answer encodings. Where a source can only be partly reproduced (gated graders, held-out answers), the limitation is documented, never papered over. See [`DATASETS.md`](DATASETS.md).

```bash
# Runs end-to-end with NO API key, NO GPU, NO network — deterministic mock backend
python -m medeval run configs/example_smoke.yaml
```

---

## ✨ Highlights

- 🧩 **Decoupled by design** — datasets, backends, and metrics depend *only* on the schema, never on each other. Add a similar dataset = edit YAML, zero code.
- 📚 **30+ live-verified benchmarks** — MCQ, open-ended (LLM-judge), safety, multimodal (舌象/影像), and interactive agents — across English & Chinese.
- 🀄 **Traditional Chinese Medicine, first-class** — 辨证证型链, 方剂结构匹配 (君臣佐使), 经络腧穴, 古籍本体, 舌象/脉象, plus a downloadable **knowledge graph** of the classics.
- 🤖 **Real agent loops** — AgentClinic (OSCE + NEJM), **MedAgentBench** against a live **FHIR** EHR server, and **MediQ** proactive questioning — scored with `pass^k`.
- 🔌 **Any backend** — local **HF/vLLM** (batched), **Poe**, and **LiteLLM** (100+ providers + the recommended judge). All swappable by one line of YAML.
- ⚖️ **Faithful grading** — HealthBench per-criterion rubric, MedAgentBench per-task FHIR-payload validation (+ official gated `refsol.py`), signed-point safety rubrics.
- ⚡ **Scales out** — embarrassingly-parallel strided sharding across **local / Ray / Slurm**; resumable, no central server.
- 📤 **Submission-ready** — export predictions to **OpenCompass** / **MedBench** upload formats.

---

## 📑 Table of contents

[Why](#-why) · [Architecture](#-architecture) · [Install](#-install) · [Quick start](#-quick-start) · [Benchmarks](#-benchmark-catalog) · [Backends](#-backends) · [Metrics](#-metrics--judge) · [Agents](#-agents) · [Multimodal](#-multimodal-舌象--影像) · [TCM](#-traditional-chinese-medicine-中医) · [Distributed](#-distributed-scheduling) · [Submission](#-leaderboard-submission) · [Extending](#-extending) · [Layout](#-project-layout) · [Citation](#-citation) · [Contributing](#-contributing) · [License](#-license)

---

## 💡 Why

Three tensions drive the design (and the dataset choices):

1. **Exam saturation vs. real clinical work** → static MCQ *and* rubric-graded open tasks *and* interactive agent environments.
2. **Single-turn QA vs. sequential care** → an agent loop (AgentClinic / MedAgentBench / MediQ) scored with **pass^k**, not one-shot.
3. **Accuracy vs. safety** (safety systematically lags) → safety is its own task type with its own rubric (CSEDB / MedSafetyBench / MTCMB-SE).

---

## 🏗 Architecture

```
Config (YAML)         declarative run spec: models / datasets / eval
Runner                schedule · concurrency · cache · resume · leaderboard
 ├ DatasetAdapter     load() -> Sample ;  parse(text) -> Prediction
 ├ ModelProvider      agenerate() ; HF overrides agenerate_many = vLLM batch
 └ Metric             score() ; aggregate()
Canonical Schema      Sample · Generation · Prediction · Score   (the bedrock)
```

The three middle layers depend **only** on the schema, never on each other — so any one can be swapped or extended in isolation. That is the whole point: **`N × M × K` composition, not `N·M·K` glue.**

---

## 📦 Install

```bash
pip install -e .                 # core (pyyaml) + the CLI
pip install -e ".[all]"          # + datasets, litellm, openai, ray (no GPU needed)
# local HF backend (GPU, optional):
pip install vllm transformers torch peft
```

> Requires **Python 3.10+**. The offline smoke test needs only `pyyaml`.

---

## 🚀 Quick start

**1 — Offline smoke test** (no keys, no GPU, no network — deterministic `mock` backend as both model and judge):

```bash
python -m medeval run configs/example_smoke.yaml
python tests/test_smoke.py && python tests/test_adapters.py     # full offline suite
```

**2 — Real datasets, mock model** (downloads the *actual* benchmarks and runs them end-to-end — great for verifying access):

```bash
python -m medeval run configs/catalog_mcq.yaml        --limit 5   # MedQA · MedMCQA · PubMedQA · MMLU · CMB · CMExam · TCMBench
python -m medeval run configs/catalog_en_med.yaml     --limit 5   # MedXpertQA · MedCalc · MedHallu · MLEC-QA · MediQ …
python -m medeval run configs/catalog_multimodal.yaml --limit 5   # MedFrameQA · SLAKE · TCM-Vision (needs a vision model)
python -m medeval run configs/example_tcm.yaml        --limit 3   # CMB + 辨证 SDT + 方剂 + 安全 (judged)
python -m medeval run configs/example_agentclinic.yaml --limit 5  # pass^k, fully offline
```

**3 — Real model + real judge** — edit `models:` in any config (see `configs/example_api_backends.yaml`):

```yaml
eval: {judge_model: deepseek-r1}
models:
  - {id: my-model,    type: litellm, model: openai/gpt-4o,            api_key_env: OPENAI_API_KEY}
  - {id: deepseek-r1, type: litellm, model: deepseek/deepseek-reasoner, api_key_env: DEEPSEEK_API_KEY, judge_only: true}
```

Outputs land in `results/<run>/`: per-sample `detail__<model>__<ds>.jsonl`, plus `leaderboard.json` and `leaderboard.md`.

```python
# …or from Python
import yaml, medeval
medeval.run_config(yaml.safe_load(open("configs/example_tcm.yaml")))
```

---

## 📊 Benchmark catalog

A representative slice (all wired & verified against live sources; **30+** documented in [`DATASETS.md`](DATASETS.md)):

| Benchmark | Adapter | Task / Metric | Access |
|---|---|---|---|
| **MedQA** (USMLE) | `hf_mcq` | mcq_accuracy | `GBaker/MedQA-USMLE-4-options` |
| **MedMCQA** | `hf_mcq` | mcq_accuracy | `openlifescienceai/medmcqa` (use *validation*) |
| **PubMedQA** | `hf_mcq` | mcq_accuracy | `qiaojin/PubMedQA` (inject yes/no/maybe) |
| **MMLU-medical** | `hf_mcq` | mcq_accuracy | `cais/mmlu` (6 subjects) |
| **CMB** | `hf_mcq` | mcq_accuracy | `FreedomIntelligence/CMB` — **full test 11,200**, GitHub gold joined by `id` |
| **CMExam** | `hf_mcq` | mcq_accuracy | `williamliujl/CMExam` (inline options, multi) |
| **TCMBench** | `tcmbench` | mcq_accuracy | `ywjawmw/TCMBench` — **all 14 demo files** (full bank gated) |
| **TCM-Ladder** | `hf_mcq` | mcq_accuracy | `timzzyus/TCM-Ladder` — 12,778 text + 8,802 image MCQ (open) |
| **HealthBench** | `local_json` | llm_judge `per_criterion` | OpenAI simple-evals — **3 variants**, faithful per-rubric grading |
| **LLMEval-Med** | `local_json` | llm_judge | `llmeval/LLMEval-Med` (checklist rubric) |
| **TCMEval-SDT** 辨证 | `local_json` | llm_judge + syndrome_chain | `zhuyan166/TCMEval` |
| **MTCMB** 方剂/安全 | `local_json` | llm_judge + prescription_match | `Wayyuanyuan/MTCMB` |
| **MedSafetyBench** | `local_json` | llm_judge (safety) | `AI4LIFE-GROUP/med-safety-bench` — **all 18 CSV** (9 AMA × 2) |
| **AgentClinic** | `agentclinic` | pass_k | `SamuelSchmidgall/AgentClinic` — MedQA **214** + NEJM **120** |
| **MedAgentBench** | `medagentbench` | pass_k | live FHIR (Docker); **per-task payload grader** (+ gated `refsol.py`) |
| **MediQ** | `mediq` | pass_k | `stellalisy/MediQ` — proactive questioning |

> **Not wirable config-only** (documented in `DATASETS.md`): **MLEC-QA** (Google-Drive sign-in), **TCM-3CEval** (MedBench held-out answers), **AgentClinic-MIMIC-IV** (PhysioNet credentialing).

---

## 🔌 Backends

| `type` | What | Notes |
|---|---|---|
| `hf` | local checkpoint / repo / LoRA | vLLM batch (overrides `agenerate_many`), transformers fallback |
| `poe` | `https://api.poe.com/v1` | bot name = model; built-in ~500 rpm throttle |
| `litellm` | 100+ providers + any OpenAI-compatible base | retries, cost; **the judge goes here** |
| `mock` | offline deterministic | smoke tests; judge / MCQ / agent-aware |

**Mode A** (default): each backend in-process — HF offline batch, Poe/LiteLLM via API. **Mode B** (production): serve HF with `vllm serve`, route everything through LiteLLM for one retry/cache/cost/limit layer.

---

## 📐 Metrics & judge

`mcq_accuracy` · `pass_k` · `llm_judge` · `f1` · `rouge` · `bleu` · `numeric_match` · plus **five structured TCM metrics**.

<details>
<summary><b>All 12 metrics in detail</b></summary>

- **`mcq_accuracy`** — robust letter/index/text extraction; single **and** multi-answer.
- **`pass_k`** — *k* independent rollouts must *all* succeed (reports pass@1 too).
- **`llm_judge`** — the judge is *just a provider*. Rubric resolves from the dataset (HealthBench points, CSEDB 分数, LLMEval checklist) or a per-task default (open_qa / **sdt 证型链** / **prescription 方剂** / **safety 安全**). Signed points honored; `per_criterion: true` runs the **faithful HealthBench algorithm** (one call per item, boolean `criteria_met`, signed-met / positive-points).
- **`f1` / `rouge` / `bleu`** — token overlap vs. a reference; **CJK-aware** tokenization (char-level Chinese, word-level Latin, `jieba` if installed).
- **`numeric_match`** — calculation tasks (MedCalc-Bench): final number within tolerance / `[lower, upper]` range.
- **`prescription_match`** — **方剂结构匹配**: herb-set P/R/F1 (君臣佐使) + formula-name + 治法 overlap, from the structured gold.
- **`syndrome_chain`** — **证型链结构分**: scores 症状→病机→证型 with **同病异治 partial credit**.
- **`meridian_acupoint`** — **经络腧穴**: set-F1 over 12 正经 + 奇经 and acupoints, with alias normalization.
- **`tongue_pulse`** — **舌象/脉象**: clause-anchored set-F1 over tongue (舌色/舌形/苔) and pulse (脉) features.
- **`classics_ontology`** — **古籍本体**: did the answer ground itself in the right classical source(s)? Set-F1 + longest-match dedup; aliases from the knowledge graph.

</details>

Multiple metrics per dataset are supported — e.g. TCMEval-SDT runs `[llm_judge, syndrome_chain, bleu, rouge]`.

---

## 🤖 Agents

The doctor agent runs the same `ModelProvider` policy inside an `AgentEnvironment(reset/step)` loop.

- **AgentClinic** — OSCE (MedQA) + image cases (NEJM). Patient / measurement / moderator run as their own LLMs via `support:` (the faithful setup), or rule-based offline.
- **MediQ** — proactive information-seeking: the doctor asks the patient for atomic facts (revealed only on a relevant question) or commits with `ANSWER: <letter>`.
- **MedAgentBench** — a **real FHIR EHR server**. The agent emits `GET <url>` / `POST <url>\n<json>` / `FINISH([...])`; scoring uses the official gated `refsol.py` (set `refsol_path`) **or** a built-in **per-task payload grader** that validates `resourceType` + `subject → Patient/{MRN}` + the right flowsheet/SNOMED/NDC/LOINC code. Conservative by design — never a false pass.

```bash
docker run -p 8080:8080 jyxsu6/medagentbench:latest          # serves :8080/fhir
python -m medeval run configs/example_medagentbench.yaml --limit 10
```

---

## 🖼 Multimodal (舌象 / 影像)

`Message` carries optional `images` (http/data URIs or local paths → auto data-URI); `to_openai()` emits OpenAI/LiteLLM **content blocks**, so LiteLLM and Poe vision models work unchanged. The `hf_mcq` adapter takes an `image` field (URL / local path / HF `Image` dict / **raw parquet bytes** / PIL). For sets that ship images as a separate `images.zip`, set `image_zip:` + `image_base:` and the adapter **auto-downloads + unzips** once (idempotent); pre-fetch with `python -m medeval fetch <url>`.

---

## 🀄 Traditional Chinese Medicine (中医)

TCM is not an afterthought — it has dedicated task types, metrics, datasets, and an ontology:

- **辨证** (syndrome differentiation) → `syndrome_chain` scores the 症状→病机→证型 reasoning chain, with 同病异治 partial credit.
- **方剂** (prescription) → `prescription_match` compares herb sets (君臣佐使), formula names, and 治法.
- **经络腧穴 · 古籍本体 · 舌象/脉象** → dedicated structured metrics.
- **Knowledge graph** — the classical-literature ontology is a real, downloadable graph (35 classics + authors + dynasties + 25 经典名方), the single source of truth for `classics_ontology`:

```bash
python -m medeval kg --out data/kg --stats          # 116 nodes, 157 edges → JSON / Turtle / GraphML
```
```python
from medeval import get_kg
get_kg().source_of_formula("银翘散")     # -> 温病条辨
```

---

## ⚡ Distributed scheduling

The grid is embarrassingly parallel, so MedEval distributes by **strided sharding** — each worker runs `samples[i::N]`, writes its own shard-scoped files, and is independently resumable. No central server.

```bash
# one box, N workers (optionally one per GPU):
python -m medeval pool configs/catalog_mcq.yaml --num-shards 4 --gpus 0,1,2,3

# Ray cluster:
python -m medeval pool cfg.yaml --num-shards 8 --backend ray --ray-num-gpus 1

# Slurm: generate + submit a job array + a dependent merge:
python -m medeval slurm cfg.yaml --num-shards 64 --partition gpu --gpus-per-task 1

# many machines (shared FS): run shards, then merge once:
python -m medeval merge /shared/run        # re-aggregates per-sample scores (not mean-of-means)
```

---

## 📤 Leaderboard submission

```bash
# OpenCompass predictions:  predictions/<model>/<dataset>.json
python -m medeval export results/mcq --format opencompass --out oc_out

# MedBench (answers held out — you generate locally, the platform scores):
python -m medeval export results/medbench --format medbench --out submission \
       --medbench-test-dir /path/to/MedBench
```

---

## 🧱 Extending

```python
from medeval import register_provider, register_dataset, register_metric

@register_provider("myllm")    # implement agenerate()
@register_dataset("mybench")   # implement load() / parse()
@register_metric("my_metric")  # implement score() / aggregate()
```

Import the module (add it to `runner.py`'s import list) and reference it by name in YAML.

---

## 📁 Project layout

```
medeval/
├── schema.py                  # canonical types (the bedrock)
├── providers/                 # hf · poe · litellm · mock
├── datasets/                  # hf_mcq · local_json · agent_env · tcmbench · medbench · medagentbench_grader
├── metrics/                   # mcq · llm_judge · text_match · prescription · syndrome · tcm_struct · numeric
├── kg/tcm_classics.py         # 经典文献 knowledge graph (JSON / Turtle / GraphML)
├── runner.py                  # orchestrator
├── distributed.py             # sharding · merge · local/Ray/Slurm
├── submit.py                  # OpenCompass / MedBench export
├── assets.py                  # auto download + unzip images.zip
└── cli.py                     # python -m medeval run|list|export|merge|pool|slurm|kg|fetch
configs/                       # declarative, live-verified run specs
tests/                         # 10 offline suites (no keys / GPU / network)
DATASETS.md                    # per-dataset access notes, caveats, field maps
```

---

## 📖 Citation

If Med-Bench-Arena helps your research, please cite it:

```bibtex
@software{med_bench_arena,
  title  = {Med-Bench-Arena: A Unified Arena for Evaluating Medical and TCM LLMs and Agents},
  author = {Med-Bench-Arena contributors},
  year   = {2026},
  url    = {https://github.com/pariskang/Med-Bench-Arena}
}
```

---

## 🤝 Contributing

Contributions are welcome! Adding a benchmark is usually **config-only** (see `DATASETS.md` for the field-map vocabulary). For a new adapter/metric/backend, register it with the decorator above and add a test under `tests/`. Please run the offline suite before opening a PR:

```bash
for t in tests/test_*.py; do python "$t"; done      # all 10 should print OK
```

---

## 📄 License

[MIT](LICENSE) © Med-Bench-Arena contributors.

---

## 🙏 Acknowledgements

Built on the shoulders of the open benchmarks it wires — MedQA, MedMCQA, PubMedQA, MMLU, CMB, CMExam, TCMBench, TCM-Ladder, HealthBench, LLMEval-Med, TCMEval, MTCMB, CSEDB, MedSafetyBench, AgentClinic, MedAgentBench, MediQ and more — and the backends that run them (HuggingFace, vLLM, Poe, LiteLLM). Thank you to every dataset author.

<div align="center"><sub>Made for rigorous, reproducible medical & TCM model evaluation. ⭐ Star us if this is useful!</sub></div>
