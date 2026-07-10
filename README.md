<div align="center">

# 🏥 Med-Bench-Arena

### A unified, config-driven arena for evaluating medical & Traditional-Chinese-Medicine LLMs and agents

*One canonical schema decouples **datasets · backends · metrics** — so any model meets any benchmark under any metric, with zero glue code.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-13%2F13%20passing-brightgreen.svg)](tests/)
[![Benchmarks](https://img.shields.io/badge/benchmarks-40%2B%20live--verified-8A2BE2.svg)](DATASETS.md)
[![TCM](https://img.shields.io/badge/中医-first--class-c1272d.svg)](#-traditional-chinese-medicine-中医)
[![Ethics & Safety](https://img.shields.io/badge/伦理·安全-first--class-2E8B57.svg)](configs/catalog_ethics_safety.yaml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/Med-Bench-Arena/blob/main/notebooks/Med_Bench_Arena_Colab.ipynb)

**English** · [简体中文](README.zh-CN.md)

</div>

---

**Med-Bench-Arena** is a reference implementation of the *MedEval protocol*: plug **40+ medical / TCM / ethics-&-safety benchmarks** into **any** LLM or agent backend and score them with **12 metrics** — from plain MCQ accuracy to rubric-graded safety, structured 方剂/辨证 matching, and interactive agent `pass^k`. A single **canonical schema** sits between datasets, model backends, and metrics, so you get free `N datasets × M backends × K metrics` composition instead of `N×M×K` hard-coding.

> 🔬 **Every** dataset entry in `configs/` was researched and **verified against its live source** (HuggingFace datasets-server / raw repo files) — real repo ids, splits, column names, and answer encodings. Where a source can only be partly reproduced (gated graders, held-out answers), the limitation is documented, never papered over. See [`DATASETS.md`](DATASETS.md).

```bash
# Runs end-to-end with NO API key, NO GPU, NO network — deterministic mock backend
python -m medeval run configs/example_smoke.yaml
```

---

## ✨ Highlights

- 🧩 **Decoupled by design** — datasets, backends, and metrics depend *only* on the schema, never on each other. Add a similar dataset = edit YAML, zero code.
- 📚 **40+ live-verified benchmarks** — MCQ, open-ended (LLM-judge), safety, multimodal (舌象/影像), and interactive agents — across English & Chinese.
- ⚖️ **Medical ethics & safety, first-class** — a dedicated [`catalog_ethics_safety.yaml`](configs/catalog_ethics_safety.yaml): principlism MCQ (MedEthicsQA · PrinciplismQA · MedEthicEval · TCM_Humanities), open 伦理两难, and AI-safety red-teaming (CARES-18K, refusal/jailbreak/over-refusal) — the axis where models lag most.
- 🀄 **Traditional Chinese Medicine, first-class** — 辨证证型链, 方剂结构匹配 (君臣佐使), 经络腧穴, 古籍本体, 舌象/脉象, real 名老中医医案, plus a downloadable **knowledge graph** of the classics.
- 🤖 **Real agent loops** — AgentClinic (OSCE + NEJM), **MedAgentBench** against a live **FHIR** EHR server, and **MediQ** proactive questioning — scored with `pass^k`.
- 🔌 **Any backend** — local **HF/vLLM** (batched), **Poe**, and **LiteLLM** (100+ providers + the recommended judge). All swappable by one line of YAML.
- ⚖️ **Faithful grading** — HealthBench per-criterion rubric, MedAgentBench per-task FHIR-payload validation (+ official gated `refsol.py`), signed-point safety rubrics.
- 🔬 **Calibrated judge** — open-ended LLM-judge scores are validated against **physician** labels (HealthBench meta-eval, balanced-F1 + κ vs the human ceiling) and demoted to an **auxiliary** tier until they clear calibration. `medeval calibrate`.
- ⚡ **Scales out** — embarrassingly-parallel strided sharding across **local / Ray / Slurm**; resumable, no central server.
- 📤 **Submission-ready** — export predictions to **OpenCompass** / **MedBench** upload formats.

---

## 📑 Table of contents

[Why](#-why) · [Architecture](#-architecture) · [Install](#-install) · [Quick start](#-quick-start) · [Reliability](#-reliability--reproducibility) · [Medical validity](#-medical-scientific-validity--what-this-does-and-does-not-measure) · [Benchmarks](#-benchmark-catalog) · [Backends](#-backends) · [Models](#-model-catalog-medical--tcm) · [Metrics](#-metrics--judge) · [Agents](#-agents) · [Multimodal](#-multimodal-舌象--影像) · [TCM](#-traditional-chinese-medicine-中医) · [Distributed](#-distributed-scheduling) · [Submission](#-leaderboard-submission) · [Extending](#-extending) · [Layout](#-project-layout) · [Citation](#-citation) · [Contributing](#-contributing) · [License](#-license)

---

## 💡 Why

Three tensions drive the design (and the dataset choices):

1. **Exam saturation vs. real clinical work** → static MCQ *and* rubric-graded open tasks *and* interactive agent environments.
2. **Single-turn QA vs. sequential care** → an agent loop (AgentClinic / MedAgentBench / MediQ) scored with **pass^k**, not one-shot.
3. **Accuracy vs. safety & ethics** (both systematically lag) → safety/ethics are their own task types with their own rubrics (CSEDB / MedSafetyBench / MTCMB-SE / **CARES-18K** red-teaming), plus a dedicated medical-ethics arena (**MedEthicsQA / PrinciplismQA / MedEthicEval / MedEthicsBench**).

---

## 🏗 Architecture

```
Config (YAML)         declarative run spec: models / datasets / eval
Runner                schedule · concurrency · cache · checkpoint-resume · live progress · leaderboard
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

> Requires **Python 3.10+**. The one-liner `example_smoke.yaml` run needs only `pyyaml`; the full offline test suite additionally needs the `datasets` extra (two suites load it) — `pip install -e ".[all]"`.

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
python -m medeval run configs/catalog_ethics_safety.yaml --limit 5 # 医学伦理 MedEthicsQA · PrinciplismQA · MedEthicEval · CARES-18K safety
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

## 🔬 Reliability & reproducibility

MCQ evaluation is only trustworthy if the data is exactly what you think it is — and a full run is only practical if it survives an interruption. Five guards:

- **Pinned revisions** — the headline MCQ catalog ([`catalog_mcq.yaml`](configs/catalog_mcq.yaml), except TCMBench-demo whose upstream publishes no stable ref — it is `demo`-tier anyway), the model-catalog MCQ slice, and TCM-Ladder are locked to immutable commits, so those eval sets can never silently change. HF repos use `revision: <sha>` (passed to `load_dataset`); raw-file sources embed the commit in the URL (`…/resolve/<sha>/…`, `raw.githubusercontent/…/<sha>/…`). The other catalogs (`catalog_en_med` / `catalog_multimodal` / `catalog_cn_tcm` / parts of `catalog_ethics_safety`) currently **track `main`** — their headers say so; pin a `revision:` before publishing numbers from them. Large pinned files download via an atomic, **HTTP-Range-resuming** fetcher — robust to proxies that truncate big responses, and a failed download never poisons the cache.
- **`preflight`** — profile every dataset *without a model*: sample count, option-count distribution, **answer-parse success rate**, and the first few examples. Run it before you spend a single token:

```bash
python -m medeval preflight configs/catalog_mcq.yaml          # all datasets, full load
python -m medeval preflight configs/catalog_mcq.yaml --strict # CI: non-zero exit if any parse < 100%
```

```
✓ cmb_test   [hf_mcq]
    样本数 samples        : 11200 of 11200 rows
    选项数 option dist     : {3: 1, 4: 1201, 5: 9956, 6: 42}
    解析率 answer parse    : 100.0%  ████████████████████
```

A parse rate below 100% means rows are being dropped (a mis-mapped `field_map`, an unexpected answer encoding, options that don't parse) — `preflight` lists them by reason so you fix the config, not the symptoms.

`preflight` also runs a pure-Python **MinHash/LSH near-duplicate scan** over each dataset's questions — a cheap first-pass signal for contamination or accidental duplication (a scrape that concatenated the same source twice, a train/test split that leaked rows). LSH banding keeps this near-linear instead of the O(n²) a naive all-pairs comparison would need, so it stays practical on an 11k-row benchmark. It flags *paraphrase-level* similarity, not just byte-identical text:

```bash
python -m medeval preflight configs/catalog_mcq.yaml --dup-threshold 0.85   # default
python -m medeval preflight configs/catalog_mcq.yaml --no-dedup             # skip it (large sets, speed)
python -m medeval preflight configs/catalog_mcq.yaml --strict-dedup        # also fail CI on any hit
```

Near-duplicates are **informational by default** (a similarity cutoff is a heuristic, not a certain contamination verdict) — `--strict` alone doesn't fail on them; pass `--strict-dedup` to make them a hard CI gate.

- **Comparability tiers (`split_type`)** — every result row carries a `split_type` so *officially-comparable* runs never get mixed with internal ones on the leaderboard. **`unverified` is the default** — a dataset must actively earn `official`; a YAML line alone can never grant it (see `medeval.eligibility`). The leaderboard renders **✅ Official** and **⚠️ Internal / non-comparable** as separate sections. Values: `official` · `validation` · `demo` · `sample` · `gated` · `approximated` · `reimplementation` · `unverified`. So **CMB-val** (validation), **TCMBench-demo** (demo), **CSEDB-sample** (sample), **MedAgentBench's built-in grader** (approximated, unless you supply the official `refsol_path`), and a partially-configured AgentClinic / any MediQ run (`reimplementation`, unless every support role is wired) are clearly fenced off from a full official run.
- **Automated on every web session + CI** — a `SessionStart` hook (`.claude/`) installs deps and runs `preflight --strict` so each Claude-Code-on-the-web session profiles the eval set up front; GitHub Actions (`.github/workflows/ci.yml`) runs the offline test suite **and** `preflight --strict` as a data gate on every push/PR.
- **Deterministic, resumable runs** — every `(model, dataset)` generation is cached on disk keyed by a **stable** hash of the model + its *effective* sampling params (`hashlib`, never the per-process-salted builtin `hash()`), so an interrupted full-dataset run (Colab timeout, spot preemption) **resumes from the last checkpoint** instead of regenerating from scratch. Progress shows **live** per-dataset bars (`tqdm`), generations are flushed to disk in batches of `checkpoint_every` (default 64 — friendly to Google Drive / network mounts), the leaderboard is rewritten after each dataset, and a crash-truncated cache line is skipped rather than poisoning the resume.
- **Content-addressed cache — never scores a stale generation against new data.** The cache filename folds in the *entire* dataset protocol (revision, field_map, prompt template, instruction, adapter class + a hand-versioned `ADAPTER_PROTOCOL_VERSION`, and — for agents — `k`/`max_turns`/support-model identities), so any of those changing routes to a fresh cache file. On top of that, every cached record carries a hash of the *actual sample content* (rendered messages + choices + reference); if an unpinned source silently drifts upstream — same `sample_id`, different question/reference — that sample's hash no longer matches and it is regenerated rather than silently scored against a changed gold answer. Metric/rubric/judge config is deliberately **excluded** from the signature (scores are always recomputed fresh each run, never cached, so tweaking a rubric can't go stale — and doesn't waste a real generation call).
- **`run_manifest.json`** — every run writes one: git commit, adapter-protocol version, the full run config (secrets redacted), and per-dataset resolved `split_type` + pin evidence + `preflight`-style load stats, and per-model resolved identity — so a leaderboard number can always be traced back to exactly what produced it.
- **Statistical rigor, not just point estimates.** Every leaderboard row carries a **95% bootstrap CI** (2000 resamples, fixed seed) for its headline metric; the table flags `n` below a 30-sample floor as noisy, and marks adjacent rows whose CIs overlap as "≈ statistically tied" rather than implying a real ranking gap. For a direct paired A/B (same held-out samples, aligned by `sample_id`) with a proper significance test:

```bash
python -m medeval compare results/ --dataset medqa_usmle --model-a gpt-4o --model-b claude
# paired-bootstrap CI for the mean difference + McNemar's test when the metric is binary (e.g. MCQ accuracy)
```

---

## ⚕️ Medical scientific validity — what this does and does not measure

Everything above is engineering rigor: it makes a reported number *trustworthy given what it measures*. It says nothing about whether what it measures is *clinical competence*. Three gaps worth being explicit about, so a leaderboard row is never over-read:

- **Dataset diversity ≠ clinical coverage.** Having 30+ benchmarks across MCQ / open-ended / agent / TCM / safety families gives *breadth of format*, not breadth of *medical practice*. USMLE-style MCQ tests recall-and-reasoning under exam conditions; it does not test the messier skills real practice needs — eliciting an ambiguous history, tolerating incomplete information, revising a working diagnosis, coordinating with other clinicians. A model that tops every catalog here has been shown to be good at *these specific tasks*, not validated for clinical deployment in any specialty, population, or care setting not represented in the data.
- **TCM structured metrics and open-ended judge scores are proxy metrics, not clinical-efficacy measures.** `syndrome_chain` scores whether a model's 辨证 (syndrome differentiation) reasoning chain matches a reference chain's structure; `prescription_match` scores herb-set/formula-name overlap against a reference prescription. Neither measures whether the differentiation was *clinically correct for the patient* or whether the prescription would *work*. Similarly, `llm_judge`'s rubric-graded open-ended score measures agreement with a rubric (itself only checked against physician **agreement**, not treatment outcomes — see "Judge calibration" in [Metrics & judge](#-metrics--judge)). A high score on these metrics means "matches the reference structure/rubric well," not "would help a patient."
- **Safety evaluation here is necessarily partial.** CSEDB / MedSafetyBench / MTCMB-SE / CARES-18K test refusal-worthy and harmful-request scenarios — they do **not** measure over-refusal (declining benign requests too), the quality of escalation advice ("see a doctor" without saying *when this is urgent* vs *when it can wait*), calibrated uncertainty communication, or longitudinal harm from repeated interactions. A model can score well on every safety benchmark here and still fail in ways this suite has no test for.

None of this is a reason not to use the numbers — it's a reason not to read more into them than they claim. Treat every result here as *capability-on-this-benchmark*, not a clinical-competence or deployment-readiness certification.

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
| **MedEthicsQA** ⚖️ | `hf_mcq` | mcq_accuracy | `JianhuiWei7/MedEthicsQA` — **5,623** ethics MCQ (bilingual), 100% parse |
| **PrinciplismQA** ⚖️ | `hf_mcq` + `local_json` | mcq_accuracy + llm_judge | `FreedomIntelligence/PrinciplismQA-Demo` — 100 MCQ + 126 rubric |
| **MedEthicEval** ⚖️ | `hf_mcq` + `local_json` | mcq_accuracy + llm_judge | `X-LANCE/MedEthicEval` (NAACL'25) — 629 知识 + 伦理两难 + 违规检测 |
| **TCM_Humanities** ⚖️ | `hf_mcq` | mcq_accuracy | `TCMLM/TCM_Humanities` — **500** 医学人文/伦理/卫生法 MCQ (multi) |
| **MedEthicsBench** ⚖️ | `local_json` | llm_judge (rubric) | `pariskang/MedEthicsBench` — key-point rubric (forward-compatible) |
| **CARES-18K** 🛡️ | `local_json` | llm_judge (safety) | `HFXM/CARES-18K` — **9,239** red-team prompts (8 principles × 4 harm × 4 strategy) |
| **real_clinical_cases** 🀄 | `local_json` | syndrome_chain + llm_judge | `TCMLM/real_clinical_cases…` — **500** 名老中医医案 (辨证论治) |
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

## 🤖 Model catalog (medical & TCM)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/Med-Bench-Arena/blob/main/notebooks/Med_Bench_Arena_Colab.ipynb)

[`configs/catalog_med_models.yaml`](configs/catalog_med_models.yaml) wires **17 medical & TCM LLMs** as ready-to-run HF/vLLM backends — every repo id, base architecture, `dtype`, context length and `trust_remote_code` flag **verified against the live HuggingFace page** (full table + per-model notes in [`MODELS.md`](MODELS.md)). Pick one per run (vLLM holds a model in GPU memory) with `--models`:

```bash
python -m medeval run configs/catalog_med_models.yaml --models zhongjing-2-1_8b --limit 20
python -m medeval run configs/catalog_med_models.yaml --models biancang        --limit 20
```

| Group | Models (`--models <id>`) |
|---|---|
| 🀄 **TCM 中医** | `zhongjing-2-1_8b` · `dao1-30b-a3b` · `biancang` · `taiyi` · `disc-medllm` |
| 🩺 **Chinese medical** | `huatuogpt2-7b` · `huatuogpt-o1-7b` · `aquilamed-rl` · `baichuan-m1-14b` · `baichuan-m2-32b` · `clinicalgpt-r1` |
| ⚕️ **English / international** | `meditron-70b` · `biomistral-7b` · `deepseek-r1-32b` · `medgemma-27b-it` · `citrus-70b` |
| 🖼 **Multimodal** | `lingshu-7b` (+ `medgemma-27b-it`) |

Quirks handled for you: **ZhongJing-2** is a LoRA on `Qwen1.5-1.8B-Chat`; **Baichuan / Taiyi / DISC / AquilaMed** need `trust_remote_code`; **Meditron / MedGemma** are gated (accept the license + `huggingface-cli login`); reasoning models (**DeepSeek-R1 / HuatuoGPT-o1 / Baichuan-M2 / ClinicalGPT-R1**) get a larger `max_tokens`; and a vLLM load failure falls back to transformers instead of crashing. *(Qibo and the Qilin-Med text model aren't publicly on HF — documented in [`MODELS.md`](MODELS.md).)*

▶️ **One-click GPU run** — open the notebook straight in Google Colab (no local setup; clone · install vLLM · pick a model · **sweep every benchmark, every question** · save all results to **Google Drive**). The notebook's `LIMIT = 0` default scores the *full* set of each dataset (set `LIMIT = 50` for a quick smoke test). A full sweep shows **live per-dataset progress**, **flushes results to Drive** every checkpoint, and **resumes where it left off** if the Colab session disconnects — just re-run the cell:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pariskang/Med-Bench-Arena/blob/main/notebooks/Med_Bench_Arena_Colab.ipynb)

---

## 📐 Metrics & judge

`mcq_accuracy` · `pass_k` · `llm_judge` · `f1` · `rouge` · `bleu` · `numeric_match` · plus **five structured TCM metrics**.

<details>
<summary><b>All 12 metrics in detail</b></summary>

- **`mcq_accuracy`** — **zero-shot CoT** prompting by default (reason step-by-step → `Answer: X`), with a structured-line-first, last-match-wins parser that ignores distractors mentioned in the reasoning and strips `<think>` traces. Robust letter/index/text extraction; single **and** multi-answer. Universal CoT is not a neutral choice — a reasoning model with its own thinking format, or one tuned for terse answers, can be disadvantaged by a prompt shape it wasn't trained for. `prompt_style: cot | direct | native` picks the *default* instruction per-dataset (an explicit `instruction:` always wins outright): `cot` (default, unchanged), `direct` (answer only, no reasoning requested), `native` (no instruction appended at all — the model's own default response shape decides; the parser's loose fallback scan still finds an answer without a structured `Answer:` line).
- **`pass_k`** — *k* independent rollouts must *all* succeed (reports pass@1 too). Reasoning-model `<think>` traces are stripped before agent action parsing.
- **`llm_judge`** — the judge is *just a provider*. Rubric resolves from the dataset (HealthBench points, CSEDB 分数, LLMEval checklist) or a per-task default (open_qa / **sdt 证型链** / **prescription 方剂** / **safety 安全**). Explicit 0/0.5/1 scoring anchors; malformed judge JSON is recovered through a **`json-repair`** pipeline; criterion keys are matched by id/text/normalized-text/position (never a silent 0). A judge that still returns nothing usable is **retried, then the sample is excluded** (`value=None`, surfaced as `judge_failures` in the aggregate) — a judge infrastructure failure is never scored as a model 0; ungraded criteria in a partial key match are likewise excluded from the ratio, not coerced to 0. Signed points honored — a **negative-point** rubric is always routed through the per-criterion path so penalties keep the right sign; `per_criterion: true` runs the **faithful HealthBench algorithm** (one call per item, boolean `criteria_met`, signed-met / positive-points).
- **`f1` / `rouge` / `bleu`** — token overlap vs. a reference; **CJK-aware** tokenization (char-level Chinese, word-level Latin, `jieba` if installed).
- **`numeric_match`** — calculation tasks (MedCalc-Bench): extracts the labeled `Answer: <number>` line (then a last-match marker, scientific notation supported) and checks it within tolerance / `[lower, upper]` range.
- **`prescription_match`** — **方剂结构匹配**: herb-set P/R/F1 (君臣佐使) + formula-name + 治法 overlap, from the structured gold.
- **`syndrome_chain`** — **证型链结构分**: scores 症状→病机→证型 with **同病异治 partial credit**.
- **`meridian_acupoint`** — **经络腧穴**: set-F1 over 12 正经 + 奇经 and acupoints, with alias normalization.
- **`tongue_pulse`** — **舌象/脉象**: clause-anchored set-F1 over tongue (舌色/舌形/苔) and pulse (脉) features.
- **`classics_ontology`** — **古籍本体**: did the answer ground itself in the right classical source(s)? Set-F1 + longest-match dedup; aliases from the knowledge graph.

> Structured metrics (方剂/证型链/经络腧穴/舌脉/古籍) **exclude** samples whose reference yields no extractable gold (`value=None`; aggregates report `n_scored` / `skipped_no_gold`) instead of scoring them 0 — an unscorable reference is not the model's fault.

</details>

Multiple metrics per dataset are supported — e.g. TCMEval-SDT runs `[llm_judge, syndrome_chain, bleu, rouge]`.

### ⚖️ Judge calibration — open-ended scores must earn the headline

An LLM-judge score is only trustworthy if the judge agrees with **human experts**.
`medeval calibrate` measures that agreement on a frozen, **physician-labeled** set and
decides whether open-ended scores may headline the leaderboard or must be reported as an
**auxiliary** metric.

The anchor is **HealthBench's official meta-evaluation** — for each *(conversation,
completion, rubric-item)* it ships the binary judgments of 2+ physicians, so we can compute
*real* judge↔human agreement offline. We replicate OpenAI simple-evals' metric verbatim
(**balanced pairwise F1** over met/unmet), report **Cohen's κ** + raw agreement with bootstrap
95% CIs, and compare every rater against the **physician-vs-physician ceiling**.

A strong frontier model graded **120 items blind** (the physician labels held out) using the
verbatim HealthBench grader. The result:

| vs physicians | Balanced F1 | Cohen's κ | Raw agree |
|---|---|---|---|
| **strong-model judge** | **0.697** [0.61, 0.77] | **0.394** [0.23, 0.54] | 0.736 |
| _physician ceiling (human)_ | _0.719_ [0.63, 0.80] | _0.437_ | _0.745_ |

The judge is **physician-equivalent** — within 0.05 of the human ceiling on *both* metrics, CIs
overlapping. But the ceiling itself is only **moderate** (κ≈0.44): rubric-item grading is
intrinsically subjective. So by policy the judge is **not headline-eligible** (absolute κ < 0.40)
and open-ended scores are kept **auxiliary** — exactly the conservative default. The leaderboard
enforces this: judge-scored datasets land in a separate *🧪 Auxiliary* tier until a calibration
report marks the judge headline-eligible (`data/calibration/calibration_report.md`).

**A calibrated judge model doesn't make a flaky RUN trustworthy.** If a run's judge fails to
grade more than **2%** of samples (empty/garbled JSON the repair pipeline couldn't recover,
refusals, API errors), that row is forced into Auxiliary regardless of calibration — a
`⚠️ N% judge-ungraded` note appears in its leaderboard row — since a high-failure subset is
plausibly the hardest/most-ambiguous samples, not a random, harmless one, and warrants manual
spot-checking before publishing.

**A calibration is bound to exactly what it measured — never a blanket "the judge is fine" pass.**
Every report carries a `signature`: the concrete judge model + revision (live `--config --judge`
mode only), and the grading prompt/protocol tested (today: the HealthBench per-criterion style).
The leaderboard promotes a row out of Auxiliary only when *that row's own* judge model+revision
and grading prompt **exactly match** the signature — calibrating `gpt-4.1` on HealthBench's
English criteria does **not** grant headline status to `deepseek-r1`, or to a Chinese TCM 辨证/
伦理/安全 dataset graded through the default (non-per-criterion) rubric prompt, even though a
`calibrated: true` report exists on disk. A `--labels` pass (a strong-model/human blind review,
not a live judge run) has no concrete judge to bind to at all, so it is **never auto-applied**
to any run — it's reported as evidence that physician-equivalence is *achievable*, not as a
live-verifiable guarantee; only `--config --judge <id>` produces a binding report.

```bash
# regenerate the frozen physician-labeled set, then score a reviewer's blind labels
python -m medeval calibrate --rebuild-from <oss_meta_eval.jsonl URL/path>
python -m medeval calibrate --labels data/calibration/healthbench_meta_strongmodel_labels.jsonl
# …or calibrate a live API judge against the same physician gold (same code path the leaderboard uses,
# and the ONLY mode that produces a binding, leaderboard-honored signature):
python -m medeval calibrate --config configs/example_open_safety.yaml --judge gpt-4.1 --strict
```

> TCMEval-SDT / MTCMB ship no physician labels, so their open-ended scores inherit **auxiliary**
> status until a TCM domain expert supplies a labels file — the harness scores it identically,
> but as a *separate* calibration bound to that dataset's own judge + prompt signature, not
> inherited from HealthBench's.

---

## 🤖 Agents

The doctor agent runs the same `ModelProvider` policy inside an `AgentEnvironment(reset/step)` loop.

- **AgentClinic** — OSCE (MedQA) + image cases (NEJM). Patient / measurement / moderator run as their own LLMs via `support:`, or rule-based offline. `split_type` reflects protocol fidelity, not a config wish: **all three** roles present → `official`; some but not all → `reimplementation`; none → `approximated` — a config can't claim the faithful setup by only wiring one role.
- **MediQ** — proactive information-seeking: the doctor asks the patient for atomic facts (revealed only on a relevant question), commits with `ANSWER: <letter>` (optionally `(confidence: NN%)`), or `ABSTAIN` when the evidence is genuinely insufficient. `pass_k` additionally reports `abstain_rate`, `avg_questions` (question efficiency), and `mean_confidence` / `confidence_brier_score` (calibration) when the doctor states one. Never `official` by default — the rule-based patient is `approximated`, an LLM patient is `reimplementation` (neither replicates MediQ's original Patient/Expert System prompts verbatim); only an explicit config override claims `official`, and that claim is on the operator, not the harness.
- **MedAgentBench** — a **real FHIR EHR server**. The agent emits `GET <url>` / `POST <url>\n<json>` / `FINISH([...])`; scoring uses the official gated `refsol.py` (set `refsol_path`) **or** a built-in **per-task payload grader** that validates `resourceType` + `subject → Patient/{MRN}` + the right flowsheet/SNOMED/NDC/LOINC code. Conservative by design — never a false pass. **Isolation risk**: every episode shares one live, unreset FHIR server by default (a write can leak into a later episode's reads) — concurrency is forced to 1 and `split_type` capped below `official` until you wire an `episode_reset` hook.

```bash
docker run -p 8080:8080 jyxsu6/medagentbench:latest          # serves :8080/fhir
python -m medeval run configs/example_medagentbench.yaml --limit 10
```

**Cost by role.** A faithful multi-agent AgentClinic run calls up to 3 extra LLMs per turn
(patient / measurement / moderator) — `model_cost_usd` is the doctor's cost only, so a leaderboard
row for a faithful run would otherwise look exactly as cheap as the fully-scripted approximation.
Every agent row with `support:` configured additionally carries `role_cost_usd: {doctor, patient,
measurement, moderator}` so single-agent and multi-agent setups are cost-comparable, not just
score-comparable.

---

## 🖼 Multimodal (舌象 / 影像)

`Message` carries optional `images` (http/data URIs or local paths → auto data-URI); `to_openai()` emits OpenAI/LiteLLM **content blocks**, so LiteLLM and Poe vision models work unchanged. Local **HF vision models** (Lingshu / MedGemma) work too: image-carrying batches are routed through **vLLM's `chat()` multimodal API** — and if the loaded backend can't run images (transformers fallback, old vLLM) the provider **raises instead of silently dropping the images** and scoring the model on text alone. The `hf_mcq` adapter takes an `image` field (URL / local path / HF `Image` dict / **raw parquet bytes** / PIL). For sets that ship images as a separate `images.zip`, set `image_zip:` + `image_base:` and the adapter **auto-downloads + unzips** once (idempotent); pre-fetch with `python -m medeval fetch <url>`.

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
├── eligibility.py              # official-tier gate: pin evidence for content adapters
├── runner.py                  # orchestrator; content-addressed cache + run_manifest.json
├── distributed.py             # sharding · merge · local/Ray/Slurm
├── submit.py                  # OpenCompass / MedBench export
├── assets.py                  # auto download + unzip images.zip
└── cli.py                     # python -m medeval run|preflight|list|export|merge|pool|slurm|kg|fetch
configs/                       # declarative, live-verified run specs (incl. catalog_med_models.yaml)
notebooks/                     # Colab runner for the medical / TCM model catalog
tests/                         # 13 offline suites (no keys / GPU / network; needs the `datasets` extra)
DATASETS.md                    # per-dataset access notes, caveats, field maps
MODELS.md                      # the 17-model catalog: verified repo ids, archs, gating, quirks
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
pip install -e ".[all]"                             # the suite needs the `datasets` extra
for t in tests/test_*.py; do python "$t"; done      # all 13 should print OK
```

---

## 📄 License

[MIT](LICENSE) © Med-Bench-Arena contributors.

---

## 🙏 Acknowledgements

Built on the shoulders of the open benchmarks it wires — MedQA, MedMCQA, PubMedQA, MMLU, CMB, CMExam, TCMBench, TCM-Ladder, HealthBench, LLMEval-Med, TCMEval, MTCMB, CSEDB, MedSafetyBench, AgentClinic, MedAgentBench, MediQ, and the ethics-&-safety sets MedEthicsBench, MedEthicsQA, PrinciplismQA, MedEthicEval, TCM_Humanities, CARES-18K, plus the TCMLM real 名老中医医案 corpus — and the backends that run them (HuggingFace, vLLM, Poe, LiteLLM). Thank you to every dataset author.

<div align="center"><sub>Made for rigorous, reproducible medical & TCM model evaluation. ⭐ Star us if this is useful!</sub></div>
