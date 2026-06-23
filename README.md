# Med-Bench-Arena · MedEval

A unified, config-driven protocol + reference implementation for plugging
**medical / Traditional-Chinese-Medicine evaluation datasets** into **any** LLM or
agent backend.

One **canonical schema** decouples three concerns — *datasets*, *model backends*,
*metrics* — so you get free `N datasets × M backends × K metrics` composition
instead of `N×M×K` hard-coding. Backends are three first-class citizens: local
**HF** (vLLM batch), **Poe**, and **LiteLLM** (100+ providers + the recommended
DeepSeek-R1 judge).

> Every dataset entry in `configs/` was researched and **verified against its live
> source** (HF datasets-server / raw repo files) on 2026-06-22 — real repo ids,
> splits, column names and answer encodings. See [`DATASETS.md`](DATASETS.md).

---

## Why

Three tensions drive the design (and the dataset choices):

1. **Exam saturation vs. real clinical work** → static MCQ *and* rubric-graded
   open tasks *and* interactive agent environments.
2. **Single-turn QA vs. sequential care** → an agent loop (AgentClinic /
   MedAgentBench) scored with **pass^k**, not one-shot.
3. **Accuracy vs. safety** (safety systematically lags) → safety is its own task
   type with its own rubric (CSEDB / MedSafetyBench / MTCMB-SE).

---

## Architecture (5 layers)

```
Config (YAML)         declarative run spec: models / datasets / eval
Runner                schedule · concurrency · cache · resume · leaderboard
 ├ DatasetAdapter     load() -> Sample ;  parse(text) -> Prediction
 ├ ModelProvider      agenerate() ; HF overrides agenerate_many = vLLM batch
 └ Metric             score() ; aggregate()
Canonical Schema      Sample · Generation · Prediction · Score  (the bedrock)
```

The three middle layers depend **only** on the schema, never on each other — so
any one can be swapped or extended in isolation.

```
medeval/
├── schema.py                  # canonical types
├── providers/{base,hf,poe,litellm_provider,mock}.py
├── datasets/{base,hf_mcq,local_json,agent_env,tcmbench,medbench,medagentbench_grader}.py
├── metrics/{base,mcq,llm_judge,text_match,prescription,syndrome,tcm_struct}.py
├── kg/tcm_classics.py         # 经典文献 knowledge graph (JSON/Turtle/GraphML)
├── runner.py                  # orchestrator
├── submit.py                  # OpenCompass / MedBench submission export
├── distributed.py             # sharding · merge · local/Ray/Slurm launchers
└── cli.py                     # python -m medeval run|list|export|merge|pool|slurm|kg
```

---

## Install

```bash
pip install -r requirements.txt          # pyyaml + datasets (+ litellm, openai)
# local HF backend (GPU, optional):  pip install vllm transformers torch peft
```

## Quickstart

**Offline smoke test** — no keys, no GPU, no network (uses the deterministic
`mock` backend as both model and judge):

```bash
python -m medeval run configs/example_smoke.yaml
python tests/test_smoke.py && python tests/test_adapters.py   # full offline suite
```

**Real datasets, mock model** — downloads the *actual* benchmarks and runs them
end to end (great for verifying access):

```bash
python -m medeval run configs/catalog_mcq.yaml      --limit 5   # MedQA/MedMCQA/PubMedQA/MMLU/CMB/CMExam/TCMBench
python -m medeval run configs/catalog_en_med.yaml   --limit 5   # MedXpertQA/MedAgentsBench/MedCalc/MedR-Bench/MedHallu/Med-HALT/MLEC-QA/MediQ
python -m medeval run configs/catalog_multimodal.yaml --limit 5 # MedFrameQA/SLAKE/TCM-Vision (needs a vision model)
python -m medeval run configs/catalog_cn_tcm.yaml   --limit 5   # PromptCBLUE/TCM-BEST4SDT/TCMEval-PA
python -m medeval run configs/example_tcm.yaml      --limit 3   # CMB + SDT + 方剂 + 安全 (judged)
python -m medeval run configs/example_open_safety.yaml --limit 3
python -m medeval run configs/example_agentclinic.yaml --limit 5   # pass^k, fully offline
```

**Real model + real judge** — edit `models:` in any config (see
`configs/example_api_backends.yaml`), e.g.:

```yaml
eval: {judge_model: deepseek-r1}
models:
  - {id: my-model,   type: litellm, model: openai/gpt-4o, api_key_env: OPENAI_API_KEY}
  - {id: deepseek-r1, type: litellm, model: deepseek/deepseek-reasoner, api_key_env: DEEPSEEK_API_KEY, judge_only: true}
```

Outputs land in `results/<run>/`: per-sample `detail__<model>__<ds>.jsonl`,
plus `leaderboard.json` and `leaderboard.md`.

```python
# or from Python
import yaml, medeval
medeval.run_config(yaml.safe_load(open("configs/example_tcm.yaml")))
```

---

## Dataset catalog (wired & verified)

| Dataset | Adapter | Task / Metric | Access |
|---|---|---|---|
| TCM-Ladder (text + 舌象/herb images) | `hf_mcq` | mcq_accuracy | `timzzyus/TCM-Ladder` (open, multimodal) |
| MedQA (USMLE) | `hf_mcq` | mcq_accuracy | `GBaker/MedQA-USMLE-4-options` |
| MedMCQA | `hf_mcq` | mcq_accuracy | `openlifescienceai/medmcqa` (use **validation**) |
| PubMedQA | `hf_mcq` | mcq_accuracy | `qiaojin/PubMedQA` / `pqa_labeled` (inject yes/no/maybe) |
| MMLU-medical | `hf_mcq` | mcq_accuracy | `cais/mmlu` (6 subjects) |
| CMB | `hf_mcq` | mcq_accuracy | `FreedomIntelligence/CMB` (raw json, **val**, multi) |
| CMExam | `hf_mcq` | mcq_accuracy | `williamliujl/CMExam` CSV (inline options, multi) |
| TCMBench | `tcmbench` | mcq_accuracy | `ywjawmw/TCMBench` `data_demo/` (embedded options) |
| HealthBench | `local_json` | llm_judge | OpenAI simple-evals JSONL (signed-points rubric) |
| LLMEval-Med | `local_json` | llm_judge | `llmeval/LLMEval-Med` (checklist rubric) |
| TCMEval-SDT | `local_json` (sdt) | llm_judge | `zhuyan166/TCMEval` (**Train** has gold) |
| MTCMB (方剂/安全) | `local_json` | llm_judge | `Wayyuanyuan/MTCMB` 12 JSONL |
| CSEDB | `local_json` (safety) | llm_judge | public 2-record sample; **full bank gated** |
| MedSafetyBench | `local_json` (safety) | llm_judge | `AI4LIFE-GROUP/med-safety-bench` CSV |
| AgentClinic | `agentclinic` | pass_k | `SamuelSchmidgall/AgentClinic` JSONL (offline-capable) |
| MedAgentBench | `medagentbench` | pass_k | live FHIR (Docker); built-in grader (+ optional `refsol.py`) |

**Not wirable config-only** (documented in `DATASETS.md`): **MLEC-QA** (Google-Drive
sign-in), **TCM-3CEval** (MedBench submission, held-out answers),
**AgentClinic-MIMIC-IV** (PhysioNet credentialing).

---

## Backends

| `type` | What | Notes |
|---|---|---|
| `hf` | local checkpoint / repo / LoRA | vLLM batch (overrides `agenerate_many`), transformers fallback |
| `poe` | `https://api.poe.com/v1` | bot name = model; built-in ~500 rpm throttle |
| `litellm` | 100+ providers + any OpenAI-compatible base | retries, cost; **judge goes here** |
| `mock` | offline deterministic | smoke tests; judge/MCQ/agent-aware |

**Mode A** (default): each backend in-process — HF offline batch, Poe/LiteLLM via
API. **Mode B** (production): serve HF with `vllm serve`, route everything
(including Poe) through LiteLLM for one retry/cache/cost/limit layer
(`configs/example_api_backends.yaml`).

## Metrics & judge

- `mcq_accuracy` — robust letter/index/text extraction; single **and** multi-answer.
- `pass_k` — k independent rollouts must *all* succeed (reports pass@1 too).
- `llm_judge` — the judge is *just a provider*. Rubric resolves from the dataset
  (HealthBench points, CSEDB 分数, LLMEval checklist) or a per-task default
  (open_qa / **sdt 证型链** / **prescription 方剂** / **safety 安全**). Signed
  points are honored (HealthBench-style). Mark a model `judge_only: true` to use
  it as a judge without ranking it.
- `f1` / `rouge` / `bleu` — token-overlap vs. a reference answer; **CJK-aware**
  tokenization (char-level for Chinese, word-level for Latin, `jieba` if installed).
  ROUGE reports 1/2/L; BLEU is smoothed sentence-BLEU-4 (1..4 in detail).
- `numeric_match` — calculation tasks (MedCalc-Bench): extracts the final number and
  checks it within tolerance or an explicit `[lower_limit, upper_limit]` range.
- `prescription_match` — **方剂结构匹配**: herb-set precision/recall/F1 (君臣佐使) +
  formula-name match + 治法 overlap, read from the structured gold (e.g. the
  MTCMB TCM-FRD `{治法, 方剂, 药物组成}` dict); herb names normalized (dosages
  stripped).
- `syndrome_chain` — **证型链结构分** for 辨证: scores the 症状→病机→证型 chain with
  **同病异治 partial credit** (multiple acceptable 证型 → recall-based credit);
  reads `syndrome` / `pathogenesis` / `reference` from the gold.
- `meridian_acupoint` — **经络腧穴**: set-F1 over canonical meridians (12 正经 + 奇经)
  and acupoints, with alias normalization (胃经 → 足阳明胃经) — targets the subdomain
  models are weakest on.
- `tongue_pulse` — **舌象/脉象**: clause-anchored set-F1 over tongue features
  (舌色/舌形/苔) and pulse features (脉) — for multimodal 舌诊/脉诊 (e.g. TCM-Ladder);
  won't grab 红 from 面色红润.
- `classics_ontology` — **古籍本体**: did the answer ground itself in the correct
  classical source(s) (《伤寒论》《黄帝内经》…)? Set-F1 + `all_sources_cited`, with
  longest-match dedup (伤寒杂病论 ≠ 伤寒论). Aliases come from the classics
  **knowledge graph** (below); lexicons extensible (`extra_*` / `lexicon_file`).

Multiple metrics per dataset are supported — e.g. TCMEval-SDT runs
`[llm_judge, syndrome_chain, bleu, rouge]` and MTCMB-FRD runs
`[llm_judge, prescription_match]`.

## Agents

The doctor agent runs the same `ModelProvider` policy inside an
`AgentEnvironment(reset/step)` loop. **AgentClinic** is wired and runs **offline**
(patient / measurement / moderator default to rule-based from the scenario; pass
`support:` to use LLM agents).

**MedAgentBench** is fully wired to a **real FHIR server**: the agent emits
`GET <url>` / `POST <url>\n<json>` / `FINISH([...])` (faithful to the upstream
harness — GET appends `&_format=json`, POST writes live), and scoring uses a
**built-in grader** (query tasks exact vs. gold `sol`; action tasks verify the
write landed for `eval_MRN`) or the **official gated `refsol.py`** if you set
`refsol_path` (called exactly as `getattr(refsol, task_id)(task, answer, base)`).
Start the EHR server first:

```bash
docker pull jyxsu6/medagentbench:latest
docker run -p 8080:8080 jyxsu6/medagentbench:latest      # serves :8080/fhir
curl http://localhost:8080/fhir/metadata                 # verify
python -m medeval run configs/example_medagentbench.yaml --limit 10
```

## Multimodal (舌象 / 脉象)

`Message` carries optional `images` (http/data URIs or local paths → auto data-URI);
`to_openai()` emits OpenAI/LiteLLM **content blocks**, so LiteLLM and Poe vision
models work unchanged. The `hf_mcq` adapter takes an `image` field (URL / local path
/ HF `Image` dict / **raw parquet bytes** / PIL), and `question_text` supplies a
constant prompt for image-classification sets without a question column.

**TCM-Ladder** (`timzzyus/TCM-Ladder`, CC-BY-4.0, open) is wired in
`configs/example_tcm_ladder.yaml`: the 12,778-item bilingual text MCQ set works
as-is, and the tongue/herb `visual.parquet` (raw JPEG bytes) drives the
bytes→data-URL→vision pipeline. Pair image 舌象 tasks with the `tongue_pulse` metric.

## Knowledge graph (古籍本体)

The classical-literature ontology is a real, downloadable **knowledge graph** —
35 classics + authors + dynasties + 25 经典名方, linked by `authored_by` /
`dynasty` / `part_of` / `from_source` / `category`. It is the single source of
truth for `classics_ontology`'s aliases. Prebuilt artifacts ship in `data/kg/`
(node-link JSON · RDF Turtle · GraphML); (re)build/query with:

```bash
python -m medeval kg --out data/kg --stats      # 116 nodes, 157 edges
```
```python
from medeval import get_kg
kg = get_kg()
kg.source_of_formula("银翘散")   # -> 温病条辨
kg.author_of("金匮要略")         # -> 张仲景
```

## Distributed scheduling

The grid is embarrassingly parallel, so MedEval distributes by **strided
sharding** — each worker runs `samples[i::N]` and writes its own shard-scoped
detail + cache files, so workers never collide and each is independently
resumable. No central server.

```bash
# one box, N workers (optionally one per GPU for data-parallel HF/vLLM):
python -m medeval pool configs/catalog_mcq.yaml --num-shards 4 --gpus 0,1,2,3

# many machines (shared filesystem): run a shard on each, then merge once:
python -m medeval run cfg.yaml --shard 0 --num-shards 8 --output /shared/run   # machine 0
python -m medeval run cfg.yaml --shard 1 --num-shards 8 --output /shared/run   # machine 1
# ...
python -m medeval merge /shared/run        # -> leaderboard.json + leaderboard.md
```

```bash
# Ray (existing cluster via RAY_ADDRESS, or local), N tasks, GPUs per task:
python -m medeval pool cfg.yaml --num-shards 8 --backend ray --ray-num-gpus 1

# Slurm: generate + submit a job array (one task per shard) + a dependent merge:
python -m medeval slurm cfg.yaml --num-shards 64 --partition gpu --gpus-per-task 1 \
    --max-parallel 16 --setup "conda activate medeval" --output /shared/run
```

`merge` **re-aggregates the per-sample scores** from every shard (not a
mean-of-means), so the result is identical to a single full run even when shards
have unequal sizes. Backends: `local` (subprocess pool), `ray` (cluster tasks),
`slurm` (job array + `afterok` merge). Programmatic: `medeval.run_pool` /
`medeval.run_ray` / `medeval.submit_slurm` / `medeval.merge_results`.

## Leaderboard submission (MedBench / OpenCompass)

Export a run's predictions into an upload-ready format:

```bash
# OpenCompass predictions:  predictions/<model>/<dataset>.json
#   = {"0": {origin_prompt, prediction, gold?}, ...}   (matches GenInferencer output)
python -m medeval export results/mcq --format opencompass --out oc_out

# MedBench: download the platform data, run the `medbench` adapter over the
# *_test.jsonl files, then fill answers back into the original records by other.id:
python -m medeval run configs/example_medbench_submit.yaml
python -m medeval export results/medbench --format medbench \
       --out submission --medbench-test-dir /path/to/MedBench
# -> submission/<set>/<set>.jsonl  with `answer` filled (letter for MCQ, text for NLG);
#    question/options/other preserved verbatim. Upload via the MedBench web UI.
```

MedBench test answers are **held out** — you generate predictions locally and the
platform scores them. The `medbench` adapter loads `{question, passage, options,
answer, other:{source,id}}` records; MCQ answers become a letter, generation tasks
the free text, joined by `other.id`.

## Extending

```python
from medeval import register_provider, register_dataset, register_metric
@register_provider("myllm")  ...   # implement agenerate()
@register_dataset("mybench")  ...  # implement load() / parse()
@register_metric("my_metric") ...  # implement score() / aggregate()
```
Import the module (add it to `runner.py`'s import list) and reference it by name
in YAML.

---

## Status

- Full offline pipeline (generate → parse → judge / pass^k → leaderboard) passes
  end-to-end (`tests/`).
- Every wired dataset was loaded from its **live** source and pushed through the
  runner; adapter edge-cases (multi-answer, inline options, CMB null slots,
  dict-of-lists, CSEDB nested explosion, signed-points rubric, AgentClinic env)
  have unit tests.
- Caveats, gating and field maps per dataset: [`DATASETS.md`](DATASETS.md).

Metrics include F1 / ROUGE / BLEU / 方剂结构匹配 / 证型链结构分 / 经络腧穴 / 舌象脉象 /
古籍本体; multimodal (舌象/herb images) via content blocks + TCM-Ladder; the classical
ontology is a downloadable knowledge graph; MedAgentBench runs against a live FHIR
server; results export to OpenCompass / MedBench; runs distribute across
**local / Ray / Slurm**. Further extension points: live/anti-contamination benchmarks
and richer multimodal scoring.
