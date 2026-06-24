# Dataset Access Reference

How each benchmark is obtained and wired, **verified against the live source on
2026-06-22** (HuggingFace datasets-server `/splits` + `/first-rows`, or fetched
raw repo files). Column names are taken from real records, not guessed. The
caveats below are the things that actually bite you.

Legend: ✅ open & config-only · ⚠️ open but needs care · 🔒 gated / manual step.

### Reproducibility — pinned revisions + `preflight`

The headline MCQ sets are **pinned to an immutable commit** so the eval set can't
change underneath you. HF-repo loads use `revision: <sha>` (→ `load_dataset`);
raw-file sources embed the commit in the URL (`…/resolve/<sha>/…` for HF,
`raw.githubusercontent/…/<sha>/…` for GitHub). Current pins (`catalog_mcq.yaml`,
`example_tcm_ladder.yaml`):

| Dataset | Pin (commit) |
|---|---|
| MedQA `GBaker/MedQA-USMLE-4-options` | `0fb93dd2…` |
| MedMCQA `openlifescienceai/medmcqa` | `91c6572c…` |
| MMLU `cais/mmlu` | `c30699e8…` |
| PubMedQA `qiaojin/PubMedQA` | `9001f285…` |
| CMB `FreedomIntelligence/CMB` (HF question / GitHub answer) | `935fbc09…` / `6c8ece46…` |
| CMExam `williamliujl/CMExam` (GitHub) | `fadb22c8…` |
| TCM-Ladder text `timzzyus/TCM-Ladder` (HF) | `4e875657…` |

Before spending tokens, profile the data with **`preflight`** (sample count ·
option-count distribution · answer-parse success rate · examples — *no model*):

```bash
python -m medeval preflight configs/catalog_mcq.yaml            # full report
python -m medeval preflight configs/catalog_mcq.yaml --strict   # CI gate (<100% → exit 1)
```

A parse rate < 100% means rows are dropped (mis-mapped `field_map`, an unexpected
answer encoding, options that don't parse); `preflight` lists drops by reason.
Verified profile (full load): MedQA 1273 · MedMCQA 4183 · PubMedQA 1000 · MMLU 1089
· CMB 11200 · CMExam 6810/6811 · TCM-Ladder text 12775/12778 — all **100%** parse
(CMExam/TCM-Ladder each have a couple of genuinely malformed source rows, reported).

> ⚠️ **YAML gotcha** (caught by `preflight`): bare `yes`/`no`/`on`/`off` are YAML
> booleans. `inject_options: [yes, no, maybe]` silently becomes `[True, False,
> "maybe"]` — **quote them**: `["yes", "no", "maybe"]`.

### Comparability tiers — `split_type`

Every dataset carries a `split_type` so *officially-comparable* runs are never
mixed with internal ones; the leaderboard splits **✅ Official** from **⚠️ Internal /
non-comparable**. Set it in the dataset config (default `official`):

| `split_type` | meaning | examples here |
|---|---|---|
| `official` | full official split + official metric/grader | CMB-test, MedQA, MMLU, MedSafetyBench (1–9), TCM-Ladder (text + visual) |
| `validation` | a dev/val split, not the held-out test | CMB-val, TCMEval-SDT (Train) |
| `demo` | a tiny demo subset shipped in lieu of the full corpus | TCMBench (14 demo items) |
| `sample` | a small public sample of an otherwise-gated set | CSEDB (2-record sample) |
| `gated` | full set needs manual access; partial here | — |
| `approximated` | a built-in/approximate grader, not the official one | MedAgentBench (built-in grader), AgentClinic (scripted offline) |

`MedAgentBench` and `AgentClinic` set this **dynamically**: official when you supply
the gated `refsol_path` / LLM `support:` agents, else `approximated`.

---

## 1. Multiple-choice (`hf_mcq`, `tcmbench`)

### MedQA (USMLE) ✅
- **Repo:** `GBaker/MedQA-USMLE-4-options` (Parquet, viewer, no remote code).
- **Split:** `test` (1,273). **Options:** dict `{"A":…,"D":…}`. **Answer:** `answer_idx`
  = letter (a key into the dict); `answer` is the full option *text*.
- Alt `bigbio/med_qa` needs `trust_remote_code` (removed in `datasets>=5`) — avoid.
```yaml
{path: GBaker/MedQA-USMLE-4-options, split: test,
 field_map: {question: question, options: options, answer: answer_idx}, answer_format: letter}
```

### MedMCQA ⚠️
- **Repo:** `openlifescienceai/medmcqa`. **Options:** 4 separate columns `opa..opd`.
  **Answer:** `cop` (int 0–3).
- **Caveat:** the **`test` split is unlabeled** (`cop = -1`). **Use `validation`** (4,183).
```yaml
{path: openlifescienceai/medmcqa, split: validation,
 field_map: {question: question, options: [opa,opb,opc,opd], answer: cop}, answer_format: index}
```

### PubMedQA ⚠️
- **Repo:** `qiaojin/PubMedQA`, config `pqa_labeled` (1,000), split **`train`** (the only
  split per config). 3-way `yes/no/maybe` in `final_decision` — **no options column**,
  so inject them. `context.contexts` holds the passages.
```yaml
{path: qiaojin/PubMedQA, name: pqa_labeled, split: train,
 field_map: {question: question, options: null, answer: final_decision, context: context},
 answer_format: text, inject_options: [yes, no, maybe]}
```

### MMLU-medical ✅
- **Repo:** `cais/mmlu` (MIT). One **config per subject** — pass a list, they concat.
  **Options:** list `choices`. **Answer:** int 0–3.
- Medical subjects: `anatomy, clinical_knowledge, college_medicine, college_biology,
  medical_genetics, professional_medicine` (test sizes 135/265/173/144/100/272).
```yaml
{path: cais/mmlu, name: [anatomy, clinical_knowledge, college_medicine, college_biology, medical_genetics, professional_medicine],
 split: test, field_map: {question: question, options: choices, answer: answer}, answer_format: index}
```

### CMB ✅ (full test 11,200, gold joined)
- **Repo:** `FreedomIntelligence/CMB`, config `CMB-Exam`. **Options:** dict `option` keyed
  A–F (unused slots are `null` → dropped by the adapter). **Answer:** letter(s); multi
  e.g. `"BCDE"` when `question_type == 多项选择题`.
- **Full test set (11,200):** the HF `CMB-test` questions ship **without** answers, but the
  gold key is **public on GitHub** (`data/CMB-test-choice-answer.json`, keyed by `id`). The
  `answer_join` config downloads it and injects `answer` by `id` → the complete, scorable
  CMB-test (not the 280-item `val`). The HF loading script is broken on `datasets>=5`, so the
  questions load via raw `data_files`. (`val` 280 / `train` 269,359 remain as commented
  alternates; `CMB-Clin` 74 free-text cases → `local_json` + `llm_judge`.)
```yaml
{path: FreedomIntelligence/CMB,
 data_files: https://huggingface.co/datasets/FreedomIntelligence/CMB/resolve/main/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json,
 field_map: {question: question, options: option, answer: answer}, answer_format: multi,
 answer_join: {data_files: https://raw.githubusercontent.com/FreedomIntelligence/CMB/main/data/CMB-test-choice-answer.json,
               key: id, value: answer}}
```

### CMExam ⚠️
- **Source:** `williamliujl/CMExam` (Apache-2.0), CSV `data/test_with_annotations.csv`
  (no `test.csv`). **Options:** ONE `Options` column, choices packed as `"A text\nB text…"`
  → `options_inline: true`. **Answer:** concatenated letters; multi (e.g. `"ABCE"`).
- HF mirror `fzkuji/CMExam` stores `Options` as `[{key,value}]` (also handled) but omits
  the annotation columns.
```yaml
{format: csv, data_files: https://raw.githubusercontent.com/williamliujl/CMExam/main/data/test_with_annotations.csv,
 options_inline: true, field_map: {question: Question, options: Options, answer: Answer}, answer_format: multi}
```

### TCMBench ⚠️ (`tcmbench` adapter — full public demo set)
- **Source:** `ywjawmw/TCMBench`, data under **`data_demo/`** (the bare `data/` 404s). The
  repo publishes a per-type **demo** set only — **one worked example per question type** —
  so the full public corpus is 14 files (3 first-level + 6 second-level + 5 third-level)
  → **20 MCQ items**. The full 5,473-Q bank is **gated** (request from the authors); drop its
  path(s) into the same `source_url` list to score it. Each file is a JSON object
  `{type, prefix_prompt, keywords, example}`; questions under `example`.
- **Options are embedded in the question text** (`A．…`); **answer** is a letter list. The
  B1/A3 files (CVR/KHC/SDT/SDT_reverse/SDT_shuffle) use a nested `share_content` +
  `question[].sub_question` shape — the adapter renders **vignette + sub-question + options
  once** (no double-render; the shared 病例 is preserved). `source_url` accepts a **list** of
  files → concatenated. Non-MCQ records (e.g. `herb_predict`) are skipped.
```yaml
{adapter: tcmbench, source_url: [
   https://raw.githubusercontent.com/ywjawmw/TCMBench/main/data_demo/first_level/CVR.json,
   https://raw.githubusercontent.com/ywjawmw/TCMBench/main/data_demo/second_level/SDT.json,
   ...all 14 demo files (see configs/example_tcm.yaml)... ]}
```

### TCM-Ladder ✅ (multimodal)
- **Repo:** `timzzyus/TCM-Ladder` (HF, **open, CC-BY-4.0**, ungated; arXiv 2505.24063).
  The HF auto-config is broken — load each parquet via `data_files`.
- **`multiChoice.parquet`** (12,778): bilingual text MCQ. Options are **5 separate
  columns `A`–`E`** (`E` empty in ~2k 4-option rows → the adapter drops the blank
  trailing slot); `answer` = letter(s), multi-correct concatenated ("ABCDE") → `multi`.
  Also `category` (subject), `type` (Single/Multiple), `lang`.
- **`visual.parquet`** (8,802): each row is a **pre-rendered 4-option MCQ image** as raw
  bytes (handled by `encode_images`), with the question text, the candidate options and the
  **A/B/C/D labels all baked into the image** (herb = a 2×2 photo grid "which is X?"; tongue =
  1 photo + 4 text options). Columns: `id, image, answer, category ∈ {herb,tongue}, lang`.
  Gold `answer` is the **letter A–D** — so the FAITHFUL task is image→letter: the config sets
  a `question_text` instruction, `prompt_template: "{question}"` (no text block — options are
  in the image), `inject_options: [A,B,C,D]` and `answer_format: letter`
  (`configs/example_tcm_ladder.yaml`). A vision model is required. ~730 MB — use `--limit` or
  stream. No pulse/audio is released on HF.
```yaml
{adapter: hf_mcq, format: parquet,
 data_files: https://huggingface.co/datasets/timzzyus/TCM-Ladder/resolve/main/multiChoice.parquet,
 field_map: {question: question, options: [A,B,C,D,E], answer: answer}, answer_format: multi}
```

### MLEC-QA 🔒
- `Judenpech/MLEC-QA` (EMNLP 2021). **Data is NOT in the repo** — a Google-Drive ZIP
  (`mlec-qa.zip`, file id `1-v4c8bcTspgBINKticTF7A9OMum0CB_y`) behind **Google sign-in**,
  no HF mirror. 5 subsets `{Clinic, Stomatology, PublicHealth, TCM, CWM}` × train/dev/test.
  Fields (README): `qtext` (stem), `options` (dict A–E), `answer` (list). After manual
  download point a local `hf_mcq` `data_files` at `./<Subset>_test.json`.

### TCM-3CEval 🔒
- arXiv 2503.07041. **No GitHub / HF release.** Only via the MedBench TCM track
  (`medbench.opencompass.org.cn`) with **held-out answer keys** → not downloadable.

---

## 2. Open-ended / rubric / safety (`local_json`)

The judge consumes the raw answer; the adapter maps `prompt` (string *or* chat
messages), an optional `rubric` (normalized from many shapes), `reference`, and `label`.

### HealthBench ✅ (faithful per-criterion grading)
- **Source:** OpenAI simple-evals JSONL (Azure blob). `prompt` = list of `{role,content}`
  chat turns. `rubrics` = list of `{criterion, points (may be NEGATIVE), tags}`.
  All three variants are wired: `2025-05-07-…_oss_eval.jsonl` (5000) · `hard_…` (1000) ·
  `consensus_…` (3671) — same schema, just swap `source_url`.
- **Faithful grading:** set the metric to `{name: llm_judge, per_criterion: true}` and the
  `llm_judge` metric runs the **official algorithm** — one judge call per rubric item using
  the verbatim simple-evals grader template, a boolean `criteria_met`, then
  `score = Σ(signed met points) / Σ(positive points)`, clipped at the example level. (Without
  `per_criterion` it falls back to a single weighted-rubric call.) Numbers still track the
  judge, so **use GPT-4.1** (HealthBench's official grader) for comparable results.
```yaml
{task: open_qa, source_url: https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl,
 field_map: {prompt: prompt, rubric: rubrics, reference: ideal_completions_data},
 metrics: [{name: llm_judge, per_criterion: true}]}   # judge: gpt-4.1
```

### HealthBench meta-eval ✅ (judge calibration — physician labels)
- **Source:** `…/healthbench/2025-05-07-06-14-12_oss_meta_eval.jsonl` (29,511 rows). Each row is
  one *(prompt, completion, rubric)* judged by **2+ physicians** (`binary_labels` +
  `anonymized_physician_ids`). This is the only one of the open-ended sets that ships **human
  gold**, so it anchors judge calibration. `medeval calibrate --rebuild-from <url>` draws a
  frozen, deterministic, class-balanced, cluster-stratified **120-item** sample into
  `data/calibration/` (a *blind* reviewer file + a held-out physician-label file).
- **Agreement metric:** OpenAI simple-evals' **balanced pairwise F1** (met/unmet), replicated
  verbatim, plus **Cohen's κ** + raw agreement with bootstrap 95% CIs, against the
  **physician-vs-physician ceiling**. A judge is *headline-eligible* only when it matches the
  human ceiling (ΔF1, Δκ ≤ 0.05) **and** reaches κ ≥ 0.40; otherwise its open-ended scores are
  **auxiliary**. See `data/calibration/calibration_report.md` and the README's *Judge
  calibration* section. Frozen result: a strong-model judge is physician-equivalent
  (F1 0.697 vs 0.719 ceiling) but only moderate absolutely (κ 0.394) → **auxiliary**.

### LLMEval-Med ✅
- **Source:** `llmeval/LLMEval-Med` `dataset/dataset.json`. **Caveat:** it's a **dict keyed
  by category**, not a flat array (the adapter flattens). `problem` = prompt, `checklist`
  = plain-text rubric, `sanswer` = reference. Public file = 667 (of 2,996); eval-only license.
```yaml
{task: open_qa, source_url: https://raw.githubusercontent.com/llmeval/LLMEval-Med/main/dataset/dataset.json,
 field_map: {prompt: problem, rubric: checklist, reference: sanswer}}
```

### TCMEval-SDT ✅ (辨证)
- **Source:** `zhuyan166/TCMEval`, `evaluation/TCMEval-SDT/data/` (note the `/data/` subdir).
  **Caveat:** **only `Train_TCM_Data_v1.json` (200) has populated gold**; Test/Val are blank.
  `Clinical Data` = input; free-text refs `Syndrome Differentiation` / `TCM Syndrome` /
  `TCM Pathogenesis`. Uses the default **证型链** rubric.
```yaml
{task: sdt, source_url: https://raw.githubusercontent.com/zhuyan166/TCMEval/main/evaluation/TCMEval-SDT/data/Train_TCM_Data_v1.json,
 field_map: {prompt: "Clinical Data", reference: "Syndrome Differentiation", label: "TCM Syndrome"}}
```

### MTCMB ✅ (方剂 / 安全)
- **Source:** `Wayyuanyuan/MTCMB`, 12 JSONL under `data/` (CC-BY). `question` = input,
  `answer` = gold (a **dict** `{治法,方剂,药物组成}` for TCM-FRD → serialized for the judge).
  Files: `10.TCM_FRD.jsonl` (方剂), `11.TCM_SE_A.jsonl` (安全 fill-in), `9.TCM_PR.jsonl`,
  `12.TCM_SE_B.jsonl` (MCQ).
```yaml
{task: prescription, source_url: https://raw.githubusercontent.com/Wayyuanyuan/MTCMB/main/data/10.TCM_FRD.jsonl,
 field_map: {prompt: question, reference: answer}}
```

### CSEDB 🔒 (安全 + 有效)
- **Source:** `Medlinker-MG/CSEDB`. **Full 2,069-item bank is gated** (email
  `it@medlinker.com`); only a **2-record sample** is open at `generate/test_sampled.json`.
  Chinese keys: input is `输入 case` (**literal space in the key**); rubric `规则判断列表`
  = `[{规则内容, 分数 1–5}]`; `考点.门类` = `安全门`/`有效门`. **Explode**
  `设计的考题内容.最具代表性的测试case` → one sample per case.
```yaml
{task: safety, source_url: https://raw.githubusercontent.com/Medlinker-MG/CSEDB/main/generate/test_sampled.json,
 explode: "设计的考题内容.最具代表性的测试case",
 field_map: {prompt: "输入 case", rubric: "规则判断列表", label: "考点.门类"}}
```

### MedSafetyBench ✅ (full 9 categories × 2 sources)
- **Source:** `AI4LIFE-GROUP/med-safety-bench`, CSV
  `datasets/test/{gpt4,llama2}/med_safety_demonstrations_category_{1..9}.csv`
  (cols `'', harmful_medical_request, safe_response`). The **9 categories = the AMA
  Principles of Medical Ethics**; the full test set is **all 18 files** (9 categories ×
  {gpt4, llama2} attacker) — `source_url` takes the **list** and concatenates them (≈900
  items). No rubric field → default **安全** rubric judges refusal/harm vs. `safe_response`.
  Research-only terms.
```yaml
{task: safety, source_url: [   # all 18 = full test set (see configs/example_open_safety.yaml)
   .../test/gpt4/med_safety_demonstrations_category_1.csv, ... category_9.csv,
   .../test/llama2/med_safety_demonstrations_category_1.csv, ... category_9.csv],
 field_map: {prompt: harmful_medical_request, reference: safe_response}}
```

---

## 3. Agent benchmarks (`agentclinic`, `medagentbench`)

### AgentClinic ✅ (MedQA + NEJM, LLM-agent faithful)
- **Source:** `SamuelSchmidgall/AgentClinic` (MIT). Full released scenario JSONL:
  `agentclinic_medqa_extended.jsonl` (**214** OSCE cases; the 107-case original is
  `agentclinic_medqa.jsonl`) and `agentclinic_nejm_extended.jsonl` (**120** image cases; the
  15-case original is `agentclinic_nejm.jsonl`). Both variants are wired (`variant: medqa|nejm`).
- **MedQA schema:** `OSCE_Examination.{Objective_for_Doctor, Patient_Actor,
  Physical_Examination_Findings, Test_Results, Correct_Diagnosis}`. **NEJM schema** (flat):
  `{question, patient_info, physical_exams, answers:[{text,correct}], image_url}` — needs a
  **vision** doctor model. **Doctor actions:** free text + `REQUEST TEST: …` / `REQUEST IMAGES`
  / `DIAGNOSIS READY: …`; ~20-turn budget. pass^k is *not* in the original repo — we add it.
- **Faithful setup (the paper):** patient + measurement + moderator are each their own LLM —
  pass `support: {patient: <id>, measurement: <id>, moderator: <id>}` and the runner resolves
  them to providers. **Offline fallback:** omit `support:` and those roles run rule-based from
  the scenario (a documented approximation; the doctor still needs a model).
- 🔒 **AgentClinic-MIMIC-IV** needs PhysioNet credentialing (not redistributable).

### MedAgentBench ⚠️ (live FHIR server required) — wired
- **Source:** `stanfordmlgroup/MedAgentBench` (300 tasks, 10 categories).
  `data/medagentbench/test_data_v2.json` (`{id, instruction, context, sol, eval_MRN}`),
  `funcs_v1.json` = FHIR tool catalog (injected into the system prompt).
- **Actions (faithful to the harness):** `GET <url>` (we append `&_format=json`), `POST
  <url>\n<json>` (we parse the JSON after line 1 and **write it to the live server**),
  `FINISH([...])` (extract `r[7:-1]`). 8 rounds (paper) / 5 (repo). **Metric: pass@1**
  (the paper explicitly avoids pass^k for healthcare).
- **Grading:** `eval(case_data, results, fhir_api_base)` →
  `getattr(refsol, case_data['id'].split('_')[0])`. We implement this exactly:
  - **official `refsol.py`** (gated, project Box link — deliberately withheld to stop
    benchmark gaming): set `refsol_path:` and it is called as
    `getattr(refsol, task_id)(task, answer, fhir_base)`. **This is the faithful path** — use
    it for real numbers.
  - **built-in grader** (default, no gated file): dispatches **per task id** and validates the
    POST **payload**, not just MRN presence. Reconstructed from the public task instructions
    + the published methodology:
    - *query* tasks (task1/2/4/6/7): must issue **no POST**; `task1` ships gold → exact /
      numeric match. Tasks 2/4/6/7 ship no gold → the **value is unverifiable offline** and is
      reported as such (never a false pass) — supply `refsol_path` to score them.
    - *action* tasks (task3/5/8/9/10): require a 2xx POST of the **expected `resourceType`**
      (Observation / MedicationRequest / ServiceRequest) with `subject → Patient/{eval_MRN}`
      and the **right code** (task3 flowsheet `BP` + value; task8 SNOMED `306181000000106` +
      SBAR note; task5/9 NDC; task9/10 LOINC). Codes are pulled from each task's text so all
      30 variants/task are covered. The **conditional** tasks (5/9/10): a placed order is
      validated by shape; a *no-op* (ordering nothing) needs live lab values to judge and is
      flagged `undecidable_noop` rather than passed.
  - The grader is **conservative — it never reports a false success**; everything that needs a
    live-FHIR gold is flagged for `refsol_path`.
- **Run it:** start the FHIR server, then point `fhir_base` at it:
  ```bash
  docker pull jyxsu6/medagentbench:latest
  docker run -p 8080:8080 jyxsu6/medagentbench:latest   # :8080/fhir, 100 patients preloaded
  ```
  ```yaml
  {adapter: medagentbench, fhir_base: http://localhost:8080/fhir, max_turns: 8,
   metrics: [pass_k], k: 1}   # optional: refsol_path: /path/to/refsol.py
  ```
- The GET/POST/FINISH loop + both grader paths are validated offline against an in-process
  mock FHIR server (`tests/test_medagentbench.py`).

---

## Batch 2 — additional datasets (verified 2026-06)

Configs live in `configs/catalog_en_med.yaml`, `catalog_multimodal.yaml`,
`catalog_cn_tcm.yaml`. ✅ = config-only · ⚠️ = one-time prep · 🔒 = gated/unreleased.

### English QA / reasoning / calc / hallucination
| Dataset | Access | Adapter / notes |
|---|---|---|
| **MedXpertQA** ✅ | `TsinghuaC3I/MedXpertQA` `Text`(2450)/`MM`(2000) | `hf_mcq`, dict options A–J, gold `label` letter. MM: `image: images` (filenames) + unzip `images.zip`, set `image_base`. |
| **MedAgentsBench** ✅ | `xk-huang/medagents-benchmark` | `hf_mcq`, 10 configs, `split: test_hard`, dict options, gold `answer_idx`. (≠ MedAgentBench.) |
| **MedCalc-Bench** ✅ | `ncbi/MedCalc-Bench-v1.2` CSV (v1.0/1.1 gated) | `local_json`+`numeric_match`; `prompt: [Patient Note, Question]`, map `lower_limit`/`upper_limit` (range check, ±5% decimal / exact else). |
| **MedR-Bench** ✅ | GitHub raw JSON (HF gated) | `local_json`+`llm_judge`; JSON keyed by PMCID → auto-flattened; gold `generate_case.diagnosis_results`. |
| **MedHallu** ✅ | `UTAustin-AIHealth/MedHallu` `pqa_labeled`/`pqa_artificial` | `hf_mcq`, 2-col options `[Ground Truth, Hallucinated Answer]`, gold = `Hallucinated Answer`, `answer_format: text`, `shuffle_options: true` (de-bias). |
| **Med-HALT** ⚠️ | `openlifescienceai/Med-HALT` | `reasoning_nota` → `hf_mcq` (`options` is a stringified dict, auto-parsed; `correct_index`). FCT (yes/no), fake (judge), IR_* (gen) need prep/judge. |
| **MLEC-QA** ✅ | `shuyuej/MLEC-QA-Benchmark` (new HF mirror, test) | `hf_mcq`, dict options A–E, gold `answer` letter. (Full 5-subset set still Google-Drive.) |
| **MediQ** ✅ | `stellalisy/MediQ` `all_dev_good.jsonl` | `hf_mcq` single-turn upper-bound, **or** `adapter: mediq` (interactive proactive-questioning: facts revealed on relevant questions; `pass_k` → accuracy + avg_turns + timeout_rate). |

### Multimodal VQA
| Dataset | Access | Adapter / notes |
|---|---|---|
| **MedFrameQA** ✅ | `SuhaoYu1020/MedFrameQA` parquet | `hf_mcq`, multi-image `image: [image_1..image_5]` (embedded HF Image), `correct_answer` letter. |
| **SLAKE-en** ✅ | `mdwiratathya/SLAKE-vqa-english` parquet | `local_json` open VQA, embedded `image`, `llm_judge`. (Bilingual `BoKelvin/SLAKE` needs `imgs.zip`.) |
| **TCM-Vision-Benchmark** ⚠️ | `FreedomIntelligence/TCM-Vision-Benchmark` JSON | `hf_mcq`, 7204 image MCQ (Tongue 768); `image` = path in `tcm_bench_images.zip` (775MB) → unzip, set `image_base`. |
| **OmniMedVQA** ✅ | `foreverbeliever/OmniMedVQA` (`OmniMedVQA.zip` 10.7GB) | `hf_mcq`, `image_zip` auto-fetch (zip holds images **and** QA JSONs); options `option_A..D`, `gt_answer` text. |
| **PMC-VQA** ✅ | `RadGenome/PMC-VQA` (`images.zip` 18.9GB) | `hf_mcq`, `image_zip` auto-fetch; `Choice A..D`, `Answer_label` letter, `Figure_path`. |
| **MedBookVQA** ✅ | `slyipae1/MedBookVQA` (`figures.zip` 839MB, CC-BY-NC) | `hf_mcq`, `image_zip` auto-fetch; options `[Answer, Distractors]` (flattened) + `shuffle_options`, `image_strip: "../"`. |
| GMAI-MMBench 🔒 | `OpenGVLab/GMAI-MMBench` (gated) | VLMEvalKit TSV; `image` is a base64 string (`encode_images` auto-detects). Use `*_VAL.tsv`. |

### Chinese-med / TCM extensions
| Dataset | Access | Adapter / notes |
|---|---|---|
| **PromptCBLUE** ✅ | `tchenglv/PromptCBLUE` `dev.json` (JSONL!) | `local_json`+`f1`/`rouge`; `{input, target, task_dataset}`. (orig `michael-wzhu` id is dead; CBLUE raw is submission-only.) |
| **TCM-BEST4SDT** ✅ | `DYJG-research/TCM-BEST4SDT` raw JSON | MCQ (3 files, `option` dict, multi) → `hf_mcq`; SDT (300, open) → `local_json`+`llm_judge`/`syndrome_chain`. |
| **TCMEval-PA** ⚠️ | Figshare `.xlsx` (`ndownloader.figshare.com/files/56581880`) | 处方审核 MCQ; convert xlsx→json, `options_inline: true`, `answer_format: multi`. (≠ TCMEval-SDT.) |
| **TCM-Text-Exams** ⚠️ | `FreedomIntelligence/TCM-Text-Exams` JSON | `hf_mcq` after flattening the dict-of-5-sections to an array (`load_dataset("json")` errors on the dict). |

Prep one-liners: **TCM-Text-Exams** `json.dump([{**r,"section":s} for s,recs in d.items() for r in recs], ...)`; **TCMEval-PA** `openpyxl` → rows → json; both then load via `hf_mcq` `data_files`.

## Cross-cutting gotchas

- **Use the labeled split:** MedMCQA → `validation`; CMB → `val`; TCMEval-SDT → `Train`.
- **Multi-answer is normal** for Chinese licensing exams (CMB, CMExam): the model parser
  accepts `"BCDE"`, `"B、C、D、E"`, `"B, C, D and E"` without grabbing capitals from prose.
- **`datasets>=5` dropped loading scripts / `trust_remote_code`:** for script-based repos
  (CMB) load the raw files via `data_files`.
- **Judge model:** DeepSeek-R1 via LiteLLM is the cost/throughput default (CSEDB-validated).
