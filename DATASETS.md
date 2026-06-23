# Dataset Access Reference

How each benchmark is obtained and wired, **verified against the live source on
2026-06-22** (HuggingFace datasets-server `/splits` + `/first-rows`, or fetched
raw repo files). Column names are taken from real records, not guessed. The
caveats below are the things that actually bite you.

Legend: ✅ open & config-only · ⚠️ open but needs care · 🔒 gated / manual step.

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

### CMB ⚠️
- **Repo:** `FreedomIntelligence/CMB`, config `CMB-Exam`. **Options:** dict `option` keyed
  A–F (unused slots are `null` → dropped by the adapter). **Answer:** letter(s); multi
  e.g. `"BCDE"` when `question_type == 多项选择题`.
- **Caveats:** (1) the **`test` split has no `answer`** → use **`val`**/`train`. (2) The HF
  **loading script is broken on `datasets>=5`** → load the raw JSON via `data_files`.
```yaml
{path: FreedomIntelligence/CMB,
 data_files: https://huggingface.co/datasets/FreedomIntelligence/CMB/resolve/main/CMB-Exam/CMB-val/CMB-val-merge.json,
 field_map: {question: question, options: option, answer: answer}, answer_format: multi}
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

### TCMBench ⚠️ (`tcmbench` adapter)
- **Source:** `ywjawmw/TCMBench`, data under **`data_demo/`** (the bare `data/` 404s; demo
  subset only — full 5,473-Q set isn't public). Each file is a JSON object
  `{type, prefix_prompt, keywords, example}`; questions under `example`.
- **Options are embedded in the question text** (`A．…`); **answer** is a letter list.
  B1/A3 files (KHC/CVR) use a nested `share_content` + `question[].sub_question` shape —
  the adapter handles both.
```yaml
{adapter: tcmbench, source_url: https://raw.githubusercontent.com/ywjawmw/TCMBench/main/data_demo/first_level/FKU.json}
```

### TCM-Ladder ✅ (multimodal)
- **Repo:** `timzzyus/TCM-Ladder` (HF, **open, CC-BY-4.0**, ungated; arXiv 2505.24063).
  The HF auto-config is broken — load each parquet via `data_files`.
- **`multiChoice.parquet`** (12,778): bilingual text MCQ. Options are **5 separate
  columns `A`–`E`** (`E` empty in ~2k 4-option rows → the adapter drops the blank
  trailing slot); `answer` = letter(s), multi-correct concatenated ("ABCDE") → `multi`.
  Also `category` (subject), `type` (Single/Multiple), `lang`.
- **`visual.parquet`** (8,802): tongue + herb images as **raw JPEG bytes** in a parquet
  `binary` column (handled by `_encode_image`), `answer` letter, `category` ∈ {herb, tongue}.
  **No question/option text** — the 4-image MCQ is assembled at eval time and its
  distractors aren't in the file; wired as image→modality classification via
  `question_text` + `inject_options: [herb, tongue]` (`config/example_tcm_ladder.yaml`).
  ~730 MB — use `--limit`. No pulse/audio is released on HF.
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

### HealthBench ✅
- **Source:** OpenAI simple-evals JSONL (Azure blob). `prompt` = list of `{role,content}`
  chat turns. `rubrics` = list of `{criterion, points (may be NEGATIVE), tags}`.
  Variants: `2025-05-07-…_oss_eval.jsonl` (5000) · `hard_…` (1000) · `consensus_…` (3671).
- **Caveat:** negative points — score = met-points / Σ(positive points), clipped (the
  `llm_judge` metric implements this).
```yaml
{task: open_qa, source_url: https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl,
 field_map: {prompt: prompt, rubric: rubrics, reference: ideal_completions_data}}
```

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

### MedSafetyBench ✅
- **Source:** `AI4LIFE-GROUP/med-safety-bench`, CSV `datasets/test/gpt4/med_safety_demonstrations_category_{1..9}.csv`
  (cols `'', harmful_medical_request, safe_response`). No rubric field → default **安全**
  rubric judges refusal/harm vs. `safe_response`. Research-only terms.
```yaml
{task: safety, source_url: https://raw.githubusercontent.com/AI4LIFE-GROUP/med-safety-bench/main/datasets/test/gpt4/med_safety_demonstrations_category_1.csv,
 field_map: {prompt: harmful_medical_request, reference: safe_response}}
```

---

## 3. Agent benchmarks (`agentclinic`, `medagentbench`)

### AgentClinic ✅ (offline-capable)
- **Source:** `SamuelSchmidgall/AgentClinic` (MIT). Scenario JSONL ships in-repo:
  `agentclinic_medqa.jsonl` (215), `agentclinic_nejm.jsonl` (120).
- **MedQA schema:** top-level `OSCE_Examination.{Objective_for_Doctor, Patient_Actor,
  Physical_Examination_Findings, Test_Results, Correct_Diagnosis}`. **Doctor actions:**
  free text + `REQUEST TEST: …` / `REQUEST IMAGES` / `DIAGNOSIS READY: …`; ~20-turn budget;
  moderator LLM judges the diagnosis. pass^k is *not* in the original repo — we add it.
- **Offline:** patient/measurement/moderator default to rule-based (read from the scenario);
  pass `support: {patient: <id>, moderator: <id>}` to use LLM agents.
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
  - **built-in grader** (default, no gated file): query tasks (have `sol`) compare the
    `FINISH` answer to gold (string-set + numeric tolerance); action tasks verify a 2xx
    write referencing `eval_MRN` landed (a documented approximation of the per-task rules).
  - **official `refsol.py`** (gated, project Box link): set `refsol_path:` and it is called
    as `getattr(refsol, task_id)(task, answer, fhir_base)`.
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
