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

### MedAgentBench ⚠️ (server + grader required)
- **Source:** `stanfordmlgroup/MedAgentBench` (on the AgentBench harness).
  `data/medagentbench/test_data_v2.json` (`{id, instruction, context, sol, eval_MRN}`),
  `funcs_v1.json` = FHIR tool catalog. **Actions:** `GET <url>` / `POST <url>\n<json>` /
  `FINISH([...])`; 8 rounds (paper) / 5 (repo config). **Metric: pass@1** (the paper
  explicitly avoids pass^k for healthcare).
- **Not self-contained:** needs (1) a running **HAPI-FHIR Docker** server on `:8080` and
  (2) the **gated `refsol.py`** grader (Box link). The `medagentbench` adapter wraps the
  GET/POST/FINISH loop; supply `grader=` to score.

---

## Cross-cutting gotchas

- **Use the labeled split:** MedMCQA → `validation`; CMB → `val`; TCMEval-SDT → `Train`.
- **Multi-answer is normal** for Chinese licensing exams (CMB, CMExam): the model parser
  accepts `"BCDE"`, `"B、C、D、E"`, `"B, C, D and E"` without grabbing capitals from prose.
- **`datasets>=5` dropped loading scripts / `trust_remote_code`:** for script-based repos
  (CMB) load the raw files via `data_files`.
- **Judge model:** DeepSeek-R1 via LiteLLM is the cost/throughput default (CSEDB-validated).
