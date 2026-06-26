# 🤖 Medical & TCM model catalog

Ready-to-run HuggingFace backends wired into Med-Bench-Arena via
[`configs/catalog_med_models.yaml`](configs/catalog_med_models.yaml). **Every repo id,
base architecture, and parameter below was verified against the live HuggingFace model
page (config.json / tokenizer_config) in 2026-06** — real ids, real `dtype`, real
context length, and the actual `trust_remote_code` requirement.

Run one model at a time (vLLM keeps a model resident in GPU memory):

```bash
python -m medeval run configs/catalog_med_models.yaml --models biancang --limit 20
python -m medeval run configs/catalog_med_models.yaml --models huatuogpt-o1-7b
```

…or one-click on a GPU with [`notebooks/Med_Bench_Arena_Colab.ipynb`](notebooks/Med_Bench_Arena_Colab.ipynb).

---

## Summary table

| id (`--models`) | HF repo | Base arch | Params | Gated | `trust_remote_code` | dtype | ctx | Multimodal |
|---|---|---|---|:--:|:--:|:--:|--:|:--:|
| `zhongjing-2-1_8b` | `CMLM/ZhongJing-2-1_8b` (LoRA) | Qwen1.5-1.8B-Chat | 1.8B | – | – | bf16 | 32768 | – |
| `meditron-70b` | `epfl-llm/meditron-70b` | LLaMA-2-70B | 70B | ✅ | – | bf16 | 4096 | – |
| `biomistral-7b` | `BioMistral/BioMistral-7B` | Mistral-7B | 7B | – | – | fp16 | 2048 | – |
| `deepseek-r1-32b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | Qwen2.5-32B | 32B | – | – | bf16 | 32768 | – |
| `medgemma-27b-it` | `google/medgemma-27b-it` | Gemma-3-27B + SigLIP | 27B | ✅ | – | bf16 | 128K | ✅ |
| `clinicalgpt-r1` | `medicalai/ClinicalGPT-R1-Qwen-7B-EN-preview` | Qwen2.5-7B | 7B | – | – | bf16 | 32768 | – |
| `huatuogpt2-7b` | `FreedomIntelligence/HuatuoGPT2-7B` | Baichuan2-7B | 7B | – | ✅ | bf16 | 4096 | – |
| `huatuogpt-o1-7b` | `FreedomIntelligence/HuatuoGPT-o1-7B` | Qwen2.5-7B | 7B | – | – | bf16 | 32768 | – |
| `aquilamed-rl` | `BAAI/AquilaMed-RL` | Aquila3 (~LLaMA) | 8B | – | ✅ | bf16 | 4096 | – |
| `disc-medllm` | `Flmc/DISC-MedLLM` | Baichuan-13B-Base | 13B | – | ✅ | fp16 | 4096 | – |
| `baichuan-m1-14b` | `baichuan-inc/Baichuan-M1-14B-Instruct` | Baichuan-M1 (scratch) | 14B | – | ✅ | bf16 | 32768 | – |
| `baichuan-m2-32b` | `baichuan-inc/Baichuan-M2-32B` | Qwen2.5-32B | 32B | – | – | bf16 | 128K | – |
| `dao1-30b-a3b` | `CMLM/Dao1-30b-a3b` | Qwen3-30B-A3B (MoE) | 30B/3B act. | – | ✅ | fp16 | 40960 | – |
| `biancang` | `QLU-NLP/BianCang-Qwen2.5-7B-Instruct` | Qwen2.5-7B | 7B | – | – | bf16 | 32768 | – |
| `taiyi` | `DUTIR-BioNLP/Taiyi-LLM` | Qwen-1 7B | 7.7B | – | ✅ | bf16 | 8192 | – |
| `citrus-70b` | `jdh-algo/Citrus1.0-llama-70B` | LLaMA-3.1-70B | 70B | – | – | bf16 | 128K | – |
| `lingshu-7b` | `lingshu-medical-mllm/Lingshu-7B` | Qwen2.5-VL-7B | 7B | – | ✅ | bf16 | ~32K | ✅ |

> Also wired (commented in the YAML, uncomment to use): `huatuogpt2-13b/34b`,
> `huatuogpt-o1-8b/70b/72b`, `biancang-14b`, `lingshu-32b`, `citrus-qwen-72b`.

---

## Per-model notes

### English / international
- **Meditron-70B** — LLaMA-2-70B continued-pretrain (EPFL). **Gated**: accept the Llama-2
  license on HF and `huggingface-cli login`. It's a *base* model with **no chat template** —
  designed for few-shot / fine-tuning. ~140GB bf16 → `tensor_parallel: 2` on 2×80GB. `meditron-7b` also exists.
- **BioMistral-7B** — continued-pretrain of Mistral-7B-Instruct-v0.1, Apache-2.0. **Short 2048
  context** (sliding window). Research-only per the authors.
- **DeepSeek-R1-Distill-Qwen-32B** — reasoning model, MIT. Authors' guidance: **no system
  prompt**, temperature ~0.6, responses should begin with `<think>\n`. ~65GB bf16.
- **MedGemma-27B-it** — Gemma-3-27B + SigLIP, **multimodal (text+image)**. **Gated** (Google
  HAI-DEF terms + login). The text-only twin is `google/medgemma-27b-text-it` (simpler for MCQ).
  vLLM needs Gemma3 multimodal support for images.
- **ClinicalGPT-R1** — `medicalai/ClinicalGPT-R1-Qwen-7B-EN-preview`, Qwen2.5-7B reasoning
  model for disease diagnosis (MedFound team, arXiv:2504.09421). AFL-3.0. Expects a clinical record in the prompt.

### Chinese medical
- **HuatuoGPT-II** — repos are named **`HuatuoGPT2-*`** (7B/13B = Baichuan2, 34B = Yi-34B).
  **`trust_remote_code: true`**; **no HF chat template** (in-house HuatuoChat format) — the
  harness applies a `ROLE: …` fallback prompt.
- **HuatuoGPT-o1** — reasoning models; 7B/72B = Qwen2.5 (bilingual), 8B/70B = LLaMA-3.1 (EN).
  Standard archs + chat templates. Emits `## Thinking … ## Final Response …`. Apache-2.0.
- **AquilaMed-RL** — BAAI. config: `architectures=["AquilaForCausalLM"]`, `model_type="aquila3"`,
  `auto_map`→Llama. **`trust_remote_code: true`** required. vLLM matches on the arch string; if a
  given vLLM build can't load `aquila3`, the `auto` backend now **auto-falls-back to transformers**
  (or set `backend: transformers`). ChatML prompts, **short 4096 ctx**. BAAI Aquila licence.
  Note: `BAAI/AquilaMed-Instruct` is a *dataset*, not a model.
- **DISC-MedLLM** — Fudan, Baichuan-13B-Base. **`trust_remote_code: true`**, **dtype float16**,
  Baichuan user/assistant token format, ctx 4096.
- **Baichuan-M1-14B-Instruct** — custom **`baichuan_m1`** architecture (trained from scratch).
  **`trust_remote_code: true`**; needs `transformers>=4.47`. ⚠️ Confirm your vLLM build supports
  `baichuan_m1`; otherwise force `backend: transformers` in the model entry.
- **Baichuan-M2-32B** — Qwen2.5-32B base (plain `qwen2` config → **no** `trust_remote_code`
  needed despite the card example). Apache-2.0, 128K ctx. vLLM ≥ 0.9: `--reasoning-parser qwen3`.

### Traditional Chinese Medicine (中医)
- **Dao1-30b-a3b** — CMLM, **Qwen3-30B-A3B MoE** (128 experts, ~3B active/token), ChatML, ctx 40960.
  vLLM pre-allocates a large KV cache and **OOMs this 30B on most GPUs**, so the Dao series defaults
  to the **transformers backend** (`backend: transformers`, `device_map=auto` which can offload,
  `dtype: float16`, `attn_implementation: eager`) with Dao's recommended sampling
  (`temperature 0.7 / top_p 0.9 / repetition_penalty 1.1`) and the **小道 / Tao** system prompt — all
  carried on the model entry (`system_prompt`, `gen`). On a big GPU switch to `backend: vllm` (bf16,
  drop `attn_implementation`) for speed. Research / non-clinical only.
- **BianCang (扁仓)** — QLU-NLP, Qwen2.5 (arXiv:2411.11027). Prefer the **-Instruct** variants;
  7B and 14B available. Standard Qwen2 → native vLLM, ChatML.
- **Taiyi (太一)** — DUTIR-BioNLP, **original Qwen-1 7B** (`QWenLMHeadModel`, custom code) →
  **`trust_remote_code: true`** genuinely required. Bilingual BioNLP, ctx 8192. (A v2,
  `DUTIR-BioNLP/Taiyi2-chat` on GLM4-9B, also exists.)
- **ZhongJing-2-1.8B (仲景)** — CMLM. The repo is a **PEFT LoRA adapter** (~6MB) on
  `Qwen/Qwen1.5-1.8B-Chat`, so the entry sets `model: Qwen/Qwen1.5-1.8B-Chat` +
  `lora: CMLM/ZhongJing-2-1_8b` (vLLM applies it at runtime). Tiny — fits a free Colab T4.

### Multimodal
- **Lingshu (灵枢)** — `lingshu-medical-mllm/Lingshu-7B` / `-32B`, Qwen2.5-VL based, MIT.
  12+ imaging modalities. **`trust_remote_code: true`**. Wired here as a text backend for MCQ;
  use `configs/catalog_multimodal.yaml` for image tasks.

### Not publicly released on HuggingFace (documented for completeness)
- **Qibo (岐黄)** — paper arXiv:2403.16056 (Tianjin Univ.), Chinese-LLaMA 7B/13B. **No HF/GitHub
  weights found** as of 2026-06. Substitute **BianCang** or **Dao1** for a downloadable TCM model.
- **Qilin-Med (麒麟)** — the text LLM (arXiv:2310.09089, Baichuan-7B) is unreleased; only the
  **multimodal** LLaVA variant `williamliu/Qilin-Med-VL-Chat` exists (see the commented entry).

---

## License & responsible-use note

Most of these checkpoints restrict use to **research / non-commercial** and **explicitly disclaim
clinical use**. Med-Bench-Arena evaluates capability on benchmarks — it is **not** a clinical
tool. Always read each model's license on its HF page before use; gated models (Meditron,
MedGemma) require accepting their terms.
