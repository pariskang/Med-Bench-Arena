"""HuggingFace / local backend (``type: hf``).

Local checkpoint, HF repo, or LoRA adapter. Prefers **vLLM** for high-throughput
batched inference (falls back to transformers). The important bit: this provider
overrides :meth:`agenerate_many` to feed the *whole* batch to vLLM in one call —
the single biggest performance difference vs. API backends. Imports are lazy so
the package stays importable on machines without a GPU / vLLM / torch.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..schema import Generation, Message, now
from .base import ModelProvider, register_provider


def _chat_to_text(tokenizer, messages: list[dict[str, str]]) -> str:
    """Apply the model chat template if available, else a simple fallback."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        parts.append("ASSISTANT:")
        return "\n".join(parts)


@register_provider("hf")
class HFProvider(ModelProvider):
    """config:
      model:            local path or HF repo id
      dtype:            bfloat16 | float16 | auto (default bfloat16)
      tensor_parallel:  vLLM tensor-parallel size (default 1)
      max_model_len:    context window cap
      lora:             optional path/repo of a LoRA adapter
      backend:          vllm | transformers | auto (default auto)
      gpu_memory_utilization: vLLM knob (default 0.9)
      trust_remote_code: load custom modeling code (Baichuan/Taiyi/… need this)
      revision:         pin a specific commit/branch of the HF repo
      attn_implementation: transformers attention impl, e.g. ``eager`` (lower memory)
      system_prompt:    default system turn prepended when the sample has none
      gen:              per-model generation overrides (temperature/top_p/…)
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model: str = config["model"]
        self.dtype: str = config.get("dtype", "bfloat16")
        self.tensor_parallel: int = int(config.get("tensor_parallel", 1))
        self.max_model_len: int | None = config.get("max_model_len")
        self.lora: str | None = config.get("lora")
        self.backend: str = config.get("backend", "auto")
        self.gpu_mem: float = float(config.get("gpu_memory_utilization", 0.9))
        self.trust_remote_code: bool = bool(config.get("trust_remote_code", False))
        self.revision: str | None = config.get("revision")
        self.attn_implementation: str | None = config.get("attn_implementation")
        self.system_prompt: str | None = config.get("system_prompt")
        self._engine = None
        self._tokenizer = None
        self._lora_request = None
        self._mode: str | None = None  # "vllm" | "transformers"

    # --- engine bootstrap -------------------------------------------------
    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        if self.backend in ("auto", "vllm"):
            try:
                self._init_vllm()
                return
            except ImportError:
                if self.backend == "vllm":
                    raise  # user explicitly asked for vLLM but it isn't installed
                # vLLM absent -> fall through to transformers
            except Exception as e:
                if self.backend == "vllm":
                    raise  # explicit vllm: surface the real error, don't mask it
                # auto: a model vLLM can't load (unsupported arch like AquilaMed's
                # `aquila3`, a too-new custom type, an OOM) shouldn't crash the run —
                # degrade to the transformers backend instead.
                import sys
                print(f"[medeval] vLLM could not load {self.model!r} "
                      f"({type(e).__name__}: {e}); falling back to transformers.",
                      file=sys.stderr)
        self._init_transformers()

    def _init_vllm(self) -> None:
        from vllm import LLM  # lazy
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        kwargs: dict[str, Any] = {
            "model": self.model,
            "tensor_parallel_size": self.tensor_parallel,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_mem,
            "enable_lora": bool(self.lora),
            "trust_remote_code": self.trust_remote_code,
        }
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len
        if self.revision:
            kwargs["revision"] = self.revision
        self._engine = LLM(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code, revision=self.revision)
        if self.lora:
            self._lora_request = LoRARequest("adapter", 1, self.lora)
        self._mode = "vllm"

    def _init_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code, revision=self.revision)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            self.dtype, "auto"
        )
        mkw: dict[str, Any] = dict(
            torch_dtype=dtype, device_map="auto",
            trust_remote_code=self.trust_remote_code, revision=self.revision,
        )
        if self.attn_implementation:
            mkw["attn_implementation"] = self.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(self.model, **mkw)
        if self.lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.lora)
        self._engine = model
        self._mode = "transformers"

    # --- generation -------------------------------------------------------
    def _sampling_params(self, gen: dict[str, Any]):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=gen.get("temperature", 0.0),
            top_p=gen.get("top_p", 1.0),
            repetition_penalty=gen.get("repetition_penalty", 1.0),
            max_tokens=gen.get("max_tokens", 1024),
            stop=gen.get("stop"),
            seed=gen.get("seed"),
        )

    def _to_chat(self, msgs: list[Message]) -> list[dict[str, str]]:
        """OpenAI-style messages, prepending the model's default system turn if the
        sample doesn't already supply one."""
        oa = [m.to_openai() for m in msgs]
        if self.system_prompt and not (oa and oa[0].get("role") == "system"):
            oa = [{"role": "system", "content": self.system_prompt}, *oa]
        return oa

    async def agenerate(self, messages: list[Message], **gen: Any) -> Generation:
        out = await self.agenerate_many([messages], **gen)
        return out[0]

    async def agenerate_many(
        self, batch: list[list[Message]], **gen: Any
    ) -> list[Generation]:
        # Run the (blocking) engine in a worker thread so we stay async-friendly.
        return await asyncio.to_thread(self._generate_batch_sync, batch, gen)

    def _generate_batch_sync(
        self, batch: list[list[Message]], gen: dict[str, Any]
    ) -> list[Generation]:
        self._ensure_engine()
        gen = self._merge_gen(gen)   # per-model overrides on top of run defaults
        prompts = [
            _chat_to_text(self._tokenizer, self._to_chat(msgs))
            for msgs in batch
        ]
        t0 = now()
        if self._mode == "vllm":
            params = self._sampling_params(gen)
            kw = {"lora_request": self._lora_request} if self._lora_request else {}
            outputs = self._engine.generate(prompts, params, **kw)
            dt = now() - t0
            results = []
            for o in outputs:
                comp = o.outputs[0]
                ptok = len(o.prompt_token_ids)
                ctok = len(comp.token_ids)
                results.append(Generation(
                    text=comp.text, model=self.model,
                    prompt_tokens=ptok, completion_tokens=ctok,
                    total_tokens=ptok + ctok, latency_s=dt / max(1, len(outputs)),
                    finish_reason=getattr(comp, "finish_reason", "") or "stop",
                ))
            return results
        return self._generate_transformers(prompts, gen, t0)

    def _generate_transformers(self, prompts, gen, t0):
        import torch
        results = []
        for prompt in prompts:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._engine.device)
            with torch.no_grad():
                out = self._engine.generate(
                    **inputs,
                    max_new_tokens=gen.get("max_tokens", 1024),
                    do_sample=gen.get("temperature", 0.0) > 0,
                    temperature=max(gen.get("temperature", 0.0), 1e-5),
                    top_p=gen.get("top_p", 1.0),
                    repetition_penalty=gen.get("repetition_penalty", 1.0),
                    use_cache=True,
                )
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(Generation(
                text=text, model=self.model,
                prompt_tokens=int(inputs["input_ids"].shape[1]),
                completion_tokens=int(gen_ids.shape[0]),
                total_tokens=int(inputs["input_ids"].shape[1] + gen_ids.shape[0]),
                latency_s=(now() - t0),
            ))
        return results
