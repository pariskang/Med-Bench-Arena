"""Local / remote JSON·JSONL·CSV adapter (``adapter: local_json``).

The workhorse for open-ended, syndrome-differentiation (辨证), prescription
(方剂) and safety (安全) data: HealthBench / LLMEval-Med / TCMEval-SDT / MTCMB /
CSEDB / MedSafetyBench. Everything is config-driven:

* ``source_url`` (downloaded + cached) or ``path`` (local file); format inferred
  from extension or set via ``format: jsonl|json|csv``.
* ``explode``: dotted path to a list — emit one sample per element (e.g. CSEDB's
  ``设计的考题内容.最具代表性的测试case``). Field paths then resolve against the
  element first, then the parent record.
* ``field_map`` values are dotted paths. ``prompt`` may resolve to a plain string
  *or* a list of ``{role, content}`` chat turns (HealthBench).
* ``rubric`` is normalized from any of: list of ``{criterion, points, tags}``
  (HealthBench), list of ``{规则内容, 分数}`` (CSEDB), a plain checklist string
  (LLMEval-Med), or ``null`` (→ llm_judge falls back to a per-task default).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from ..schema import Message, Prediction, Sample, TaskType, encode_images
from .base import DatasetAdapter, register_dataset

CACHE_DIR = Path(os.environ.get("MEDEVAL_CACHE", "data/cache"))

# rubric key aliases seen across datasets
_CRIT_KEYS = ("criterion", "规则内容", "text", "description", "rule", "内容")
_POINT_KEYS = ("points", "分数", "weight", "score", "权重")


def _get_path(obj: Any, path: str) -> Any:
    """Resolve a dotted path; returns None if any hop is missing."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_stringify(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify(v) for v in value)
    return str(value)


@register_dataset("local_json")
class LocalJSONAdapter(DatasetAdapter):
    """config: see module docstring. Key fields:
      task:        open_qa | sdt | prescription | safety
      source_url / path / hf:{repo,file}
      format:      jsonl | json | csv  (default: infer from extension)
      explode:     dotted path to a list (one sample per element)
      field_map:   {prompt, rubric?, reference?, label?, options?}
      metrics:     default [llm_judge]
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.task = TaskType(config.get("task", "open_qa"))
        self.source_url = config.get("source_url")
        self.path = config.get("path")
        self.hf = config.get("hf")
        self.format = config.get("format")
        self.explode = config.get("explode")
        self.fm = dict(config.get("field_map", {}))
        self.system_prompt = config.get("system_prompt")
        self.image_base = config.get("image_base", "")
        self.image_zip = config.get("image_zip")   # auto download+unzip images archive
        self.image_strip = config.get("image_strip", "")
        self.prompt_template = config.get("prompt_template")
        if not self.metric_specs:
            self.metric_specs = [("llm_judge", {})]
            self.metrics = ["llm_judge"]

    # --- source acquisition ----------------------------------------------
    def _resolve_files(self) -> list[Path]:
        """One or more source files. ``source_url`` / ``path`` may be a list (e.g.
        MedSafetyBench's 9 category CSVs) — they are concatenated."""
        if self.path:
            paths = self.path if isinstance(self.path, list) else [self.path]
            return [Path(p) for p in paths]
        urls = self.source_url
        if not urls and self.hf:
            repo, fname = self.hf["repo"], self.hf["file"]
            branch = self.hf.get("revision", "main")
            urls = f"https://huggingface.co/datasets/{repo}/resolve/{branch}/{fname}"
        if not urls:
            raise ValueError(f"{self.id}: need source_url, path, or hf")
        urls = urls if isinstance(urls, list) else [urls]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out: list[Path] = []
        for url in urls:
            h = hashlib.sha256(url.encode()).hexdigest()[:16]
            dest = CACHE_DIR / f"{self.id}_{h}{self._infer_ext(url)}"
            if not dest.exists():
                req = urllib.request.Request(url, headers={"User-Agent": "medeval/1.0"})
                with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                    f.write(r.read())
            out.append(dest)
        return out

    def _infer_ext(self, url_or_path: str) -> str:
        low = url_or_path.lower()
        for ext in (".jsonl", ".json", ".csv", ".tsv", ".parquet"):
            if ext in low:
                return ext
        return ".json"

    def _read_records(self, fp: Path) -> list[dict[str, Any]]:
        fmt = self.format or self._infer_ext(str(fp)).lstrip(".")
        if fmt == "parquet":  # binary; embedded-image open VQA (e.g. SLAKE-en)
            import datasets as _ds
            return [dict(r) for r in _ds.Dataset.from_parquet(str(fp))]
        text = fp.read_text(encoding="utf-8")
        if fmt in ("jsonl",):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        if fmt in ("csv", "tsv"):
            delim = "\t" if fmt == "tsv" else ","
            return list(csv.DictReader(text.splitlines(), delimiter=delim))
        data = json.loads(text)
        if isinstance(data, dict):
            # common wrappers: {"data":[...]} / {"examples":[...]}
            for k in ("data", "examples", "questions", "rows"):
                if isinstance(data.get(k), list):
                    return data[k]
            # dict-of-lists grouped by category (e.g. LLMEval-Med keyed by 类别)
            list_vals = [v for v in data.values() if isinstance(v, list)]
            if list_vals:
                flat: list[dict[str, Any]] = []
                for v in list_vals:
                    flat.extend(x for x in v if isinstance(x, dict))
                if flat:
                    return flat
            # dict keyed by id with dict values (e.g. MedR-Bench keyed by PMCID)
            dict_vals = [v for v in data.values() if isinstance(v, dict)]
            if dict_vals and len(dict_vals) == len(data):
                for k, v in zip(data.keys(), dict_vals):
                    v.setdefault("id", k)
                return dict_vals
            return [data]
        return data

    # --- loading ----------------------------------------------------------
    def load(self) -> list[Sample]:
        if self.image_zip:
            from ..assets import ensure_image_base
            self.image_base = ensure_image_base(self.image_zip, self.image_base or None)
        records: list[dict[str, Any]] = []
        for fp in self._resolve_files():
            records.extend(self._read_records(fp))
        samples: list[Sample] = []
        for ridx, root in enumerate(records):
            items = self._explode(root)
            for iidx, item in enumerate(items):
                s = self._build_sample(root, item, ridx, iidx)
                if s is not None:
                    samples.append(s)
                if self.limit and len(samples) >= self.limit:
                    return samples
        return samples

    def _explode(self, root: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.explode:
            return [root]
        lst = _get_path(root, self.explode)
        return lst if isinstance(lst, list) and lst else [root]

    def _resolve(self, field: str | None, root: dict, item: dict) -> Any:
        """Resolve a field path against the exploded item first, then the root."""
        if not field:
            return None
        v = _get_path(item, field)
        if v is None and item is not root:
            v = _get_path(root, field)
        return v

    def _build_sample(self, root: dict, item: dict, ridx: int, iidx: int) -> Sample | None:
        pfield = self.fm.get("prompt")
        if isinstance(pfield, list):  # join several columns (e.g. Patient Note + Question)
            parts = [_stringify(self._resolve(p, root, item)) for p in pfield]
            raw_prompt = "\n\n".join(p for p in parts if p)
        else:
            raw_prompt = self._resolve(pfield, root, item)
        if raw_prompt is None or raw_prompt == "":
            return None
        messages = self._to_messages(raw_prompt, root, item)
        if not messages:
            return None

        rubric = self._normalize_rubric(self._resolve(self.fm.get("rubric"), root, item))
        raw_ref = self._resolve(self.fm.get("reference"), root, item)
        reference = self._stringify_ref(raw_ref)
        label = self._resolve(self.fm.get("label"), root, item)

        ref_dict: dict[str, Any] = {}
        if rubric is not None:
            ref_dict["rubric"] = rubric
        if reference:
            ref_dict["reference"] = reference
        if isinstance(raw_ref, (dict, list)):
            ref_dict["reference_raw"] = raw_ref   # structured form (e.g. 方剂 dict)
        if label is not None:
            ref_dict["label"] = label
            if self.task == TaskType.SDT:
                ref_dict["syndrome"] = _stringify(label)
        # any extra field_map key (beyond the standard ones) becomes a reference
        # field, e.g. `pathogenesis: "TCM Pathogenesis"` for the 证型链 metric, or
        # `meridians`/`acupoints`/`classics` lists for the structured TCM metrics.
        for key, src in self.fm.items():
            if key not in ("prompt", "rubric", "reference", "label", "options"):
                val = self._resolve(src, root, item)
                if val is not None:
                    ref_dict[key] = val if isinstance(val, (list, dict)) else _stringify(val)

        rid = root.get("id", root.get("prompt_id", ridx))
        sid = f"{self.id}:{rid}" + (f":{iidx}" if self.explode else "")
        return Sample(
            id=sid, task_type=self.task, messages=messages,
            reference=ref_dict,
            meta={"source": self.source_url or self.path, "label": label},
        )

    def _to_messages(self, raw_prompt: Any, root: dict, item: dict) -> list[Message]:
        msgs: list[Message] = []
        if self.system_prompt:
            msgs.append(Message("system", self.system_prompt))
        # chat-style prompt (HealthBench): list of {role, content}
        if isinstance(raw_prompt, list) and raw_prompt and isinstance(raw_prompt[0], dict):
            for m in raw_prompt:
                role = m.get("role", "user")
                content = m.get("content", "")
                if content:
                    msgs.append(Message(role, str(content)))
        else:
            text = _stringify(raw_prompt)
            if self.prompt_template:
                text = self.prompt_template.format(prompt=text)
            msgs.append(Message("user", text))
        # multimodal: attach images (open VQA) to the last user turn
        imgs = encode_images(self._resolve(self.fm.get("image"), root, item),
                             self.image_base, self.image_strip)
        if imgs:
            for m in reversed(msgs):
                if m.role == "user":
                    m.images = imgs
                    break
        return msgs

    def _stringify_ref(self, value: Any) -> str:
        # MTCMB answers can be a dict {治法,方剂,药物组成} or a stringified list
        return _stringify(value)

    def _normalize_rubric(self, value: Any) -> list[dict[str, Any]] | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            # a plain checklist (LLMEval-Med) -> a single weighted criterion
            return [{"id": "checklist", "points": 1, "criterion": value}]
        if isinstance(value, dict):
            value = [value]
        out: list[dict[str, Any]] = []
        for i, it in enumerate(value):
            if isinstance(it, str):
                out.append({"id": f"c{i}", "points": 1, "criterion": it})
                continue
            crit = next((it[k] for k in _CRIT_KEYS if it.get(k)), "")
            pts = next((it[k] for k in _POINT_KEYS if it.get(k) is not None), 1)
            try:
                pts = float(pts)
            except (TypeError, ValueError):
                pts = 1.0
            cid = it.get("id") or (it.get("tags") or [f"c{i}"])[0]
            out.append({"id": str(cid), "points": pts, "criterion": str(crit),
                        "tags": it.get("tags")})
        return out or None

    # --- parsing ----------------------------------------------------------
    def parse(self, sample: Sample, text: str) -> Prediction:
        from ..schema import Generation
        # open-ended: the judge consumes the raw text; nothing to extract
        return Prediction(sample_id=sample.id, generation=Generation(text=text),
                          parsed=text)
