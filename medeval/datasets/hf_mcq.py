"""Config-driven HuggingFace multiple-choice adapter (``adapter: hf_mcq``).

Covers MedQA / MedMCQA / PubMedQA / MMLU-medical / CMB / CMExam / TCMBench / ...
Adding another lettered-MCQ dataset is *just config* — point ``path`` at the HF
repo and describe the columns in ``field_map``.

It normalizes the three option layouts seen in the wild...
  * dict column   ``{"A": "...", "B": "..."}``         -> options: <col>
  * N columns     ``opa, opb, opc, opd``               -> options: [opa, opb, opc, opd]
  * list column   ``["...", "...", ...]``              -> options: <col>
...and the three answer encodings: ``letter`` ("B"), ``index`` (int 0-based),
``text`` (the option's full text, e.g. PubMedQA yes/no/maybe).
"""
from __future__ import annotations

import json
import re
import string
from typing import Any

from ..schema import Message, Prediction, Sample, TaskType
from .base import DatasetAdapter, register_dataset

LETTERS = string.ascii_uppercase


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


@register_dataset("hf_mcq")
class HFMCQAdapter(DatasetAdapter):
    """config:
      path:          HF dataset repo id
      name:          config name | list of config names (concatenated) | null
      split:         split name (e.g. test, validation)
      field_map:     {question, options, answer, context?}
                       options: dict-col name | list-col name | [col, col, ...]
      answer_format: letter | index | text | multi
      inject_options: list[str]   # when there is no options column (e.g. yes/no/maybe)
      system_prompt: optional system message
      prompt_template: optional; placeholders {question} {options}
      instruction:   trailing instruction (default: answer with a letter)
      trust_remote_code: bool
      limit:         cap number of samples
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.path = config.get("path")
        self.name = config.get("name")
        self.split = config.get("split")  # default chosen per-mode in load()
        # data_files escape hatch: load raw json/csv/parquet directly, bypassing a
        # dataset's (possibly broken / removed) loading script. e.g. CMB on
        # datasets>=5 errors via the script but loads fine as raw json data_files.
        self.data_files = config.get("data_files")
        # merge gold from a separate file by a key column (e.g. CMB test answers
        # live on GitHub keyed by id, joined to the HF question file):
        #   answer_join: {data_files: <url>, key: id, value: answer}
        self.answer_join = config.get("answer_join")
        self._answer_map: dict[str, Any] | None = None
        self.format = config.get("format", "json")
        self.fm = dict(config.get("field_map", {}))
        self.answer_format = config.get("answer_format", "letter")
        self.inject_options = config.get("inject_options")
        self.options_inline = bool(config.get("options_inline", False))
        # deterministically shuffle option order per sample (removes position bias,
        # e.g. MedHallu's fixed correct-first 2-option layout)
        self.shuffle_options = bool(config.get("shuffle_options", False))
        self.system_prompt = config.get("system_prompt")
        self.prompt_template = config.get("prompt_template")
        self.instruction = config.get(
            "instruction",
            "Answer with the letter of the correct option (e.g. \"A\"). "
            "请只回答正确选项的字母。",
        )
        self.trust_remote_code = bool(config.get("trust_remote_code", False))
        # multimodal: field_map.image may name a column (or list) of images;
        # image_base is prepended to relative paths/URLs.
        self.image_base = config.get("image_base", "")
        # auto-download + unzip an images.zip and use its dir as image_base
        self.image_zip = config.get("image_zip")
        self.image_strip = config.get("image_strip", "")   # drop a path prefix (e.g. "../")
        # constant question when the data has no question column (e.g. an
        # image-classification set: TCM-Ladder visual = image + category label)
        self.question_text = config.get("question_text", "")
        if not self.metric_specs:
            self.metric_specs = [("mcq_accuracy", {})]
            self.metrics = ["mcq_accuracy"]

    # --- loading ----------------------------------------------------------
    def load(self) -> list[Sample]:
        from datasets import load_dataset  # lazy

        if self.image_zip:  # fetch + unzip the images archive once
            from ..assets import ensure_image_base
            self.image_base = ensure_image_base(self.image_zip, self.image_base or None)
        if self.answer_join:
            self._answer_map = self._load_answer_map()
        samples: list[Sample] = []
        if self.data_files:  # raw-file mode (json/csv/parquet): single -> "train"
            ds = load_dataset(self.format, data_files=self.data_files,
                              split=self.split or "train")
            self._ingest(ds, None, samples)
            return samples

        configs = self.name if isinstance(self.name, list) else [self.name]
        for cfg in configs:
            kwargs: dict[str, Any] = {"split": self.split or "test"}
            if cfg:
                kwargs["name"] = cfg
            if self.trust_remote_code:
                kwargs["trust_remote_code"] = True
            ds = load_dataset(self.path, **kwargs)
            if self._ingest(ds, cfg, samples):
                return samples
        return samples

    def _load_answer_map(self) -> dict[str, Any]:
        import hashlib
        import urllib.request
        from .local_json import CACHE_DIR
        spec = self.answer_join
        url = spec["data_files"]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        dest = CACHE_DIR / f"{self.id}_answers_{h}.json"
        if not dest.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "medeval/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
        data = json.loads(dest.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", list(data.values()))
        key, val = spec.get("key", "id"), spec.get("value", "answer")
        return {str(rec[key]): rec[val] for rec in data if key in rec and val in rec}

    def _ingest(self, ds, cfg, samples: list[Sample]) -> bool:
        """Append samples from one loaded split; return True if the limit was hit."""
        for i, row in enumerate(ds):
            if self._answer_map is not None:  # inject joined gold by key
                row = dict(row)
                rid = str(row.get(self.answer_join.get("key", "id")))
                if rid in self._answer_map:
                    row[self.fm["answer"]] = self._answer_map[rid]
            s = self._row_to_sample(row, cfg, i)
            if s is not None:
                samples.append(s)
            if self.limit and len(samples) >= self.limit:
                return True
        return False

    def _resolve_options(self, row: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Return (choices, keys) where keys are the source letter/keys if any.

        Handles the option layouts seen across datasets: N separate columns;
        a dict column ``{"A": ...}`` (dropping null slots like CMB's ``F: null``);
        a list column; a list of ``{key, value}`` objects (fzkuji/CMExam mirror);
        and a single inline lettered string (CMExam CSV).
        """
        opt = self.fm.get("options")
        if opt is None and self.inject_options:
            return list(self.inject_options), []
        if isinstance(opt, list):  # N columns; list-valued cols are flattened
            vals: list[str] = []                 # (e.g. MedBookVQA [Answer, Distractors])
            for c in opt:
                v = row.get(c)
                if isinstance(v, (list, tuple)):
                    vals.extend("" if x is None else str(x) for x in v)
                else:
                    vals.append("" if v is None else str(v))
            while len(vals) > 2 and not vals[-1].strip():  # drop trailing empty slots (blank E)
                vals.pop()
            return vals, []
        val = row[opt]
        if isinstance(val, dict):  # dict column {"A": ...} — drop null option slots
            keys = [k for k in val.keys() if val[k] is not None and str(val[k]).strip()]
            if all(len(k) == 1 and k.upper() in LETTERS for k in keys):
                keys = sorted(keys, key=lambda k: k.upper())
            return [str(val[k]) for k in keys], keys
        if isinstance(val, (list, tuple)):  # list column
            if val and isinstance(val[0], dict):  # [{"key":"A","value":"..."}]
                choices = [str(d.get("value", d.get("text", ""))) for d in val]
                ks = [str(d.get("key", "")) for d in val]
                return choices, ([k.upper() for k in ks] if all(len(k) == 1 for k in ks) else [])
            return [str(x) for x in val], []
        # single string that is a stringified dict/list (e.g. Med-HALT
        # "{'0': '...', '1': '...'}") -> parse and re-resolve
        sval = str(val).strip()
        if sval[:1] in ("{", "[") and sval[-1:] in ("}", "]"):
            import ast
            try:
                parsed = ast.literal_eval(sval)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, dict):
                parsed = {k: v for k, v in parsed.items()
                          if str(k).strip().lower() != "correct answer"}
                keys = list(parsed.keys())
                letterkeys = keys if all(len(str(k)) == 1 and str(k).upper() in LETTERS
                                         for k in keys) else []
                return [str(parsed[k]) for k in keys], letterkeys
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed], []
        # single string: try inline lettered options (CMExam), else wrap
        if self.options_inline or self._looks_inline(sval):
            parsed = self._split_inline(sval)
            if len(parsed) >= 2:
                return [t for _, t in parsed], [k for k, _ in parsed]
        return [sval], []

    @staticmethod
    def _looks_inline(s: str) -> bool:
        return len(re.findall(r"(?m)^\s*[A-Za-z][\s．.、:：)]", s)) >= 2

    @staticmethod
    def _split_inline(s: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for line in re.split(r"[\n\r]+", s):
            m = re.match(r"^\s*([A-Za-z])[\s．.、:：)]+\s*(.+)$", line)
            if m:
                out.append((m.group(1).upper(), m.group(2).strip()))
        return out

    def _gold_index(self, row: dict[str, Any], choices: list[str], keys: list[str]):
        ans = row[self.fm["answer"]]
        fmt = self.answer_format
        if fmt == "index":
            try:
                return int(ans)
            except (TypeError, ValueError):
                return None
        if fmt == "letter":
            letter = str(ans).strip().upper()
            if keys:  # dict-keyed options: map by key identity
                up = [k.upper() for k in keys]
                if letter in up:
                    return up.index(letter)
            if len(letter) == 1 and letter in LETTERS:
                idx = LETTERS.index(letter)
                return idx if idx < len(choices) else None
            return None
        if fmt == "text":
            target = _norm(ans)
            for j, c in enumerate(choices):
                if _norm(c) == target:
                    return j
            return None
        if fmt == "multi":
            return None  # handled by _gold_indices
        return None

    def _gold_indices(self, row: dict[str, Any], choices: list[str], keys: list[str]) -> list[int]:
        ans = str(row[self.fm["answer"]]).strip().upper()
        letters = re.findall(r"[A-Z]", ans)
        out = []
        for L in letters:
            if keys:
                up = [k.upper() for k in keys]
                if L in up:
                    out.append(up.index(L))
            elif L in LETTERS and LETTERS.index(L) < len(choices):
                out.append(LETTERS.index(L))
        return sorted(set(out))

    def _row_to_sample(self, row: dict[str, Any], cfg: str | None, i: int) -> Sample | None:
        try:
            choices, keys = self._resolve_options(row)
        except (KeyError, TypeError):
            return None
        if not choices:
            return None
        qfield = self.fm.get("question")
        question = str(row[qfield]) if qfield else self.question_text
        ctx_field = self.fm.get("context")
        context = ""
        if ctx_field and ctx_field in row and row[ctx_field]:
            cval = row[ctx_field]
            if isinstance(cval, dict) and "contexts" in cval:
                context = "\n".join(str(x) for x in cval["contexts"])
            elif isinstance(cval, (list, tuple)):  # e.g. MedHallu Knowledge: list[str]
                context = "\n".join(str(x) for x in cval)
            else:
                context = str(cval)

        reference: dict[str, Any] = {}
        if self.answer_format == "multi":
            idxs = self._gold_indices(row, choices, keys)
            if not idxs:
                return None
            reference["indices"] = idxs
        else:
            gi = self._gold_index(row, choices, keys)
            if gi is None or gi >= len(choices):
                return None
            reference = {"index": gi, "letter": LETTERS[gi], "text": choices[gi]}

        sid = f"{self.id}:{cfg or 'default'}:{row.get('id', i)}"
        if self.shuffle_options:  # de-bias fixed option order (e.g. MedHallu correct-first)
            choices, reference = self._shuffle(sid, choices, reference)

        user = self._render(question, choices, context)
        msgs = []
        if self.system_prompt:
            msgs.append(Message("system", self.system_prompt))
        images = self._resolve_images(row)
        msgs.append(Message("user", user, images=images or None))
        return Sample(
            id=sid, task_type=TaskType.MCQ, messages=msgs,
            choices=choices, reference=reference,
            meta={"path": self.path, "config": cfg, "subject": row.get("subject")},
        )

    @staticmethod
    def _shuffle(sid: str, choices: list[str], reference: dict[str, Any]
                 ) -> tuple[list[str], dict[str, Any]]:
        import hashlib
        import random
        seed = int(hashlib.md5(sid.encode("utf-8")).hexdigest()[:8], 16)  # stable across runs
        perm = list(range(len(choices)))
        random.Random(seed).shuffle(perm)
        new_choices = [choices[i] for i in perm]
        pos = {old: new for new, old in enumerate(perm)}  # old index -> new index
        if "indices" in reference:
            ref = {"indices": sorted(pos[i] for i in reference["indices"])}
        else:
            ni = pos[reference["index"]]
            ref = {"index": ni, "letter": LETTERS[ni], "text": new_choices[ni]}
        return new_choices, ref

    # --- multimodal -------------------------------------------------------
    def _resolve_images(self, row: dict[str, Any]) -> list[str]:
        field = self.fm.get("image")
        if not field:
            return []
        cols = field if isinstance(field, list) else [field]
        out: list[str] = []
        for c in cols:
            if c in row and row[c] is not None:
                out.extend(self._encode_image(row[c]))
        return out

    def _encode_image(self, val: Any) -> list[str]:
        from ..schema import encode_images
        return encode_images(val, self.image_base, self.image_strip)

    def _render(self, question: str, choices: list[str], context: str) -> str:
        opt_block = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
        if self.prompt_template:
            return self.prompt_template.format(
                question=question, options=opt_block, context=context
            )
        head = f"{context}\n\n" if context else ""
        return f"{head}{question}\n\n{opt_block}\n\n{self.instruction}"

    # --- parsing ----------------------------------------------------------
    def parse(self, sample: Sample, text: str) -> Prediction:
        from ..schema import Generation, Prediction as P
        n = len(sample.choices or [])
        if "indices" in sample.reference:
            parsed: Any = self._extract_multi(text, n, sample.choices)
        else:
            parsed = self._extract_single(text, n, sample.choices or [])
        return P(sample_id=sample.id, generation=Generation(text=text), parsed=parsed)

    @staticmethod
    def _letters_for(n: int) -> list[str]:
        return list(LETTERS[:n])

    def _extract_single(self, text: str, n: int, choices: list[str]) -> int | None:
        if not text:
            return None
        letters = self._letters_for(n)
        t = text.strip()
        # 1) explicit "answer is X" / "答案是X" / "正确选项为X"
        #    The letter must be standalone (not the first letter of an option word
        #    like the "A" in "Atrophy") -> require it is not followed by a letter.
        m = re.search(
            r"(?:answer|correct option|正确答案|答案|选项)\s*(?:is|:|：|为|是)?\s*\(?([A-Z])\)?(?![A-Za-z])",
            t, re.IGNORECASE,
        )
        if m and m.group(1).upper() in letters:
            return letters.index(m.group(1).upper())
        # 2) a parenthesized or punctuated standalone letter, prefer the last
        cands = re.findall(r"(?:^|[^A-Za-z])\(?([A-Z])\)?(?:[.):、]|\b)", t)
        cands = [c.upper() for c in cands if c.upper() in letters]
        if cands:
            return letters.index(cands[-1])
        # 3) fall back to option-text containment
        tl = _norm(t)
        for j, c in enumerate(choices):
            if c and _norm(c) in tl:
                return j
        return None

    def _extract_multi(self, text: str, n: int, choices: list[str] | None) -> list[int]:
        """Extract multiple selected options without grabbing capitals from prose
        words. Handles concatenated runs ('BCDE') and delimited lists
        ('B, C, D and E' / 'B、C、D、E')."""
        letters = self._letters_for(n)
        rng = re.escape("".join(letters))
        t = text or ""
        picks: set[int] = set()
        # contiguous runs of >=2 option letters, e.g. "BCDE"
        for run in re.findall(rf"(?<![A-Za-z])([{rng}]{{2,}})(?![A-Za-z])", t):
            picks.update(letters.index(c) for c in run)
        # standalone single option letters, e.g. "B, C, D and E"
        for c in re.findall(rf"(?<![A-Za-z])([{rng}])(?![A-Za-z])", t):
            picks.add(letters.index(c))
        if picks:
            return sorted(picks)
        one = self._extract_single(t, n, choices)
        return [one] if one is not None else []
