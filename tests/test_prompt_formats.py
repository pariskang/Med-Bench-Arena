"""Unit tests for prompt_format modes and few_shot_examples.

Covers all four hf_mcq prompt_format values (zero_shot, zero_shot_cot,
few_shot, few_shot_cot) and the local_json few_shot_examples feature.
All tests are fully offline — no network, no GPU, no HuggingFace downloads.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from medeval.datasets.hf_mcq import HFMCQAdapter
from medeval.datasets.local_json import LocalJSONAdapter
from medeval.schema import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mcq(prompt_format: str = "zero_shot_cot",
         few_shot_examples: list | None = None,
         answer_format: str = "letter") -> HFMCQAdapter:
    cfg: dict = {
        "id": "t", "path": "x",
        "field_map": {"question": "q", "options": "o", "answer": "a"},
        "answer_format": answer_format,
        "prompt_format": prompt_format,
    }
    if few_shot_examples is not None:
        cfg["few_shot_examples"] = few_shot_examples
    return HFMCQAdapter(cfg)


_EXAMPLE = {
    "question": "Which is the powerhouse of the cell?",
    "options": {"A": "Nucleus", "B": "Mitochondria", "C": "Ribosome"},
    "answer": "B",
    "explanation": "Mitochondria generate ATP via oxidative phosphorylation, making them the energy source of the cell.",
}

_ROW = {"q": "What causes pneumonia?", "o": {"A": "Virus", "B": "Bacteria", "C": "Fungus"}, "a": "B"}


# ---------------------------------------------------------------------------
# 1. zero_shot — no instruction, no examples
# ---------------------------------------------------------------------------

def test_zero_shot_no_instruction():
    ad = _mcq("zero_shot")
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    assert "Think" not in prompt, "zero_shot must not include CoT instruction"
    assert "Answer:" not in prompt, "zero_shot must not include 'Answer:' placeholder"
    assert "A. Virus" in prompt
    assert "B. Bacteria" in prompt


def test_zero_shot_no_examples():
    ad = _mcq("zero_shot", few_shot_examples=[_EXAMPLE])
    # When prompt_format is zero_shot, examples SHOULD still be prepended
    # (the format only controls the instruction; examples are always rendered)
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    assert "powerhouse" in prompt, "few_shot_examples must be prepended even in zero_shot mode"
    assert "Think" not in prompt, "zero_shot still omits the CoT instruction"


# ---------------------------------------------------------------------------
# 2. zero_shot_cot (default) — instruction present, no examples
# ---------------------------------------------------------------------------

def test_zero_shot_cot_default_instruction():
    ad = _mcq("zero_shot_cot")
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    assert "Think step by step" in prompt
    assert "Answer: A" in prompt


def test_zero_shot_cot_is_default():
    """Omitting prompt_format gives the same prompt as zero_shot_cot."""
    ad_default = HFMCQAdapter({
        "id": "t", "path": "x",
        "field_map": {"question": "q", "options": "o", "answer": "a"},
        "answer_format": "letter",
    })
    ad_explicit = _mcq("zero_shot_cot")
    s1 = ad_default._row_to_sample(_ROW, None, 0)
    s2 = ad_explicit._row_to_sample(_ROW, None, 0)
    assert s1.messages[-1].content == s2.messages[-1].content


def test_zero_shot_cot_multi_answer():
    ad = _mcq("zero_shot_cot", answer_format="multi")
    ad.fm["options"] = "o"
    sample = ad._row_to_sample(
        {"q": "Which are antibiotics?", "o": {"A": "Amoxicillin", "B": "Aspirin", "C": "Azithromycin"},
         "a": "AC"}, None, 0)
    prompt = sample.messages[-1].content
    assert "Answer: BC" in prompt or "Answer: AC" in prompt  # multi-answer placeholder


# ---------------------------------------------------------------------------
# 3. few_shot — examples prepended inline, no instruction at end
# ---------------------------------------------------------------------------

def test_few_shot_examples_appear_in_prompt():
    ad = _mcq("few_shot", few_shot_examples=[_EXAMPLE])
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    assert "powerhouse" in prompt
    assert "Mitochondria" in prompt
    assert "Answer: B" in prompt   # example answer line


def test_few_shot_no_instruction_at_end():
    ad = _mcq("few_shot", few_shot_examples=[_EXAMPLE])
    sample = ad._row_to_sample(_ROW, None, 0)
    # The CoT instruction must NOT follow the actual question
    last_block = sample.messages[-1].content.split("Answer: B")[-1]
    assert "Think step by step" not in last_block


def test_few_shot_two_examples_ordered():
    examples = [
        {"question": "Q1", "options": {"A": "Opt1"}, "answer": "A"},
        {"question": "Q2", "options": ["Yes", "No"], "answer": "A"},
    ]
    ad = _mcq("few_shot", few_shot_examples=examples)
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    q1_pos = prompt.index("Q1")
    q2_pos = prompt.index("Q2")
    assert q1_pos < q2_pos, "examples must appear in config order"


def test_few_shot_dict_options_sorted_by_key():
    ex = {"question": "Q", "options": {"C": "Third", "A": "First", "B": "Second"}, "answer": "A"}
    ad = _mcq("few_shot", few_shot_examples=[ex])
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    a_pos = prompt.index("A. First")
    b_pos = prompt.index("B. Second")
    c_pos = prompt.index("C. Third")
    assert a_pos < b_pos < c_pos


def test_few_shot_list_options():
    ex = {"question": "Q", "choices": ["Alpha", "Beta", "Gamma"], "answer": "B"}
    ad = _mcq("few_shot", few_shot_examples=[ex])
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    assert "A. Alpha" in prompt
    assert "B. Beta" in prompt


# ---------------------------------------------------------------------------
# 4. few_shot_cot — examples include reasoning + instruction at end
# ---------------------------------------------------------------------------

def test_few_shot_cot_explanation_in_example():
    ad = _mcq("few_shot_cot", few_shot_examples=[_EXAMPLE])
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    assert "oxidative phosphorylation" in prompt  # explanation text
    assert "Answer: B" in prompt


def test_few_shot_cot_instruction_after_actual_question():
    ad = _mcq("few_shot_cot", few_shot_examples=[_EXAMPLE])
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    # The actual question ("What causes pneumonia?") must appear AFTER the example
    example_end = prompt.index("Answer: B")
    actual_question_pos = prompt.index("What causes pneumonia?")
    assert actual_question_pos > example_end
    # CoT instruction must appear AFTER the actual question options
    instruction_pos = prompt.rfind("Think step by step")
    assert instruction_pos > actual_question_pos


def test_few_shot_cot_no_explanation_falls_back_to_plain_answer():
    ex_no_expl = {k: v for k, v in _EXAMPLE.items() if k != "explanation"}
    ad = _mcq("few_shot_cot", few_shot_examples=[ex_no_expl])
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    # Without explanation, the example still shows "Answer: B"
    assert "Answer: B" in prompt
    # But no explanation text
    assert "oxidative phosphorylation" not in prompt


def test_few_shot_cot_explanation_not_shown_in_zero_shot_cot_mode():
    """Explanation text in examples is ONLY rendered in few_shot_cot mode."""
    ad = _mcq("zero_shot_cot", few_shot_examples=[_EXAMPLE])
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    # Example is still prepended; but explanation text must NOT appear
    assert "powerhouse" in prompt  # example question appears
    assert "oxidative phosphorylation" not in prompt  # explanation suppressed


# ---------------------------------------------------------------------------
# 5. Custom instruction override works in all CoT modes
# ---------------------------------------------------------------------------

def test_custom_instruction_override():
    ad = HFMCQAdapter({
        "id": "t", "path": "x",
        "field_map": {"question": "q", "options": "o", "answer": "a"},
        "answer_format": "letter",
        "prompt_format": "zero_shot_cot",
        "instruction": "Just pick the letter.",
    })
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    assert "Just pick the letter." in prompt
    assert "Think step by step" not in prompt


def test_custom_instruction_in_few_shot_cot():
    ad = HFMCQAdapter({
        "id": "t", "path": "x",
        "field_map": {"question": "q", "options": "o", "answer": "a"},
        "answer_format": "letter",
        "prompt_format": "few_shot_cot",
        "instruction": "Explain, then say Answer: X.",
        "few_shot_examples": [_EXAMPLE],
    })
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    assert "Explain, then say Answer: X." in prompt


# ---------------------------------------------------------------------------
# 6. prompt_template ignores prompt_format instruction but keeps prefix
# ---------------------------------------------------------------------------

def test_prompt_template_with_few_shot_prefix():
    ad = HFMCQAdapter({
        "id": "t", "path": "x",
        "field_map": {"question": "q", "options": "o", "answer": "a"},
        "answer_format": "letter",
        "prompt_format": "few_shot",
        "prompt_template": "Q: {question}\nChoices:\n{options}\nYour answer:",
        "few_shot_examples": [_EXAMPLE],
    })
    sample = ad._row_to_sample(_ROW, None, 0)
    prompt = sample.messages[-1].content
    assert "powerhouse" in prompt       # example still prepended
    assert "Your answer:" in prompt     # template used for actual question


# ---------------------------------------------------------------------------
# 7. local_json few_shot_examples → interleaved user/assistant turns
# ---------------------------------------------------------------------------

def _lj(few_shot_examples: list | None = None, **extra) -> LocalJSONAdapter:
    cfg = {
        "id": "lj", "adapter": "local_json", "task": "open_qa",
        "path": "placeholder",
        "field_map": {"prompt": "q", "reference": "a"},
    }
    if few_shot_examples is not None:
        cfg["few_shot_examples"] = few_shot_examples
    cfg.update(extra)
    return LocalJSONAdapter(cfg)


def test_local_json_few_shot_interleaved_turns():
    examples = [
        {"prompt": "What is aspirin?", "answer": "A non-steroidal anti-inflammatory drug."},
        {"prompt": "What is metformin?", "answer": "A first-line oral hypoglycaemic agent."},
    ]
    data = [{"q": "Explain beta-blockers.", "a": "..."}]
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "x.jsonl"
        fp.write_text(json.dumps(data[0]), encoding="utf-8")
        ad = _lj(few_shot_examples=examples, path=str(fp))
        samples = ad.load()
    assert len(samples) == 1
    msgs = samples[0].messages
    # Expected: user(ex1), assistant(ex1), user(ex2), assistant(ex2), user(actual)
    assert len(msgs) == 5
    assert msgs[0].role == "user"   and "aspirin" in msgs[0].content
    assert msgs[1].role == "assistant" and "anti-inflammatory" in msgs[1].content
    assert msgs[2].role == "user"   and "metformin" in msgs[2].content
    assert msgs[3].role == "assistant" and "hypoglycaemic" in msgs[3].content
    assert msgs[4].role == "user"   and "beta-blockers" in msgs[4].content


def test_local_json_few_shot_with_system_prompt():
    examples = [{"prompt": "Q1", "answer": "A1"}]
    data = [{"q": "Q2", "a": "A2"}]
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "x.jsonl"
        fp.write_text(json.dumps(data[0]), encoding="utf-8")
        ad = _lj(few_shot_examples=examples, path=str(fp), system_prompt="You are a doctor.")
        samples = ad.load()
    msgs = samples[0].messages
    # Expected: system, user(ex1), assistant(ex1), user(actual)
    assert msgs[0].role == "system" and "doctor" in msgs[0].content
    assert msgs[1].role == "user"   and "Q1" in msgs[1].content
    assert msgs[2].role == "assistant"
    assert msgs[3].role == "user"   and "Q2" in msgs[3].content


def test_local_json_few_shot_instruction_only_on_actual_question():
    """Instruction is appended to the final question, not to example questions.

    With 1 example and no system prompt the message order is:
      msgs[0] user(example Q, no instruction)
      msgs[1] assistant(example A)
      msgs[2] user(actual question + instruction)
    """
    examples = [{"prompt": "Example Q", "answer": "Example A"}]
    data = [{"q": "Actual question", "a": "ref"}]
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "x.jsonl"
        fp.write_text(json.dumps(data[0]), encoding="utf-8")
        ad = _lj(few_shot_examples=examples, path=str(fp),
                 instruction="Think carefully and cite evidence.")
        samples = ad.load()
    msgs = samples[0].messages
    assert len(msgs) == 3
    assert msgs[0].role == "user"      and "Think carefully" not in msgs[0].content
    assert msgs[1].role == "assistant"
    assert msgs[2].role == "user"      and "Think carefully" in msgs[2].content


def test_local_json_no_few_shot_unchanged():
    """Without few_shot_examples, behaviour is identical to before."""
    data = [{"q": "Q?", "a": "A."}]
    with tempfile.TemporaryDirectory() as d:
        fp = Path(d) / "x.jsonl"
        fp.write_text(json.dumps(data[0]), encoding="utf-8")
        ad = _lj(path=str(fp))
        samples = ad.load()
    msgs = samples[0].messages
    assert len(msgs) == 1
    assert msgs[0].role == "user"


# ---------------------------------------------------------------------------
# 8. Parser handles all prompt_format outputs correctly
# ---------------------------------------------------------------------------

def test_parser_zero_shot_bare_letter():
    """In zero_shot mode the model may return just the letter; parser must handle it."""
    ad = _mcq("zero_shot")
    ad.fm["options"] = "o"
    sample = ad._row_to_sample(_ROW, None, 0)
    parsed = ad.parse(sample, "B").parsed
    assert parsed == 1  # B is index 1


def test_parser_zero_shot_cot_structured_line():
    """In zero_shot_cot mode the model returns reasoning + 'Answer: B'."""
    ad = _mcq("zero_shot_cot")
    ad.fm["options"] = "o"
    sample = ad._row_to_sample(_ROW, None, 0)
    reply = "Bacteria are the most common cause.\n\nAnswer: B"
    assert ad.parse(sample, reply).parsed == 1


def test_parser_few_shot_cot_multi_step_self_correct():
    """Self-correcting few-shot CoT: last 'Answer:' line wins."""
    ad = _mcq("few_shot_cot", few_shot_examples=[_EXAMPLE])
    ad.fm["options"] = "o"
    sample = ad._row_to_sample(_ROW, None, 0)
    reply = "Initially Answer: A ... wait, reconsidering. Answer: B"
    assert ad.parse(sample, reply).parsed == 1  # last wins


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("OK: all prompt-format tests passed")
