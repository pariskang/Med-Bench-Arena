"""Multimodal (舌象/脉象 image) plumbing: image encoding, content blocks, the
hf_mcq image field, and an end-to-end image MCQ run with the mock vision model."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import medeval
from medeval import Message
from medeval.schema import image_to_url
from medeval.datasets.hf_mcq import HFMCQAdapter

# a minimal valid 1x1 PNG
_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00"
        b"\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")


def test_image_to_url_and_message_blocks():
    assert image_to_url("https://x/y.jpg") == "https://x/y.jpg"
    assert image_to_url("data:image/png;base64,AAA") == "data:image/png;base64,AAA"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tongue.png"
        p.write_bytes(_PNG)
        url = image_to_url(str(p))
        assert url.startswith("data:image/png;base64,")
        m = Message("user", "描述舌象", images=[str(p)])
        oc = m.to_openai()
        assert isinstance(oc["content"], list)
        assert oc["content"][0] == {"type": "text", "text": "描述舌象"}
        assert oc["content"][1]["type"] == "image_url"
        assert oc["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    # text-only messages stay plain strings (backward compatible)
    assert Message("user", "hi").to_openai() == {"role": "user", "content": "hi"}


def test_hf_mcq_image_encoding_variants():
    ad = HFMCQAdapter({"id": "t", "path": "x", "image_base": "https://host/imgs/",
                       "field_map": {"question": "q", "options": "o", "answer": "a", "image": "img"}})
    assert ad._encode_image("http://x/t.png") == ["http://x/t.png"]      # URL passthrough
    assert ad._encode_image("a/b.png") == ["https://host/imgs/a/b.png"]  # base prepend
    assert ad._encode_image({"url": "http://u/i.jpg"}) == ["http://u/i.jpg"]
    assert ad._encode_image({"bytes": _PNG})[0].startswith("data:image/png;base64,")
    # raw bytes (TCM-Ladder visual.parquet): PNG vs JPEG mime sniffed from magic
    assert ad._encode_image(_PNG)[0].startswith("data:image/png;base64,")
    assert ad._encode_image(b"\xff\xd8\xff\xe0jpegdata")[0].startswith("data:image/jpeg;base64,")
    assert ad._encode_image([{"url": "http://u/1.png"}, "2.png"]) == \
        ["http://u/1.png", "https://host/imgs/2.png"]


def test_image_strip_and_option_flatten():
    from medeval.schema import encode_images
    # image_strip drops a leading prefix (MedBookVQA ../figures/x.jpg)
    assert encode_images("../figures/x.jpg", "/d/", strip="../") == ["/d/figures/x.jpg"]
    assert encode_images({"path": "../figures/y.jpg"}, "/d/", strip="../") == ["/d/figures/y.jpg"]
    # list-valued option columns are flattened (MedBookVQA [Answer, Distractors])
    ad = HFMCQAdapter({"id": "t", "path": "x", "answer_format": "text",
                       "field_map": {"question": "q", "options": ["Answer", "Distractors"], "answer": "Answer"}})
    choices, keys = ad._resolve_options({"Answer": "correct", "Distractors": ["w1", "w2", "w3"]})
    assert choices == ["correct", "w1", "w2", "w3"] and keys == []


def test_multimodal_mcq_end_to_end_with_mock_vision():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img = d / "t.png"
        img.write_bytes(_PNG)
        recs = [{"q": "此舌象提示的证型是？", "o": ["气虚", "血瘀", "湿热"], "a": "C", "img": str(img)},
                {"q": "此舌象舌色为？", "o": ["淡白", "红绛", "青紫"], "a": "A", "img": str(img)}]
        fp = d / "vqa.jsonl"
        fp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")

        cfg = {
            "run": {"output_dir": str(d / "out"), "cache": False},
            "eval": {"gen": {"temperature": 0.0, "max_tokens": 32}},
            "models": [{"id": "mock-vision", "type": "mock", "behavior": "auto"}],
            "datasets": [{"id": "tongue_vqa", "adapter": "hf_mcq", "format": "json",
                          "data_files": str(fp),
                          "field_map": {"question": "q", "options": "o", "answer": "a", "image": "img"},
                          "answer_format": "letter", "metrics": ["mcq_accuracy"]}],
        }
        rows = medeval.run_config(cfg)
    r = [x for x in rows if x["dataset"] == "tongue_vqa"][0]
    assert r["n"] == 2 and "accuracy" in r["metrics"]["mcq_accuracy"]


if __name__ == "__main__":
    test_image_to_url_and_message_blocks()
    test_hf_mcq_image_encoding_variants()
    test_multimodal_mcq_end_to_end_with_mock_vision()
    print("OK: multimodal tests passed")
