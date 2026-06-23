"""TCM classics knowledge graph: build, query, export/load, and metric wiring."""
from __future__ import annotations

import tempfile
import xml.dom.minidom
from pathlib import Path

import medeval
from medeval.kg.tcm_classics import build_classics_kg, export_kg, load_kg


def test_build_stats_and_queries():
    kg = build_classics_kg()
    st = kg.stats()
    assert st["nodes"] > 100 and st["edges"] > 100
    for t in ("node:classic", "node:author", "node:formula", "node:dynasty", "node:category"):
        assert st[t] > 0
    # typed-relation queries
    assert kg.source_of_formula("麻黄汤") == "伤寒论"
    assert kg.source_of_formula("银翘散") == "温病条辨"
    assert kg.author_of("金匮要略") == "张仲景"
    assert kg.neighbors("伤寒论", "part_of") == ["伤寒杂病论"]
    assert kg.neighbors("叶桂", "dynasty") == ["清"]


def test_alias_map_and_metric_use_kg():
    amap = build_classics_kg().classics_alias_map()
    assert amap["金匮"] == "金匮要略" and amap["内经"] == "黄帝内经"
    # the classics_ontology metric pulls its aliases from the KG
    import asyncio
    from medeval import Sample, TaskType, Message, Prediction, Generation, create_metric
    s = Sample(id="s", task_type=TaskType.OPEN_QA, messages=[Message("u", "q")],
               reference={"reference": "出自《金匮要略》"})
    sc = asyncio.run(create_metric("classics_ontology").score(
        s, Prediction("s", Generation(text="该条文见于《金匮要略》。"), parsed="x")))
    assert sc.detail["gold_sources"] == ["金匮要略"] and sc.value == 1.0


def test_export_roundtrip_and_formats():
    kg = build_classics_kg()
    with tempfile.TemporaryDirectory() as d:
        paths = export_kg(kg, d)
        names = {p.name for p in paths}
        assert names == {"tcm_classics_kg.json", "tcm_classics_kg.ttl", "tcm_classics_kg.graphml"}
        # JSON roundtrip preserves the graph
        kg2 = load_kg(Path(d) / "tcm_classics_kg.json")
        assert len(kg2.nodes) == len(kg.nodes) and len(kg2.edges) == len(kg.edges)
        assert kg2.source_of_formula("桂枝汤") == "伤寒论"
        # Turtle has the prefixes + a known triple; GraphML is valid XML
        ttl = (Path(d) / "tcm_classics_kg.ttl").read_text(encoding="utf-8")
        assert "@prefix tcm:" in ttl and "tcm:authored_by" in ttl
        xml.dom.minidom.parseString((Path(d) / "tcm_classics_kg.graphml").read_text(encoding="utf-8"))


def test_shipped_artifact_exists():
    # the repo ships a prebuilt, downloadable KG
    assert (Path(medeval.__file__).resolve().parents[1] / "data/kg/tcm_classics_kg.json").exists()


if __name__ == "__main__":
    test_build_stats_and_queries()
    test_alias_map_and_metric_use_kg()
    test_export_roundtrip_and_formats()
    test_shipped_artifact_exists()
    print("OK: knowledge graph tests passed")
