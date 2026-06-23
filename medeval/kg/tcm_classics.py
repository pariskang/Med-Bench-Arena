"""TCM classical-literature knowledge graph (经典文献本体 → 知识图谱).

A curated graph of the classics: 经典著作 (with 朝代/作者/别名/分类), 医家, 朝代,
经典名方 (with source classic), and 分类/学派 — linked by typed relations
(``authored_by`` / ``dynasty`` / ``part_of`` / ``from_source`` / ``category`` /
``aka``). Build it, export it (node-link JSON · RDF Turtle · GraphML) for download,
load it back, and query it. The ``classics_ontology`` metric derives its alias map
from this graph, so the KG is the single source of truth.
"""
from __future__ import annotations

import json
import xml.sax.saxutils as sx
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- curated source data ------------------------------------------------------
# classic: (dynasty, author, [aliases], category, part_of|None)
CLASSICS: dict[str, tuple[str, str, list[str], str, str | None]] = {
    "黄帝内经": ("战国秦汉", "托名黄帝", ["内经"], "医经", None),
    "素问": ("战国秦汉", "托名黄帝", ["黄帝内经素问"], "医经", "黄帝内经"),
    "灵枢": ("战国秦汉", "托名黄帝", ["灵枢经"], "医经", "黄帝内经"),
    "难经": ("秦汉", "托名扁鹊", ["黄帝八十一难经"], "医经", None),
    "神农本草经": ("秦汉", "托名神农", ["本草经", "本经"], "本草", None),
    "伤寒杂病论": ("东汉", "张仲景", [], "经方", None),
    "伤寒论": ("东汉", "张仲景", ["伤寒"], "经方", "伤寒杂病论"),
    "金匮要略": ("东汉", "张仲景", ["金匮"], "经方", "伤寒杂病论"),
    "脉经": ("西晋", "王叔和", [], "诊法", None),
    "针灸甲乙经": ("西晋", "皇甫谧", ["甲乙经"], "针灸", None),
    "诸病源候论": ("隋", "巢元方", ["病源"], "病机", None),
    "备急千金要方": ("唐", "孙思邈", ["千金要方", "千金方"], "方书", None),
    "千金翼方": ("唐", "孙思邈", [], "方书", None),
    "外台秘要": ("唐", "王焘", ["外台"], "方书", None),
    "太平惠民和剂局方": ("宋", "陈师文", ["和剂局方", "局方"], "方书", None),
    "小儿药证直诀": ("宋", "钱乙", [], "儿科", None),
    "三因极一病证方论": ("宋", "陈言", ["三因方"], "病因", None),
    "脾胃论": ("金", "李杲", [], "内伤", None),
    "内外伤辨惑论": ("金", "李杲", [], "内伤", None),
    "素问玄机原病式": ("金", "刘完素", [], "运气", None),
    "儒门事亲": ("金", "张从正", [], "攻邪", None),
    "丹溪心法": ("元", "朱震亨", [], "杂病", None),
    "格致余论": ("元", "朱震亨", [], "医论", None),
    "本草纲目": ("明", "李时珍", ["纲目"], "本草", None),
    "濒湖脉学": ("明", "李时珍", [], "诊法", None),
    "景岳全书": ("明", "张介宾", [], "综合", None),
    "温疫论": ("明", "吴有性", ["瘟疫论"], "温病", None),
    "医宗金鉴": ("清", "吴谦", [], "综合", None),
    "温热论": ("清", "叶桂", [], "温病", None),
    "温病条辨": ("清", "吴瑭", [], "温病", None),
    "湿热条辨": ("清", "薛雪", ["湿热病篇"], "温病", None),
    "医林改错": ("清", "王清任", [], "瘀血", None),
    "血证论": ("清", "唐宗海", [], "血证", None),
    "证治准绳": ("明", "王肯堂", [], "综合", None),
    "医学心悟": ("清", "程国彭", [], "综合", None),
}

# author -> (dynasty, [aliases/字号])
AUTHORS: dict[str, tuple[str, list[str]]] = {
    "张仲景": ("东汉", ["张机", "医圣"]), "王叔和": ("西晋", ["王熙"]),
    "皇甫谧": ("西晋", []), "巢元方": ("隋", []), "孙思邈": ("唐", ["药王"]),
    "王焘": ("唐", []), "钱乙": ("宋", ["钱仲阳"]), "陈言": ("宋", ["陈无择"]),
    "李杲": ("金", ["李东垣", "东垣"]), "刘完素": ("金", ["刘河间", "河间"]),
    "张从正": ("金", ["张子和", "子和"]), "朱震亨": ("元", ["朱丹溪", "丹溪"]),
    "李时珍": ("明", ["李东璧", "濒湖"]), "张介宾": ("明", ["张景岳", "景岳"]),
    "吴有性": ("明", ["吴又可"]), "吴谦": ("清", []), "叶桂": ("清", ["叶天士", "天士"]),
    "吴瑭": ("清", ["吴鞠通", "鞠通"]), "薛雪": ("清", ["薛生白"]),
    "王清任": ("清", ["王勋臣"]), "唐宗海": ("清", ["唐容川"]),
    "王肯堂": ("明", []), "程国彭": ("清", ["程钟龄"]),
}

# 经典名方 -> source classic
FORMULAS: dict[str, str] = {
    "桂枝汤": "伤寒论", "麻黄汤": "伤寒论", "小柴胡汤": "伤寒论", "白虎汤": "伤寒论",
    "四逆汤": "伤寒论", "理中丸": "伤寒论", "大承气汤": "伤寒论", "五苓散": "伤寒论",
    "炙甘草汤": "伤寒论", "肾气丸": "金匮要略", "大黄牡丹汤": "金匮要略",
    "麦门冬汤": "金匮要略", "当归芍药散": "金匮要略", "六味地黄丸": "小儿药证直诀",
    "补中益气汤": "脾胃论", "清暑益气汤": "脾胃论", "银翘散": "温病条辨",
    "桑菊饮": "温病条辨", "清营汤": "温病条辨", "安宫牛黄丸": "温病条辨",
    "血府逐瘀汤": "医林改错", "补阳还五汤": "医林改错", "越鞠丸": "丹溪心法",
    "逍遥散": "太平惠民和剂局方", "四君子汤": "太平惠民和剂局方",
}


# --- graph types --------------------------------------------------------------
@dataclass
class Node:
    id: str
    type: str
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    relation: str


class KnowledgeGraph:
    NS = "http://medeval.org/tcm/"

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, id: str, type: str, label: str | None = None, **attrs: Any) -> None:
        if id in self.nodes:
            self.nodes[id].attrs.update(attrs)
        else:
            self.nodes[id] = Node(id, type, label or id, dict(attrs))

    def add_edge(self, source: str, target: str, relation: str) -> None:
        self.edges.append(Edge(source, target, relation))

    # --- queries ---
    def neighbors(self, node_id: str, relation: str | None = None) -> list[str]:
        return [e.target for e in self.edges
                if e.source == node_id and (relation is None or e.relation == relation)]

    def by_type(self, type: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.type == type]

    def source_of_formula(self, formula: str) -> str | None:
        ns = self.neighbors(formula, "from_source")
        return ns[0] if ns else None

    def author_of(self, classic: str) -> str | None:
        ns = self.neighbors(classic, "authored_by")
        return ns[0] if ns else None

    def classics_alias_map(self) -> dict[str, str]:
        """alias/label -> canonical classic id (for the classics_ontology metric)."""
        out: dict[str, str] = {}
        for n in self.by_type("classic"):
            out[n.label] = n.id
            for a in n.attrs.get("aliases", []):
                out[a] = n.id
        return out

    def stats(self) -> dict[str, int]:
        from collections import Counter
        nt = Counter(n.type for n in self.nodes.values())
        et = Counter(e.relation for e in self.edges)
        return {"nodes": len(self.nodes), "edges": len(self.edges),
                **{f"node:{k}": v for k, v in nt.items()},
                **{f"rel:{k}": v for k, v in et.items()}}

    # --- exports ---
    def to_node_link(self) -> dict[str, Any]:
        return {
            "directed": True, "namespace": self.NS,
            "nodes": [{"id": n.id, "type": n.type, "label": n.label, **n.attrs}
                      for n in self.nodes.values()],
            "links": [{"source": e.source, "target": e.target, "relation": e.relation}
                      for e in self.edges],
        }

    def to_turtle(self) -> str:
        def iri(x: str) -> str:
            return f"<{self.NS}{x}>"
        lines = [f"@prefix tcm: <{self.NS}> .",
                 "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                 "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .", ""]
        for n in self.nodes.values():
            lines.append(f'{iri(n.id)} rdf:type tcm:{n.type} ;')
            lines.append(f'    rdfs:label "{n.label}" ;')
            for k, v in n.attrs.items():
                if isinstance(v, list):
                    for item in v:
                        lines.append(f'    tcm:{k} "{item}" ;')
                else:
                    lines.append(f'    tcm:{k} "{v}" ;')
            for e in (e for e in self.edges if e.source == n.id):
                lines.append(f'    tcm:{e.relation} {iri(e.target)} ;')
            lines[-1] = lines[-1][:-2] + "."  # close the statement
        return "\n".join(lines) + "\n"

    def to_graphml(self) -> str:
        q = sx.quoteattr
        out = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
               '<key id="type" for="node" attr.name="type" attr.type="string"/>',
               '<key id="label" for="node" attr.name="label" attr.type="string"/>',
               '<key id="relation" for="edge" attr.name="relation" attr.type="string"/>',
               '<graph edgedefault="directed">']
        for n in self.nodes.values():
            out.append(f'<node id={q(n.id)}><data key="type">{sx.escape(n.type)}</data>'
                       f'<data key="label">{sx.escape(n.label)}</data></node>')
        for i, e in enumerate(self.edges):
            out.append(f'<edge id="e{i}" source={q(e.source)} target={q(e.target)}>'
                       f'<data key="relation">{sx.escape(e.relation)}</data></edge>')
        out += ["</graph>", "</graphml>"]
        return "\n".join(out) + "\n"


# --- builder ------------------------------------------------------------------
def build_classics_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for author, (dyn, aliases) in AUTHORS.items():
        kg.add_node(author, "author", aliases=aliases)
        kg.add_node(dyn, "dynasty")
        kg.add_edge(author, dyn, "dynasty")
    for c, (dyn, author, aliases, cat, part_of) in CLASSICS.items():
        kg.add_node(c, "classic", dynasty=dyn, category=cat, aliases=aliases)
        kg.add_node(dyn, "dynasty")
        kg.add_node(cat, "category")
        kg.add_edge(c, dyn, "dynasty")
        kg.add_edge(c, cat, "category")
        if author not in kg.nodes:
            kg.add_node(author, "author", aliases=[])
        kg.add_edge(c, author, "authored_by")
        if part_of:
            kg.add_edge(c, part_of, "part_of")
    for formula, source in FORMULAS.items():
        kg.add_node(formula, "formula")
        kg.add_edge(formula, source, "from_source")
    return kg


def export_kg(kg: KnowledgeGraph, out_dir: str | Path, formats: list[str] | None = None
              ) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    formats = formats or ["json", "turtle", "graphml"]
    written: list[Path] = []
    if "json" in formats:
        p = out / "tcm_classics_kg.json"
        p.write_text(json.dumps(kg.to_node_link(), ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p)
    if "turtle" in formats:
        p = out / "tcm_classics_kg.ttl"
        p.write_text(kg.to_turtle(), encoding="utf-8")
        written.append(p)
    if "graphml" in formats:
        p = out / "tcm_classics_kg.graphml"
        p.write_text(kg.to_graphml(), encoding="utf-8")
        written.append(p)
    return written


def load_kg(path: str | Path) -> KnowledgeGraph:
    """Load a node-link JSON KG back into a :class:`KnowledgeGraph`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    kg = KnowledgeGraph()
    for n in data["nodes"]:
        nid, typ, label = n["id"], n.get("type", ""), n.get("label", n["id"])
        attrs = {k: v for k, v in n.items() if k not in ("id", "type", "label")}
        kg.add_node(nid, typ, label, **attrs)
    for e in data["links"]:
        kg.add_edge(e["source"], e["target"], e["relation"])
    return kg


# the metric imports this; build once, cache
_KG: KnowledgeGraph | None = None


def get_kg() -> KnowledgeGraph:
    global _KG
    if _KG is None:
        _KG = build_classics_kg()
    return _KG


def classics_alias_map() -> dict[str, str]:
    return get_kg().classics_alias_map()
