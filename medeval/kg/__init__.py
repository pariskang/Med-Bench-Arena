"""Knowledge graphs (经典文献本体). The classics KG is the single source of truth
for the ``classics_ontology`` metric's alias map and is exportable for download."""
from .tcm_classics import (
    KnowledgeGraph, build_classics_kg, classics_alias_map, export_kg, get_kg, load_kg,
)

__all__ = ["KnowledgeGraph", "build_classics_kg", "classics_alias_map",
           "export_kg", "get_kg", "load_kg"]
