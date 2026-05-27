"""In-process port of the Epicure ``/api/graph`` algorithm.

Builds a two-tiered pairing graph for one or more seed ingredients:

1. **Primary nodes**: top-k ingredients by cosine to the seed centroid,
   filtered for dietary compatibility, word-overlap with the seeds, and
   penalised for piling up identical categories (meat/fish, sweet, fat).
2. **Secondary nodes**: top-4 cosine neighbours of each
   ``mean(seeds, primary)`` centroid, with two follow-up batches to
   guarantee per-primary diversity and a globally capped total.
3. **Clusters**: primaries grouped by Jaccard overlap of their secondary
   sets (>=0.1).
4. **Bridges**: secondaries connected to two or more primaries.

The algorithm follows the new-epicure paper-branch implementation
verbatim (``src/api/routes/core/graph_routes.py``) except that the
upstream API's HTTP plumbing and logging are stripped out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data_loader import Bundle, IngredientData

JACCARD_THRESHOLD = 0.1

_MEAT_CATEGORIES = frozenset({"Meat", "Fish", "Seafood"})
_SWEET_CATEGORIES = frozenset({"Sweet"})
_FAT_CATEGORIES = frozenset({"Fat", "Nuts"})


def _categories_for(ing: IngredientData, node_id: int) -> set[str]:
    """Return the set of category strings for a node, supporting either a
    single primary category or a comma-joined ``categories`` field."""
    primary = ing.primary_category.get(node_id, "")
    if not primary or primary == "Unknown":
        primary = ing.food_group.get(node_id, "Unknown")
    cats = {c.strip() for c in primary.split(",") if c.strip()}
    if not cats:
        cats = {"Unknown"}
    return cats


def _category_penalty(
    candidate_cats: set[str],
    has_meat: bool,
    has_sweet: bool,
    has_fat: bool,
) -> float:
    penalty = 0.0
    if has_meat and (candidate_cats & _MEAT_CATEGORIES):
        penalty += 0.35 if len(candidate_cats) == 1 else 0.15
    if has_sweet and (candidate_cats & _SWEET_CATEGORIES):
        penalty += 0.30 if len(candidate_cats) == 1 else 0.10
    if has_fat and (candidate_cats & _FAT_CATEGORIES):
        penalty += 0.25 if len(candidate_cats) == 1 else 0.12
    return min(penalty, 0.60)


def _has_word_overlap(candidate_name: str, seed_names: list[str]) -> bool:
    candidate_words = set(candidate_name.lower().replace("_", " ").split())
    for seed in seed_names:
        seed_words = set(seed.lower().replace("_", " ").split())
        if candidate_words & seed_words:
            return True
    return False


def _meets_dietary(
    ing: IngredientData, node_id: int, is_vegan: bool, is_vegetarian: bool
) -> bool:
    if not (is_vegan or is_vegetarian):
        return True
    if is_vegan:
        return ing.is_vegan.get(node_id, False)
    return ing.is_vegetarian.get(node_id, False)


@dataclass
class _Primary:
    node_id: int
    row: int
    name: str
    primary_category: str
    similarity_to_center: float


@dataclass
class _SecondaryLink:
    source_node_id: int  # primary node id
    target_node_id: int
    target_name: str
    target_primary_category: str
    similarity: float


@dataclass
class PairingGraph:
    seeds: list[str]
    primaries: list[_Primary]
    secondary_links: list[_SecondaryLink]
    sec_ids_per_primary: dict[int, set[int]]


def _candidate_mask(
    ing: IngredientData,
    excluded_rows: set[int],
    is_vegan: bool,
    is_vegetarian: bool,
) -> np.ndarray:
    """Boolean mask over all ingredients excluding the seed rows and any
    rows whose dietary flags disqualify them."""
    n = ing.normed.shape[0]
    mask = np.ones(n, dtype=bool)
    if excluded_rows:
        mask[list(excluded_rows)] = False
    if is_vegan or is_vegetarian:
        flags = ing.is_vegan if is_vegan else ing.is_vegetarian
        for i, nid in enumerate(ing.node_ids):
            if mask[i] and not flags.get(int(nid), False):
                mask[i] = False
    return mask


def _scored_candidates(
    ing: IngredientData,
    query_vec: np.ndarray,
    excluded_rows: set[int],
    seed_names: list[str],
    is_vegan: bool,
    is_vegetarian: bool,
    has_meat: bool,
    has_sweet: bool,
    has_fat: bool,
) -> list[tuple[int, float]]:
    """Return ``(row_index, penalised_similarity)`` for every candidate
    that passes the dietary + word-overlap filters."""
    norm = float(np.linalg.norm(query_vec))
    q = query_vec / norm if norm > 0 else query_vec
    sims = ing.normed @ q
    out: list[tuple[int, float]] = []
    for i, raw in enumerate(sims):
        if i in excluded_rows:
            continue
        node_id = int(ing.node_ids[i])
        if not _meets_dietary(ing, node_id, is_vegan, is_vegetarian):
            continue
        name = str(ing.names[i])
        if _has_word_overlap(name, seed_names):
            continue
        cats = _categories_for(ing, node_id)
        penalty = _category_penalty(cats, has_meat, has_sweet, has_fat)
        out.append((i, float(raw) * (1.0 - penalty)))
    return out


def build_graph(
    bundle: Bundle,
    seed_rows: list[int],
    seed_names: list[str],
    *,
    is_vegan: bool = False,
    is_vegetarian: bool = False,
    max_primary_nodes: int = 8,
    max_secondary_nodes: int = 8,
) -> PairingGraph:
    """Pure-numpy port of the ``/api/graph`` algorithm.

    ``seed_rows`` are row indices into ``bundle.ingredients.normed``;
    ``seed_names`` are the resolved canonical names in the same order.
    """
    ing = bundle.ingredients

    seed_categories: set[str] = set()
    for r in seed_rows:
        seed_categories |= _categories_for(ing, int(ing.node_ids[r]))
    has_meat = bool(seed_categories & _MEAT_CATEGORIES)
    has_sweet = bool(seed_categories & _SWEET_CATEGORIES)
    has_fat = bool(seed_categories & _FAT_CATEGORIES)

    seed_vecs = ing.normed[seed_rows]
    combined_seed = seed_vecs.mean(axis=0)

    seed_set = set(seed_rows)
    primary_scored = _scored_candidates(
        ing,
        combined_seed,
        seed_set,
        seed_names,
        is_vegan,
        is_vegetarian,
        has_meat,
        has_sweet,
        has_fat,
    )
    primary_scored.sort(key=lambda x: -x[1])

    primaries: list[_Primary] = []
    for row, sim in primary_scored:
        if len(primaries) >= max_primary_nodes:
            break
        node_id = int(ing.node_ids[row])
        name = str(ing.names[row])
        primaries.append(
            _Primary(
                node_id=node_id,
                row=row,
                name=name,
                primary_category=ing.primary_category.get(node_id, "Unknown"),
                similarity_to_center=sim,
            )
        )

    excluded_for_secondaries = seed_set | {p.row for p in primaries}

    secondary_links: list[_SecondaryLink] = []
    sec_ids_per_primary: dict[int, set[int]] = {p.node_id: set() for p in primaries}
    chosen_secondary_rows: set[int] = set()

    def _record(primary: _Primary, target_row: int, similarity: float) -> None:
        target_nid = int(ing.node_ids[target_row])
        secondary_links.append(
            _SecondaryLink(
                source_node_id=primary.node_id,
                target_node_id=target_nid,
                target_name=str(ing.names[target_row]),
                target_primary_category=ing.primary_category.get(target_nid, "Unknown"),
                similarity=similarity,
            )
        )
        sec_ids_per_primary[primary.node_id].add(target_nid)
        chosen_secondary_rows.add(target_row)

    def _scored_for_primary(primary: _Primary, exclude_rows: set[int]) -> list[tuple[int, float]]:
        centroid = np.vstack([seed_vecs, ing.normed[primary.row : primary.row + 1]]).mean(axis=0)
        cand = _scored_candidates(
            ing,
            centroid,
            exclude_rows,
            seed_names,
            is_vegan,
            is_vegetarian,
            has_meat,
            has_sweet,
            has_fat,
        )
        cand.sort(key=lambda x: -x[1])
        return cand

    # Batch 1: top-4 per primary (shared targets allowed).
    for p in primaries:
        for row, sim in _scored_for_primary(p, excluded_for_secondaries)[:4]:
            _record(p, row, sim)

    # Batch 2: top up each primary to 4 unique connections.
    for p in primaries:
        already_connected = {
            link.target_node_id for link in secondary_links if link.source_node_id == p.node_id
        }
        # Number of unique-to-this-primary connections so far.
        shared_count = 0
        for nid in already_connected:
            sources = {
                lk.source_node_id for lk in secondary_links if lk.target_node_id == nid
            }
            if len(sources) > 1:
                shared_count += 1
        unique_needed = max(0, 4 - shared_count)
        if unique_needed == 0:
            continue
        candidates = _scored_for_primary(
            p,
            excluded_for_secondaries
            | {
                ing.nid_to_row[link.target_node_id]
                for link in secondary_links
                if link.source_node_id == p.node_id
            },
        )
        added = 0
        for row, sim in candidates:
            if added >= unique_needed:
                break
            _record(p, row, sim)
            added += 1

    # Batch 3: global fill until we hit max_secondary_nodes unique targets.
    while len(chosen_secondary_rows) < max_secondary_nodes and primaries:
        # Primary with fewest connections gets the next slot.
        primary_to_extend = min(
            primaries,
            key=lambda p: sum(
                1 for link in secondary_links if link.source_node_id == p.node_id
            ),
        )
        candidates = _scored_for_primary(
            primary_to_extend, excluded_for_secondaries | chosen_secondary_rows
        )
        if not candidates:
            break
        row, sim = candidates[0]
        _record(primary_to_extend, row, sim)

    return PairingGraph(
        seeds=seed_names,
        primaries=primaries,
        secondary_links=secondary_links,
        sec_ids_per_primary=sec_ids_per_primary,
    )


def _format_name(name: str) -> str:
    return name.replace("_", " ")


def format_graph_text(graph: PairingGraph, pairing_stats: dict[str, float]) -> str:
    """Render the prototype's text layout (CLUSTERS / CONNECTIONS / BRIDGES)."""
    lines: list[str] = [
        f"Pairing graph for: {', '.join(_format_name(s) for s in graph.seeds)}",
        (
            f"(reference scores: p10={pairing_stats['p10']:.2f}, "
            f"median={pairing_stats['median']:.2f}, "
            f"p90={pairing_stats['p90']:.2f})"
        ),
        "",
    ]

    clusters: list[list[_Primary]] = []
    assigned: set[int] = set()
    for p in graph.primaries:
        if p.node_id in assigned:
            continue
        cluster = [p]
        assigned.add(p.node_id)
        for q in graph.primaries:
            if q.node_id in assigned:
                continue
            sp = graph.sec_ids_per_primary.get(p.node_id, set())
            sq = graph.sec_ids_per_primary.get(q.node_id, set())
            union = sp | sq
            jaccard = len(sp & sq) / len(union) if union else 0.0
            if jaccard >= JACCARD_THRESHOLD:
                cluster.append(q)
                assigned.add(q.node_id)
        clusters.append(cluster)

    lines.append("CLUSTERS (primaries grouped by shared secondary connections):")
    for i, cluster in enumerate(clusters):
        cats = [p.primary_category for p in cluster]
        dominant = max(set(cats), key=cats.count) if cats else "Unknown"
        density = "dense" if len(cluster) > 1 else "isolated"
        members = ", ".join(
            f"{_format_name(p.name)} ({p.similarity_to_center:.3f})" for p in cluster
        )
        label = chr(ord("A") + i)
        plural = "s" if len(cluster) != 1 else ""
        lines.append(
            f"  {label} [{dominant} - {len(cluster)} node{plural}, {density}]: {members}"
        )

    if len(clusters) > 1:
        all_pairs_isolated = all(
            not (
                graph.sec_ids_per_primary.get(p.node_id, set())
                & graph.sec_ids_per_primary.get(q.node_id, set())
            )
            for ci, c1 in enumerate(clusters)
            for cj, c2 in enumerate(clusters)
            if ci < cj
            for p in c1
            for q in c2
        )
        if all_pairs_isolated:
            lines.append(
                "  -> Clusters share 0 secondaries - completely different flavor directions."
            )
    lines.append("")

    sec_links_per_primary: dict[int, list[_SecondaryLink]] = {
        p.node_id: [] for p in graph.primaries
    }
    for link in graph.secondary_links:
        sec_links_per_primary.setdefault(link.source_node_id, []).append(link)
    for pid in sec_links_per_primary:
        sec_links_per_primary[pid].sort(key=lambda lk: -lk.similarity)

    lines.append("CONNECTIONS (each primary -> its secondaries):")
    for p in graph.primaries:
        secs = sec_links_per_primary.get(p.node_id, [])
        sec_strs = [f"{_format_name(lk.target_name)} {lk.similarity:.2f}" for lk in secs]
        lines.append(f"  {_format_name(p.name)}: {', '.join(sec_strs)}")
    lines.append("")

    sec_to_primaries: dict[int, list[str]] = {}
    primary_name_by_id = {p.node_id: p.name for p in graph.primaries}
    for link in graph.secondary_links:
        sec_to_primaries.setdefault(link.target_node_id, []).append(
            primary_name_by_id.get(link.source_node_id, "?")
        )
    bridges = [
        (tid, pnames)
        for tid, pnames in sec_to_primaries.items()
        if len(set(pnames)) >= 2
    ]
    bridges.sort(key=lambda x: -len(set(x[1])))

    if bridges:
        lines.append("BRIDGES (secondaries connecting multiple primaries):")
        target_name_by_id = {lk.target_node_id: lk.target_name for lk in graph.secondary_links}
        n_primaries = len(graph.primaries)
        for tid, pnames in bridges:
            uniq = list(dict.fromkeys(pnames))
            lines.append(
                f"  {_format_name(target_name_by_id[tid])} -> "
                f"{', '.join(_format_name(n) for n in uniq)} "
                f"({len(uniq)} of {n_primaries} primaries)"
            )
    return "\n".join(lines)


def find_pairings(
    bundle: Bundle,
    ingredients: list[str],
    *,
    is_vegan: bool = False,
    is_vegetarian: bool = False,
    max_primary_nodes: int = 8,
    max_secondary_nodes: int = 8,
) -> dict[str, Any]:
    """Entry point used by the MCP tool.

    Returns a dict with either ``error`` or ``resolved`` + ``text``.
    """
    resolved_rows: list[int] = []
    resolved_names: list[str] = []
    unresolved: list[str] = []
    for raw in ingredients:
        m = bundle.matcher.resolve(raw)
        if m is None:
            unresolved.append(raw)
            continue
        resolved_rows.append(bundle.ingredients.nid_to_row[m.node_id])
        resolved_names.append(m.name)
    if not resolved_rows:
        return {"error": f"Could not resolve any ingredients: {ingredients}"}

    graph = build_graph(
        bundle,
        resolved_rows,
        resolved_names,
        is_vegan=is_vegan,
        is_vegetarian=is_vegetarian,
        max_primary_nodes=max_primary_nodes,
        max_secondary_nodes=max_secondary_nodes,
    )
    text = format_graph_text(graph, bundle.pairing_stats)
    return {
        "resolved": resolved_names,
        "unresolved": unresolved,
        "text": text,
    }
