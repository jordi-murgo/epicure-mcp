"""closest_mode: which named region of the atlas does the ingredient live in?"""

from __future__ import annotations

from typing import Any

from ..data_loader import get_bundle

DESCRIPTION = (
    "Use when the user asks what named flavour region an ingredient "
    "belongs to ('what family is miso in?', 'what cluster does kimchi "
    "fit?'). Returns the top-k emergent GMM modes that best describe "
    "the ingredient, each with a Claude-labelled name (e.g. 'East Asian "
    "umami pantry staples'), cosine similarity to the mode's centroid, "
    "and the mode's top members. Pass property=... to restrict to one "
    "axis family (cuisine, sensory, etc.). For 2-D visual placement use "
    "where_on_atlas instead; for cuisine-only questions use "
    "cultural_profile."
)


def run(ingredient: str, property: str | None = None, top_k: int = 3) -> dict[str, Any]:
    bundle = get_bundle()
    if bundle.modes is None:
        return {"error": "No mode data bundled."}
    m = bundle.matcher.resolve(ingredient)
    if m is None:
        return {"error": f"Could not resolve ingredient '{ingredient}'"}

    row = bundle.ingredients.nid_to_row[m.node_id]
    vec = bundle.ingredients.normed[row]
    sims = bundle.modes.pole_vectors @ vec

    candidates: list[tuple[int, str, int, float]] = []
    for (prop, mid), idx in bundle.modes.pole_index.items():
        if property is not None and prop != property:
            continue
        candidates.append((idx, prop, mid, float(sims[idx])))
    if not candidates:
        return {
            "resolved": m.name,
            "modes": [],
            "note": "No modes matched the requested filter.",
        }
    candidates.sort(key=lambda x: -x[3])
    k = max(1, int(top_k))
    selected = candidates[:k]

    modes_payload: list[dict[str, Any]] = []
    for _idx, prop, mid, score in selected:
        modes_list = bundle.modes.by_property.get(prop, [])
        mode = next((mm for mm in modes_list if mm.mode_id == mid), None)
        if mode is None:
            continue
        modes_payload.append(
            {
                "property": prop,
                "mode_id": mid,
                "label": mode.label,
                "cos_to_pole": round(score, 4),
                "size": mode.size,
                "dominant_cuisine": mode.dominant_cuisine,
                "dominant_food_group": mode.dominant_food_group,
                "top_members": mode.member_names[:10],
            }
        )

    return {
        "resolved": m.name,
        "property_filter": property,
        "modes": modes_payload,
    }
