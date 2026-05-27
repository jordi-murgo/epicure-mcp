"""where_on_atlas: precomputed UMAP coordinates + nearest-in-2D neighbours."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data_loader import get_bundle

DESCRIPTION = (
    "Use when the user wants visual / spatial context for an ingredient "
    "('where does miso sit on the atlas?', 'what surrounds saffron "
    "visually?'). Returns the precomputed 2-D UMAP coordinate (x, y) "
    "plus the closest neighbours in 2-D space with their cuisine and "
    "food-group labels -- useful for describing the local neighbourhood. "
    "For 'what named flavour cluster is X in?' use closest_mode instead "
    "(named regions, not coordinates)."
)


def run(ingredient: str, top_k_neighbors_2d: int = 5) -> dict[str, Any]:
    bundle = get_bundle()
    if bundle.atlas is None:
        return {"error": "No UMAP atlas bundled."}

    m = bundle.matcher.resolve(ingredient)
    if m is None:
        return {"error": f"Could not resolve ingredient '{ingredient}'"}

    if m.name not in bundle.atlas.name_to_row:
        return {
            "error": f"Atlas coordinates missing for '{m.name}'.",
        }
    row_atlas = bundle.atlas.name_to_row[m.name]
    xy = bundle.atlas.coords[row_atlas]
    deltas = bundle.atlas.coords - xy
    dists = np.linalg.norm(deltas, axis=1)
    dists[row_atlas] = np.inf
    k = max(1, int(top_k_neighbors_2d))
    order = np.argsort(dists)[:k]

    inv_atlas = {i: n for n, i in bundle.atlas.name_to_row.items()}
    neighbors_2d: list[dict[str, Any]] = []
    for i in order:
        name = inv_atlas.get(int(i), "?")
        nid = bundle.ingredients.name_to_row.get(name)
        cuisine = None
        food_group = None
        if nid is not None:
            node_id = int(bundle.ingredients.node_ids[nid])
            cuisine = bundle.ingredients.cuisine_region.get(node_id)
            food_group = bundle.ingredients.food_group.get(node_id)
        neighbors_2d.append(
            {
                "name": name,
                "dist_2d": round(float(dists[int(i)]), 4),
                "cuisine_region": cuisine,
                "food_group": food_group,
            }
        )

    node_id = m.node_id
    return {
        "ingredient": m.name,
        "atlas_xy": [round(float(xy[0]), 4), round(float(xy[1]), 4)],
        "cuisine_region": bundle.ingredients.cuisine_region.get(node_id),
        "food_group": bundle.ingredients.food_group.get(node_id),
        "neighbors_in_2d": neighbors_2d,
    }
