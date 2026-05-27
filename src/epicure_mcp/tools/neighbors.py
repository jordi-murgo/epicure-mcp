"""neighbors: top-k cosine neighbours of an ingredient."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data_loader import get_bundle

DESCRIPTION = (
    "Use when the user asks what is similar to one specific ingredient "
    "('what's like miso?', 'similar to gochujang'). Returns the top-k "
    "cosine-nearest ingredients to a single seed -- no graph structure, "
    "no clustering. For multi-seed pairing exploration ('what goes with "
    "miso AND ginger?') use find_pairings instead, which handles seed "
    "centroids, dietary filters, and category penalties."
)


def run(ingredient: str, top_k: int = 5) -> dict[str, Any]:
    bundle = get_bundle()
    m = bundle.matcher.resolve(ingredient)
    if m is None:
        return {"error": f"Could not resolve ingredient '{ingredient}'"}
    row = bundle.ingredients.nid_to_row[m.node_id]
    sims = bundle.ingredients.normed @ bundle.ingredients.normed[row]
    sims[row] = -np.inf
    k = max(1, int(top_k))
    order = np.argsort(-sims)[:k]
    neighbors = [
        {
            "name": str(bundle.ingredients.names[int(i)]),
            "sim": round(float(sims[int(i)]), 4),
            "rank": rank + 1,
        }
        for rank, i in enumerate(order)
    ]
    return {"ingredient": m.name, "neighbors": neighbors}
