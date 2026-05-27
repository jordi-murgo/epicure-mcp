"""ingredient_on_factor: signed projection of an ingredient onto an ICA factor."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data_loader import get_bundle

DESCRIPTION = (
    "Project an ingredient onto a residualised ICA factor. Returns the "
    "signed projection (positive = toward pole A, negative = toward "
    "pole B), the matching pole context (Claude label + top "
    "ingredients), and the projection percentile across all "
    "ingredients."
)


def run(ingredient: str, factor: int) -> dict[str, Any]:
    bundle = get_bundle()
    if bundle.factors is None:
        return {"error": "No factor data bundled."}
    if factor < 0 or factor >= bundle.factors.directions.shape[0]:
        return {
            "error": f"factor must be in [0, {bundle.factors.directions.shape[0] - 1}].",
        }

    m = bundle.matcher.resolve(ingredient)
    if m is None:
        return {"error": f"Could not resolve ingredient '{ingredient}'"}

    direction = bundle.factors.directions[factor]
    projections = bundle.ingredients.normed @ direction
    row = bundle.ingredients.nid_to_row[m.node_id]
    value = float(projections[row])
    rank = int(np.sum(projections < value))
    percentile = round(100.0 * rank / max(1, projections.size - 1), 1)
    side = "a" if value >= 0 else "b"
    entry = bundle.factors.labels.get(factor, {})
    return {
        "ingredient": m.name,
        "factor": factor,
        "projection": round(value, 4),
        "side": side,
        "percentile": percentile,
        "axis": entry.get("axis", {}),
        "pole_a": {
            "label": (entry.get("pole_a") or {}).get("label"),
            "top_ingredients": entry.get("pole_a_top", []),
        },
        "pole_b": {
            "label": (entry.get("pole_b") or {}).get("label"),
            "top_ingredients": entry.get("pole_b_top", []),
        },
    }
