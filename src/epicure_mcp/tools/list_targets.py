"""list_targets: enumerate every valid `morph` target.

Returns supervised directions (cuisine, sensory, nutrient, NOVA, diet)
side-by-side with emergent GMM modes, plus a primer on the
``angle_deg`` parameter.
"""

from __future__ import annotations

from typing import Any

from ..data_loader import get_bundle

DESCRIPTION = (
    "Use before any call to morph or compare_on_axis with a free-text "
    "axis or target you have not already verified. Enumerates every "
    "valid target: supervised directions (cuisines, cf_* sensory "
    "descriptors, usda_* nutrients, NOVA, diet) and emergent GMM mode "
    "poles. Each entry includes a one-line description. The response "
    "also explains how the `angle_deg` parameter works on morph. Filter "
    "with kind='direction' or kind='mode' if you only need one family."
)

ANGLE_DEG_PRIMER = (
    "`angle_deg` controls how far the seed is rotated toward the "
    "target on the unit sphere. Cosine-to-seed = cos(angle_deg). "
    "Useful guideposts: "
    "0deg = seed unchanged (cos=1.00); "
    "15deg = subtle nudge (cos=0.97); "
    "30deg = mild morph (cos=0.87); "
    "45deg = balanced morph (cos=0.71); "
    "60deg = strong morph (cos=0.50); "
    "90deg = orthogonal - the seed's identity is dropped entirely (cos=0)."
)


_DIRECTION_FAMILY_HINTS = {
    "cf_": "Sensory descriptor (Cooks Foundry tags).",
    "usda_": "USDA nutrient axis.",
}

_CUISINE_KEYS = {
    "Japanese",
    "East_Asian",
    "Southeast_Asian",
    "South_Asian",
    "Latin_American",
    "Mediterranean",
    "Eastern_European",
    "Western_Atlantic",
}


def _describe_direction(name: str) -> str:
    if name in _CUISINE_KEYS:
        return f"Cuisine direction for {name.replace('_', ' ')}."
    if name == "nova":
        return "NOVA processing axis (1 = whole foods, 4 = ultra-processed)."
    if name == "diet":
        return "Diet axis (animal-derived -> plant-derived)."
    for prefix, hint in _DIRECTION_FAMILY_HINTS.items():
        if name.startswith(prefix):
            return hint
    return "Supervised direction extracted from the embedding space."


def run(kind: str | None = None) -> dict[str, Any]:
    bundle = get_bundle()

    directions_payload: list[dict[str, Any]] = []
    if kind in (None, "direction"):
        for name in sorted(bundle.directions.keys()):
            stats = bundle.direction_stats.get(name, {})
            directions_payload.append(
                {
                    "kind": "direction",
                    "name": name,
                    "description": _describe_direction(name),
                    "p10": stats.get("p10"),
                    "p90": stats.get("p90"),
                }
            )

    modes_payload: list[dict[str, Any]] = []
    if kind in (None, "mode") and bundle.modes is not None:
        for prop in bundle.modes.properties:
            for mode in bundle.modes.by_property.get(prop, []):
                modes_payload.append(
                    {
                        "kind": "mode",
                        "property": prop,
                        "mode_id": mode.mode_id,
                        "label": mode.label,
                        "size": mode.size,
                        "dominant_cuisine": mode.dominant_cuisine,
                        "dominant_food_group": mode.dominant_food_group,
                    }
                )

    return {
        "angle_deg_primer": ANGLE_DEG_PRIMER,
        "directions": directions_payload,
        "modes": modes_payload,
        "summary": {
            "n_directions": len(directions_payload),
            "n_modes": len(modes_payload),
        },
    }
