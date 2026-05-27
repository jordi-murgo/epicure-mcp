"""list_factors: ICA factor catalogue with Claude-labelled poles."""

from __future__ import annotations

from typing import Any

from ..data_loader import get_bundle

DESCRIPTION = (
    "Use to browse the 20 emergent flavour factors before calling "
    "ingredient_on_factor or pareto_navigate ('what factors exist in "
    "this model?', 'show me the high-coherence flavour axes'). Each "
    "factor exposes a named axis (e.g. 'Indonesian Culinary Identity') "
    "plus pole_a / pole_b Claude labels, coherence rating, and the "
    "top-10 anchoring ingredients per pole. Filter with min_coherence "
    "('high' | 'moderate' | 'low')."
)

_COHERENCE_RANK = {"high": 3, "moderate": 2, "low": 1, "incoherent": 0}


def _pole_payload(pole: dict[str, Any], top_names: list[str]) -> dict[str, Any]:
    return {
        "label": pole.get("label"),
        "coherence": pole.get("coherence"),
        "themes": pole.get("themes", []),
        "top_ingredients": top_names,
    }


def _meets(coherence: str | None, threshold: int) -> bool:
    if coherence is None:
        return False
    return _COHERENCE_RANK.get(str(coherence).lower(), 0) >= threshold


def run(min_coherence: str = "low") -> dict[str, Any]:
    bundle = get_bundle()
    if bundle.factors is None:
        return {"error": "No factor data bundled."}

    threshold = _COHERENCE_RANK.get(min_coherence.lower(), 1)
    factors_out: list[dict[str, Any]] = []
    for idx in sorted(bundle.factors.labels.keys()):
        entry = bundle.factors.labels[idx]
        pole_a = entry.get("pole_a") or {}
        pole_b = entry.get("pole_b") or {}
        coh_a = pole_a.get("coherence")
        coh_b = pole_b.get("coherence")
        if not (_meets(coh_a, threshold) or _meets(coh_b, threshold)):
            continue
        factors_out.append(
            {
                "factor": idx,
                "axis": entry.get("axis", {}),
                "pole_a": _pole_payload(pole_a, entry.get("pole_a_top", [])),
                "pole_b": _pole_payload(pole_b, entry.get("pole_b_top", [])),
                "secondary_lens": entry.get("secondary_lens"),
            }
        )

    return {
        "model": "cooc",
        "method": "ica",
        "n": bundle.factors.directions.shape[0],
        "min_coherence": min_coherence,
        "factors": factors_out,
    }
