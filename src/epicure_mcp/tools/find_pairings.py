"""find_pairings: two-tiered pairing graph for one or more ingredients.

Computes the graph in-process from the bundled embeddings + ingredient
metadata. The algorithm is a direct port of the new-epicure paper-branch
``/api/graph`` endpoint, with no external HTTP call.
"""

from __future__ import annotations

from ..data_loader import get_bundle
from ..pairings import find_pairings as _find_pairings

DESCRIPTION = (
    "Find ingredients that pair best with one or more ingredients. "
    "Returns three sections: CLUSTERS (primaries grouped by shared "
    "secondaries), CONNECTIONS (each primary's secondary pairings), "
    "and BRIDGES (secondaries that connect multiple primaries). "
    "Category penalties (meat/sweet/fat stacking) promote diversity. "
    "Set is_vegan or is_vegetarian to filter dietary-incompatible "
    "ingredients out of the graph. Computed locally - no external API."
)


def run(
    ingredients: list[str] | str,
    *,
    is_vegan: bool = False,
    is_vegetarian: bool = False,
) -> str:
    if isinstance(ingredients, str):
        ingredients = [ingredients]
    bundle = get_bundle()
    result = _find_pairings(
        bundle,
        list(ingredients),
        is_vegan=is_vegan,
        is_vegetarian=is_vegetarian,
    )
    if "error" in result:
        return f"Error: {result['error']}"
    return result["text"]
