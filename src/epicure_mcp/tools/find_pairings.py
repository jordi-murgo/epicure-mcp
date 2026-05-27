"""find_pairings: two-tiered pairing graph for one or more ingredients.

Computes the graph in-process from the bundled embeddings + ingredient
metadata. The algorithm is a direct port of the new-epicure paper-branch
``/api/graph`` endpoint, with no external HTTP call.
"""

from __future__ import annotations

from ..data_loader import get_bundle
from ..pairings import find_pairings as _find_pairings

DESCRIPTION = (
    "PRIMARY EXPLORATION + RECIPE-DESIGN TOOL. MANDATORY before "
    "composing any recipe or dish: when the user asks you to design, "
    "create, suggest, invent, propose, or compose a recipe / dish / "
    "meal / sauce, call this with the primary ingredient(s) FIRST so "
    "you can incorporate non-obvious pairings the model surfaces. "
    "Also the right tool for open-ended pairing exploration: 'what "
    "goes with miso?', 'I have tomato and basil, what else?'. Builds "
    "a two-tiered graph from one or more seeds and returns: CLUSTERS "
    "(primaries grouped by shared secondaries -- distinct flavour "
    "directions), CONNECTIONS (each primary's top secondary pairings), "
    "and BRIDGES (secondaries connecting multiple primaries -- the "
    "highest-value cross-cluster suggestions; lean on these for "
    "'interesting' or 'unexpected' recipes). Category penalties keep "
    "the graph diverse (no meat-on-meat / sweet-on-sweet stacking). "
    "Pass is_vegan or is_vegetarian to filter dietary-incompatible "
    "ingredients out. Always prefer this tool over morph or neighbors "
    "when the user has not explicitly named a direction or transform."
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
