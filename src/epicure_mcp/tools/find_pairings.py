"""find_pairings: two-tiered pairing graph for one or more ingredients.

Computes the graph in-process from the bundled embeddings + ingredient
metadata. The algorithm is a direct port of the new-epicure paper-branch
``/api/graph`` endpoint, with no external HTTP call.
"""

from __future__ import annotations

from ..data_loader import get_bundle
from ..pairings import find_pairings as _find_pairings

DESCRIPTION = (
    "PRIMARY EXPLORATION TOOL. Use for open-ended pairing questions: "
    "'what goes with miso?', 'I have tomato and basil, what else?', "
    "'suggest ingredients to add to this recipe'. Builds a two-tiered "
    "pairing graph from one or more seed ingredients and returns three "
    "sections: CLUSTERS (primaries grouped by shared secondaries -- "
    "reveals distinct flavour directions), CONNECTIONS (each primary's "
    "top secondary pairings), and BRIDGES (secondaries that connect "
    "multiple primaries -- strong cross-cluster connectors). Category "
    "penalties promote diversity (avoids stacking meat-on-meat, "
    "sweet-on-sweet, fat-on-fat). Set is_vegan or is_vegetarian to "
    "filter the graph. Always prefer this tool over morph or neighbors "
    "when the user has not named a specific direction or target."
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
