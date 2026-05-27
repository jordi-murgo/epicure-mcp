"""cultural_profile: cosine similarity of an ingredient to cuisine directions."""

from __future__ import annotations

from typing import Any

from ..data_loader import get_bundle

DESCRIPTION = (
    "Use when the question is specifically cuisine-related: 'what "
    "cuisine does miso belong to?', 'how Japanese is this ingredient?', "
    "'rank ginger across cuisines'. Returns cosine similarity of one "
    "ingredient against all 8 macro-region cuisine directions "
    "(Japanese, East_Asian, Southeast_Asian, South_Asian, Latin_American, "
    "Mediterranean, Eastern_European, Western_Atlantic) in a single "
    "call. For non-cuisine 'where is X?' questions use where_on_atlas "
    "or closest_mode instead."
)

_CUISINE_KEYS = (
    "Japanese",
    "East_Asian",
    "Southeast_Asian",
    "South_Asian",
    "Latin_American",
    "Mediterranean",
    "Eastern_European",
    "Western_Atlantic",
)


def run(ingredient: str) -> dict[str, Any]:
    bundle = get_bundle()
    match = bundle.matcher.resolve(ingredient)
    if match is None:
        return {"error": f"Could not resolve ingredient '{ingredient}'"}

    row = bundle.ingredients.nid_to_row[match.node_id]
    vec = bundle.ingredients.normed[row]
    scores: list[tuple[str, dict[str, Any]]] = []
    for name in _CUISINE_KEYS:
        if name not in bundle.directions:
            continue
        d = bundle.directions[name]
        s = float(vec @ d)
        stats = bundle.direction_stats[name]
        scores.append(
            (
                name,
                {
                    "score": round(s, 4),
                    "range": {"p10": stats["p10"], "p90": stats["p90"]},
                },
            )
        )
    scores.sort(key=lambda kv: -kv[1]["score"])
    return {
        "resolved": match.name,
        "cuisines": {k: v for k, v in scores},
        "note": (
            "Cosine similarity to each cuisine direction. Use p10/p90 to "
            "judge whether a score is typical or extreme."
        ),
    }
