"""flavour_correlations: which named axes correlate with each other."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data_loader import get_bundle

DESCRIPTION = (
    "Use when the user asks about the global structure of the flavour "
    "space ('what correlates with sweetness?', 'is umami the same as "
    "salty?', 'explain the trade-offs in this embedding'). Returns "
    "cosine between every pair of axis vectors; positive = correlated, "
    "negative = anti-correlated. Only axis pairs with |r| > 0.3 are "
    "returned. Useful for explaining substitution trade-offs (e.g. "
    "moving sweeter usually moves more-processed too)."
)


def run() -> dict[str, Any]:
    bundle = get_bundle()
    names = sorted(bundle.directions.keys())
    if not names:
        return {"notable_correlations": [], "note": "No supervised directions bundled."}
    vecs = np.stack([bundle.directions[n] for n in names])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = vecs / norms
    corr = normed @ normed.T

    notable: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = float(corr[i, j])
            if abs(c) > 0.3:
                notable.append(
                    {
                        "axis_a": names[i],
                        "axis_b": names[j],
                        "correlation": round(c, 3),
                    }
                )
    notable.sort(key=lambda x: -abs(x["correlation"]))
    return {
        "n_axes": len(names),
        "notable_correlations": notable,
        "note": (
            "Cosine between unit-direction vectors. Use to understand "
            "trade-offs (e.g. sweet vs nova) when substituting ingredients."
        ),
    }
