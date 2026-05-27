"""Matcher unit tests against the synthetic mini bundle."""

from __future__ import annotations

from pathlib import Path

from epicure_mcp.matcher import Matcher, normalize


def _build(mini: Path) -> Matcher:
    return Matcher(mini / "ingredient_list.csv", mini / "consolidated_nodes.csv")


def test_normalize() -> None:
    assert normalize("Soy_Sauce") == "soy sauce"
    assert normalize("  Olive-Oil  ") == "olive oil"


def test_exact_match(mini_data_dir: Path) -> None:
    m = _build(mini_data_dir)
    r = m.resolve("miso")
    assert r is not None
    assert r.name == "miso"
    assert r.method == "vocab"
    assert r.score == 1.0


def test_modifier_stripped_match(mini_data_dir: Path) -> None:
    m = _build(mini_data_dir)
    r = m.resolve("dried miso")
    assert r is not None
    assert r.name == "miso"
    assert r.method in {"vocab_stripped", "substring"}


def test_variant_lookup_via_consolidation(mini_data_dir: Path) -> None:
    m = _build(mini_data_dir)
    r = m.resolve("fresh ginger")
    assert r is not None
    assert r.name == "ginger"


def test_unknown_returns_none(mini_data_dir: Path) -> None:
    m = _build(mini_data_dir)
    assert m.resolve("nonexistent_xyz") is None


def test_empty_returns_none(mini_data_dir: Path) -> None:
    m = _build(mini_data_dir)
    assert m.resolve("") is None
