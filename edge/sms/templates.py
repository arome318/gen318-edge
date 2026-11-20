"""Load and render SMS templates (one YAML file per language)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

TEMPLATE_DIR = Path(__file__).parent / "templates"
FALLBACK = "en"
MAX_CHARS = 306  # two concatenated GSM-7 SMS segments


class Templates:
    def __init__(self, directory: Path = TEMPLATE_DIR) -> None:
        self._langs: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.yaml")):
            self._langs[path.stem] = yaml.safe_load(path.read_text()) or {}
        if FALLBACK not in self._langs:
            raise RuntimeError("English templates (en.yaml) are required")

    @property
    def languages(self) -> list[str]:
        return sorted(self._langs)

    def _lookup(self, lang: str, *path: str) -> Any:
        for candidate in (lang, FALLBACK):
            node: Any = self._langs.get(candidate, {})
            for part in path:
                node = node.get(part) if isinstance(node, dict) else None
            if node is not None:
                return node
        raise KeyError("/".join(path))

    def render(self, lang: str, *path: str, **values: Any) -> str:
        text = str(self._lookup(lang, *path)).format(**values)
        text = text.encode("ascii", "replace").decode("ascii")
        return text[:MAX_CHARS]

    def raw(self, lang: str, *path: str) -> Any:
        return self._lookup(lang, *path)
